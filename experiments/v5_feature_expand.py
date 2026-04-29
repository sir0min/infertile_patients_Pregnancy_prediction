"""
IVF 임신 성공 예측 - Solution v5 (Feature-Augmented Stacking)
================================================================
v4 -> v5 핵심 개선:
  1. 클래스별 최적 가중치 블렌딩 (Optuna 3D: w_lgb, w_xgb, w_cat)
     - v3 연구에서 최적 비율이 LGB<<XGB<CAT 임을 확인
     - 단순 평균(0.74018)보다 높은 OOF AUC 기대
  2. Feature-Augmented Meta-LGB
     - 9개 OOF 예측 + 상위 40개 원본 피처를 동시 입력
     - 메타 러너가 "어떤 환경에서 어떤 모델이 유리한지" 학습
  3. 3-way 최종 블렌딩: Opt-Blend + FA-Meta-LGB + Meta-LR

목표: OOF AUC 0.7405+ (v4 베이스라인 0.74018 초과)
"""

import io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocess import load_data, preprocess_full

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET  = "임신 성공 여부"
ID_COL  = "ID"
N_FOLDS = 5
SEEDS   = [42, 777, 2024]
SEED    = 42

# ── 베이스 모델 파라미터 ─────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "device": "gpu",
    "learning_rate": 0.05, "num_leaves": 127, "max_depth": 8,
    "min_child_samples": 20, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "tree_method": "hist", "device": "cuda",
    "learning_rate": 0.05, "max_depth": 6,
    "min_child_weight": 5, "subsample": 0.8,
    "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "n_estimators": 2000, "n_jobs": -1, "verbosity": 0,
    "early_stopping_rounds": 100,
}

CAT_PARAMS = {
    "iterations": 2000, "learning_rate": 0.05, "depth": 7,
    "l2_leaf_reg": 5, "subsample": 0.8,
    "eval_metric": "AUC", "early_stopping_rounds": 100,
    "verbose": False, "bootstrap_type": "Bernoulli", "task_type": "GPU",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 피처 엔지니어링 (v4 동일)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    return (num / denom.where(denom > 0, other=np.nan)).fillna(0)


def feature_engineering_v4(df: pd.DataFrame, fe_stats: dict = None) -> tuple:
    df = df.copy()
    s = dict(fe_stats) if fe_stats else {}

    gen    = df["총 생성 배아 수"]
    trans  = df["이식된 배아 수"]
    stored = df["저장된 배아 수"]
    thawed = df["해동된 배아 수"]
    mixed  = df["혼합된 난자 수"]
    fresh  = df["수집된 신선 난자 수"]
    icsi_e = df["미세주입된 난자 수"]
    icsi_b = df["미세주입에서 생성된 배아 수"]
    icsi_t = df["미세주입 배아 이식 수"]
    partner= df["파트너 정자와 혼합된 난자 수"]
    stored_fresh = df["저장된 신선 난자 수"]
    frozen = df["동결 배아 사용 여부"]
    fresh_ = df["신선 배아 사용 여부"]

    et_day  = df["배아 이식 경과일"]
    mix_day = df["난자 혼합 경과일"]

    age         = df["시술 당시 나이"].clip(lower=0)
    t_proc      = df["총 시술 횟수"].clip(lower=0)
    t_preg      = df["총 임신 횟수"].clip(lower=0)
    ivf_p       = df["IVF 시술 횟수"].clip(lower=0)
    ivf_pg      = df["IVF 임신 횟수"].clip(lower=0)
    t_brth      = df["총 출산 횟수"].clip(lower=0)
    clinic_proc = df["클리닉 내 총 시술 횟수"].clip(lower=0)

    age_raw    = df["시술 당시 나이"]
    t_proc_raw = df["총 시술 횟수"]
    ivf_pg_raw = df["IVF 임신 횟수"]
    t_brth_raw = df["총 출산 횟수"]

    # [v2] 7개
    df["이식_효율"]        = _safe_div(trans, gen)
    df["과거_임신_성공률"] = t_preg / (t_proc + 1)
    df["수정_효율"]        = _safe_div(gen, mixed)
    df["나이x난자수"]      = age * fresh

    if "embryo_max" not in s:
        s["embryo_max"] = max(gen.max() + 1, 9)
    df["배아_풍요도"] = pd.cut(
        gen, bins=[-1, 0, 3, 7, s["embryo_max"]], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    if "egg_median" not in s:
        s["egg_median"] = fresh.median()
    df["고령저반응"]   = ((age_raw >= 3) & (fresh < s["egg_median"])).astype(int)
    df["순수동결_주기"] = ((frozen == 1) & (fresh_ == 0)).astype(int)

    # [v3] 20개
    df["순수신선_주기"]   = ((fresh_ == 1) & (frozen == 0)).astype(int)
    df["첫_시술"]        = (t_proc_raw == 0).astype(int)
    df["반복_시술"]      = (t_proc_raw >= 3).astype(int)
    df["IVF_임신률"]     = ivf_pg / (ivf_p + 1)
    df["출산률"]         = t_brth / (t_proc + 1)
    df["ICSI_생존율"]    = _safe_div(icsi_b, icsi_e)
    df["배아_활용률"]    = _safe_div(trans + stored, gen)
    df["동결_활용률"]    = _safe_div(thawed, stored)
    df["잉여배아_비율"]  = stored / (trans + 1)
    df["이식수_카테고리"] = pd.cut(
        trans, bins=[-1, 0, 1, 2, float("inf")], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)
    df["나이x시술횟수"]  = age * t_proc
    df["나이x배아수"]    = age * gen
    df["자극반응_지수"]  = fresh / (age + 2)
    df["배아품질_복합지수"] = df["ICSI_생존율"] * df["이식_효율"]
    df["ICSI_이식비율"]  = _safe_div(icsi_t, trans)
    df["기증난자_사용"]  = (df["난자 출처"] == 1).astype(int)
    df["이전_출산"]      = (t_brth_raw > 0).astype(int)
    df["이전_IVF_성공"]  = (ivf_pg_raw > 0).astype(int)

    cause_cols = [c for c in df.columns if "불임 원인" in c]
    if cause_cols:
        df["불임원인_수"] = df[cause_cols].clip(0, 1).sum(axis=1)

    male_cols = [c for c in df.columns if "남성 주 불임 원인" in c
                 or "불임 원인 - 남성 요인" in c or "불임 원인 - 정자" in c]
    if male_cols:
        df["남성_불임_복합"] = df[male_cols].clip(0, 1).sum(axis=1)

    # [v4] 신규 이식 경과일 피처
    df["혼합_이식_간격"]    = (et_day - mix_day).fillna(0)
    df["이식일_D5이상"]     = (et_day >= 5).astype(int)
    df["이식일x이식수"]     = et_day * trans
    df["이식일_결측"]       = (et_day == 0).astype(int)
    df["정확히_D5"]         = (et_day == 5).astype(int)
    df["D5_최적이식_복합"]  = (et_day == 5).astype(int) * trans
    df["나이xD5"]           = age * (et_day == 5).astype(int)
    df["D5_신선주기"]       = ((et_day == 5) & (fresh_ == 1)).astype(int)
    df["D5_동결주기"]       = ((et_day == 5) & (frozen == 1)).astype(int)

    df["파트너정자_활용률"]   = _safe_div(partner, mixed)
    df["신선난자_저장비율"]   = _safe_div(stored_fresh, fresh)
    df["전체_배아_효율"]      = _safe_div(trans + stored, mixed + 1)
    df["생성배아_품질복합"]   = df["이식_효율"] * df["수정_효율"]
    df["해동_이식_비율"]      = _safe_div(thawed, trans + thawed + 1)
    df["클리닉_임신_집중도"]  = _safe_div(t_preg, clinic_proc + 1)
    df["임신_출산_전환율"]    = _safe_div(t_brth, t_preg + 1)
    df["나이x수정률"]         = age * df["수정_효율"]
    df["나이x클리닉횟수"]     = age * clinic_proc
    df["최적연령_D5"]         = ((age_raw == 0) & (et_day == 5)).astype(int)

    return df, s


# ══════════════════════════════════════════════════════════════════════════════
# 2. Level-0 베이스 모델 학습
# ══════════════════════════════════════════════════════════════════════════════

def train_base_models(X_train, X_test, y):
    n_tr, n_te = len(X_train), len(X_test)
    all_oof  = {}
    all_pred = {}

    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        # LightGBM
        key = f"LGB_s{seed}"
        oof = np.zeros(n_tr); pred = np.zeros(n_te)
        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            p = {**LGB_PARAMS, "seed": seed}
            m = lgb.train(p, lgb.Dataset(X_tr, y_tr),
                          num_boost_round=2000,
                          valid_sets=[lgb.Dataset(X_val, y_val)],
                          callbacks=[lgb.early_stopping(100, verbose=False),
                                     lgb.log_evaluation(-1)])
            oof[val_i] = m.predict(X_val)
            pred      += m.predict(X_test) / N_FOLDS
        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        all_oof[key] = oof; all_pred[key] = pred

        # XGBoost
        key = f"XGB_s{seed}"
        oof = np.zeros(n_tr); pred = np.zeros(n_te)
        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            p = {**XGB_PARAMS, "random_state": seed}
            m = xgb.XGBClassifier(**p)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof[val_i] = m.predict_proba(X_val)[:, 1]
            pred      += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        all_oof[key] = oof; all_pred[key] = pred

        # CatBoost
        key = f"CAT_s{seed}"
        oof = np.zeros(n_tr); pred = np.zeros(n_te)
        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            p = {**CAT_PARAMS, "random_seed": seed}
            m = CatBoostClassifier(**p)
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            oof[val_i] = m.predict_proba(X_val)[:, 1]
            pred      += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        all_oof[key] = oof; all_pred[key] = pred

    # 단순 평균 앙상블
    simple_avg_oof  = np.mean(list(all_oof.values()),  axis=0)
    simple_avg_pred = np.mean(list(all_pred.values()), axis=0)
    print(f"\n  단순 평균 OOF AUC: {roc_auc_score(y, simple_avg_oof):.5f}")

    return all_oof, all_pred, simple_avg_oof, simple_avg_pred


# ══════════════════════════════════════════════════════════════════════════════
# 3. 클래스별 최적 가중치 (v3 방식 확장)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_class_blend(all_oof, y):
    """
    각 모델 클래스(LGB/XGB/CAT)별 집계 OOF를 Optuna 3D 탐색으로 최적 가중치 찾기.
    반환: (w_lgb, w_xgb, w_cat, opt_oof, best_auc)
    """
    lgb_oof = np.mean([v for k, v in all_oof.items() if k.startswith("LGB")], axis=0)
    xgb_oof = np.mean([v for k, v in all_oof.items() if k.startswith("XGB")], axis=0)
    cat_oof = np.mean([v for k, v in all_oof.items() if k.startswith("CAT")], axis=0)

    def objective(trial):
        w1 = trial.suggest_float("w_lgb", 0.0, 1.0)
        w2 = trial.suggest_float("w_xgb", 0.0, 1.0)
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            return 0.0
        oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
        return roc_auc_score(y, oof)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=500, show_progress_bar=False)

    w1, w2 = study.best_params["w_lgb"], study.best_params["w_xgb"]
    w3 = 1.0 - w1 - w2
    best_auc = study.best_value
    opt_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cat_oof
    print(f"  클래스별 최적 가중치: LGB={w1:.3f}  XGB={w2:.3f}  CAT={w3:.3f}")
    print(f"  최적 클래스 블렌드 OOF AUC: {best_auc:.5f}")
    return w1, w2, w3, opt_oof, best_auc


# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature-Augmented Meta-LGB
# ══════════════════════════════════════════════════════════════════════════════

FA_META_PARAMS = {
    "objective": "binary", "metric": "auc", "verbosity": -1,
    "learning_rate": 0.01, "num_leaves": 31, "max_depth": 4,
    "min_child_samples": 50, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5,
    "reg_alpha": 0.5, "reg_lambda": 1.0,
    "n_jobs": -1, "seed": SEED,
}


def get_top_features(X_train, y, n=40):
    """LGB feature importance (gain) 기준 상위 n개 피처명 반환"""
    p = {**LGB_PARAMS, "seed": SEED, "num_leaves": 63, "verbosity": -1}
    m = lgb.train(p, lgb.Dataset(X_train, y), num_boost_round=300)
    imp = pd.Series(m.feature_importance("gain"), index=X_train.columns)
    return imp.nlargest(n).index.tolist()


def train_fa_meta_lgb(all_oof, all_pred, y, X_train, X_test, top_feats):
    """
    Feature-Augmented Meta-LGB:
    입력 = [9 OOF 예측] + [상위 N개 원본 피처]
    """
    meta_tr = np.column_stack(list(all_oof.values()))
    meta_te = np.column_stack(list(all_pred.values()))

    X_top_tr = X_train[top_feats].values.astype(np.float32)
    X_top_te = X_test[top_feats].values.astype(np.float32)

    # 스케일링 없이 그대로 concat (LGB은 scale 불필요)
    fa_tr = np.hstack([meta_tr, X_top_tr])
    fa_te = np.hstack([meta_te, X_top_te])

    n_tr = len(fa_tr); n_te = len(fa_te)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_fa  = np.zeros(n_tr)
    pred_fa = np.zeros(n_te)

    for fold, (tr_i, val_i) in enumerate(cv.split(fa_tr, y)):
        m = lgb.train(
            FA_META_PARAMS,
            lgb.Dataset(fa_tr[tr_i], y[tr_i]),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(fa_tr[val_i], y[val_i])],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        oof_fa[val_i]  = m.predict(fa_tr[val_i])
        pred_fa       += m.predict(fa_te) / N_FOLDS

    auc = roc_auc_score(y, oof_fa)
    print(f"  FA-Meta-LGB OOF AUC: {auc:.5f}")
    return oof_fa, pred_fa, auc


# ══════════════════════════════════════════════════════════════════════════════
# 5. Meta-LR (기본 메타 러너)
# ══════════════════════════════════════════════════════════════════════════════

def train_meta_lr(all_oof, all_pred, y):
    meta_tr = np.column_stack(list(all_oof.values()))
    meta_te = np.column_stack(list(all_pred.values()))
    n_tr = len(meta_tr); n_te = len(meta_te)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_lr  = np.zeros(n_tr)
    pred_lr = np.zeros(n_te)

    for fold, (tr_i, val_i) in enumerate(cv.split(meta_tr, y)):
        sc  = StandardScaler()
        mtr = sc.fit_transform(meta_tr[tr_i])
        mva = sc.transform(meta_tr[val_i])
        mte = sc.transform(meta_te)
        lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
        lr.fit(mtr, y[tr_i])
        oof_lr[val_i]  = lr.predict_proba(mva)[:, 1]
        pred_lr       += lr.predict_proba(mte)[:, 1] / N_FOLDS

    auc = roc_auc_score(y, oof_lr)
    print(f"  Meta-LR  OOF AUC: {auc:.5f}")
    return oof_lr, pred_lr, auc


# ══════════════════════════════════════════════════════════════════════════════
# 6. 최종 3-way 블렌딩 (Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_final_blend(oof_opt, oof_fa, oof_lr, y):
    """
    3-way Optuna blend: Opt-Blend + FA-Meta-LGB + Meta-LR
    """
    def objective(trial):
        w1 = trial.suggest_float("w_opt",  0.0, 1.0)
        w2 = trial.suggest_float("w_fa",   0.0, 1.0)
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            return 0.0
        oof = w1 * oof_opt + w2 * oof_fa + w3 * oof_lr
        return roc_auc_score(y, oof)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=500, show_progress_bar=False)

    w1, w2 = study.best_params["w_opt"], study.best_params["w_fa"]
    w3 = max(0.0, 1.0 - w1 - w2)
    best_auc = study.best_value
    print(f"\n  최적 블렌딩: Opt={w1:.3f}  FA-LGB={w2:.3f}  LR={w3:.3f}")
    print(f"  스태킹 최종 OOF AUC: {best_auc:.5f}")
    return w1, w2, w3, best_auc


# ══════════════════════════════════════════════════════════════════════════════
# 7. 성능 비교 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(scores: dict):
    print("\n" + "=" * 60)
    print("  성능 비교")
    print("=" * 60)
    max_auc = max(scores.values())
    for name, auc in sorted(scores.items(), key=lambda x: -x[1]):
        bar_len = int((auc - 0.72) / (max_auc - 0.72) * 20)
        bar = "█" * bar_len
        star = "  ★" if auc == max_auc else ""
        print(f"  {name:<34} {auc:.5f}  {bar}{star}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def main(data_dir="data", output_path="submission_v5.csv"):
    print("=" * 60)
    print(" IVF 임신 성공 예측 - Solution v5 (FA-Stacking)")
    print("=" * 60)

    # 데이터 로드
    print("\n[1] 데이터 로드 및 전처리")
    train_df, test_df = load_data(data_dir)
    y        = train_df[TARGET].values
    X_raw    = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    X_te_raw = test_df.drop(columns=[ID_COL], errors="ignore")
    test_ids = test_df[ID_COL].values
    print(f"  Train {train_df.shape}  Test {test_df.shape}  양성율 {y.mean():.3f}")

    X_pp,    le, tr_stats = preprocess_full(X_raw)
    X_te_pp, _,  _        = preprocess_full(X_te_raw,
                                            fit_label_encoders=le,
                                            train_stats=tr_stats)

    # 피처 엔지니어링
    print("\n[2] 피처 엔지니어링 (v4)")
    X_train, fe_stats = feature_engineering_v4(X_pp)
    X_test,  _        = feature_engineering_v4(X_te_pp, fe_stats=fe_stats)

    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].reset_index(drop=True).fillna(0)
    X_test  = X_test[common].reset_index(drop=True).fillna(0)
    print(f"  피처 수: {X_train.shape[1]}")

    # Level-0: 9개 베이스 모델
    print(f"\n[3] Level-0 베이스 모델 학습 (3 seeds x 3 모델 = 9개)")
    all_oof, all_pred, simple_oof, simple_pred = train_base_models(X_train, X_test, y)

    # 클래스별 최적 가중치
    print(f"\n[4] 클래스별 최적 가중치 탐색 (Optuna 500 trials)")
    w_lgb, w_xgb, w_cat, opt_oof, opt_auc = optimize_class_blend(all_oof, y)

    lgb_pred = np.mean([v for k, v in all_pred.items() if k.startswith("LGB")], axis=0)
    xgb_pred = np.mean([v for k, v in all_pred.items() if k.startswith("XGB")], axis=0)
    cat_pred = np.mean([v for k, v in all_pred.items() if k.startswith("CAT")], axis=0)
    opt_pred = w_lgb * lgb_pred + w_xgb * xgb_pred + w_cat * cat_pred

    # 상위 피처 추출 (FA 메타 입력용)
    print(f"\n[5] 상위 40개 피처 추출 (LGB importance)")
    top_feats = get_top_features(X_train, y, n=40)
    print(f"  상위 피처 예시: {top_feats[:5]}")

    # Feature-Augmented Meta-LGB
    print(f"\n[6] Feature-Augmented Meta-LGB 학습")
    oof_fa, pred_fa, fa_auc = train_fa_meta_lgb(
        all_oof, all_pred, y, X_train, X_test, top_feats
    )

    # Meta-LR
    print(f"\n[7] Meta-LR 학습")
    oof_lr, pred_lr, lr_auc = train_meta_lr(all_oof, all_pred, y)

    # 최종 3-way 블렌딩
    print(f"\n[8] 최종 3-way 블렌딩 (Optuna 500 trials)")
    w_opt, w_fa, w_lr, final_auc = optimize_final_blend(opt_oof, oof_fa, oof_lr, y)

    # 성능 비교
    scores = {
        "스태킹 최종 (3-way)": final_auc,
        "클래스 최적 블렌드":   opt_auc,
        "FA-Meta-LGB":          fa_auc,
        "Meta-LR":              lr_auc,
        "단순 평균":            roc_auc_score(y, simple_oof),
    }
    for k, v in all_oof.items():
        scores[k] = roc_auc_score(y, v)
    print_comparison(scores)

    # 제출 파일
    final_preds = w_opt * opt_pred + w_fa * pred_fa + w_lr * pred_lr
    sub = pd.DataFrame({"ID": test_ids, "probability": final_preds})
    sub.to_csv(output_path, index=False)

    print(f"\n  제출 파일 저장: {output_path}")
    print(f"  예측 범위: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    print("=" * 60)

    return sub, final_auc


if __name__ == "__main__":
    main()
