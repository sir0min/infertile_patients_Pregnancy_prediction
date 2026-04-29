"""
IVF 임신 성공 예측 - Solution v8 (CatBoost depth=8 + TabNet cat-emb)
======================================================================
v7 -> v8 변경:
  - CatBoost depth: 7 -> 8 (더 깊은 트리, 복잡한 패턴 학습)
  - 나머지 구성은 v7 동일 유지

근거:
  - CAT_s2024 = 0.73975 (현재 최강 단일 모델)
  - depth 증가로 추가 패턴 학습 가능
  - 단, 과적합 위험 있으므로 OOF로 검증
"""

import io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
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
TAB_SEEDS = [42, 777, 2024]
SEED    = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

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
    "iterations": 2000, "learning_rate": 0.05, "depth": 8,  # v7: 7 -> 8
    "l2_leaf_reg": 3, "subsample": 0.8,                      # l2: 5 -> 3 (depth 증가 보완)
    "eval_metric": "AUC", "early_stopping_rounds": 100,
    "verbose": False, "bootstrap_type": "Bernoulli", "task_type": "GPU",
}


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


def prepare_tabnet_features(X_train_df, X_test_df, max_cat_cardinality=15):
    X_tr = X_train_df.copy().astype(float)
    X_te = X_test_df.copy().astype(float)
    cat_idxs = []; cat_dims = []; cont_idxs = []

    for i, col in enumerate(X_train_df.columns):
        vals = X_train_df[col].dropna()
        if not np.all(vals.values == vals.values.astype(np.int64)):
            cont_idxs.append(i); continue
        n_unique = int(vals.nunique())
        if n_unique <= max_cat_cardinality:
            min_val = int(vals.min())
            if min_val < 0:
                shift = int(-min_val)
                X_tr[col] += shift; X_te[col] += shift
            cat_idxs.append(i)
            cat_dims.append(int(X_tr[col].max()) + 1)
        else:
            cont_idxs.append(i)

    cols = list(X_train_df.columns)
    cont_cols = [cols[i] for i in cont_idxs]
    if cont_cols:
        sc = StandardScaler()
        X_tr[cont_cols] = sc.fit_transform(X_tr[cont_cols])
        X_te[cont_cols] = sc.transform(X_te[cont_cols])

    print(f"  카테고리 피처: {len(cat_idxs)}개  연속형 피처: {len(cont_idxs)}개")
    return X_tr.values.astype(np.float32), X_te.values.astype(np.float32), cat_idxs, cat_dims


def train_tree_models(X_train, X_test, y):
    n_tr, n_te = len(X_train), len(X_test)
    all_oof = {}; all_pred = {}

    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        key = f"LGB_s{seed}"; oof = np.zeros(n_tr); pred = np.zeros(n_te)
        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            m = lgb.train({**LGB_PARAMS, "seed": seed}, lgb.Dataset(X_tr, y_tr),
                          num_boost_round=2000, valid_sets=[lgb.Dataset(X_val, y_val)],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[val_i] = m.predict(X_val); pred += m.predict(X_test) / N_FOLDS
        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        all_oof[key] = oof; all_pred[key] = pred

        key = f"XGB_s{seed}"; oof = np.zeros(n_tr); pred = np.zeros(n_te)
        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            m = xgb.XGBClassifier(**{**XGB_PARAMS, "random_state": seed})
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof[val_i] = m.predict_proba(X_val)[:, 1]; pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        all_oof[key] = oof; all_pred[key] = pred

        key = f"CAT_s{seed}"; oof = np.zeros(n_tr); pred = np.zeros(n_te)
        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            m = CatBoostClassifier(**{**CAT_PARAMS, "random_seed": seed})
            m.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            oof[val_i] = m.predict_proba(X_val)[:, 1]; pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        all_oof[key] = oof; all_pred[key] = pred

    simple_oof  = np.mean(list(all_oof.values()),  axis=0)
    simple_pred = np.mean(list(all_pred.values()), axis=0)
    print(f"\n  트리 단순 평균 OOF AUC: {roc_auc_score(y, simple_oof):.5f}")
    return all_oof, all_pred, simple_oof, simple_pred


def train_tabnet_models(X_tr_np, X_te_np, y, cat_idxs, cat_dims):
    n_tr, n_te = len(X_tr_np), len(X_te_np)
    tab_oof = {}; tab_pred = {}

    for seed in TAB_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        key = f"TAB_s{seed}"; oof = np.zeros(n_tr); pred = np.zeros(n_te)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold, (tr_i, val_i) in enumerate(cv.split(X_tr_np, y)):
            X_tr, X_val = X_tr_np[tr_i], X_tr_np[val_i]
            y_tr, y_val = y[tr_i], y[val_i]
            tab = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5, gamma=1.3,
                n_independent=2, n_shared=2,
                cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=3,
                lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params=dict(step_size=50, gamma=0.5),
                mask_type="sparsemax", verbose=0,
                device_name=DEVICE, seed=int(seed + fold * 100),
            )
            tab.fit(X_tr, y_tr.astype(np.int64),
                    eval_set=[(X_val, y_val.astype(np.int64))],
                    eval_name=["val"], eval_metric=["auc"],
                    max_epochs=300, patience=40,
                    batch_size=4096, virtual_batch_size=256, drop_last=False)
            oof[val_i] = tab.predict_proba(X_val)[:, 1]
            pred      += tab.predict_proba(X_te_np)[:, 1] / N_FOLDS
            del tab
            if DEVICE == "cuda": torch.cuda.empty_cache()

        print(f"  {key}  OOF AUC: {roc_auc_score(y, oof):.5f}")
        tab_oof[key] = oof; tab_pred[key] = pred

    tab_avg = np.mean(list(tab_oof.values()), axis=0)
    print(f"  TAB 평균 OOF AUC: {roc_auc_score(y, tab_avg):.5f}")
    return tab_oof, tab_pred, tab_avg, np.mean(list(tab_pred.values()), axis=0)


def optimize_4class_blend(tree_oof, tree_pred, tab_oof, tab_pred, y):
    lgb_oof = np.mean([v for k,v in tree_oof.items() if k.startswith("LGB")], axis=0)
    xgb_oof = np.mean([v for k,v in tree_oof.items() if k.startswith("XGB")], axis=0)
    cat_oof = np.mean([v for k,v in tree_oof.items() if k.startswith("CAT")], axis=0)
    tab_avg = np.mean(list(tab_oof.values()), axis=0)
    lgb_p = np.mean([v for k,v in tree_pred.items() if k.startswith("LGB")], axis=0)
    xgb_p = np.mean([v for k,v in tree_pred.items() if k.startswith("XGB")], axis=0)
    cat_p = np.mean([v for k,v in tree_pred.items() if k.startswith("CAT")], axis=0)
    tab_p = np.mean(list(tab_pred.values()), axis=0)

    def objective(trial):
        w1 = trial.suggest_float("w_lgb", 0.0, 1.0)
        w2 = trial.suggest_float("w_xgb", 0.0, 1.0)
        w3 = trial.suggest_float("w_cat", 0.0, 1.0)
        w4 = 1.0 - w1 - w2 - w3
        if w4 < 0: return 0.0
        return roc_auc_score(y, w1*lgb_oof + w2*xgb_oof + w3*cat_oof + w4*tab_avg)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=500, show_progress_bar=False)
    w1, w2, w3 = study.best_params["w_lgb"], study.best_params["w_xgb"], study.best_params["w_cat"]
    w4 = max(0.0, 1.0 - w1 - w2 - w3)
    opt_oof  = w1*lgb_oof + w2*xgb_oof + w3*cat_oof + w4*tab_avg
    opt_pred = w1*lgb_p   + w2*xgb_p   + w3*cat_p   + w4*tab_p
    print(f"  최적 가중치: LGB={w1:.3f} XGB={w2:.3f} CAT={w3:.3f} TAB={w4:.3f}")
    print(f"  4-class 블렌드 OOF AUC: {study.best_value:.5f}")
    return opt_oof, opt_pred, study.best_value


def train_meta_lr(all_oof, all_pred, y):
    meta_tr = np.column_stack(list(all_oof.values()))
    meta_te = np.column_stack(list(all_pred.values()))
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_lr = np.zeros(len(meta_tr)); pred_lr = np.zeros(len(meta_te))
    for _, (tr_i, val_i) in enumerate(cv.split(meta_tr, y)):
        sc = StandardScaler()
        mtr = sc.fit_transform(meta_tr[tr_i])
        lr  = LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)
        lr.fit(mtr, y[tr_i])
        oof_lr[val_i]  = lr.predict_proba(sc.transform(meta_tr[val_i]))[:, 1]
        pred_lr       += lr.predict_proba(sc.transform(meta_te))[:, 1] / N_FOLDS
    auc = roc_auc_score(y, oof_lr)
    print(f"  Meta-LR AUC: {auc:.5f}")
    return oof_lr, pred_lr, auc


def optimize_final_blend(oof_opt, oof_lr, y):
    def objective(trial):
        w = trial.suggest_float("w", 0.0, 1.0)
        return roc_auc_score(y, w*oof_opt + (1-w)*oof_lr)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=300)
    w = study.best_params["w"]
    print(f"  최종 블렌딩: Opt={w:.3f}  LR={1-w:.3f}  AUC={study.best_value:.5f}")
    return w, study.best_value


def print_comparison(scores):
    print("\n" + "=" * 60)
    print("  성능 비교")
    print("=" * 60)
    max_auc = max(scores.values())
    for name, auc in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "█" * int((auc - 0.72) / (max_auc - 0.72) * 20)
        star = "  ★" if auc == max_auc else ""
        print(f"  {name:<36} {auc:.5f}  {bar}{star}")
    print("=" * 60)


def main(data_dir="data", output_path="submission_v8.csv"):
    print("=" * 60)
    print(" IVF 임신 성공 예측 - Solution v8 (CatBoost depth=8)")
    print("=" * 60)

    print("\n[1] 데이터 로드 및 전처리")
    train_df, test_df = load_data(data_dir)
    y = train_df[TARGET].values
    X_raw    = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    X_te_raw = test_df.drop(columns=[ID_COL], errors="ignore")
    test_ids = test_df[ID_COL].values
    X_pp,    le, tr_stats = preprocess_full(X_raw)
    X_te_pp, _,  _        = preprocess_full(X_te_raw, fit_label_encoders=le, train_stats=tr_stats)

    print("\n[2] 피처 엔지니어링")
    X_train, fe_stats = feature_engineering_v4(X_pp)
    X_test,  _        = feature_engineering_v4(X_te_pp, fe_stats=fe_stats)
    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].reset_index(drop=True).fillna(0)
    X_test  = X_test[common].reset_index(drop=True).fillna(0)
    print(f"  피처 수: {X_train.shape[1]}")

    print("\n[3] TabNet 피처 준비")
    X_tr_np, X_te_np, cat_idxs, cat_dims = prepare_tabnet_features(X_train, X_test)

    print(f"\n[4] 트리 모델 학습 (CatBoost depth=8)")
    tree_oof, tree_pred, tree_avg_oof, tree_avg_pred = train_tree_models(X_train, X_test, y)

    print(f"\n[5] TabNet 학습")
    tab_oof, tab_pred, tab_avg_oof, tab_avg_pred = train_tabnet_models(X_tr_np, X_te_np, y, cat_idxs, cat_dims)

    print(f"\n[6] 4-class 최적 가중치 탐색")
    opt_oof, opt_pred, opt_auc = optimize_4class_blend(tree_oof, tree_pred, tab_oof, tab_pred, y)

    print(f"\n[7] Meta-LR 학습")
    all_oof_c  = {**tree_oof,  **tab_oof}
    all_pred_c = {**tree_pred, **tab_pred}
    oof_lr, pred_lr, lr_auc = train_meta_lr(all_oof_c, all_pred_c, y)

    print(f"\n[8] 최종 블렌딩")
    w_opt, final_auc = optimize_final_blend(opt_oof, oof_lr, y)

    scores = {
        "스태킹 최종 (v8)":     final_auc,
        "4-class 블렌드":       opt_auc,
        "트리 단순 평균":       roc_auc_score(y, tree_avg_oof),
        "TAB 평균":             roc_auc_score(y, tab_avg_oof),
        "Meta-LR":              lr_auc,
        "[v7 최고] 0.74027":    0.74027,
    }
    for k, v in tree_oof.items():
        scores[k] = roc_auc_score(y, v)
    print_comparison(scores)

    final_preds = w_opt * opt_pred + (1 - w_opt) * pred_lr
    sub = pd.DataFrame({"ID": test_ids, "probability": final_preds})
    sub.to_csv(output_path, index=False)
    print(f"\n  제출 파일: {output_path}")
    print(f"  예측 범위: [{final_preds.min():.4f}, {final_preds.max():.4f}]")
    print("=" * 60)
    return sub, final_auc


if __name__ == "__main__":
    main()
