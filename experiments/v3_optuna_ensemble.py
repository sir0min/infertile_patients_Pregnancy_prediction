"""
IVF 임신 성공 예측 - 고급 앙상블 솔루션 v3
==============================================
현재 베스트: OOF AUC 0.7400  (CatBoost 0.75 + LGB 0.25, 7 파생변수)

개선 전략:
  1. 의료 도메인 기반 파생변수 확장 (20개 추가)
  2. Optuna 하이퍼파라미터 최적화 (LGB + XGB + CatBoost, 3모델)
  3. 시드 앙상블 (5 시드 × 3 모델 = 15 모델) → 분산 감소
  4. Optuna 기반 최적 앙상블 가중치 탐색 (3D 그리드 탐색)

데이터 유출 방지:
  - 모든 통계(중앙값 등)는 train 데이터에서만 계산
  - Optuna: 훈련 샘플(50%)에서 3-Fold CV로 빠른 탐색 → 최적 파라미터로 전체 5-Fold 재학습
"""

import sys
import io
import warnings
from pathlib import Path

# Windows cp949 환경에서 한글 출력 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from src.preprocess import load_data, preprocess_full

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 상수 ─────────────────────────────────────────────────────────────────────
TARGET   = "임신 성공 여부"
ID_COL   = "ID"
SEED     = 42
N_FOLDS  = 5
SEEDS    = [42, 123, 456, 789, 1024]   # 시드 앙상블
N_TRIALS = 60                           # Optuna 탐색 횟수 (GPU라서 늘림)
OPTUNA_SAMPLE_FRAC = 1.0               # GPU: 전체 데이터로 탐색
OPTUNA_FOLDS = 5                        # GPU: 5-Fold 풀 CV로 탐색


# ══════════════════════════════════════════════════════════════════════════════
# 1. 확장 파생변수 (의료 도메인 지식 기반)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """분모가 0인 경우 0을 반환하는 안전 나눗셈 (pandas Series 반환 보장)."""
    return (num / denom.where(denom > 0, other=np.nan)).fillna(0)


def feature_engineering_v3(df: pd.DataFrame, fe_stats: dict = None) -> tuple:
    """
    v2(7개) 파생변수 + 의료 도메인 지식 기반 파생변수 20개 추가.
    fe_stats: None이면 현재 df(=train)에서 통계 계산, dict이면 test용 재사용.
    반환: (df, fe_stats)

    설계 원칙:
    - 순서형 인코딩된 열(-1=알 수 없음)은 계산 시 clip(lower=0) 처리
      → 알 수 없음을 0(가장 낮은 범주)으로 취급 (보수적 가정)
    - 불리언 플래그는 원본 값 기준 (>0 → 1회 이상, -1도 False)
    - pandas Series 연산만 사용하여 .fillna() 안전성 보장
    """
    df = df.copy()
    s = dict(fe_stats) if fe_stats else {}

    # ── 수치형 열 (전처리 후 NaN 없음: FILL_ZERO_COLS / cont_cols 처리됨) ────
    gen    = df["총 생성 배아 수"]
    trans  = df["이식된 배아 수"]
    stored = df["저장된 배아 수"]
    thawed = df["해동된 배아 수"]
    mixed  = df["혼합된 난자 수"]
    fresh  = df["수집된 신선 난자 수"]
    icsi_e = df["미세주입된 난자 수"]
    icsi_b = df["미세주입에서 생성된 배아 수"]
    icsi_t = df["미세주입 배아 이식 수"]
    frozen = df["동결 배아 사용 여부"]
    fresh_ = df["신선 배아 사용 여부"]

    # ── 순서형 인코딩 열: 계산용 (clip) vs 불리언 플래그용 (원본) ─────────────
    # clip(-1 → 0): 알 수 없음을 0회(기저값)로 처리
    age    = df["시술 당시 나이"].clip(lower=0)
    t_proc = df["총 시술 횟수"].clip(lower=0)
    t_preg = df["총 임신 횟수"].clip(lower=0)
    ivf_p  = df["IVF 시술 횟수"].clip(lower=0)
    ivf_pg = df["IVF 임신 횟수"].clip(lower=0)
    t_brth = df["총 출산 횟수"].clip(lower=0)

    # 불리언 플래그용 원본 (−1=알 수 없음 → False 처리됨)
    age_raw    = df["시술 당시 나이"]
    t_proc_raw = df["총 시술 횟수"]
    ivf_pg_raw = df["IVF 임신 횟수"]
    t_brth_raw = df["총 출산 횟수"]

    # ── [v2] 핵심 파생변수 7개 ────────────────────────────────────────────────

    # 이식 효율: 생성 배아 대비 이식 배아
    df["이식_효율"] = _safe_div(trans, gen)

    # 과거 임신 성공률: 총 임신 / (총 시술 + 1)
    df["과거_임신_성공률"] = t_preg / (t_proc + 1)

    # 수정 효율: 생성 배아 / 혼합 난자
    df["수정_효율"] = _safe_div(gen, mixed)

    # 나이 × 신선 난자 수 교호작용
    df["나이x난자수"] = age * fresh

    # 배아 풍요도 구간화 (train max 기반)
    if "embryo_max" not in s:
        s["embryo_max"] = max(gen.max() + 1, 9)
    df["배아_풍요도"] = pd.cut(
        gen, bins=[-1, 0, 3, 7, s["embryo_max"]], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    # 고령저반응: 만40세 이상 + 난자 수 < train 중앙값
    if "egg_median" not in s:
        s["egg_median"] = fresh.median()
    df["고령저반응"] = ((age_raw >= 3) & (fresh < s["egg_median"])).astype(int)

    # 순수 동결 배아 이식 주기 (FET)
    df["순수동결_주기"] = ((frozen == 1) & (fresh_ == 0)).astype(int)

    # ── [v3] 추가 파생변수 20개 ───────────────────────────────────────────────

    # 순수 신선 배아 이식 주기 (fresh only)
    df["순수신선_주기"] = ((fresh_ == 1) & (frozen == 0)).astype(int)

    # 첫 번째 시술 여부 (총 시술 횟수 == 0 → 첫 시도)
    df["첫_시술"] = (t_proc_raw == 0).astype(int)

    # 반복 시술 (3회 이상): 반복 실패 패턴
    df["반복_시술"] = (t_proc_raw >= 3).astype(int)

    # IVF 임신률: IVF 임신 / (IVF 시술 + 1)
    df["IVF_임신률"] = ivf_pg / (ivf_p + 1)

    # 출산률: 총 출산 / (총 시술 + 1)
    df["출산률"] = t_brth / (t_proc + 1)

    # ICSI 생존율: ICSI 생성 배아 / ICSI 주입 난자
    df["ICSI_생존율"] = _safe_div(icsi_b, icsi_e)

    # 배아 활용률: (이식 + 저장) / 생성
    df["배아_활용률"] = _safe_div(trans + stored, gen)

    # 동결 배아 활용률: 해동 / 저장
    df["동결_활용률"] = _safe_div(thawed, stored)

    # 잉여 배아 비율: 저장 / (이식 + 1)
    df["잉여배아_비율"] = stored / (trans + 1)

    # 이식 배아 수 카테고리: 0개 / 1개 / 2개 / 3개+
    df["이식수_카테고리"] = pd.cut(
        trans, bins=[-1, 0, 1, 2, float("inf")], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    # 나이 × 총 시술 횟수 (고령 반복 시술은 예후 불량)
    df["나이x시술횟수"] = age * t_proc

    # 나이 × 생성 배아 수
    df["나이x배아수"] = age * gen

    # 자극 반응 지수: 신선 난자 / (나이 + 2) — age+2: 0(알 수 없음) 포함 안전
    df["자극반응_지수"] = fresh / (age + 2)

    # 배아 품질 복합지수: ICSI 생존율 × 이식 효율
    df["배아품질_복합지수"] = df["ICSI_생존율"] * df["이식_효율"]

    # ICSI 이식 비율: ICSI 이식 / 전체 이식
    df["ICSI_이식비율"] = _safe_div(icsi_t, trans)

    # 기증 난자 사용 여부
    df["기증난자_사용"] = (df["난자 출처"] == 1).astype(int)

    # 이전 출산 경험 여부 (총 출산 횟수 > 0, -1은 False)
    df["이전_출산"] = (t_brth_raw > 0).astype(int)

    # IVF 이전 임신 성공 여부
    df["이전_IVF_성공"] = (ivf_pg_raw > 0).astype(int)

    # 불임 원인 수 (케이스 복잡도)
    cause_cols = [c for c in df.columns if "불임 원인" in c]
    if cause_cols:
        df["불임원인_수"] = df[cause_cols].clip(0, 1).sum(axis=1)

    # 남성 인자 복합 지수 (남성 불임 → ICSI 유효성 지표)
    male_cols = [c for c in df.columns
                 if "남성 주 불임 원인" in c
                 or "불임 원인 - 남성 요인" in c
                 or "불임 원인 - 정자" in c]
    if male_cols:
        df["남성_불임_복합"] = df[male_cols].clip(0, 1).sum(axis=1)

    return df, s


# ══════════════════════════════════════════════════════════════════════════════
# 2. Optuna 목적 함수 (각 모델별)
# ══════════════════════════════════════════════════════════════════════════════

def _lgb_objective(trial, X, y):
    params = {
        "objective":         "binary",
        "metric":            "auc",
        "verbosity":         -1,
        "boosting_type":     "gbdt",
        "device":            "gpu",       # GPU 사용
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 63, 255),
        "max_depth":         trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "n_jobs": -1, "seed": SEED,
    }
    cv = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr_i, val_i in cv.split(X, y):
        d_tr  = lgb.Dataset(X.iloc[tr_i],  label=y[tr_i])
        d_val = lgb.Dataset(X.iloc[val_i], label=y[val_i])
        m = lgb.train(
            params, d_tr, num_boost_round=1500,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
        )
        scores.append(roc_auc_score(y[val_i], m.predict(X.iloc[val_i])))
    return np.mean(scores)


def _xgb_objective(trial, X, y):
    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "tree_method":      "hist",
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-6, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-6, 5.0, log=True),
        "gamma":            trial.suggest_float("gamma", 1e-6, 1.0, log=True),
        "n_estimators": 1500, "n_jobs": -1, "random_state": SEED, "verbosity": 0,
        "tree_method": "hist",
        "device":      "cuda",            # GPU 사용 (XGBoost 2.0+)
        "early_stopping_rounds": 80,      # XGBoost 3.x: 생성자에 전달
    }
    cv = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr_i, val_i in cv.split(X, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X.iloc[tr_i], y[tr_i],
              eval_set=[(X.iloc[val_i], y[val_i])], verbose=False)
        scores.append(roc_auc_score(y[val_i], m.predict_proba(X.iloc[val_i])[:, 1]))
    return np.mean(scores)


def _cat_objective(trial, X, y):
    params = {
        "iterations":       1500,
        "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth":            trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        # colsample_bylevel(rsm)은 GPU 분류 모드 미지원 → 제외
        "eval_metric": "AUC", "early_stopping_rounds": 80,
        "random_seed": SEED, "verbose": False,
        "bootstrap_type": "Bernoulli", "task_type": "GPU",
    }
    cv = StratifiedKFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr_i, val_i in cv.split(X, y):
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[tr_i], y[tr_i],
              eval_set=(X.iloc[val_i], y[val_i]), verbose=False)
        scores.append(roc_auc_score(y[val_i], m.predict_proba(X.iloc[val_i])[:, 1]))
    return np.mean(scores)


def run_optuna(X_train, y):
    """50% 샘플 + 3-Fold로 빠른 탐색, 최적 파라미터 반환."""
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(len(X_train), size=int(len(X_train) * OPTUNA_SAMPLE_FRAC),
                            replace=False)
    Xs = X_train.iloc[sample_idx].reset_index(drop=True)
    ys = y[sample_idx]

    print(f"\n[Optuna] 탐색 샘플: {len(Xs):,}행, {N_TRIALS}회 × 3 모델")

    study_lgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(lambda t: _lgb_objective(t, Xs, ys), n_trials=N_TRIALS, show_progress_bar=True)
    best_lgb = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "device": "gpu", "n_jobs": -1, **study_lgb.best_params
    }
    print(f"  LGB 최적 AUC: {study_lgb.best_value:.4f}  params: {study_lgb.best_params}")

    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(lambda t: _xgb_objective(t, Xs, ys), n_trials=N_TRIALS, show_progress_bar=True)
    best_xgb = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "device": "cuda",
        "n_estimators": 2000, "n_jobs": -1, "verbosity": 0,
        "early_stopping_rounds": 100,
        **study_xgb.best_params
    }
    print(f"  XGB 최적 AUC: {study_xgb.best_value:.4f}  params: {study_xgb.best_params}")

    study_cat = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study_cat.optimize(lambda t: _cat_objective(t, Xs, ys), n_trials=N_TRIALS, show_progress_bar=True)
    best_cat = {
        "iterations": 2000, "eval_metric": "AUC",
        "early_stopping_rounds": 100, "verbose": False,
        "bootstrap_type": "Bernoulli", "task_type": "GPU",
        **study_cat.best_params
    }
    print(f"  CAT 최적 AUC: {study_cat.best_value:.4f}  params: {study_cat.best_params}")

    return best_lgb, best_xgb, best_cat


# ══════════════════════════════════════════════════════════════════════════════
# 3. 시드 앙상블 학습
# ══════════════════════════════════════════════════════════════════════════════

def train_seed_ensemble(X_train, X_test, y, best_lgb, best_xgb, best_cat):
    """
    5 시드 × 3 모델 × 5 Fold = 75 모델 학습.
    각 모델의 OOF/Test 예측을 평균내어 반환.
    """
    n_tr, n_te = len(X_train), len(X_test)
    oof_lgb = np.zeros(n_tr);  test_lgb = np.zeros(n_te)
    oof_xgb = np.zeros(n_tr);  test_xgb = np.zeros(n_te)
    oof_cat = np.zeros(n_tr);  test_cat = np.zeros(n_te)

    for seed in SEEDS:
        print(f"\n─── Seed {seed} ───")
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        oof_l = np.zeros(n_tr); test_l = np.zeros(n_te)
        oof_x = np.zeros(n_tr); test_x = np.zeros(n_te)
        oof_c = np.zeros(n_tr); test_c = np.zeros(n_te)

        for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y)):
            X_tr, X_val = X_train.iloc[tr_i], X_train.iloc[val_i]
            y_tr, y_val = y[tr_i], y[val_i]

            # LightGBM
            lgb_p = {**best_lgb, "seed": seed}
            d_tr  = lgb.Dataset(X_tr, label=y_tr)
            d_val = lgb.Dataset(X_val, label=y_val)
            m_lgb = lgb.train(
                lgb_p, d_tr, num_boost_round=2000,
                valid_sets=[d_val],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
            )
            oof_l[val_i] = m_lgb.predict(X_val)
            test_l      += m_lgb.predict(X_test) / N_FOLDS

            # XGBoost
            xgb_p = {**best_xgb, "random_state": seed}
            m_xgb = xgb.XGBClassifier(**xgb_p)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof_x[val_i] = m_xgb.predict_proba(X_val)[:, 1]
            test_x      += m_xgb.predict_proba(X_test)[:, 1] / N_FOLDS

            # CatBoost
            cat_p = {**best_cat, "random_seed": seed}
            m_cat = CatBoostClassifier(**cat_p)
            m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
            oof_c[val_i] = m_cat.predict_proba(X_val)[:, 1]
            test_c      += m_cat.predict_proba(X_test)[:, 1] / N_FOLDS

            l_s = roc_auc_score(y_val, oof_l[val_i])
            x_s = roc_auc_score(y_val, oof_x[val_i])
            c_s = roc_auc_score(y_val, oof_c[val_i])
            print(f"  Fold {fold+1}  LGB={l_s:.4f}  XGB={x_s:.4f}  CAT={c_s:.4f}")

        auc_l = roc_auc_score(y, oof_l)
        auc_x = roc_auc_score(y, oof_x)
        auc_c = roc_auc_score(y, oof_c)
        print(f"  [Seed {seed} OOF]  LGB={auc_l:.4f}  XGB={auc_x:.4f}  CAT={auc_c:.4f}")

        oof_lgb += oof_l / len(SEEDS);  test_lgb += test_l / len(SEEDS)
        oof_xgb += oof_x / len(SEEDS);  test_xgb += test_x / len(SEEDS)
        oof_cat += oof_c / len(SEEDS);  test_cat += test_c / len(SEEDS)

    return (oof_lgb, oof_xgb, oof_cat), (test_lgb, test_xgb, test_cat)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 앙상블 가중치 최적화 (3D 그리드 탐색)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_weights(oof_lgb, oof_xgb, oof_cat, y):
    best_auc, best_w = 0.0, (0.33, 0.33, 0.34)
    step = 0.05
    for w_l in np.arange(0, 1.0 + step, step):
        for w_x in np.arange(0, 1.0 - w_l + step, step):
            w_c = round(1.0 - w_l - w_x, 4)
            if w_c < 0:
                continue
            ens = w_l * oof_lgb + w_x * oof_xgb + w_c * oof_cat
            auc = roc_auc_score(y, ens)
            if auc > best_auc:
                best_auc = auc
                best_w   = (round(w_l, 2), round(w_x, 2), round(w_c, 2))
    print(f"\n[앙상블 가중치]  LGB={best_w[0]}  XGB={best_w[1]}  CAT={best_w[2]}"
          f"  OOF AUC={best_auc:.6f}")
    return best_w, best_auc


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def main(data_dir="data", output_path="submission_v3.csv"):
    print("=" * 60)
    print(" IVF 임신 성공 예측 - Solution v3")
    print("=" * 60)

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print("\n[1] 데이터 로드 및 전처리")
    train_df, test_df = load_data(data_dir)
    y = train_df[TARGET].values
    X_raw      = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    X_test_raw = test_df.drop(columns=[ID_COL], errors="ignore")
    test_ids   = test_df[ID_COL].values
    print(f"  Train: {train_df.shape}  Test: {test_df.shape}  양성율: {y.mean():.3f}")

    # ── 전처리 (데이터 유출 없음: test에 train 통계 적용) ────────────────────
    X_pp,       le, tr_stats = preprocess_full(X_raw)
    X_test_pp,  _,  _        = preprocess_full(X_test_raw,
                                               fit_label_encoders=le,
                                               train_stats=tr_stats)

    # ── 파생변수 생성 (v3) ───────────────────────────────────────────────────
    print("\n[2] 파생변수 생성 (v3: 27개)")
    X_train, fe_stats = feature_engineering_v3(X_pp)
    X_test,  _        = feature_engineering_v3(X_test_pp, fe_stats=fe_stats)

    common  = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common].reset_index(drop=True)
    X_test  = X_test[common].reset_index(drop=True)
    print(f"  피처 수: {X_train.shape[1]}  (Train {X_train.shape[0]:,} | Test {X_test.shape[0]:,})")

    # NaN 최종 보정 (안전장치)
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)

    # ── Optuna 하이퍼파라미터 탐색 ──────────────────────────────────────────
    print("\n[3] Optuna 하이퍼파라미터 탐색")
    best_lgb, best_xgb, best_cat = run_optuna(X_train, y)

    # ── 시드 앙상블 학습 ─────────────────────────────────────────────────────
    print(f"\n[4] 시드 앙상블 학습 ({len(SEEDS)} 시드 × 3 모델 × {N_FOLDS} Fold)")
    (oof_lgb, oof_xgb, oof_cat), (test_lgb, test_xgb, test_cat) = train_seed_ensemble(
        X_train, X_test, y, best_lgb, best_xgb, best_cat
    )

    # ── 앙상블 가중치 최적화 ─────────────────────────────────────────────────
    print("\n[5] 앙상블 가중치 최적화")
    print(f"  개별 OOF AUC  LGB={roc_auc_score(y, oof_lgb):.4f}"
          f"  XGB={roc_auc_score(y, oof_xgb):.4f}"
          f"  CAT={roc_auc_score(y, oof_cat):.4f}")
    (w_l, w_x, w_c), ens_auc = optimize_weights(oof_lgb, oof_xgb, oof_cat, y)

    # ── 최종 예측 및 제출 파일 ───────────────────────────────────────────────
    test_preds = w_l * test_lgb + w_x * test_xgb + w_c * test_cat

    sub = pd.DataFrame({"ID": test_ids, "probability": test_preds})
    sub.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" [최종 결과]  앙상블 OOF AUC = {ens_auc:.6f}")
    print(f" 제출 파일 저장: {output_path}")
    print(f" 예측 범위: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print("=" * 60)

    return sub, ens_auc


if __name__ == "__main__":
    main()
