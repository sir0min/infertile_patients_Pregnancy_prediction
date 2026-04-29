import warnings
warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ── 설정 ─────────────────────────────────────────────────────────────────────
N_FOLDS        = 5
OPTUNA_TRIALS  = 30
RANDOM_STATE   = 42
TUNE_SAMPLE    = 0.3   # Optuna 탐색 시 사용할 데이터 비율 (속도/정확도 트레이드오프)
TARGET         = "임신 성공 여부"
ID_COL         = "ID"

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

y        = train[TARGET].copy()
train_df = train.drop(columns=[ID_COL, TARGET])
test_df  = test.drop(columns=[ID_COL])
all_df   = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
n_train  = len(train_df)

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Positive rate: {y.mean():.3f}")

# ── 전처리 헬퍼 ──────────────────────────────────────────────────────────────
def parse_count(series):
    return pd.to_numeric(
        series.astype(str).str.replace("회", "", regex=False).str.strip(),
        errors="coerce"
    )

# '횟수' 컬럼 숫자 변환
count_cols = [c for c in all_df.columns if all_df[c].astype(str).str.contains("회").any()]
for col in count_cols:
    all_df[col] = parse_count(all_df[col])

# ── 피처 엔지니어링 ──────────────────────────────────────────────────────────
def add_features(df):
    d = df.copy()

    # 1. 이력 기반 비율 피처
    d["임신율_이력"]       = d["총 임신 횟수"] / (d["총 시술 횟수"] + 1)
    d["IVF_임신율_이력"]   = d["IVF 임신 횟수"] / (d["IVF 시술 횟수"] + 1)
    d["DI_임신율_이력"]    = d["DI 임신 횟수"]  / (d["DI 시술 횟수"]  + 1)
    d["출산율_이력"]       = d["총 출산 횟수"]  / (d["총 임신 횟수"]  + 1)

    # 2. 배아 효율 피처
    d["배아_이식률"]       = d["이식된 배아 수"]           / (d["총 생성 배아 수"] + 1)
    d["배아_저장률"]       = d["저장된 배아 수"]           / (d["총 생성 배아 수"] + 1)
    d["미세주입_성공률"]   = d["미세주입에서 생성된 배아 수"] / (d["미세주입된 난자 수"] + 1)
    d["난자_수정률"]       = d["혼합된 난자 수"]           / (d["수집된 신선 난자 수"] + 1)

    # 3. 불임 원인 복잡도 (해당 컬럼 합산)
    cause_cols = [c for c in df.columns if c.startswith("불임 원인 -")]
    d["불임원인_총개수"] = df[cause_cols].sum(axis=1)

    # 4. 나이 → 순서형 숫자
    age_order = {
        "만18-34세": 1, "만35-37세": 2, "만38-39세": 3,
        "만40-42세": 4, "만43-44세": 5, "만45-50세": 6,
    }
    d["나이_순서형"] = d["시술 당시 나이"].map(age_order)

    # 5. 클리닉 경험 비율
    d["클리닉_경험비율"] = d["클리닉 내 총 시술 횟수"] / (d["총 시술 횟수"] + 1)

    # 6. 이식 배아 대비 미세주입 비율
    d["미세주입_이식비율"] = d["미세주입 배아 이식 수"] / (d["이식된 배아 수"] + 1)

    return d

all_df = add_features(all_df)
print(f"피처 엔지니어링 완료. 총 피처 수: {all_df.shape[1]}")

# ── 범주형 인코딩 (train 데이터만으로 fit) ───────────────────────────────────
cat_cols = all_df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    train_vals = all_df.iloc[:n_train][col].astype(str)
    le.fit(train_vals)                          # train만으로 fit
    known = set(le.classes_)
    # test에 train에 없는 카테고리가 있으면 train 최빈값으로 대체
    most_freq = train_vals.mode()[0]
    all_df[col] = all_df[col].astype(str).map(
        lambda x: x if x in known else most_freq
    )
    all_df[col] = le.transform(all_df[col])

# ── 결측값 처리 ──────────────────────────────────────────────────────────────
medians = all_df.iloc[:n_train].median()
all_df  = all_df.fillna(medians)

X      = all_df.iloc[:n_train].values
X_test = all_df.iloc[n_train:].values
feature_names = all_df.columns.tolist()

# ── Optuna 목적함수 (샘플링으로 속도 향상) ────────────────────────────────────
cv      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

rng = np.random.default_rng(RANDOM_STATE)
tune_idx = rng.choice(len(y), size=int(len(y) * TUNE_SAMPLE), replace=False)
X_tune = X[tune_idx]
y_tune = y.iloc[tune_idx].reset_index(drop=True)

def lgbm_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 700),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth":        trial.suggest_int("max_depth", 4, 10),
        "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
    }
    scores = []
    for tr_idx, val_idx in cv_tune.split(X_tune, y_tune):
        m = LGBMClassifier(**params)
        m.fit(X_tune[tr_idx], y_tune.iloc[tr_idx])
        pred = m.predict_proba(X_tune[val_idx])[:, 1]
        scores.append(roc_auc_score(y_tune.iloc[val_idx], pred))
    return np.mean(scores)

def catboost_objective(trial):
    params = {
        "iterations":        trial.suggest_int("iterations", 200, 500),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth":             trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        "min_data_in_leaf":  trial.suggest_int("min_data_in_leaf", 5, 50),
        "random_seed": RANDOM_STATE, "verbose": 0,
    }
    scores = []
    for tr_idx, val_idx in cv_tune.split(X_tune, y_tune):
        m = CatBoostClassifier(**params)
        m.fit(X_tune[tr_idx], y_tune.iloc[tr_idx])
        pred = m.predict_proba(X_tune[val_idx])[:, 1]
        scores.append(roc_auc_score(y_tune.iloc[val_idx], pred))
    return np.mean(scores)

# ── LightGBM 튜닝 ─────────────────────────────────────────────────────────────
print(f"\n[1/2] LightGBM Optuna 튜닝 ({OPTUNA_TRIALS} trials)...")
lgbm_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgbm_study.optimize(lgbm_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best LightGBM AUC: {lgbm_study.best_value:.4f}")
print(f"  Best params: {lgbm_study.best_params}")

# ── CatBoost 튜닝 ─────────────────────────────────────────────────────────────
print(f"\n[2/2] CatBoost Optuna 튜닝 ({OPTUNA_TRIALS} trials)...")
cat_study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
cat_study.optimize(catboost_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best CatBoost AUC: {cat_study.best_value:.4f}")
print(f"  Best params: {cat_study.best_params}")

# ── 앙상블 가중치 최적화 (OOF 기반) ─────────────────────────────────────────
print("\n앙상블 OOF 예측 생성 중...")

lgbm_best = LGBMClassifier(**lgbm_study.best_params, random_state=RANDOM_STATE,
                            n_jobs=-1, verbose=-1)
cat_best  = CatBoostClassifier(**cat_study.best_params, random_seed=RANDOM_STATE, verbose=0)

oof_lgbm = np.zeros(len(y))
oof_cat  = np.zeros(len(y))
pred_lgbm = np.zeros(len(X_test))
pred_cat  = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=" ", flush=True)

    lgbm_best.fit(X[tr_idx], y.iloc[tr_idx])
    cat_best.fit(X[tr_idx], y.iloc[tr_idx])

    oof_lgbm[val_idx] = lgbm_best.predict_proba(X[val_idx])[:, 1]
    oof_cat[val_idx]  = cat_best.predict_proba(X[val_idx])[:, 1]

    pred_lgbm += lgbm_best.predict_proba(X_test)[:, 1] / N_FOLDS
    pred_cat  += cat_best.predict_proba(X_test)[:, 1]  / N_FOLDS
    print("done")

auc_lgbm = roc_auc_score(y, oof_lgbm)
auc_cat  = roc_auc_score(y, oof_cat)

# 성능 비례 가중치
w_lgbm = auc_lgbm / (auc_lgbm + auc_cat)
w_cat  = auc_cat  / (auc_lgbm + auc_cat)
oof_ensemble  = w_lgbm * oof_lgbm  + w_cat * oof_cat
pred_ensemble = w_lgbm * pred_lgbm + w_cat * pred_cat
auc_ensemble  = roc_auc_score(y, oof_ensemble)

print("\n" + "="*45)
print(f"  LightGBM OOF AUC  : {auc_lgbm:.4f}  (가중치 {w_lgbm:.2f})")
print(f"  CatBoost OOF AUC  : {auc_cat:.4f}  (가중치 {w_cat:.2f})")
print(f"  Ensemble OOF AUC  : {auc_ensemble:.4f}  ← 최종")
print("="*45)

# ── 제출 파일 저장 ────────────────────────────────────────────────────────────
submission = pd.DataFrame({ID_COL: test[ID_COL], "probability": pred_ensemble})
submission.to_csv("submission_ensemble.csv", index=False)
print(f"\nsubmission_ensemble.csv 저장 완료 ({len(submission)}행)")
