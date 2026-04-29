import warnings
warnings.filterwarnings("ignore")
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pathlib import Path

# ── 설정 ─────────────────────────────────────────────────────────────────────
N_FOLDS       = 5
OPTUNA_TRIALS = 50
TUNE_SAMPLE   = 0.5   # Optuna 탐색용 샘플 비율
RANDOM_STATE  = 42
TARGET        = "임신 성공 여부"
ID_COL        = "ID"
NOTEBOOK_SUB  = Path("submission/submission.csv")   # 노트북 제출 파일 경로

# ── 전처리 상수 (노트북 기준) ─────────────────────────────────────────────────
AGE_MAP = {
    "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
    "만40-42세": 3, "만43-44세": 4, "만45-50세": 5, "알 수 없음": -1,
}
COUNT_MAP = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}
DONOR_AGE_MAP = {
    "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2,
    "만31-35세": 3,  "만36-40세": 4, "만41-45세": 5, "알 수 없음": -1,
}
COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
DROP_COLS = [
    "착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부",
    "불임 원인 - 여성 요인", "난자 채취 경과일", "난자 해동 경과일",
]
REASON_CATS   = ["현재 시술용", "배아 저장용", "난자 저장용", "기증용", "연구용"]
PROCEDURE_TYPES = ["IVF", "ICSI", "IUI", "ICI", "GIFT", "FER", "BLASTOCYST", "AH", "Generic DI", "IVI"]
BINARY_COLS = [
    "배란 자극 여부", "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부",
    "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
]
CONT_COLS = [
    "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
    "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
    "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
    "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
    "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일",
]

# ── 전처리 함수 ───────────────────────────────────────────────────────────────
def expand_reason(df):
    col = "배아 생성 주요 이유"
    for cat in REASON_CATS:
        df[f"이유_{cat}"] = df[col].fillna("").str.contains(cat).astype(int)
    return df

def expand_procedure(df):
    col = "특정 시술 유형"
    filled = df[col].fillna("Unknown")
    for pt in PROCEDURE_TYPES:
        df[f"시술_{pt}"] = (
            filled.str.upper().str.replace(" ", "", regex=False)
            .str.contains(pt.upper()).astype(int)
        )
    return df

def preprocess(df, train_stats=None):
    df = df.copy()
    stats = dict(train_stats) if train_stats else {}

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 임신 시도 연수 - 결측 지시 변수 + -1
    if "임신 시도 또는 마지막 임신 경과 연수" in df.columns:
        df["임신_시도_연수_결측"] = df["임신 시도 또는 마지막 임신 경과 연수"].isnull().astype(int)
        df["임신 시도 또는 마지막 임신 경과 연수"] = df["임신 시도 또는 마지막 임신 경과 연수"].fillna(-1)

    # 배아 해동 경과일 - 결측 지시 변수 + 0
    if "배아 해동 경과일" in df.columns:
        df["배아_해동_결측"] = df["배아 해동 경과일"].isnull().astype(int)
        df["배아 해동 경과일"] = df["배아 해동 경과일"].fillna(0)

    # IVF 전용 컬럼 결측 → 0
    fill_zero = [
        "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부",
        "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
        "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
        "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
        "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
        "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
    ]
    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 경과일 - train 중앙값 대체
    for col in ["난자 혼합 경과일", "배아 이식 경과일"]:
        if col in df.columns:
            key = f"{col}_med"
            if key not in stats:
                stats[key] = df[col].median()
            df[col] = df[col].fillna(stats[key])

    # 순서형 인코딩
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(-1).astype(int)
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP).fillna(-1).astype(int)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(-1).astype(int)

    # 범주형 → 정수
    if "시술 유형" in df.columns:
        df["시술 유형"] = (df["시술 유형"] == "IVF").astype(int)

    induction_map = {"알 수 없음": 0, "기록되지 않은 시행": 1, "생식선 자극 호르몬": 2, "세트로타이드 (억제제)": 3}
    if "배란 유도 유형" in df.columns:
        df["배란 유도 유형"] = df["배란 유도 유형"].map(induction_map).fillna(0).astype(int)

    egg_map   = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": -1}
    sperm_map = {"배우자 제공": 0, "기증 제공": 1, "배우자 및 기증 제공": 2, "미할당": -1}
    if "난자 출처" in df.columns:
        df["난자 출처"] = df["난자 출처"].map(egg_map).fillna(-1).astype(int)
    if "정자 출처" in df.columns:
        df["정자 출처"] = df["정자 출처"].map(sperm_map).fillna(-1).astype(int)

    code_map = {v: i for i, v in enumerate(
        ["TRCMWS", "TRDQAZ", "TRJXFG", "TRVNRY", "TRXQMD", "TRYBLT", "TRZKPL"]
    )}
    if "시술 시기 코드" in df.columns:
        df["시술 시기 코드"] = df["시술 시기 코드"].map(code_map).fillna(-1).astype(int)

    # 이진 컬럼 결측 → 0
    for c in BINARY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 연속형 결측 → train 중앙값
    for c in CONT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            key = f"{c}_med"
            if key not in stats:
                stats[key] = df[c].median() if df[c].notna().any() else 0
            df[c] = df[c].fillna(stats[key])

    return df, stats


def preprocess_full(df, train_stats=None):
    df = expand_reason(df.copy())
    df = expand_procedure(df)
    df.drop(columns=["배아 생성 주요 이유", "특정 시술 유형"], inplace=True, errors="ignore")
    return preprocess(df, train_stats)


# ── 피처 엔지니어링 (노트북 7개 + 추가 6개) ──────────────────────────────────
def add_features(df, fe_stats=None):
    d = df.copy()
    stats = dict(fe_stats) if fe_stats else {}

    # 1. 이식 효율
    d["이식_효율"] = np.where(d["총 생성 배아 수"] > 0,
                               d["이식된 배아 수"] / d["총 생성 배아 수"], 0)
    # 2. 과거 임신 성공률
    d["과거_임신_성공률"] = np.where(d["총 시술 횟수"] > 0,
                                      d["총 임신 횟수"] / (d["총 시술 횟수"] + 1), 0)
    # 3. 수정 효율
    d["수정_효율"] = np.where(d["혼합된 난자 수"] > 0,
                               d["총 생성 배아 수"] / d["혼합된 난자 수"], 0)
    # 4. 나이 × 난자 수
    age_num = d["시술 당시 나이"].replace(-1, np.nan)
    d["나이x난자수"] = (age_num * d["수집된 신선 난자 수"]).fillna(0)

    # 5. 배아 풍요도
    if "embryo_max" not in stats:
        stats["embryo_max"] = max(d["총 생성 배아 수"].max() + 1, 9)
    d["배아_풍요도"] = pd.cut(
        d["총 생성 배아 수"], bins=[-1, 0, 3, 7, stats["embryo_max"]],
        labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    # 6. 고령 저반응
    if "egg_median" not in stats:
        stats["egg_median"] = d["수집된 신선 난자 수"].median()
    d["고령저반응"] = (
        (age_num >= 3) & (d["수집된 신선 난자 수"] < stats["egg_median"])
    ).fillna(False).astype(int)

    # 7. 순수 동결 주기
    d["순수동결_주기"] = (
        (d["동결 배아 사용 여부"] == 1) & (d["신선 배아 사용 여부"] == 0)
    ).astype(int)

    # 8. IVF 임신율 이력
    d["IVF_임신율_이력"] = np.where(d["IVF 시술 횟수"] > 0,
                                     d["IVF 임신 횟수"] / (d["IVF 시술 횟수"] + 1), 0)
    # 9. 출산율 이력
    d["출산율_이력"] = np.where(d["총 임신 횟수"] > 0,
                                 d["총 출산 횟수"] / (d["총 임신 횟수"] + 1), 0)
    # 10. 배아 저장률
    d["배아_저장률"] = np.where(d["총 생성 배아 수"] > 0,
                                 d["저장된 배아 수"] / d["총 생성 배아 수"], 0)
    # 11. 미세주입 성공률
    d["미세주입_성공률"] = np.where(d["미세주입된 난자 수"] > 0,
                                     d["미세주입에서 생성된 배아 수"] / d["미세주입된 난자 수"], 0)
    # 12. 불임 원인 총 개수
    cause_cols = [c for c in d.columns if c.startswith("불임 원인 -")]
    d["불임원인_총개수"] = d[cause_cols].sum(axis=1)

    # 13. 클리닉 경험 비율
    d["클리닉_경험비율"] = np.where(d["총 시술 횟수"] > 0,
                                     d["클리닉 내 총 시술 횟수"] / (d["총 시술 횟수"] + 1), 0)

    return d, stats


# ── 데이터 로드 & 전처리 ──────────────────────────────────────────────────────
print("데이터 로드 및 전처리 중...")
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

y          = train[TARGET].copy()
X_raw      = train.drop(columns=[ID_COL, TARGET])
X_test_raw = test.drop(columns=[ID_COL])

X_pp,      train_stats = preprocess_full(X_raw)
X_test_pp, _           = preprocess_full(X_test_raw, train_stats=train_stats)

X_train, fe_stats = add_features(X_pp)
X_test,  _        = add_features(X_test_pp, fe_stats=fe_stats)

common  = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common]
X_test  = X_test[common]

print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Positive rate: {y.mean():.3f}")

X_arr      = X_train.values
X_test_arr = X_test.values

# ── Optuna 샘플링 준비 ────────────────────────────────────────────────────────
rng      = np.random.default_rng(RANDOM_STATE)
tune_idx = rng.choice(len(y), size=int(len(y) * TUNE_SAMPLE), replace=False)
X_tune   = X_arr[tune_idx]
y_tune   = y.iloc[tune_idx].reset_index(drop=True)
cv_tune  = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_score(model_cls, params, X, y_s, cv_obj):
    scores = []
    for tr, val in cv_obj.split(X, y_s):
        m = model_cls(**params)
        m.fit(X[tr], y_s.iloc[tr])
        scores.append(roc_auc_score(y_s.iloc[val], m.predict_proba(X[val])[:, 1]))
    return np.mean(scores)


# ── LightGBM 튜닝 ─────────────────────────────────────────────────────────────
print(f"\n[1/3] LightGBM Optuna ({OPTUNA_TRIALS} trials, {TUNE_SAMPLE*100:.0f}% sample)...")

def lgbm_obj(trial):
    p = dict(
        n_estimators    = trial.suggest_int("n_estimators", 300, 800),
        learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth       = trial.suggest_int("max_depth", 4, 10),
        num_leaves      = trial.suggest_int("num_leaves", 31, 255),
        subsample       = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        reg_alpha       = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda      = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    return cv_score(LGBMClassifier, p, X_tune, y_tune, cv_tune)

lgbm_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
lgbm_study.optimize(lgbm_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best LightGBM AUC: {lgbm_study.best_value:.4f}")

# ── CatBoost 튜닝 ─────────────────────────────────────────────────────────────
print(f"\n[2/3] CatBoost Optuna ({OPTUNA_TRIALS} trials, {TUNE_SAMPLE*100:.0f}% sample)...")

def cat_obj(trial):
    p = dict(
        iterations       = trial.suggest_int("iterations", 200, 500),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        depth            = trial.suggest_int("depth", 4, 8),
        l2_leaf_reg      = trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bylevel= trial.suggest_float("colsample_bylevel", 0.6, 1.0),
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 50),
        random_seed=RANDOM_STATE, verbose=0,
    )
    return cv_score(CatBoostClassifier, p, X_tune, y_tune, cv_tune)

cat_study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
cat_study.optimize(cat_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best CatBoost AUC: {cat_study.best_value:.4f}")

# ── XGBoost 튜닝 ─────────────────────────────────────────────────────────────
print(f"\n[3/3] XGBoost Optuna ({OPTUNA_TRIALS} trials, {TUNE_SAMPLE*100:.0f}% sample)...")

def xgb_obj(trial):
    p = dict(
        n_estimators    = trial.suggest_int("n_estimators", 300, 800),
        learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth       = trial.suggest_int("max_depth", 4, 10),
        min_child_weight= trial.suggest_int("min_child_weight", 1, 20),
        subsample       = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha       = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda      = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    return cv_score(XGBClassifier, p, X_tune, y_tune, cv_tune)

xgb_study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
xgb_study.optimize(xgb_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best XGBoost AUC: {xgb_study.best_value:.4f}")

# ── 최적 파라미터로 OOF 학습 (전체 데이터 5-Fold) ────────────────────────────
print("\n최적 파라미터로 전체 데이터 OOF 학습 중...")

lgbm_best = LGBMClassifier(**lgbm_study.best_params,
                            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
cat_best  = CatBoostClassifier(**cat_study.best_params,
                               random_seed=RANDOM_STATE, verbose=0)
xgb_best  = XGBClassifier(**xgb_study.best_params,
                           eval_metric="logloss", random_state=RANDOM_STATE,
                           n_jobs=-1, verbosity=0)

oof   = {m: np.zeros(len(y)) for m in ["lgb", "cat", "xgb"]}
preds = {m: np.zeros(len(X_test_arr)) for m in ["lgb", "cat", "xgb"]}

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_arr, y), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=" ", flush=True)
    Xtr, Xval = X_arr[tr_idx], X_arr[val_idx]
    ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]

    lgbm_best.fit(Xtr, ytr)
    cat_best.fit(Xtr, ytr)
    xgb_best.fit(Xtr, ytr)

    for key, m in [("lgb", lgbm_best), ("cat", cat_best), ("xgb", xgb_best)]:
        oof[key][val_idx]  = m.predict_proba(Xval)[:, 1]
        preds[key]        += m.predict_proba(X_test_arr)[:, 1] / N_FOLDS
    print("done")

auc = {k: roc_auc_score(y, oof[k]) for k in oof}
print(f"\n  OOF AUC  LGB: {auc['lgb']:.4f}  CAT: {auc['cat']:.4f}  XGB: {auc['xgb']:.4f}")

# ── 앙상블 가중치 최적화 (Grid Search on OOF) ─────────────────────────────────
print("\n앙상블 가중치 최적화 중...")
best_auc, best_w = 0, None
for wl in np.arange(0, 1.01, 0.05):
    for wc in np.arange(0, 1.01 - wl, 0.05):
        wx = round(1.0 - wl - wc, 5)
        if wx < 0:
            continue
        blend = wl * oof["lgb"] + wc * oof["cat"] + wx * oof["xgb"]
        a = roc_auc_score(y, blend)
        if a > best_auc:
            best_auc, best_w = a, (wl, wc, wx)

wl, wc, wx = best_w
pred_3model = wl * preds["lgb"] + wc * preds["cat"] + wx * preds["xgb"]

print(f"  최적 가중치 - LGB: {wl:.2f}  CAT: {wc:.2f}  XGB: {wx:.2f}")
print(f"  3모델 앙상블 OOF AUC: {best_auc:.4f}")

# ── 노트북 submission과 블렌딩 ────────────────────────────────────────────────
if NOTEBOOK_SUB.exists():
    nb_sub    = pd.read_csv(NOTEBOOK_SUB)
    pred_nb   = nb_sub.set_index("ID").loc[test["ID"], "probability"].values
    NB_AUC    = 0.7400   # 노트북 OOF AUC

    # AUC 비례 가중치
    w_new = best_auc / (best_auc + NB_AUC)
    w_nb  = NB_AUC  / (best_auc + NB_AUC)
    pred_final = w_new * pred_3model + w_nb * pred_nb

    print(f"\n  노트북 블렌딩 - 새 모델: {w_new:.2f}  노트북: {w_nb:.2f}")
    print(f"  (노트북 OOF AUC {NB_AUC:.4f} 기준)")
else:
    pred_final = pred_3model
    print("\n  노트북 submission 파일 없음 - 3모델 앙상블만 사용")

# ── 제출 파일 저장 ────────────────────────────────────────────────────────────
submission = pd.DataFrame({"ID": test[ID_COL], "probability": pred_final})
submission.to_csv("submission_final.csv", index=False)

print("\n" + "="*50)
print(f"  LightGBM OOF AUC : {auc['lgb']:.4f}")
print(f"  CatBoost OOF AUC : {auc['cat']:.4f}")
print(f"  XGBoost  OOF AUC : {auc['xgb']:.4f}")
print(f"  3모델 앙상블 AUC  : {best_auc:.4f}")
print(f"  submission_final.csv 저장 완료 ({len(submission)}행)")
print("="*50)
