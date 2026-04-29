import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import time

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

TARGET = "임신 성공 여부"
ID_COL = "ID"

X = train.drop(columns=[ID_COL, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID_COL])

print(f"Train shape: {X.shape}, Positive rate: {y.mean():.3f}")

# ── 전처리 ───────────────────────────────────────────────────────────────────
# '횟수' 문자열 → 숫자 변환 (ex. "3회" → 3, "0회" → 0)
def parse_count(series):
    return pd.to_numeric(
        series.astype(str).str.replace("회", "", regex=False).str.strip(),
        errors="coerce"
    )

count_cols = [c for c in X.columns if X[c].astype(str).str.contains("회").any()]
for col in count_cols:
    X[col]      = parse_count(X[col])
    X_test[col] = parse_count(X_test[col])

# 범주형 → Label Encoding
cat_cols = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col]      = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    encoders[col] = le

# 수치형 결측값 중앙값으로 채우기
num_cols = X.select_dtypes(include="number").columns
medians  = X[num_cols].median()
X[num_cols]      = X[num_cols].fillna(medians)
X_test[num_cols] = X_test[num_cols].fillna(medians)

print(f"결측값 처리 완료. Feature 수: {X.shape[1]}")

# ── 모델 정의 ────────────────────────────────────────────────────────────────
models = {}

models["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
models["RandomForest"]       = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
models["GradientBoosting"]   = GradientBoostingClassifier(n_estimators=300, random_state=42)

try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0
    )
except ImportError:
    print("XGBoost 미설치 — 건너뜀")

try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
except ImportError:
    print("LightGBM 미설치 — 건너뜀")

try:
    from catboost import CatBoostClassifier
    models["CatBoost"] = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0
    )
except ImportError:
    print("CatBoost 미설치 — 건너뜀")

# ── 5-Fold CV 비교 ───────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n" + "="*55)
print(f"{'모델':<22} {'AUC-ROC':>10} {'Std':>8} {'시간(초)':>10}")
print("="*55)

for name, model in models.items():
    t0     = time.time()
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - t0
    results.append({"모델": name, "AUC": scores.mean(), "Std": scores.std(), "시간": elapsed})
    print(f"{name:<22} {scores.mean():.4f}  ±{scores.std():.4f}  {elapsed:>8.1f}s")

print("="*55)

# ── 결과 정리 ────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results).sort_values("AUC", ascending=False).reset_index(drop=True)
print("\n[최종 순위]")
print(df_results.to_string(index=False))

best_name  = df_results.iloc[0]["모델"]
best_model = models[best_name]

print(f"\n최고 모델: {best_name} (AUC {df_results.iloc[0]['AUC']:.4f})")

# ── 최고 모델로 제출 파일 생성 ────────────────────────────────────────────────
print(f"\n{best_name} 전체 학습 후 예측 중...")
best_model.fit(X, y)
preds = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"ID": test[ID_COL], TARGET: preds})
submission.to_csv("submission.csv", index=False)
print(f"submission.csv 저장 완료 ({len(submission)}행)")
