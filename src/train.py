"""
==============================================================
 실험 기록 (Experiment Log)
==============================================================
 사용한 모델:
   - LightGBM  (5-Fold Stratified CV)
   - CatBoost  (5-Fold Stratified CV)
   - 앙상블    CatBoost × LightGBM  (가중 평균 w_cat=0.75, w_lgb=0.25)

 적용한 Feature Engineering (핵심 파생변수 7개 / feature_engineering_v2):
   1. 이식_효율          : 이식 배아 수 / 총 생성 배아 수
   2. 과거_임신_성공률   : 총 임신 횟수 / (총 시술 횟수 + 1)
   3. 수정_효율          : 총 생성 배아 수 / 혼합된 난자 수
   4. 나이x난자수        : 시술 당시 나이 × 수집된 신선 난자 수 (교호작용)
   5. 배아_풍요도        : 생성 배아 수 구간화 (0=없음, 1=1-3, 2=4-7, 3=8+)
   6. 고령저반응         : 나이 >= 만40세 AND 난자 수 < train 중앙값 (플래그)
   7. 순수동결_주기      : 동결 배아만 사용 (FER 주기 플래그)

 주요 파라미터:
   LightGBM  num_leaves=127, learning_rate=0.05, feature_fraction=0.8
   CatBoost  depth=6,        learning_rate=0.05, subsample=0.8
   공통      num_boost_round=2000, early_stopping=100, n_splits=5

 모델 성능 (OOF AUC, 5-Fold):
   LightGBM  : 0.7384
   CatBoost  : 0.7398
   앙상블    : 0.7400  ← best

 실험 메모:
   - v1(파생변수 34개, 총 110 feature) 대비 v2(핵심 7개)가 동등 성능 유지
   - Data Leakage 수정: 결측치 대체 median, egg_median, embryo_max 모두 train 통계 사용
   - 앙상블 최적 가중치는 OOF AUC 기준 0.05 단위 그리드 탐색으로 결정
==============================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.preprocess import load_data, preprocess_full, feature_engineering_v2

TARGET    = "임신 성공 여부"
ID_COL    = "ID"
N_SPLITS  = 5
SEED      = 42
W_CAT     = 0.75   # best ensemble weight (CatBoost)
W_LGB     = 0.25   # best ensemble weight (LightGBM)

LGB_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "learning_rate":     0.05,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 20,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "seed":              SEED,
    "verbose":           -1,
}

CAT_PARAMS = {
    "iterations":            2000,
    "learning_rate":         0.05,
    "depth":                 6,
    "l2_leaf_reg":           3,
    "eval_metric":           "AUC",
    "early_stopping_rounds": 100,
    "random_seed":           SEED,
    "verbose":               500,
    "task_type":             "CPU",
    "bootstrap_type":        "Bernoulli",
    "subsample":             0.8,
    "colsample_bylevel":     0.8,
}


def train(data_dir="data", model_dir="models"):
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)

    # ── 데이터 로드 및 전처리 ────────────────────────────────────────────────
    print("데이터 로드 중...")
    train_df, test_df = load_data(data_dir)

    y = train_df[TARGET]
    X_raw      = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    X_test_raw = test_df.drop(columns=[ID_COL], errors="ignore")

    X_pp, label_encoders, train_stats = preprocess_full(X_raw)
    X_test_pp, _, _ = preprocess_full(
        X_test_raw, fit_label_encoders=label_encoders, train_stats=train_stats
    )

    X_train, fe_stats = feature_engineering_v2(X_pp)
    X_test,  _        = feature_engineering_v2(X_test_pp, train_stats=fe_stats)

    common = [c for c in X_train.columns if c in X_test.columns]
    X_train, X_test = X_train[common], X_test[common]
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # ── LightGBM 학습 ────────────────────────────────────────────────────────
    print("\nLightGBM 학습 시작...")
    oof_lgb  = np.zeros(len(X_train))
    pred_lgb = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx],       y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        m = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )
        oof_lgb[val_idx]  = m.predict(X_val)
        pred_lgb          += m.predict(X_test) / N_SPLITS
        m.save_model(str(model_dir / f"lgb_fold_{fold}.txt"))

        auc = roc_auc_score(y_val, oof_lgb[val_idx])
        print(f"  Fold {fold+1}  best_iter={m.best_iteration:>4}  AUC={auc:.4f}")

    print(f"  LightGBM OOF AUC: {roc_auc_score(y, oof_lgb):.4f}")

    # ── CatBoost 학습 ────────────────────────────────────────────────────────
    print("\nCatBoost 학습 시작...")
    oof_cat  = np.zeros(len(X_train))
    pred_cat = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx],       y.iloc[val_idx]

        m = CatBoostClassifier(**CAT_PARAMS)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

        oof_cat[val_idx]  = m.predict_proba(X_val)[:, 1]
        pred_cat          += m.predict_proba(X_test)[:, 1] / N_SPLITS
        m.save_model(str(model_dir / f"cat_fold_{fold}.cbm"))

        auc = roc_auc_score(y_val, oof_cat[val_idx])
        print(f"  Fold {fold+1}  best_iter={m.best_iteration_:>4}  AUC={auc:.4f}")

    print(f"  CatBoost OOF AUC: {roc_auc_score(y, oof_cat):.4f}")

    # ── 앙상블 평가 ──────────────────────────────────────────────────────────
    oof_ens = W_LGB * oof_lgb + W_CAT * oof_cat
    ens_auc = roc_auc_score(y, oof_ens)
    print(f"\n  앙상블 OOF AUC (w_cat={W_CAT}): {ens_auc:.4f}")

    # ── 메타 정보 저장 ───────────────────────────────────────────────────────
    meta = {
        "w_cat":         W_CAT,
        "w_lgb":         W_LGB,
        "oof_auc_lgb":   float(roc_auc_score(y, oof_lgb)),
        "oof_auc_cat":   float(roc_auc_score(y, oof_cat)),
        "oof_auc_ens":   float(ens_auc),
        "n_features":    len(common),
        "feature_names": common,
        "train_stats":   {k: float(v) for k, v in train_stats.items()},
        "fe_stats":      {k: float(v) for k, v in fe_stats.items()},
    }
    with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료: {model_dir}/")
    return oof_lgb, oof_cat, pred_lgb, pred_cat


if __name__ == "__main__":
    train()
