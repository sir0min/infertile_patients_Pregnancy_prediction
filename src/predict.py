"""저장된 모델로 예측 및 submission 생성"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.preprocess import load_data, preprocess_full, feature_engineering_v2

ID_COL   = "ID"
N_SPLITS = 5


def predict(data_dir="data", model_dir="models", output_dir="submission"):
    output_dir = Path(output_dir)
    model_dir  = Path(model_dir)
    output_dir.mkdir(exist_ok=True)

    # ── 메타 정보 로드 ───────────────────────────────────────────────────────
    with open(model_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    W_CAT = meta["w_cat"]
    W_LGB = meta["w_lgb"]
    feature_names = meta["feature_names"]
    print(f"앙상블 가중치  LGB={W_LGB}  CAT={W_CAT}")
    print(f"OOF AUC  LGB={meta['oof_auc_lgb']:.4f}  CAT={meta['oof_auc_cat']:.4f}  ENS={meta['oof_auc_ens']:.4f}")

    # ── 테스트 데이터 전처리 ─────────────────────────────────────────────────
    print("\n테스트 데이터 로드 및 전처리 중...")
    _, test_df = load_data(data_dir)
    test_ids   = test_df[ID_COL]
    X_test_raw = test_df.drop(columns=[ID_COL], errors="ignore")

    train_stats = meta.get("train_stats", {})
    fe_stats    = meta.get("fe_stats", {})
    X_test_pp, _, _ = preprocess_full(X_test_raw, train_stats=train_stats)
    X_test, _       = feature_engineering_v2(X_test_pp, train_stats=fe_stats)
    X_test       = X_test[[c for c in feature_names if c in X_test.columns]]
    print(f"  테스트 피처: {X_test.shape}")

    # ── 모델 로드 및 예측 ────────────────────────────────────────────────────
    pred_lgb = np.zeros(len(X_test))
    pred_cat = np.zeros(len(X_test))

    print("\nLightGBM 예측 중...")
    for i in range(N_SPLITS):
        m = lgb.Booster(model_file=str(model_dir / f"lgb_fold_{i}.txt"))
        pred_lgb += m.predict(X_test) / N_SPLITS
        print(f"  Fold {i+1} 완료")

    print("\nCatBoost 예측 중...")
    for i in range(N_SPLITS):
        m = CatBoostClassifier()
        m.load_model(str(model_dir / f"cat_fold_{i}.cbm"))
        pred_cat += m.predict_proba(X_test)[:, 1] / N_SPLITS
        print(f"  Fold {i+1} 완료")

    # ── 앙상블 및 저장 ───────────────────────────────────────────────────────
    pred_ensemble = W_LGB * pred_lgb + W_CAT * pred_cat

    submission = pd.DataFrame({"ID": test_ids, "probability": pred_ensemble})
    out_path   = output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)

    print(f"\nsubmission 저장 완료: {out_path}")
    print(f"  예측 확률  min={pred_ensemble.min():.4f}  max={pred_ensemble.max():.4f}  mean={pred_ensemble.mean():.4f}")
    print(submission.head())


if __name__ == "__main__":
    predict()
