"""
train.py — 모델 학습 (ML + DL Seed 앙상블 + 2단계 스태킹)
"""

import warnings, os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import shap

from preprocess import (
    TARGET, ID_COL, DATA_DIR, RANDOM_STATE,
    full_pipeline,
)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
N_SPLITS    = 5
N_OPTUNA    = 30
USE_CACHED_PARAMS = True   # True → Optuna 재탐색 생략

np.random.seed(RANDOM_STATE)

USE_GPU    = torch.cuda.is_available()
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
LGB_DEVICE = "gpu"  if USE_GPU else "cpu"
CAT_TASK   = "GPU"  if USE_GPU else "CPU"
DEVICE     = torch.device("cuda" if USE_GPU else "cpu")
BATCH_SIZE = 8192 if USE_GPU else 2048

print(f"Device: {DEVICE}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# Optuna 캐시 파라미터 (exp22 최종)
# ─────────────────────────────────────────────
CACHED_XGB = {
    "n_estimators": 1800, "learning_rate": 0.01681590206217679,
    "max_depth": 6, "subsample": 0.7219206134659216,
    "colsample_bytree": 0.6872602657464894, "reg_alpha": 7.452848201378795,
    "reg_lambda": 9.211281496853013, "min_child_weight": 9,
    "gamma": 4.14379616379793, "scale_pos_weight": 2.064196846705621,
}
CACHED_CAT = {
    "iterations": 1200, "learning_rate": 0.02441661265474018,
    "depth": 7, "l2_leaf_reg": 7.790676508309964,
    "bagging_temperature": 1.0157813933009723,
    "random_strength": 1.1252522473159237, "border_count": 91,
}
CACHED_LGB = {
    "n_estimators": 1300, "learning_rate": 0.020263188298172464,
    "num_leaves": 150, "max_depth": 4,
    "subsample": 0.91206952631544, "colsample_bytree": 0.9428380005720536,
    "reg_alpha": 0.9061803492912888, "reg_lambda": 3.7279273338107584,
    "min_child_samples": 100, "boosting_type": "gbdt",
}

# DL 이진형 피처 집합
DL_BINARY_SET = {
    "시술 유형", "배란 자극 여부", "단일 배아 이식 여부",
    "착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부",
    "착상 전 유전 진단 사용 여부", "동결 배아 사용 여부", "신선 배아 사용 여부",
    "기증 배아 사용 여부", "대리모 여부",
    "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
    "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
    "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", "불임 원인 - 여성 요인",
    "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
    "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
    "is_D5", "수정률_양호", "ICSI수정률_양호", "이식수_최적", "최적이식_D5",
    "고령저반응", "젊고고수확", "고령IVF", "젊은DI", "고령반복시술", "순수동결_주기",
    "다배아이식_플래그", "초회시술", "초회IVF", "출산_경험", "IVF_출산_경험",
    "복합_불임_여부", "이식일_D5이상", "정확히_D5", "D5_최적이식_복합",
    "이식일_결측", "난자_채취_결측", "배아_해동_결측", "임신_시도_연수_결측",
    "수정률_결측", "ICSI_미시행", "D5_결측",
    "누적실패_패턴", "최적_시술_콤보", "출산후_재시술", "고위험_복합", "IVF_고령_초회",
}


# ─────────────────────────────────────────────
# DL 모델 정의
# ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []; in_d = n_in
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
    def __init__(self, n_feat, d=64, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.emb = nn.Linear(n_feat, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=d * 4,
                                          dropout=drop, batch_first=True)
        self.tf  = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(d // 2, 1)
        )

    def forward(self, x):
        return self.cls(self.tf(self.emb(x).unsqueeze(1)).squeeze(1)).squeeze(-1)


def scale_dl(X_tr, X_va, X_te, cont_idx):
    """fold 내 train split에서만 fit → validation/test는 transform만 (leakage 없음)"""
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    sc = RobustScaler()
    X_tr[:, cont_idx] = sc.fit_transform(X_tr[:, cont_idx])
    X_va[:, cont_idx] = sc.transform(X_va[:, cont_idx])
    X_te[:, cont_idx] = sc.transform(X_te[:, cont_idx])
    return X_tr, X_va, X_te, sc


def train_nn(model, X_tr, y_tr, X_va, y_va, epochs=100, lr=1e-3, patience=15):
    dev   = next(model.parameters()).device
    pos_w = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], dtype=torch.float32).to(dev)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds    = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32))
    dl    = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                       pin_memory=USE_GPU, num_workers=2 if USE_GPU else 0)

    best_auc, best_state, no_imp = 0, None, 0
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(dev)

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(X_va_t)).cpu().numpy()
        auc = roc_auc_score(y_va, vp)
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def predict_nn(model, X, batch=8192):
    model.eval()
    dev   = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(dev)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)


# ─────────────────────────────────────────────
# Optuna 목적 함수
# ─────────────────────────────────────────────
def make_xgb_obj(X_ml, y_arr, skf):
    def xgb_obj(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 500, 3000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10., log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10., log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 0, 5),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.5, 4.0),
            tree_method="hist", device=XGB_DEVICE, eval_metric="auc",
            early_stopping_rounds=50, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        oof = np.zeros(len(X_ml))
        for tr, va in skf.split(X_ml, y_arr):
            m = xgb.XGBClassifier(**p)
            m.fit(X_ml[tr], y_arr[tr], eval_set=[(X_ml[va], y_arr[va])], verbose=False)
            oof[va] = m.predict_proba(X_ml[va])[:, 1]
        return roc_auc_score(y_arr, oof)
    return xgb_obj


def make_cat_obj(X_ml, y_arr, skf):
    def cat_obj(trial):
        p = dict(
            iterations=trial.suggest_int("iterations", 500, 3000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1., 10.),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0, 2),
            random_strength=trial.suggest_float("random_strength", 0, 2),
            border_count=trial.suggest_int("border_count", 32, 255),
            auto_class_weights="Balanced", eval_metric="AUC", task_type=CAT_TASK,
            early_stopping_rounds=50, random_seed=RANDOM_STATE, verbose=0,
        )
        oof = np.zeros(len(X_ml))
        for tr, va in skf.split(X_ml, y_arr):
            m = CatBoostClassifier(**p)
            m.fit(X_ml[tr], y_arr[tr], eval_set=(X_ml[va], y_arr[va]), use_best_model=True)
            oof[va] = m.predict_proba(X_ml[va])[:, 1]
        return roc_auc_score(y_arr, oof)
    return cat_obj


def make_lgb_obj(X_ml, y_arr, skf):
    def lgb_obj(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 500, 3000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10., log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10., log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            boosting_type=trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            class_weight="balanced", device=LGB_DEVICE,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        oof = np.zeros(len(X_ml))
        for tr, va in skf.split(X_ml, y_arr):
            m = lgb.LGBMClassifier(**p)
            m.fit(X_ml[tr], y_arr[tr], eval_set=[(X_ml[va], y_arr[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)])
            oof[va] = m.predict_proba(X_ml[va])[:, 1]
        return roc_auc_score(y_arr, oof)
    return lgb_obj


# ─────────────────────────────────────────────
# 학습 메인
# ─────────────────────────────────────────────
def train_all(X_pp, X_test_pp, y, train_medians):
    """
    Returns:
        oof_ml, pred_ml  : ML 모델별 OOF / test 예측
        oof_dl, pred_dl  : DL 모델별 OOF / test 예측
        oof_final        : 스태킹 최종 OOF 예측
        pred_final       : 스태킹 최종 test 예측
        feat_imp         : SHAP 피처 중요도 DataFrame
    """
    y_arr = y.values
    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    X_arr_full  = X_pp.fillna(0).values.astype(np.float32)
    X_test_full = X_test_pp.fillna(0).values.astype(np.float32)

    # ── SHAP 피처 중요도 계산 ───────────────────────────
    print("\nSHAP 계산용 LGB 학습 중...")
    m_fi = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    )
    m_fi.fit(X_arr_full, y_arr, callbacks=[lgb.log_evaluation(False)])

    print("SHAP 값 계산 중...")
    explainer   = shap.TreeExplainer(m_fi)
    shap_values = explainer.shap_values(X_arr_full)
    sv       = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_imp = np.abs(sv).mean(axis=0)
    lgb_imp  = m_fi.feature_importances_

    feat_imp = pd.DataFrame({
        "feature":  X_pp.columns,
        "shap_imp": shap_imp,
        "lgb_imp":  lgb_imp,
    }).sort_values("shap_imp", ascending=False).reset_index(drop=True)
    feat_imp["shap_rank"]       = range(1, len(feat_imp) + 1)
    feat_imp["shap_cumulative"] = (feat_imp["shap_imp"].cumsum() / feat_imp["shap_imp"].sum() * 100).round(2)
    feat_imp.to_csv("feature_importance_vC_shap.csv", index=False)
    print("✅ feature_importance_vC_shap.csv 저장")

    # 피처셋: SHAP > 0
    ml_features = feat_imp[feat_imp["shap_imp"] > 0]["feature"].tolist()
    dl_features = ml_features.copy()
    print(f"ML/DL 피처셋 (SHAP>0): {len(ml_features)}개")

    X_ml      = X_pp[ml_features].fillna(0).values.astype(np.float32)
    X_test_ml = X_test_pp[ml_features].fillna(0).values.astype(np.float32)
    X_dl_raw  = X_pp[dl_features].fillna(0).values.astype(np.float32)
    X_test_dl = X_test_pp[dl_features].fillna(0).values.astype(np.float32)

    cont_idx = [i for i, c in enumerate(dl_features) if c not in DL_BINARY_SET]

    # ── ML Optuna ──────────────────────────────────────
    if USE_CACHED_PARAMS:
        bp_xgb, bp_cat, bp_lgb = CACHED_XGB, CACHED_CAT, CACHED_LGB
        print("캐시 파라미터 사용 (XGB / CAT / LGB)")
    else:
        print("\n=== XGBoost Optuna ===")
        s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        s.optimize(make_xgb_obj(X_ml, y_arr, skf), n_trials=N_OPTUNA, show_progress_bar=True)
        bp_xgb = s.best_params; print(f"XGB best AUC={s.best_value:.5f}")

        print("\n=== CatBoost Optuna ===")
        s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        s.optimize(make_cat_obj(X_ml, y_arr, skf), n_trials=N_OPTUNA, show_progress_bar=True)
        bp_cat = s.best_params; print(f"CAT best AUC={s.best_value:.5f}")

        print("\n=== LightGBM Optuna ===")
        s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        s.optimize(make_lgb_obj(X_ml, y_arr, skf), n_trials=N_OPTUNA, show_progress_bar=True)
        bp_lgb = s.best_params; print(f"LGB best AUC={s.best_value:.5f}")

    # ── ML Seed 앙상블 (9모델) ─────────────────────────
    print(f"\n=== ML Seed 앙상블 (9모델, 피처 {len(ml_features)}개) ===")
    xgb_fp = {**bp_xgb, "tree_method": "hist", "device": XGB_DEVICE,
               "eval_metric": "auc", "early_stopping_rounds": 100,
               "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0}
    cat_fp = {**bp_cat, "auto_class_weights": "Balanced", "eval_metric": "AUC",
               "task_type": CAT_TASK, "early_stopping_rounds": 100,
               "random_seed": RANDOM_STATE, "verbose": 0}
    lgb_fp = {**bp_lgb, "class_weight": "balanced", "device": LGB_DEVICE, "n_jobs": -1, "verbose": -1}

    ml_configs = {
        "XGB_42":   ("xgb", {**xgb_fp, "random_state": 42}),
        "XGB_2024": ("xgb", {**xgb_fp, "random_state": 2024}),
        "XGB_777":  ("xgb", {**xgb_fp, "random_state": 777}),
        "CAT_42":   ("cat", {**cat_fp, "random_seed": 42}),
        "CAT_2024": ("cat", {**cat_fp, "random_seed": 2024}),
        "CAT_777":  ("cat", {**cat_fp, "random_seed": 777}),
        "LGB_42":   ("lgb", {**lgb_fp, "random_state": 42}),
        "LGB_2024": ("lgb", {**lgb_fp, "random_state": 2024}),
        "LGB_777":  ("lgb", {**lgb_fp, "random_state": 777}),
    }

    oof_ml = {}; pred_ml = {}
    for name, (mtype, p) in ml_configs.items():
        oof  = np.zeros(len(X_ml))
        pred = np.zeros(len(X_test_ml))
        for tr, va in skf.split(X_ml, y_arr):
            if mtype == "xgb":
                m = xgb.XGBClassifier(**p)
                m.fit(X_ml[tr], y_arr[tr], eval_set=[(X_ml[va], y_arr[va])], verbose=False)
            elif mtype == "cat":
                m = CatBoostClassifier(**p)
                m.fit(X_ml[tr], y_arr[tr], eval_set=(X_ml[va], y_arr[va]), use_best_model=True)
            else:
                m = lgb.LGBMClassifier(**p)
                m.fit(X_ml[tr], y_arr[tr], eval_set=[(X_ml[va], y_arr[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)])
            oof[va]  = m.predict_proba(X_ml[va])[:, 1]
            pred    += m.predict_proba(X_test_ml)[:, 1] / N_SPLITS
        oof_ml[name]  = oof
        pred_ml[name] = pred
        print(f"  {name:<12} OOF AUC={roc_auc_score(y_arr, oof):.5f}")

    n_ml = len(ml_configs)
    print(f"ML 앙상블 AUC={roc_auc_score(y_arr, np.mean(list(oof_ml.values()), axis=0)):.5f}")

    # ── DL 모델 ────────────────────────────────────────
    n_dl = X_dl_raw.shape[1]
    oof_dl = {}; pred_dl = {}

    # MLP
    print("\n=== MLP 학습 ===")
    oof_mlp = np.zeros(len(X_dl_raw)); pred_mlp = np.zeros(len(X_test_dl))
    for fold, (tr, va) in enumerate(skf.split(X_dl_raw, y_arr), 1):
        X_tr, X_va, X_te, _ = scale_dl(X_dl_raw[tr], X_dl_raw[va], X_test_dl, cont_idx)
        model = MLP(n_dl).to(DEVICE)
        model, best_auc = train_nn(model, X_tr, y_arr[tr], X_va, y_arr[va], epochs=100, patience=15)
        oof_mlp[va] = predict_nn(model, X_va)
        pred_mlp   += predict_nn(model, X_te) / N_SPLITS
        print(f"  Fold {fold} MLP AUC={best_auc:.5f}")
        if USE_GPU: torch.cuda.empty_cache()
    oof_dl["MLP"] = oof_mlp; pred_dl["MLP"] = pred_mlp
    print(f"MLP OOF AUC={roc_auc_score(y_arr, oof_mlp):.5f}")

    # TabNet
    print("\n=== TabNet 학습 ===")
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        oof_tn = np.zeros(len(X_dl_raw)); pred_tn = np.zeros(len(X_test_dl))
        for fold, (tr, va) in enumerate(skf.split(X_dl_raw, y_arr), 1):
            X_tr, X_va, X_te, _ = scale_dl(X_dl_raw[tr], X_dl_raw[va], X_test_dl, cont_idx)
            clf = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5, gamma=1.3,
                seed=RANDOM_STATE, verbose=0,
                device_name="cuda" if USE_GPU else "cpu",
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
                momentum=0.02
            )
            clf.fit(X_tr, y_arr[tr], eval_set=[(X_va, y_arr[va])], eval_metric=["auc"],
                    max_epochs=200, patience=20, batch_size=1024, virtual_batch_size=128, weights=1)
            oof_tn[va] = clf.predict_proba(X_va)[:, 1]
            pred_tn   += clf.predict_proba(X_te)[:, 1] / N_SPLITS
            print(f"  Fold {fold} TabNet AUC={roc_auc_score(y_arr[va], oof_tn[va]):.5f}")
            if USE_GPU: torch.cuda.empty_cache()
        oof_dl["TabNet"] = oof_tn; pred_dl["TabNet"] = pred_tn
        print(f"TabNet OOF AUC={roc_auc_score(y_arr, oof_tn):.5f}")
    except ImportError:
        print("pytorch-tabnet 미설치 — TabNet 제외")
        oof_dl["TabNet"] = np.zeros(len(X_dl_raw)); pred_dl["TabNet"] = np.zeros(len(X_test_dl))

    # FT-Transformer
    print("\n=== FT-Transformer 학습 ===")
    oof_ftt = np.zeros(len(X_dl_raw)); pred_ftt = np.zeros(len(X_test_dl))
    for fold, (tr, va) in enumerate(skf.split(X_dl_raw, y_arr), 1):
        X_tr, X_va, X_te, _ = scale_dl(X_dl_raw[tr], X_dl_raw[va], X_test_dl, cont_idx)
        model = FTTransformer(n_dl).to(DEVICE)
        model, best_auc = train_nn(model, X_tr, y_arr[tr], X_va, y_arr[va], epochs=80, patience=15, lr=5e-4)
        oof_ftt[va] = predict_nn(model, X_va)
        pred_ftt   += predict_nn(model, X_te) / N_SPLITS
        print(f"  Fold {fold} FTT AUC={best_auc:.5f}")
        if USE_GPU: torch.cuda.empty_cache()
    oof_dl["FT_Transformer"] = oof_ftt; pred_dl["FT_Transformer"] = pred_ftt
    print(f"FT-Transformer OOF AUC={roc_auc_score(y_arr, oof_ftt):.5f}")

    # ── 2단계 스태킹 ────────────────────────────────────
    print("\n=== 2단계 스태킹 ===")
    all_oof  = {**oof_ml, **oof_dl}
    all_pred = {**pred_ml, **pred_dl}
    n_models = len(all_oof)
    ml_idx   = list(range(n_ml))
    dl_idx   = list(range(n_ml, n_models))

    def build_meta(mat, ml_idx, dl_idx):
        ml_m = mat[:, ml_idx].mean(axis=1, keepdims=True)
        dl_m = mat[:, dl_idx].mean(axis=1, keepdims=True) if dl_idx else np.zeros((len(mat), 1))
        return np.hstack([mat, ml_m, dl_m,
                          mat.std(axis=1, keepdims=True),
                          mat[:, ml_idx].std(axis=1, keepdims=True),
                          ml_m - dl_m])

    meta_train_base = np.column_stack(list(all_oof.values()))
    meta_test_base  = np.column_stack(list(all_pred.values()))
    meta_train = build_meta(meta_train_base, ml_idx, dl_idx)
    meta_test  = build_meta(meta_test_base,  ml_idx, dl_idx)
    print(f"메타 피처: {meta_train.shape[1]}개")

    mp = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, max_depth=4,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
              class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    oof_mlgb = np.zeros(len(meta_train)); pred_mlgb = np.zeros(len(meta_test))
    for tr, va in skf.split(meta_train, y_arr):
        m = lgb.LGBMClassifier(**mp)
        m.fit(meta_train[tr], y_arr[tr], eval_set=[(meta_train[va], y_arr[va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)])
        oof_mlgb[va]  = m.predict_proba(meta_train[va])[:, 1]
        pred_mlgb    += m.predict_proba(meta_test)[:, 1] / N_SPLITS
    print(f"Meta-LGB OOF AUC={roc_auc_score(y_arr, oof_mlgb):.5f}")

    oof_mlr = np.zeros(len(meta_train)); pred_mlr = np.zeros(len(meta_test))
    for tr, va in skf.split(meta_train, y_arr):
        sc  = StandardScaler()
        mtr = sc.fit_transform(meta_train[tr])
        mva = sc.transform(meta_train[va])
        mte = sc.transform(meta_test)
        lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
        lr.fit(mtr, y_arr[tr])
        oof_mlr[va]  = lr.predict_proba(mva)[:, 1]
        pred_mlr    += lr.predict_proba(mte)[:, 1] / N_SPLITS
    print(f"Meta-LR  OOF AUC={roc_auc_score(y_arr, oof_mlr):.5f}")

    def neg_auc(w):
        w = np.array(w); w = w / w.sum()
        return -roc_auc_score(y_arr, w[0] * oof_mlgb + w[1] * oof_mlr)

    res          = minimize(neg_auc, x0=[0.7, 0.3], bounds=[(0, 1)] * 2, method="L-BFGS-B")
    w_f          = np.array(res.x) / np.array(res.x).sum()
    oof_final    = w_f[0] * oof_mlgb  + w_f[1] * oof_mlr
    pred_final   = w_f[0] * pred_mlgb + w_f[1] * pred_mlr
    print(f"\n[최종] OOF AUC={roc_auc_score(y_arr, oof_final):.5f}")
    print(f"Meta-LGB={w_f[0]:.3f}, Meta-LR={w_f[1]:.3f}")

    return oof_ml, pred_ml, oof_dl, pred_dl, oof_final, pred_final, feat_imp


# ─────────────────────────────────────────────
# 직접 실행 시
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    test_df  = pd.read_csv(DATA_DIR / "test.csv",  encoding="utf-8-sig")
    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns  = [c.strip() for c in test_df.columns]

    X_raw      = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    y          = train_df[TARGET]
    X_test_raw = test_df.drop(columns=[ID_COL], errors="ignore")

    X_pp,      train_medians = full_pipeline(X_raw,      fit=True)
    X_test_pp, _             = full_pipeline(X_test_raw, medians=train_medians, fit=False)

    common_cols = [c for c in X_pp.columns if c in X_test_pp.columns]
    X_pp      = X_pp[common_cols]
    X_test_pp = X_test_pp[common_cols]

    (oof_ml, pred_ml, oof_dl, pred_dl,
     oof_final, pred_final, feat_imp) = train_all(X_pp, X_test_pp, y, train_medians)

    # 결과 저장 (predict.py에서 사용)
    np.save("pred_final.npy",  pred_final)
    np.save("pred_ml_mean.npy", np.mean(list(pred_ml.values()), axis=0))
    feat_imp.to_csv("feature_importance_vC_shap.csv", index=False)
    print("\n학습 완료 — pred_final.npy, pred_ml_mean.npy 저장")
