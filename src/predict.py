"""
predict.py — 예측 및 submission 파일 생성
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_auc_score,
)

try:
    import koreanize_matplotlib
except ImportError:
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

from preprocess import TARGET, ID_COL, DATA_DIR, RANDOM_STATE, full_pipeline
from train import train_all

# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    """Precision-Recall 곡선 기반 최적 F1 임계값 탐색"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s      = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s.argmax()
    return float(thresholds[min(best_idx, len(thresholds) - 1)]), float(f1s[best_idx])


def plot_confusion_matrix(ax, cm, title, threshold, auc, f1):
    total = cm.sum()
    cell_colors = [["#D6EAF8", "#FADBD8"], ["#FADBD8", "#D5F5E3"]]
    for i, j in itertools.product(range(2), range(2)):
        count = cm[i, j]
        pct   = count / total * 100
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                     facecolor=cell_colors[i][j], edgecolor="white", linewidth=2))
        ax.text(j + 0.5, 1.5 - i, f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=13, fontweight="bold")

    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["예측: 음성", "예측: 양성"], fontsize=11)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["실제: 양성", "실제: 음성"], fontsize=11)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_title(f"{title}\nAUC={auc:.5f}  F1={f1:.4f}  threshold={threshold:.3f}",
                 fontsize=12, fontweight="bold", pad=10)
    for label, pos in [("TN", (0.12, 1.88)), ("FP", (1.12, 1.88)),
                        ("FN", (0.12, 0.88)), ("TP", (1.12, 0.88))]:
        color = "#2980B9" if label in ("TN", "TP") else "#C0392B"
        ax.text(*pos, label, fontsize=9, color=color, alpha=0.7)


# ─────────────────────────────────────────────
# 성능 리포트
# ─────────────────────────────────────────────
def print_performance(oof_ml, oof_dl, oof_final, y_arr):
    print("=" * 60)
    print("  성능 비교  (OOF AUC)")
    print("=" * 60)
    results = {
        "ML 앙상블 (9모델)": roc_auc_score(y_arr, np.mean(list(oof_ml.values()), axis=0)),
        "스태킹 최종":       roc_auc_score(y_arr, oof_final),
    }
    for name, oof in oof_dl.items():
        if oof.sum() != 0:
            results[f"DL {name}"] = roc_auc_score(y_arr, oof)

    for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
        bar    = "█" * int((v - 0.72) * 500)
        marker = " ★" if k == "스태킹 최종" else ""
        print(f"  {k:<34} {v:.5f}  {bar}{marker}")
    print("=" * 60)


# ─────────────────────────────────────────────
# 혼동행렬 시각화
# ─────────────────────────────────────────────
def plot_all_confusion_matrices(oof_ml, oof_dl, oof_final, y_arr):
    eval_targets = {
        "ML 앙상블\n(XGB+CAT+LGB)":   np.mean(list(oof_ml.values()), axis=0),
        "스태킹 최종\n(Meta-LGB+LR)": oof_final,
    }
    for dl_name in ["MLP", "FT_Transformer", "TabNet"]:
        if dl_name in oof_dl and oof_dl[dl_name].sum() != 0:
            eval_targets[f"DL {dl_name}"] = oof_dl[dl_name]

    n_plots = len(eval_targets)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("혼동행렬 비교 (OOF 기준, 최적 임계값 적용)", fontsize=15, fontweight="bold", y=1.02)

    print("=" * 62)
    print("  혼동행렬 요약 (OOF 기준)")
    print("=" * 62)
    print(f"  {'모델':<22} {'threshold':>10}  {'F1':>8}  {'정밀도':>8}  {'재현율':>8}")
    print("-" * 62)

    for ax, (name, y_prob) in zip(axes, eval_targets.items()):
        auc_val   = roc_auc_score(y_arr, y_prob)
        thr, f1   = find_best_threshold(y_arr, y_prob)
        y_pred    = (y_prob >= thr).astype(int)
        cm        = confusion_matrix(y_arr, y_pred)
        report    = classification_report(y_arr, y_pred, output_dict=True)
        precision = report["1"]["precision"]
        recall    = report["1"]["recall"]
        plot_confusion_matrix(ax, cm, name.replace("\n", " "), thr, auc_val, f1)
        print(f"  {name.replace(chr(10), ' '):<22} {thr:>10.3f}  {f1:>8.4f}  {precision:>8.4f}  {recall:>8.4f}")

    print("=" * 62)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ confusion_matrix.png 저장")

    # 최종 모델 상세 리포트
    print("\n=== 스태킹 최종 모델 상세 분류 리포트 ===")
    thr_final, _ = find_best_threshold(y_arr, oof_final)
    y_pred_final = (oof_final >= thr_final).astype(int)
    print(classification_report(y_arr, y_pred_final, target_names=["음성(0)", "양성(1)"]))


# ─────────────────────────────────────────────
# submission 생성
# ─────────────────────────────────────────────
def make_submission(pred_final, pred_ml_mean, sub_df, prefix="vC"):
    sub_final = sub_df.copy()
    sub_final["probability"] = pred_final
    path_final = f"submission_{prefix}_final.csv"
    sub_final.to_csv(path_final, index=False)
    print(f"✅ {path_final} 저장 완료")

    sub_ml = sub_df.copy()
    sub_ml["probability"] = pred_ml_mean
    path_ml = f"submission_{prefix}_ml_only.csv"
    sub_ml.to_csv(path_ml, index=False)
    print(f"✅ {path_ml} 저장 완료")

    print(sub_final.head())
    return sub_final


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def main():
    # 데이터 로드
    train_df = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    test_df  = pd.read_csv(DATA_DIR / "test.csv",  encoding="utf-8-sig")
    sub_df   = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns  = [c.strip() for c in test_df.columns]
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    # 전처리
    X_raw      = train_df.drop(columns=[ID_COL, TARGET], errors="ignore")
    y          = train_df[TARGET]
    X_test_raw = test_df.drop(columns=[ID_COL], errors="ignore")

    X_pp,      train_medians = full_pipeline(X_raw,      fit=True)
    X_test_pp, _             = full_pipeline(X_test_raw, medians=train_medians, fit=False)

    common_cols = [c for c in X_pp.columns if c in X_test_pp.columns]
    X_pp      = X_pp[common_cols]
    X_test_pp = X_test_pp[common_cols]
    print(f"전체 피처: {X_pp.shape[1]}개")

    # 학습
    (oof_ml, pred_ml, oof_dl, pred_dl,
     oof_final, pred_final, feat_imp) = train_all(X_pp, X_test_pp, y, train_medians)

    y_arr = y.values

    # 성능 비교 출력
    print_performance(oof_ml, oof_dl, oof_final, y_arr)

    # 혼동행렬 시각화
    plot_all_confusion_matrices(oof_ml, oof_dl, oof_final, y_arr)

    # submission 저장
    pred_ml_mean = np.mean(list(pred_ml.values()), axis=0)
    make_submission(pred_final, pred_ml_mean, sub_df, prefix="vC")


if __name__ == "__main__":
    main()
