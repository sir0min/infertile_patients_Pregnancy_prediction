# 2. 모델링

## 1. 단독 모델

| 모델 | OOF AUC | 비고 |
| --- | --- | --- |
| LightGBM | - | GBDT 계열, 기본 feature |
| XGBoost | - | GBDT 계열, 기본 feature |
| CatBoost | - | GBDT 계열, 기본 feature |
| RandomForest | - | 트리 기반 |
| MLP (PyTorch) | - | 딥러닝 |
| **앙상블 (baseline)** | **0.6736** | **개별 모델 평균** |

> GBDT 계열(LGB, XGB, CAT)이 RandomForest, MLP 대비 압도적으로 우수함을 확인

---

## 2. 앙상블 모델

### 4/24 (금) — XGBoost × CatBoost 앙상블 방법론 비교

#### OOF (Out-of-Fold) 개념

- **정의**: 교차 검증(K-Fold) 과정에서 모델이 훈련에 사용하지 않은, 즉 검증(Validation) 폴드 데이터에 대해 예측한 값들의 집합
- **작동 방식**: 데이터를 K개의 폴드로 나누어 K번 학습할 때, 각 학습 과정에서 검증 데이터로 남겨둔 부분에 대한 예측을 합쳐 전체 학습 데이터에 대한 'OOF 예측값'을 생성
- **목적**: 훈련 데이터 전체에 대한 예측을 통해 모델 성능을 정확히 평가하거나, OOF 값을 새로운 피처로 활용하여 스태킹 모델의 성능을 높이는 데 활용

#### 앙상블 방법 비교

| 방법 | 핵심 아이디어 |
| --- | --- |
| A. Simple Average | 두 모델을 동등하게 1/2씩 평균 |
| B. Weighted Average | OOF AUC가 높은 모델에 더 많은 비중 부여 |
| C. Stacking (LR) | 메타 모델이 "어느 상황에서 XGB vs CAT 중 누구를 믿을지" 학습 |
| D. Rank Average | 확률 대신 순위로 변환 후 평균 — 두 모델의 스케일 차이 제거 |

**C. Stacking (LR) 구조**

```
방법 A/B/D                    방법 C (Stacking)
─────────────────             ─────────────────────────────
XGBoost  ──┐                  XGBoost  ──┐
           ├→ 수식으로 계산                ├→ OOF 예측값 → Logistic Regression
CatBoost ──┘                  CatBoost ──┘        (메타 모델, 3번째 모델)
                                                       ↓
                                                  최종 예측
```

**D. Rank Average 원리**

두 모델의 예측 확률 분포가 다를 때 단순 평균은 분포가 넓은 모델의 극단값에 지배될 수 있음.

- XGBoost 예측: 대부분 **0.3~0.6** 사이에 몰림
- CatBoost 예측: **0.1~0.9**로 넓게 퍼짐

**단계별 처리:**
1. **원래 예측 확률** — 두 모델의 분포 범위가 다름 (XGB: 0.20~0.82, CAT: 0.05~0.95)
2. **순위(rank)로 변환** — 각 모델 내에서 낮은 순서대로 순위를 매긴 후 `(순위 - 1) / (샘플 수 - 1)` 공식으로 [0,1] 척도로 통일
3. **순위 평균 → 최종 예측** — 두 모델의 순위를 평균 내고 다시 [0,1]로 정규화

---

### 4/25 (토) — 점수 결과 불량

- 앙상블 시도 결과 기대치 미달

---

### 4/26 (일) — 파생변수 추가 및 Seed 앙상블

- **파생변수 29개 추가** (기존 피처 기반 도메인 피처)
- **Seed 앙상블** 도입 — 동일 모델을 여러 랜덤 시드로 학습 후 평균

---

### 4/27 (월) — 딥러닝 도입 및 피처 엔지니어링 확장

#### [exp16] ML + DL (TabNet + MLP + FT-Transformer)
- 결측치 처리 재검토 → 도메인 관점 해석 후 재채움

#### [exp17] ML + DL (TabNet + MLP + FT-Transformer)

- LGBM feature importance 기반 피처셋 추출 (상위 중요도 피처만 선택)

**`시술 시기 코드` 피처 중요도가 높은 이유 분석:**

> "이식 경과일 분포"와 "환자 나이 분포"를 동시에 압축해서 담고 있기 때문

코드가 시술이 이루어진 기간(연도/분기)을 나타내며:
- **D5(배반포) 비율이 점점 줄고 D3 비율이 점점 늘어나는 단조로운 패턴** — 시술 트렌드 변화 반영
- 평균 나이 지수도 오래된 코드(TRDQAZ 1.386) → 최신 코드(TRVNRY 1.247)로 갈수록 낮아짐

**관련 파생 변수 생성:**
- 개별 상관계수는 작지만 `코드_시간×나이`가 −0.111로 의미 있는 신호 확인
- "오래된 시기의 고령 환자일수록 성공률이 낮다"는 패턴 포착
- 트리 모델과 딥러닝 모두에서 교호작용을 통한 추가 성능 향상 가능성 있음

**exp17 코드 분석 — DL 모델 성능:**

```
스태킹 최종                         0.74058  ★   
베이스라인 (seed 앙상블)              0.73998  ↑베이스   
DL MLP                           0.73983   
DL FT_Transformer                0.73959  
ML 앙상블 (Optuna+Seed)            0.73955
DL TabNet                        0.73253
```

> **TabNet이 0.73253으로 ML 대비 -0.007 낮고, FTT/MLP도 ML보다 낮음**  
> → 현재 DL이 스태킹에서 노이즈로 작용하고 있을 가능성이 높음

#### [exp18] TabNet 제거 + 스태킹 재계산

- **결론: 제거하면 안 됨**
- TabNet을 제거하면 오히려 성능이 약간 낮아짐
- TabNet의 OOF AUC(0.73253)는 낮지만, 다른 모델들과 **상관관계가 낮아 다양성 기여** 중
- → TabNet 다시 포함

---

### 4/28 (화) — 도메인 피처 추가 및 모델 확장

#### [exp19] 도메인 피처 추가 & HistGBM 추가

**1. 도메인 피처 6개 추가 (`add_new_features` 그룹 F)**
- `누적실패_패턴`, `최적_시술_콤보`, `출산후_재시술`, `고위험_복합`, `수정률_D5_복합`, `IVF_고령_초회`
- 전체 피처: 162개 → **168개**

**2. HistGradientBoosting × 3 seed 추가** (`HGB_42`, `HGB_2024`, `HGB_777`)
- ML 모델: 9개 → **12개**

```
HGB_42       OOF AUC=0.73849
HGB_2024     OOF AUC=0.73868
HGB_777      OOF AUC=0.73864
ML 앙상블 AUC=0.73951
```

- **결론: HistGBM 없어도 됨** — HGB 3개 모두 0.738대로 앙상블 평균을 끌어내림
- 이유: 현재 피처셋은 이미 결측치를 전처리 단계에서 모두 채워놔 HGB의 native 결측치 처리 장점이 발휘되지 않음 → 즉시 제거

#### [exp20] ML 피처셋 Top65 확대 & DL 피처셋 Top85
- TabNet `batch_size=1024`, `vbs=128` 재학습
- LGBM feature importance 기반 피처셋 추출

#### [exp21] ML 피처셋 Top65 & DL 피처셋 전체
- TabNet `batch_size=1024`, `vbs=128` 재학습
- **결론: 딥러닝은 전체 피처 학습이 중요함**

#### [exp22] ML_DL_shap_feature_importance
- LGBM feature importance → **SHAP feature importance**로 전환
- 150개 피처 전체 사용 (중요도 0인 14개 제거)

#### [exp23] ML Top66 / DL Top66
- SHAP 기반 피처 선택
- ML Top66 / DL Top66 동일 피처셋

#### [exp24] ML Top45 / DL Top66
- SHAP 기반 피처 선택
- ML과 DL에 서로 다른 피처셋 적용

---

### 4/29 (수) — 최종 튜닝 및 마무리

#### [exp25] ML + DL (메타4모델 + DL seed앙상블 + OOF TargetEncoding + OneHot)
- **결론: 효과 없음**

#### [exp26] ML + DL (Optuna 재탐색)

- Optuna 재탐색으로 최적 파라미터 갱신

```python
# XGB best AUC=0.74033
CACHED_XGB = {
    'n_estimators': 2100, 'learning_rate': 0.01014, 'max_depth': 6,
    'subsample': 0.7387, 'colsample_bytree': 0.7325,
    'reg_alpha': 4.272, 'reg_lambda': 6.904,
    'min_child_weight': 7, 'gamma': 4.735, 'scale_pos_weight': 1.531
}

# CAT best AUC=0.74033
CACHED_CAT = {
    'iterations': 2200, 'learning_rate': 0.04300, 'depth': 6,
    'l2_leaf_reg': 6.788, 'bagging_temperature': 1.133,
    'random_strength': 1.740, 'border_count': 49
}

# LGB best AUC=0.73959
CACHED_LGB = {
    'n_estimators': 2600, 'learning_rate': 0.02576, 'num_leaves': 189,
    'max_depth': 5, 'subsample': 0.9402, 'colsample_bytree': 0.8963,
    'reg_alpha': 0.176, 'reg_lambda': 0.140,
    'min_child_samples': 86, 'boosting_type': 'goss'
}
```

#### [exp27] 새로운 feature 추가 + Data Leakage 수정
- SHAP 기준 154개 피처 선택 후 학습
- **exp22 Data leakage 의심 사례 수정**

#### [exp28] exp22 발전 — SHAP 기준 150개 피처

```
feature 중요도 0인 14개 제거
ML 피처셋 (SHAP>0): 150개
DL 피처셋 (SHAP>0): 150개
```

- **exp22 Data leakage 의심 사례 수정**

---

## 3. 민석

### 4월 24일 — 프로젝트 기반 구축

`src/preprocess.py`, `src/train.py`, `src/predict.py` 세 모듈로 공통 전처리 파이프라인을 구성하였습니다. Label Encoding, 결측치 처리, 학습·추론 분리 등 이후 모든 버전의 공통 기반이 되는 코드입니다. 데이터 리키지 방지를 위해 train 통계만 사용하는 구조를 이 단계에서 확립했습니다.

---

### 4월 25, 26일 — 모델 탐색 및 Optuna 하이퍼파라미터 튜닝

`model_comparison.py`에서 LogisticRegression, RandomForest, GradientBoosting, LightGBM, CatBoost 등 여러 모델의 5-fold OOF AUC를 비교하며 GBDT 계열이 압도적으로 우수함을 확인했습니다. 이어 `ensemble_optuna.py`에서 LGB·CAT에 Optuna 하이퍼파라미터 탐색을 적용하고, `final_model.py`에서 XGBoost를 추가한 LGB·XGB·CAT 3-way 앙상블로 발전시켰습니다. `advanced_model.py`에서는 ExtraTrees, HistGradientBoosting 등 추가 모델을 실험했으나 GBDT 3종 대비 유의미한 향상이 없어 이후 버전에서 제외했습니다.

---

### 4월 27일 — 시드 앙상블·스태킹 체계 도입 및 피처 엔지니어링 확장

#### solution_v3.py — OOF AUC 0.74020

Optuna로 최적화된 LGB·XGB·CAT를 4개 시드로 반복 학습한 후 가중치 최적화 앙상블을 적용하였습니다.

- 최적 가중치: LGB=0.05, XGB=0.40, CAT=0.55
- **OOF AUC: 0.74020**

#### solution_v4.py — OOF AUC 0.74015

피처 엔지니어링을 27개에서 122개로 대폭 확장하고, 3 seed × 3 모델 = 9개 기본 모델에 Meta-LGB + Meta-LR 2단계 스태킹을 도입했습니다.

- Meta-LGB: 과적합으로 **0.73965** 기록
- Meta-LR: 단순 평균보다 낮은 **0.74015** 기록
- **결론: 스태킹이 동일 계열(GBDT) 모델만으로는 효과 없음을 확인**

---

### 4월 28일 — TabNet 도입, 카테고리 임베딩 최적화, 추가 실험

#### solution_v5.py — OOF AUC 0.74023

Meta-LGB를 제거하고 Optuna로 LGB·XGB·CAT의 3차원 최적 가중치를 탐색하여 성능 반등을 이뤄냈습니다.

#### solution_v6.py — TabNet 최초 도입

모델 다양성 확보를 위해 TabNet을 추가했으나, 카테고리 임베딩 없는 기본 구성(n_d=16)으로는 TAB avg OOF 0.73338에 불과해 블렌딩 기여가 0.4%에 그쳤습니다.

#### solution_v7.py — OOF AUC 0.74027 ★ 최고 기록

TabNet에 카테고리 임베딩(`cat_idxs`, `cat_dims`, `cat_emb_dim=3`)을 적용하고 모델 크기를 `n_d=32`로 확장하였습니다.

- TAB avg OOF: 0.73338 → **0.73689**로 향상
- TabNet 기여도: 0.4% → **10.3%**로 증가
- **OOF AUC: 0.74027** (전체 최고 기록)

#### solution_v8.py — OOF AUC 0.74025

CatBoost depth를 7→8로 키웠으나 오히려 소폭 하락했습니다.

#### solution_v9.py — OOF AUC 0.74023

중요도=0 피처 14개와 상관계수 0.97 이상 중복 피처 5개를 제거하여 122→103개로 축소했으나 역시 하락했습니다.

> **결론: 제거된 피처들도 앙상블 내에서 소량의 신호를 제공하고 있었음**

---

### 민석 실험 기록

| 실험 | 모델 | Feature Engineering | 주요 파라미터 | OOF AUC | 비고 |
| --- | --- | --- | --- | --- | --- |
| solution_v3 | LGB + XGB + CAT (4 seed) | 기본 feature | Optuna 최적화, 가중치 LGB=0.05/XGB=0.40/CAT=0.55 | 0.74020 | 시드 앙상블 도입 |
| solution_v4 | LGB + XGB + CAT + Meta-LGB + Meta-LR (스태킹) | 피처 27→122개 확장 | 3 seed × 3 모델 = 9개 + 2단계 스태킹 | 0.74015 | 스태킹 효과 없음 확인 |
| solution_v5 | LGB + XGB + CAT (가중치 최적화) | 피처 122개 | Optuna 3차원 가중치 탐색 | 0.74023 | Meta-LGB 제거 후 반등 |
| solution_v6 | LGB + XGB + CAT + TabNet | 피처 122개 | TabNet n_d=16, 임베딩 없음 | - | TabNet 기여도 0.4% |
| solution_v7 | LGB + XGB + CAT + TabNet | 피처 122개 | TabNet n_d=32, cat_emb_dim=3 | **0.74027** | **★ 최고 기록** |
| solution_v8 | LGB + XGB + CAT + TabNet | 피처 122개 | CatBoost depth=8 | 0.74025 | depth 증가 효과 없음 |
| solution_v9 | LGB + XGB + CAT + TabNet | 피처 122→103개 | 저중요도 피처 19개 제거 | 0.74023 | 피처 제거 시 성능 하락 |

> **현재 제출 최고: `submission_v7.csv` (OOF AUC 0.74027)**

---

## 4. Phase별 최종 성능 비교

```
============================================================
  Phase별 OOF AUC 비교
============================================================
  Phase1 XGB×Cat (Optuna×60)         0.74004  █████
  Phase2 15-model 앙상블                0.74004  █████
  Phase4 최종 (Prob+Rank)              0.74000  ████
  Phase4 Meta-Ridge                  0.73982  ████
  Phase4 Meta-LGB                    0.73961  ████
  Phase4 Meta-LR                     0.73934  ████
  Phase3 FT-Transformer              0.73690  ███
  Phase3 TabNet                      0.73056  
============================================================
```

| 실험 | 모델 | Feature Engineering | 주요 파라미터 | OOF AUC | Public Score | 비고 |
| --- | --- | --- | --- | --- | --- | --- |
| exp1 | LightGBM, XGBoost, CatBoost, RandomForest, MLP (PyTorch) | 기본 feature | default | - | 0.6736 | baseline |
| exp2 | XGB + CAT + LGB 앙상블 | 기본 feature | default | - | 0.7414 | 앙상블 도입 |
| exp3 | LGB + CAT 앙상블 | 전처리 수정 | default | - | 0.6735 | 전처리 후 성능 하락 |
| exp4 | LGB + CAT 앙상블 | 전처리 수정 | default | - | 0.7414 | submission_lgb_cat_ensemble |
| exp5 | XGB × Cat (Optuna×60) | 기본 feature | Optuna 60회 | 0.74004 | - | Phase1 |
| exp6 | 15-model 앙상블 | 기본 feature | 15개 모델 | 0.74004 | 0.7417 | **Phase2 best public** |
| exp7 | TabNet | 기본 feature | default | 0.73056 | - | Phase3, 성능 미달 |
| exp8 | FT-Transformer | 기본 feature | default | 0.73690 | - | Phase3, 성능 미달 |
| exp9 | Meta-LR (스태킹) | Phase2 OOF 활용 | Logistic Regression | 0.73934 | - | Phase4 |
| exp10 | Meta-LGB (스태킹) | Phase2 OOF 활용 | LightGBM meta | 0.73961 | - | Phase4 |
| exp11 | Meta-Ridge (스태킹) | Phase2 OOF 활용 | Ridge | 0.73982 | - | Phase4 |
| exp12 | 최종 앙상블 (Prob+Rank) | Phase2 OOF 활용 | Prob+Rank 혼합 | 0.74000 | 0.7417 | **Phase4 제출, submission_final.csv** |

> **최종 제출**: `submission_final.csv` (90067행, [0.0116, 0.8546])  
> **Public Score: 0.7417**

---

## 5. 주요 인사이트

1. **앙상블 > 단독 모델**: XGB+CAT+LGB 앙상블이 개별 모델 대비 큰 폭으로 성능 향상 (0.6736 → 0.7414)
2. **딥러닝은 단독으론 열세, 다양성에서 기여**: TabNet(0.73056), FT-Transformer(0.73690) 모두 ML 대비 낮지만, 앙상블 내 상관관계가 낮아 다양성 기여로 포함 유지
3. **스태킹은 동질 모델 간에 비효과적**: Meta-LGB, Meta-LR 등 모든 스태킹 실험이 Phase2 단순 앙상블(0.74004) 대비 성능 하락
4. **`시술 시기 코드` 피처**: 시간 순서 + 나이 분포를 동시에 인코딩하는 중요 피처로 확인
5. **SHAP 기반 피처 선택**: LGBM importance → SHAP importance로 전환하여 더 정확한 피처 선택 가능
6. **Data leakage 주의**: exp22에서 의심 사례 발견 후 exp27/exp28에서 수정
