# IVF 임신 성공 여부 예측 - 해커톤 솔루션

시험관 아기(IVF) 시술 데이터를 기반으로 임신 성공 여부를 이진 분류로 예측합니다.

- **평가 지표**: ROC-AUC
- **모델**: LightGBM + CatBoost 앙상블 (가중치 0.25 : 0.75)
- **검증 전략**: Stratified 5-Fold Cross Validation
- **최고 OOF AUC**: 0.7400 (v9, 피처 선택 적용)

---

## 폴더 구조

```
.
├── src/                          # 핵심 파이프라인
│   ├── preprocess.py             # 데이터 전처리 & 피처 엔지니어링
│   ├── train.py                  # 모델 학습 (5-Fold CV)
│   └── predict.py                # 예측 & 제출 파일 생성
│
├── experiments/                  # 버전별 실험 스크립트
│   ├── v3_optuna_ensemble.py     # Optuna HPO + 3모델 앙상블
│   ├── v4_ensemble_tuning.py
│   ├── v5_feature_expand.py
│   ├── v6_seed_ensemble.py
│   ├── v7_stacking.py
│   ├── v8_simplified_ensemble.py
│   ├── v9_feature_selection.py   # 최고 성능 (zero-importance / 고상관 피처 제거)
│   ├── advanced_features.py
│   ├── final_submission.py
│   ├── ensemble_optuna.py
│   └── model_comparison.py
│
├── notebooks/
│   ├── 01_eda.ipynb              # 탐색적 데이터 분석
│   ├── 02_baseline_model.ipynb   # 기본 모델 (LGB / XGB / CatBoost)
│   ├── 03_feature_engineering.ipynb
│   ├── 04_seed_ensemble_tabnet.ipynb
│   └── 05_catboost_lgbm_analysis.ipynb
│
├── data/
│   ├── raw/                      # 원본 데이터 (gitignored: train.csv, test.csv)
│   │   ├── train.csv             # 학습 데이터 (256,351행 × 69열)
│   │   ├── test.csv              # 테스트 데이터 (90,067행 × 68열)
│   │   ├── sample_submission.csv # 제출 양식
│   │   └── data_specification.xlsx # 변수 명세서
│   └── submissions/              # 버전별 제출 파일
│
├── models/                       # 저장된 모델 바이너리 (gitignored)
├── logs/                         # 학습 로그 & 피처 중요도 분석
├── requirements.txt
├── CLAUDE.md                     # 대회 규칙
└── .gitignore
```

---

## 실행 방법

### 1. 환경 설정

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 데이터 준비

대회 공식 페이지에서 데이터를 다운로드한 후 `data/raw/` 에 배치합니다.

```
data/raw/
├── train.csv
├── test.csv
└── sample_submission.csv
```

### 3. 학습 & 예측

```bash
# 모델 학습 (models/ 에 fold별 모델 저장)
python -m src.train

# 예측 및 제출 파일 생성 (data/submissions/ 에 저장)
python -m src.predict
```

---

## 실험 이력

| 버전 | 스크립트 | 핵심 내용 | OOF AUC |
|------|---------|----------|---------|
| exp3 | XGBoost | 기본 + 파생변수 110개 | 0.7391 |
| exp4 | LightGBM | 기본 + 파생변수 110개 | 0.7384 |
| exp5 | CatBoost | 기본 + 파생변수 110개 | 0.7398 |
| exp6 | CatBoost × LGB | 앙상블 (w=0.75:0.25) | 0.7400 |
| v3 | v3_optuna_ensemble | Optuna HPO + 3모델 앙상블 | - |
| v4 | v4_ensemble_tuning | 앙상블 가중치 그리드 탐색 | - |
| v5 | v5_feature_expand | 피처 확장 실험 | - |
| v6 | v6_seed_ensemble | 시드 앙상블 (5 seeds) | - |
| v7 | v7_stacking | 스태킹 구조 실험 | - |
| v8 | v8_simplified_ensemble | 구조 단순화 | - |
| **v9** | **v9_feature_selection** | **zero-importance 14개 + 고상관 5쌍 제거** | **0.7400** |

---

## 핵심 파생변수 (7가지)

| 변수 | 설명 |
|------|------|
| `이식_효율` | 이식 배아 / 총 생성 배아 |
| `과거_임신_성공률` | 총 임신 횟수 / (총 시술 횟수 + 1) |
| `수정_효율` | 총 생성 배아 / 혼합된 난자 수 |
| `나이x난자수` | 시술 당시 나이 × 수집된 신선 난자 수 (교호작용) |
| `배아_풍요도` | 생성 배아 수 구간화 (0~3) |
| `고령저반응` | 나이 ≥ 만40세 AND 난자 수 < 중앙값 (플래그) |
| `순수동결_주기` | 동결 배아만 사용한 주기 (FER 플래그) |

---

## 데이터 설명

| 항목 | 내용 |
|------|------|
| 학습 데이터 | 256,351건, 69개 컬럼 |
| 테스트 데이터 | 90,067건, 68개 컬럼 |
| 타겟 | `임신 성공 여부` (0: 실패, 1: 성공) |
| 양성 비율 | 약 25.8% |
| 평가 지표 | ROC-AUC |

---

## Data Leakage 방지 원칙

모든 인코딩·스케일링·결측치 통계는 **학습 데이터 기준**으로만 계산하며,  
테스트 데이터에는 학습 시 산출한 통계값을 그대로 적용합니다.  
`src/preprocess.py`의 `preprocess()` / `preprocess_full()` 함수가 이를 보장합니다.
