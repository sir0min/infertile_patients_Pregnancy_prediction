# 🏥 불임 시술 임신 성공 여부 예측 — 12조

> **해커톤 최종 리더보드 점수: 0.74206**  
> ML(XGB + CAT + LGB Seed 앙상블) + DL(TabNet + MLP + FT-Transformer) + 2단계 스태킹

---

## 📁 프로젝트 구조

```
src/
├── preprocess.py     # 데이터 전처리 및 피처 엔지니어링
├── train.py          # 모델 학습 (ML + DL + 스태킹)
├── predict.py        # 예측 및 submission 생성
├── requirements.txt  # 의존 패키지
└── README.md
```

---

## ⚙️ 설치 및 실행

```bash
pip install -r requirements.txt
python predict.py
```

> **데이터 파일** (`train.csv`, `test.csv`, `sample_submission.csv`)을 `src/`와 같은 디렉토리에 위치시켜 주세요.

`predict.py`는 `preprocess.py`와 `train.py`를 import해서 순서대로 호출하는 **진입점(entry point)** 입니다.

```
predict.py
  ├── from preprocess import full_pipeline   → 전처리 실행
  ├── from train import train_all            → 모델 학습 실행
  └── make_submission()                      → submission 저장
```

각 파일을 개별 실행하는 것도 가능합니다.

```bash
python preprocess.py   # 전처리 결과 확인만 할 때
python train.py        # 학습 후 pred_final.npy 저장
python predict.py      # 전처리 + 학습 + submission 생성 전체 실행
```

---

## 🔧 파이프라인 개요

### 1. 전처리 (`preprocess.py`)

| 단계 | 내용 |
|------|------|
| 결측치 처리 | 결측률 80%+ 컬럼 이진화, 연속형 → train 기준 중앙값 대체 |
| 순서형 인코딩 | 나이대, 시술 횟수, 기증자 나이 등 |
| 멀티레이블 분리 | 배아 생성 이유(5개), 특정 시술 유형(10개) 이진 컬럼으로 확장 |
| 시술 시기 코드 | D5 비율·D3 비율·성공률 등 고정 맵으로 파생 (leakage 방지) |
| 파생변수 | 총 **150개+** 피처 생성 (도메인 파생 · 배아 품질 · 교호작용 등) |

**Data Leakage 방지 처리**
- `egg_median`, `egg_q75`, `emax` — train에서만 계산 후 `medians` 딕셔너리에 저장, test는 저장된 값으로 transform만 수행

### 2. 학습 (`train.py`)

| 구성 | 내용 |
|------|------|
| 피처 선택 | SHAP 중요도 > 0 피처만 사용 |
| ML 앙상블 | XGBoost × 3 seed + CatBoost × 3 seed + LightGBM × 3 seed = **9모델** |
| DL 앙상블 | **MLP** + **TabNet** + **FT-Transformer** (RobustScaler, fold-wise fit) |
| 스태킹 | Meta-LGB + Meta-LR → L-BFGS-B 최적 가중치 혼합 |
| 하이퍼파라미터 | Optuna 탐색 결과 캐싱 (`USE_CACHED_PARAMS=True`로 재탐색 생략 가능) |

### 3. 예측 (`predict.py`)

- 최종 submission 저장: `submission_vC_final.csv` (스태킹), `submission_vC_ml_only.csv` (ML only)
- OOF 기반 혼동행렬 시각화 및 최적 임계값 자동 탐색

---

## 📊 실험 로그

### 민석

| 실험 | 모델 | Feature Engineering | OOF AUC | Public Score | 비고 |
|------|------|---------------------|---------|-------------|------|
| exp1 | LightGBM, XGBoost, CatBoost, RandomForest, MLP (PyTorch) | 기본 feature | - | 0.6736 | baseline |
| exp2 | XGB + CAT + LGB 앙상블 | 기본 feature | - | 0.7414 | 앙상블 도입 |
| exp3 | LGB + CAT 앙상블 | 전처리 수정 | - | 0.6735 | 전처리 후 성능 하락 |
| exp4 | LGB + CAT 앙상블 | 전처리 수정 | - | 0.7414 | submission_lgb_cat_ensemble |
| exp5 | XGB × CAT (Optuna×60) | 기본 feature | 0.74004 | - | Phase1 |
| exp6 | 15-model 앙상블 | 기본 feature | 0.74004 | 0.7417 | Phase2 best public |
| exp7 | TabNet | 기본 feature | 0.73056 | - | Phase3, 성능 미달 |
| exp8 | FT-Transformer | 기본 feature | 0.73690 | - | Phase3, 성능 미달 |
| exp9 | Meta-LR (스태킹) | Phase2 OOF 활용 | 0.73934 | - | Phase4 |
| exp10 | Meta-LGB (스태킹) | Phase2 OOF 활용 | 0.73961 | - | Phase4 |
| exp11 | Meta-Ridge (스태킹) | Phase2 OOF 활용 | 0.73982 | - | Phase4 |
| exp12 | 최종 앙상블 (Prob+Rank) | Phase2 OOF 활용 | 0.74000 | 0.7417 | 서영 최종 제출 |
| solution_v3 | LGB+XGB+CAT (4 seed) | 기본 feature | 0.74020 | - | 시드 앙상블 도입, 가중치 LGB=0.05/XGB=0.40/CAT=0.55 |
| solution_v4 | LGB+XGB+CAT+Meta-LGB+Meta-LR | 피처 27→122개 | 0.74015 | - | 3 seed × 3 모델 + 2단계 스태킹, 스태킹 효과 없음 확인 |
| solution_v5 | LGB+XGB+CAT | 피처 122개 | 0.74023 | - | Optuna 3차원 가중치 탐색, Meta-LGB 제거 후 반등 |
| solution_v6 | LGB+XGB+CAT+TabNet | 피처 122개 | - | - | TabNet n_d=16, 임베딩 없음, 기여도 0.4% |
| solution_v7 | LGB+XGB+CAT+TabNet | 피처 122개 | **0.74027** | - | TabNet n_d=32, cat_emb_dim=3, **민석 최고 기록** |
| solution_v8 | LGB+XGB+CAT+TabNet | 피처 122개 | 0.74025 | - | CatBoost depth=8, depth 증가 효과 없음 |
| solution_v9 | LGB+XGB+CAT+TabNet | 피처 122→103개 | 0.74023 | - | 저중요도 피처 19개 제거, 피처 제거 시 성능 하락 |
| solution_v4_final | LGB+XGB+CAT (Prob+Rank) | 피처 147→112개 (신규 12개) | 0.73986 | - | v2 대비 -0.00018 하락 |

### 서영

| 실험 | 모델 | Feature Engineering | OOF AUC | 리더보드 | 비고 |
|------|------|---------------------|---------|---------|------|
| exp0 | LightGBM | 기본 feature + 파생변수 110개 | 0.7385 | 0.5 | - |
| exp1 | XGBoost | 기본 feature + 파생변수 110개 | 0.7392 | 0.5 | - |
| exp2 | CatBoost | 기본 feature + 파생변수 110개 | 0.7397 | 0.7412 | - |
| exp3 | XGBoost × CatBoost 앙상블 | 기본 feature + 파생변수 110개 | 0.7399 | - | A. Simple Average |
| exp4 | XGBoost × CatBoost 앙상블 | 기본 feature + 파생변수 110개 | 0.7399 | - | B. Weighted Average |
| exp5 | XGBoost × CatBoost 앙상블 | 기본 feature + 파생변수 110개 | 0.7399 | - | C. Stacking (메타 모델 : LR) |
| exp6 | XGBoost × CatBoost 앙상블 | 기본 feature + 파생변수 110개 | 0.7399 | - | D. Rank Average |
| exp7 | XGBoost (Optuna) | 기본 feature + 파생변수 110개 | 0.7397 | - | depth=5 |
| exp8 | CatBoost (Optuna) | 기본 feature + 파생변수 110개 | 0.7397 | - | depth=5 |
| exp9 | XGB × CAT 앙상블 (Optuna) | 기본 feature + 파생변수 110개 | 0.7399 | - | A. Simple Average |
| exp10 | XGB × CAT 앙상블 (Optuna) | 기본 feature + 파생변수 110개 | 0.7399 | 0.74162 | D. Rank Average |
| exp11 | LGB × XGB × CAT 앙상블 (Optuna) | 기본 feature + 파생변수 110개 | 0.7400 | 0.74140 | LGBM max_depth=3 / XGB max_depth=5 / Cat depth=6 |
| exp12 | XGB+CAT+LGB+RF+ExtraTree 앙상블 | 기본 feature + 파생변수 110개 | 0.7398 | - | Stacking (메타모델 : LR) |
| exp13 | XGB+CAT+LGB 앙상블 | 파생변수 110개 + 신규 29개 | 0.73788 | - | - |
| exp14 | seed 앙상블 | 파생변수 110개 + 신규 29개 | 0.73998 | 0.74182 | - |
| exp15 | seed 앙상블 + TabNet + FT-Transformer | 파생변수 110개 + 신규 29개 | 0.73970 | - | 2단계 스태킹 앙상블 |
| exp16 | ML+DL feature split | 파생변수 110+29 + 복구 6개 | - | - | feature split 도입 |
| exp17 | ML+DL feature split | 파생변수 110+29 + 복구 6개 + 시기코드 5개 | 0.74058 | 0.74183 | feature split |
| exp18 | ML + MLP + FT-Transformer | 파생변수 110+29 + 복구 6개 + 시기코드 5개 | 0.74047 | - | TabNet 제거 → 결론: 포함 필요 |
| exp19 | ML+DL feature split | 위 + 파생변수 6개 추가 | 0.74053 | - | HistGradientBoosting 추가 → 결론: 제거 |
| exp20 | ML+DL feature split | 위 + 파생변수 6개 추가 | 0.74052 | - | ML Top65 / DL Top85 |
| exp21 | ML+DL feature split | 위 + 파생변수 6개 추가 | 0.74065 | - | ML Top65 / DL 중요도 >0 |
| exp22 | ML+DL feature split | 위 + 파생변수 6개 추가 | 0.74076 | **0.74202** | SHAP Top150 피처 사용 |
| exp23 | ML+DL feature split | 위 + 파생변수 6개 추가 | 0.74026 | - | SHAP 누적 95% → Top66 |
| exp24 | ML+DL feature split | 위 + 파생변수 6개 추가 | 0.74030 | - | ML Top45 / DL Top66 |
| exp25 | ML+DL (메타4모델+DL seed앙상블) | 위 + 파생변수 6개 추가 | 0.74073 | - | OOF TargetEncoding + OneHot |
| exp26 | ML+DL (Optuna 재탐색) | 위 + 파생변수 4개 추가 | - | - | 컴퓨터 성능으로 인한 실패 |
| exp27 | ML+DL | 위 + 파생변수 4개 추가 | 0.74067 | - | exp26 Optuna 결과 활용 |
| exp28 | ML+DL | 위 + 파생변수 4개 추가 | 0.74067 | **0.74206** | SHAP Top150, **최종 제출** |

### 수빈

| 실험 | 모델 | 핵심 내용 | OOF AUC |
|------|------|----------|---------|
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

## 🏆 최종 결과

| 버전 | OOF AUC | 리더보드 | 설명 |
|------|---------|---------|------|
| exp22 (서영) | 0.74076 | 0.74202 | SHAP Top150 피처 |
| **exp28 (서영)** | **0.74067** | **0.74206** | **최종 제출 — 최고 점수** |
| solution_v7 (민석) | 0.74027 | 0.7413949427 | ML+TabNet |

---

## 💡 핵심 인사이트

1. **피처 엔지니어링이 가장 큰 기여** — 기본 27개 → 150개+ 파생변수로 확장
2. **ML + DL feature split** — ML과 DL에 서로 다른 피처셋을 제공하는 전략이 단일 피처셋 대비 성능 향상
3. **SHAP 기반 피처 선택** — 중요도 > 0 피처만 사용해 노이즈 제거
4. **Data Leakage 수정** — `고령저반응`, `젊고고수확`, `배아_풍요도` 기준값을 train에서만 계산하도록 수정
5. **TabNet 포함 필수** — exp18에서 TabNet 제거 시 성능 하락 확인
