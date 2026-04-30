"""
preprocess.py — 데이터 전처리 및 피처 엔지니어링
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────
TARGET   = "임신 성공 여부"
ID_COL   = "ID"
DATA_DIR = Path(".")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

AGE_MAP = {
    "만18-34세": 0, "만35-37세": 1, "만38-39세": 2,
    "만40-42세": 3, "만43-44세": 4, "만45-50세": 5,
    "알 수 없음": -1,
}
COUNT_MAP = {"0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6}
DONOR_AGE_MAP = {
    "만20세 이하": 0, "만21-25세": 1, "만26-30세": 2,
    "만31-35세": 3, "만36-40세": 4, "만41-45세": 5,
    "알 수 없음": -1,
}
INDUCTION_MAP = {
    "알 수 없음": 0, "기록되지 않은 시행": 1,
    "생식선 자극 호르몬": 2, "세트로타이드 (억제제)": 3,
}
EGG_MAP   = {"본인 제공": 0, "기증 제공": 1, "알 수 없음": -1}
SPERM_MAP = {"배우자 제공": 0, "기증 제공": 1, "배우자 및 기증 제공": 2, "미할당": -1}

COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
REASON_CATS = ["현재 시술용", "배아 저장용", "난자 저장용", "기증용", "연구용"]
PROC_TYPES  = ["IVF", "ICSI", "IUI", "ICI", "GIFT", "FER", "BLASTOCYST", "AH", "Generic DI", "IVI"]

# 시술 시기 코드 파생변수 고정 맵 (train 통계 기반 상수 — leakage 없음)
CODE_D5_RATE    = {"TRDQAZ": 0.376, "TRCMWS": 0.365, "TRYBLT": 0.357, "TRJXFG": 0.339, "TRVNRY": 0.301, "TRZKPL": 0.256, "TRXQMD": 0.217}
CODE_D3_RATE    = {"TRDQAZ": 0.147, "TRCMWS": 0.174, "TRYBLT": 0.205, "TRJXFG": 0.231, "TRVNRY": 0.256, "TRZKPL": 0.287, "TRXQMD": 0.295}
CODE_TIME_ORDER = {"TRXQMD": 0, "TRZKPL": 1, "TRVNRY": 2, "TRJXFG": 3, "TRYBLT": 4, "TRCMWS": 5, "TRDQAZ": 6}
CODE_SUCCESS    = {"TRDQAZ": 0.245, "TRCMWS": 0.257, "TRYBLT": 0.269, "TRJXFG": 0.266, "TRVNRY": 0.260, "TRZKPL": 0.255, "TRXQMD": 0.256}


# ─────────────────────────────────────────────
# 멀티레이블 분리 (배아 이유 / 시술 유형)
# ─────────────────────────────────────────────
def expand_reason(df: pd.DataFrame) -> pd.DataFrame:
    for cat in REASON_CATS:
        df[f"이유_{cat}"] = df["배아 생성 주요 이유"].fillna("").str.contains(cat).astype(int)
    return df


def expand_procedure(df: pd.DataFrame) -> pd.DataFrame:
    filled = df["특정 시술 유형"].fillna("Unknown")
    for pt in PROC_TYPES:
        df[f"시술_{pt}"] = (
            filled.str.upper().str.replace(" ", "", regex=False)
            .str.contains(pt.upper()).astype(int)
        )
    return df


# ─────────────────────────────────────────────
# 기본 전처리
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, medians: dict = None, fit: bool = False):
    df = df.copy()
    if medians is None:
        medians = {}

    # 유전 검사 관련 → 결측 여부 이진화
    for col in ["착상 전 유전 검사 사용 여부", "PGD 시술 여부", "PGS 시술 여부"]:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)

    # 난자 채취 경과일 결측 처리
    if "난자 채취 경과일" in df.columns:
        df["난자_채취_결측"] = df["난자 채취 경과일"].isnull().astype(int)
        df["난자 채취 경과일"] = df["난자 채취 경과일"].fillna(0)

    # 배아 해동 경과일 결측 처리
    if "배아 해동 경과일" in df.columns:
        df["배아_해동_결측"] = df["배아 해동 경과일"].isnull().astype(int)
        df["배아 해동 경과일"] = df["배아 해동 경과일"].fillna(0)

    # 임신 시도 연수 결측 처리
    col_y = "임신 시도 또는 마지막 임신 경과 연수"
    if col_y in df.columns:
        df["임신_시도_연수_결측"] = df[col_y].isnull().astype(int)
        df[col_y] = df[col_y].fillna(-1)

    # DI 시술행 IVF 전용 컬럼 → 0 대체
    fill_zero = [
        "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부",
        "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
        "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
        "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
        "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
        "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
    ]
    for c in fill_zero:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # 경과일 컬럼 — train 기준 중앙값 저장/적용
    for col in ["난자 혼합 경과일", "배아 이식 경과일"]:
        if col not in df.columns:
            continue
        if fit:
            medians[col] = df[col].median()
        df[col] = df[col].fillna(medians.get(col, 0))

    # 연속형 수치 결측 → train 기준 중앙값
    cont_cols = [
        "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
        "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
        "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
        "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
        "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수", "배아 해동 경과일",
    ]
    for c in cont_cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if fit:
            medians[c] = df[c].median() if df[c].notna().any() else 0
        df[c] = df[c].fillna(medians.get(c, 0))

    # 순서형 인코딩
    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(-1).astype(int)
    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP).fillna(-1).astype(int)
    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(-1).astype(int)

    # 시술 유형 이진화
    if "시술 유형" in df.columns:
        df["시술 유형"] = (df["시술 유형"] == "IVF").astype(int)

    # 배란 유도 유형
    if "배란 유도 유형" in df.columns:
        df["배란 유도 유형"] = df["배란 유도 유형"].map(INDUCTION_MAP).fillna(0).astype(int)

    # 난자/정자 출처
    if "난자 출처" in df.columns:
        df["난자 출처"] = df["난자 출처"].map(EGG_MAP).fillna(-1).astype(int)
    if "정자 출처" in df.columns:
        df["정자 출처"] = df["정자 출처"].map(SPERM_MAP).fillna(-1).astype(int)

    # 시술 시기 코드 파생변수 → 고정 맵 적용
    if "시술 시기 코드" in df.columns:
        cs = df["시술 시기 코드"].copy()
        df["코드_D5비율"]    = cs.map(CODE_D5_RATE).fillna(0)
        df["코드_D3비율"]    = cs.map(CODE_D3_RATE).fillna(0)
        df["코드_시간순서"]  = cs.map(CODE_TIME_ORDER).fillna(3)
        df["코드_성공률"]    = cs.map(CODE_SUCCESS).fillna(0)
        age_n = df["시술 당시 나이"].replace(-1, 0).clip(lower=0)
        df["코드_시간x나이"] = df["코드_시간순서"] * age_n
        df.drop(columns=["시술 시기 코드"], inplace=True)

    # 이진형 컬럼 결측 → 0
    binary_cols = [
        "배란 자극 여부", "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부",
        "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
        "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
        "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
        "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", "불임 원인 - 여성 요인",
        "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
        "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
    ]
    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    return df, medians


# ─────────────────────────────────────────────
# 피처 엔지니어링 (도메인 파생변수)
# ─────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame, medians: dict = None, fit: bool = False):
    """
    Data Leakage 방지:
      - 고령저반응/젊고고수확의 median·quantile은 train에서만 계산 → medians 저장
      - 배아_풍요도 emax도 동일 처리
    """
    if medians is None:
        medians = {}
    df = df.copy()

    # 핵심 수치 강제 변환
    for c in [
        "총 생성 배아 수", "이식된 배아 수", "저장된 배아 수", "수집된 신선 난자 수",
        "미세주입에서 생성된 배아 수", "미세주입된 난자 수", "해동된 배아 수",
        "미세주입 후 저장된 배아 수", "총 임신 횟수", "총 시술 횟수",
        "클리닉 내 총 시술 횟수", "IVF 시술 횟수",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부",
        "시술 유형", "시술 당시 나이",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 불임 원인 합계
    cause_cols = [c for c in df.columns if "불임 원인 -" in c]
    df["불임_원인_합계"]      = df[cause_cols].sum(axis=1)
    primary = [c for c in ["남성 주 불임 원인", "여성 주 불임 원인", "부부 주 불임 원인"] if c in df.columns]
    df["주요_불임_원인_합계"] = df[primary].sum(axis=1)

    # 이식 효율 / ICSI 비율 / 과거 임신 성공률
    df["이식_효율"]       = np.where(df["총 생성 배아 수"] > 0, df["이식된 배아 수"] / df["총 생성 배아 수"], 0)
    df["ICSI_비율"]       = np.where(df["총 생성 배아 수"] > 0, df["미세주입에서 생성된 배아 수"] / df["총 생성 배아 수"], 0)
    df["과거_임신_성공률"] = np.where(df["총 시술 횟수"] > 0, df["총 임신 횟수"] / (df["총 시술 횟수"] + 1), 0)

    # 배아 전략
    for col in ["동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부"]:
        if col not in df.columns:
            df[col] = 0
    df["배아_전략"] = df["동결 배아 사용 여부"] * 1 + df["신선 배아 사용 여부"] * 0 + df["기증 배아 사용 여부"] * 2
    df["저장_비율"] = np.where(df["총 생성 배아 수"] > 0, df["저장된 배아 수"] / df["총 생성 배아 수"], 0)
    df["기증_사용"] = (
        (df.get("난자 출처", pd.Series(0, index=df.index)) == 1) |
        (df.get("정자 출처", pd.Series(0, index=df.index)) == 1)
    ).astype(int)

    # 동결 배아 관련
    if "해동된 배아 수" in df.columns:
        df["동결배아_활용률"]   = np.where(df["저장된 배아 수"] > 0, df["해동된 배아 수"] / df["저장된 배아 수"], 0)
        df["ICSI동결_이식비율"] = np.where(df["이식된 배아 수"] > 0, df["미세주입 후 저장된 배아 수"] / df["이식된 배아 수"], 0)
        df["순수동결_주기"]     = ((df["동결 배아 사용 여부"] == 1) & (df["신선 배아 사용 여부"] == 0)).astype(int)
        remaining = (df["저장된 배아 수"] - df["해동된 배아 수"]).clip(lower=0)
        df["동결배아_잉여율"]   = np.where(df["저장된 배아 수"] > 0, remaining / df["저장된 배아 수"], 0)
        df["동결배아_총량"]     = df["저장된 배아 수"] + df["해동된 배아 수"]

    # 나이 × 난자 수 (Data Leakage 방지: median·q75 train에서만 계산)
    if "수집된 신선 난자 수" in df.columns and "시술 당시 나이" in df.columns:
        age_num = df["시술 당시 나이"].replace(-1, np.nan)
        egg_cnt = df["수집된 신선 난자 수"]

        if fit:
            medians["egg_median"] = float(egg_cnt.median())
            medians["egg_q75"]    = float(egg_cnt.quantile(0.75))

        egg_median = medians.get("egg_median", float(egg_cnt.median()))
        egg_q75    = medians.get("egg_q75",    float(egg_cnt.quantile(0.75)))

        df["나이x난자수"]       = (age_num * egg_cnt).fillna(0)
        df["나이보정_난자효율"] = egg_cnt / (age_num.fillna(0) + 1)
        df["고령저반응"]        = ((age_num >= 3) & (egg_cnt < egg_median)).fillna(False).astype(int)
        df["젊고고수확"]        = ((age_num <= 1) & (egg_cnt >= egg_q75)).fillna(False).astype(int)

    # 배아_풍요도 emax — train에서만 계산 후 저장
    if fit:
        medians["emx"] = max(int(df["총 생성 배아 수"].max()) + 1, 9)
    emax = medians.get("emx", 9)

    df["배아_손실수"]      = (df["총 생성 배아 수"] - df["이식된 배아 수"] - df["저장된 배아 수"]).clip(lower=0)
    df["배아_총활용률"]    = np.where(df["총 생성 배아 수"] > 0, (df["이식된 배아 수"] + df["저장된 배아 수"]) / df["총 생성 배아 수"], 0).clip(0, 1)
    df["배아_풍요도"]      = pd.cut(df["총 생성 배아 수"], bins=[-1, 0, 3, 7, emax], labels=[0, 1, 2, 3]).astype(float).fillna(0).astype(int)
    df["다배아이식_플래그"] = (df["이식된 배아 수"] >= 3).astype(int)
    df["ICSI배아_우위비율"] = np.where(df["총 생성 배아 수"] > 0, df["미세주입에서 생성된 배아 수"] / df["총 생성 배아 수"], 0)
    denom = df["이식된 배아 수"] + df["저장된 배아 수"]
    df["이식_집중도"] = np.where(denom > 0, df["이식된 배아 수"] / denom, 0)

    # 나이 × 시술 유형
    if "시술 유형" in df.columns and "시술 당시 나이" in df.columns:
        age2 = df["시술 당시 나이"].replace(-1, 0)
        df["나이x시술유형"]      = age2 * df["시술 유형"]
        df["고령IVF"]            = ((df["시술 당시 나이"] >= 3) & (df["시술 유형"] == 1)).astype(int)
        df["젊은DI"]             = ((df["시술 당시 나이"] <= 1) & (df["시술 유형"] == 0)).astype(int)
        df["나이_시술_복합코드"] = df["시술 당시 나이"].clip(lower=0) * 10 + df["시술 유형"]

    # 나이 × 시술 횟수
    if "총 시술 횟수" in df.columns and "시술 당시 나이" in df.columns:
        age3  = df["시술 당시 나이"].replace(-1, 0)
        trial = df["총 시술 횟수"].replace(-1, 0)
        df["나이x총시술횟수"]     = age3 * trial
        df["나이x클리닉시술횟수"] = age3 * df["클리닉 내 총 시술 횟수"].replace(-1, 0)
        ivf = df["IVF 시술 횟수"].replace(-1, 0)
        df["나이xIVF횟수"]   = age3 * ivf
        df["시술부담_지수"]  = age3 * (trial + ivf) / 2
        df["고령반복시술"]   = ((df["시술 당시 나이"] >= 3) & (trial >= 3)).astype(int)

    return df, medians


# ─────────────────────────────────────────────
# 배아 품질 피처
# ─────────────────────────────────────────────
def add_embryo_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["총 생성 배아 수", "수집된 신선 난자 수", "미세주입에서 생성된 배아 수", "미세주입된 난자 수", "이식된 배아 수"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if "배아 이식 경과일" in out.columns:
        out["배아 이식 경과일"] = pd.to_numeric(out["배아 이식 경과일"], errors="coerce")

    out["수정률"]      = np.where(out["수집된 신선 난자 수"] > 0, out["총 생성 배아 수"] / out["수집된 신선 난자 수"], np.nan).clip(0, 1)
    out["수정률_결측"] = (out["수집된 신선 난자 수"] == 0).astype(int)
    out["수정률"]      = out["수정률"].fillna(0)
    out["수정률_양호"] = (out["수정률"] >= 0.5).astype(int)
    out["수정률_구간"] = pd.cut(out["수정률"], bins=[-0.01, 0.3, 0.5, 0.7, 1.01], labels=[0, 1, 2, 3]).astype(float)

    out["is_D5"]    = np.where(out["배아 이식 경과일"].isnull(), np.nan, (out["배아 이식 경과일"] >= 5).astype(float))
    out["D5_결측"]  = out["배아 이식 경과일"].isnull().astype(int)
    out["is_D5"]    = out["is_D5"].fillna(0)
    out["이식단계"] = np.where(out["배아 이식 경과일"].isnull(), -1,
                               np.where(out["배아 이식 경과일"] <= 3, 0,
                               np.where(out["배아 이식 경과일"] >= 5, 1, 0)))

    out["ICSI수정률"]      = np.where(out["미세주입된 난자 수"] > 0, out["미세주입에서 생성된 배아 수"] / out["미세주입된 난자 수"], np.nan).clip(0, 1)
    out["ICSI_미시행"]     = (out["미세주입된 난자 수"] == 0).astype(int)
    out["ICSI수정률"]      = out["ICSI수정률"].fillna(0)
    out["ICSI수정률_양호"] = (out["ICSI수정률"] >= 0.7).astype(int)
    out["ICSI수정률_구간"] = pd.cut(out["ICSI수정률"], bins=[-0.01, 0.5, 0.7, 0.9, 1.01], labels=[0, 1, 2, 3]).astype(float)

    out["이식수_최적"]  = out["이식된 배아 수"].isin([1.0, 2.0]).astype(int)
    out["최적이식_D5"] = out["이식수_최적"] * out["is_D5"]
    out["배아질_복합"] = out["수정률"].clip(0, 1) * 0.4 + out["is_D5"] * 0.4 + out["이식수_최적"] * 0.2
    return out


# ─────────────────────────────────────────────
# 추가 도메인 피처 (exp19+ 신규 파생변수)
# ─────────────────────────────────────────────
def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "총 생성 배아 수", "이식된 배아 수", "저장된 배아 수", "수집된 신선 난자 수",
        "저장된 신선 난자 수", "혼합된 난자 수", "해동된 배아 수", "파트너 정자와 혼합된 난자 수",
        "배아 이식 경과일", "난자 혼합 경과일", "시술 당시 나이", "시술 유형",
        "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수",
        "총 임신 횟수", "총 출산 횟수", "IVF 임신 횟수", "IVF 출산 횟수",
        "DI 임신 횟수", "DI 출산 횟수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    fert = np.where(out["수집된 신선 난자 수"] > 0, out["총 생성 배아 수"] / out["수집된 신선 난자 수"], 0).clip(0, 1)

    # 그룹 A. 배아 이식 경과일
    out["혼합_이식_간격"]   = (out["배아 이식 경과일"] - out["난자 혼합 경과일"]).fillna(0)
    out["이식일_D5이상"]    = (out["배아 이식 경과일"] >= 5).fillna(False).astype(int)
    out["이식일x이식수"]    = out["배아 이식 경과일"].fillna(0) * out["이식된 배아 수"].fillna(0)
    out["이식일_결측"]      = (out["배아 이식 경과일"] == 0).astype(int)
    out["정확히_D5"]        = (out["배아 이식 경과일"] == 5.0).astype(int)
    out["D5_최적이식_복합"] = ((out["배아 이식 경과일"] >= 5) & (out["이식된 배아 수"].isin([1.0, 2.0]))).fillna(False).astype(int)

    # 그룹 B. 배아/난자 품질
    out["생성배아_품질복합"] = out["총 생성 배아 수"] * fert
    out["난자_풍부도"]       = pd.cut(out["수집된 신선 난자 수"], bins=[-1, 0, 4, 8, 12, 999], labels=[0, 1, 2, 3, 4]).astype(float).fillna(0).astype(int)
    out["신선난자_저장비율"] = np.where(out["수집된 신선 난자 수"] > 0, out["저장된 신선 난자 수"] / out["수집된 신선 난자 수"], 0)
    out["파트너정자_활용률"] = np.where(out["혼합된 난자 수"] > 0, out["파트너 정자와 혼합된 난자 수"] / out["혼합된 난자 수"], 0)
    out["전체_배아_효율"]    = np.where(out["수집된 신선 난자 수"] > 0, (out["이식된 배아 수"] + out["저장된 배아 수"]) / out["수집된 신선 난자 수"], 0).clip(0, 5)

    # 그룹 C. 시술 이력
    out["초회시술"]         = (out["총 시술 횟수"] == 0).astype(int)
    out["초회IVF"]          = ((out["IVF 시술 횟수"] == 0) & (out["총 시술 횟수"] <= 1)).astype(int)
    out["클리닉_집중도"]    = np.where(out["총 시술 횟수"] > 0, out["클리닉 내 총 시술 횟수"] / out["총 시술 횟수"], 1.0).clip(0, 1)
    out["IVF_임신_전환율"]  = np.where(out["IVF 시술 횟수"] > 0, out["IVF 임신 횟수"] / out["IVF 시술 횟수"], 0).clip(0, 1)
    out["임신_출산_전환율"] = np.where(out["총 임신 횟수"] > 0, out["총 출산 횟수"] / out["총 임신 횟수"], 0).clip(0, 1)
    out["출산_경험"]        = (out["총 출산 횟수"] > 0).astype(int)
    out["IVF_출산_경험"]    = (out["IVF 출산 횟수"] > 0).astype(int)

    # 그룹 D. 불임 원인
    mc = [c for c in ["남성 주 불임 원인", "남성 부 불임 원인", "불임 원인 - 남성 요인",
                       "불임 원인 - 정자 농도", "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
                       "불임 원인 - 정자 면역학적 요인"] if c in out.columns]
    fc = [c for c in ["여성 주 불임 원인", "여성 부 불임 원인", "불임 원인 - 난관 질환",
                       "불임 원인 - 배란 장애", "불임 원인 - 자궁내막증", "불임 원인 - 자궁경부 문제"] if c in out.columns]
    out["남성불임_복합"]  = out[mc].fillna(0).sum(axis=1)
    out["여성불임_복합"]  = out[fc].fillna(0).sum(axis=1)
    out["복합_불임_여부"] = ((out["남성불임_복합"] > 0) & (out["여성불임_복합"] > 0)).astype(int)
    cause_all = [c for c in out.columns if "불임 원인 -" in c]
    age_num   = out["시술 당시 나이"].replace(-1, 0)
    out["나이x불임원인수"] = age_num * out[cause_all].fillna(0).sum(axis=1)

    # 그룹 E. 나이 교호작용
    out["나이xD5"]         = age_num * out["이식일_D5이상"]
    out["나이x수정률"]     = age_num * fert
    out["나이x클리닉횟수"] = age_num * out["클리닉 내 총 시술 횟수"].replace(-1, 0)

    # 그룹 F. 도메인 파생변수 (행 내부 연산 — Leakage 없음)
    trial = out["총 시술 횟수"].replace(-1, 0)
    out["누적실패_패턴"]   = ((trial >= 3) & (out["총 임신 횟수"] == 0)).astype(int)
    out["최적_시술_콤보"]  = ((out["이식일_D5이상"] == 1) & (out["이식된 배아 수"] == 1) & (out["시술 유형"] == 1)).astype(int)
    out["출산후_재시술"]   = ((out["총 출산 횟수"] > 0) & (trial > 1)).astype(int)
    out["고위험_복합"]     = ((out["시술 당시 나이"] >= 3) & (trial >= 3) & (out["총 임신 횟수"] == 0) & (out["시술 유형"] == 1)).astype(int)
    out["수정률_D5_복합"]  = (out["수정률"] if "수정률" in out.columns else fert) * out["이식일_D5이상"]
    out["IVF_고령_초회"]   = ((out["시술 유형"] == 1) & (out["시술 당시 나이"] >= 3) & (out["IVF 시술 횟수"] == 0)).astype(int)

    return out


# ─────────────────────────────────────────────
# 전체 파이프라인
# ─────────────────────────────────────────────
def full_pipeline(df: pd.DataFrame, medians: dict = None, fit: bool = False):
    """
    fit=True  → train: 모든 통계 기준값 계산 & 저장
    fit=False → test:  저장된 기준값으로 transform만 수행
    """
    if medians is None:
        medians = {}
    df = expand_reason(df.copy())
    df = expand_procedure(df)
    df.drop(columns=["배아 생성 주요 이유", "특정 시술 유형"], inplace=True, errors="ignore")
    df, medians = preprocess(df, medians=medians, fit=fit)
    df, medians = feature_engineering(df, medians=medians, fit=fit)
    df = add_embryo_quality_features(df)
    df = add_new_features(df)
    return df, medians


# ─────────────────────────────────────────────
# 데이터 로딩 & 전처리 실행 (직접 실행 시)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig")
    test  = pd.read_csv(DATA_DIR / "test.csv",  encoding="utf-8-sig")
    train.columns = [c.strip() for c in train.columns]
    test.columns  = [c.strip() for c in test.columns]
    print(f"Train: {train.shape} | Test: {test.shape}")
    print(f"양성 비율: {train[TARGET].mean():.4f}")

    X_raw      = train.drop(columns=[ID_COL, TARGET], errors="ignore")
    y          = train[TARGET]
    X_test_raw = test.drop(columns=[ID_COL], errors="ignore")

    X_pp,      train_medians = full_pipeline(X_raw,      fit=True)
    X_test_pp, _             = full_pipeline(X_test_raw, medians=train_medians, fit=False)

    common_cols = [c for c in X_pp.columns if c in X_test_pp.columns]
    X_pp        = X_pp[common_cols]
    X_test_pp   = X_test_pp[common_cols]

    print(f"\n전체 피처: {X_pp.shape[1]}개")
    print(f"잔여 결측치 (Train): {X_pp.isnull().sum().sum()}")
    print(f"egg_median: {train_medians.get('egg_median', 'N/A'):.2f}")
    print(f"egg_q75   : {train_medians.get('egg_q75',    'N/A'):.2f}")
    print(f"emax      : {train_medians.get('emx',        'N/A')}")
