"""데이터 전처리 및 핵심 파생변수 생성 모듈"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── 순서형 인코딩 매핑 ────────────────────────────────────────────────────────
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
COUNT_COLS = [
    "총 시술 횟수", "클리닉 내 총 시술 횟수",
    "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수",
    "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수",
]
DROP_COLS = [
    "착상 전 유전 검사 사용 여부",
    "PGD 시술 여부",
    "PGS 시술 여부",
    "불임 원인 - 여성 요인",
    "난자 채취 경과일",
    "난자 해동 경과일",
]
REASON_CATEGORIES = ["현재 시술용", "배아 저장용", "난자 저장용", "기증용", "연구용"]
PROCEDURE_TYPES   = ["IVF", "ICSI", "IUI", "ICI", "GIFT", "FER",
                     "BLASTOCYST", "AH", "Generic DI", "IVI"]


def expand_reason(df: pd.DataFrame) -> pd.DataFrame:
    col = "배아 생성 주요 이유"
    for cat in REASON_CATEGORIES:
        df[f"이유_{cat}"] = df[col].fillna("").str.contains(cat).astype(int)
    return df


def expand_procedure(df: pd.DataFrame) -> pd.DataFrame:
    col = "특정 시술 유형"
    filled = df[col].fillna("Unknown")
    for pt in PROCEDURE_TYPES:
        df[f"시술_{pt}"] = (
            filled.str.upper().str.replace(" ", "", regex=False)
            .str.contains(pt.upper()).astype(int)
        )
    return df


def preprocess(df: pd.DataFrame, fit_label_encoders=None, train_stats=None):
    """
    train_stats: train 전처리 시 계산한 통계값 dict.
                 None이면 현재 df(=train)에서 계산 후 반환.
                 dict이면 test용 — train 통계값을 그대로 사용.
    반환: (df, label_encoders, stats)
    """
    df = df.copy()
    # train_stats가 없으면 새로 계산(train), 있으면 그대로 사용(test)
    stats = dict(train_stats) if train_stats else {}

    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    col_years = "임신 시도 또는 마지막 임신 경과 연수"
    if col_years in df.columns:
        df["임신_시도_연수_결측"] = df[col_years].isnull().astype(int)
        df[col_years] = df[col_years].fillna(-1)

    col_thaw = "배아 해동 경과일"
    if col_thaw in df.columns:
        df["배아_해동_결측"] = df[col_thaw].isnull().astype(int)
        df[col_thaw] = df[col_thaw].fillna(0)

    FILL_ZERO_COLS = [
        "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부", "배아 생성 주요 이유",
        "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
        "이식된 배아 수", "미세주입 배아 이식 수", "저장된 배아 수",
        "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
        "수집된 신선 난자 수", "저장된 신선 난자 수", "혼합된 난자 수",
        "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
    ]
    for col in FILL_ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ── [Fix 1] 경과일 결측치: train 중앙값 사용 ─────────────────────────────
    for col in ["난자 혼합 경과일", "배아 이식 경과일"]:
        if col in df.columns:
            key = f"{col}_median"
            if key not in stats:                      # train: 계산 후 저장
                stats[key] = df[col].median()
            df[col] = df[col].fillna(stats[key])      # test: train 중앙값 사용

    if "시술 당시 나이" in df.columns:
        df["시술 당시 나이"] = df["시술 당시 나이"].map(AGE_MAP).fillna(-1).astype(int)

    for col in COUNT_COLS:
        if col in df.columns:
            df[col] = df[col].map(COUNT_MAP).fillna(-1).astype(int)

    for col in ["난자 기증자 나이", "정자 기증자 나이"]:
        if col in df.columns:
            df[col] = df[col].map(DONOR_AGE_MAP).fillna(-1).astype(int)

    if "시술 유형" in df.columns:
        df["시술 유형"] = (df["시술 유형"] == "IVF").astype(int)

    induction_map = {
        "알 수 없음": 0, "기록되지 않은 시행": 1,
        "생식선 자극 호르몬": 2, "세트로타이드 (억제제)": 3,
    }
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

    LABEL_ENC_COLS = ["배아 생성 주요 이유", "특정 시술 유형"]
    label_encoders = fit_label_encoders or {}
    for col in LABEL_ENC_COLS:
        if col not in df.columns:
            continue
        if col not in label_encoders:
            le = LabelEncoder()
            le.fit(df[col].fillna("Unknown").astype(str))
            label_encoders[col] = le
        df[col] = label_encoders[col].transform(df[col].fillna("Unknown").astype(str))

    binary_cols = [
        "배란 자극 여부", "단일 배아 이식 여부", "착상 전 유전 진단 사용 여부",
        "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인",
        "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
        "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애",
        "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증",
        "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인",
        "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태",
        "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부",
    ]
    for c in binary_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ── [Fix 2] 연속형 변수 결측치: train 중앙값 사용 ─────────────────────────
    cont_cols = [
        "총 생성 배아 수", "미세주입된 난자 수", "미세주입에서 생성된 배아 수",
        "이식된 배아 수", "미세주입 배아 이식 수",
        "저장된 배아 수", "미세주입 후 저장된 배아 수",
        "해동된 배아 수", "해동 난자 수",
        "수집된 신선 난자 수", "저장된 신선 난자 수",
        "혼합된 난자 수", "파트너 정자와 혼합된 난자 수", "기증자 정자와 혼합된 난자 수",
        "난자 혼합 경과일", "배아 이식 경과일", "배아 해동 경과일",
    ]
    for c in cont_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            key = f"{c}_median"
            if key not in stats:                      # train: 계산 후 저장
                stats[key] = df[c].median() if df[c].notna().any() else -1
            df[c] = df[c].fillna(stats[key])          # test: train 중앙값 사용

    return df, label_encoders, stats


def preprocess_full(df: pd.DataFrame, fit_label_encoders=None, train_stats=None):
    """반환: (df, label_encoders, train_stats)"""
    df = expand_reason(df.copy())
    df = expand_procedure(df)
    df.drop(columns=["배아 생성 주요 이유", "특정 시술 유형"], inplace=True, errors="ignore")
    df, le, stats = preprocess(df, fit_label_encoders, train_stats)
    return df, le, stats


def feature_engineering_v2(df: pd.DataFrame, train_stats=None):
    """
    핵심 파생변수 7개.
    train_stats: None이면 현재 df(=train)에서 통계 계산 후 반환.
                 dict이면 test용 — train 통계값 사용.
    반환: (df, fe_stats)
    """
    df = df.copy()
    fe_stats = dict(train_stats) if train_stats else {}

    df["이식_효율"] = np.where(
        df["총 생성 배아 수"] > 0,
        df["이식된 배아 수"] / df["총 생성 배아 수"], 0
    )

    if "총 임신 횟수" in df.columns and "총 시술 횟수" in df.columns:
        df["과거_임신_성공률"] = np.where(
            df["총 시술 횟수"] > 0,
            df["총 임신 횟수"] / (df["총 시술 횟수"] + 1), 0
        )

    if "혼합된 난자 수" in df.columns:
        df["수정_효율"] = np.where(
            df["혼합된 난자 수"] > 0,
            df["총 생성 배아 수"] / df["혼합된 난자 수"], 0
        )

    if "수집된 신선 난자 수" in df.columns and "시술 당시 나이" in df.columns:
        age_num = df["시술 당시 나이"].replace(-1, np.nan)
        df["나이x난자수"] = (age_num * df["수집된 신선 난자 수"]).fillna(0)

    # ── [Fix 3] pd.cut bins: train max 사용 ──────────────────────────────────
    if "embryo_max" not in fe_stats:
        fe_stats["embryo_max"] = max(df["총 생성 배아 수"].max() + 1, 9)
    embryo_max = fe_stats["embryo_max"]
    df["배아_풍요도"] = pd.cut(
        df["총 생성 배아 수"], bins=[-1, 0, 3, 7, embryo_max], labels=[0, 1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    # ── [Fix 4] 고령저반응: train egg_median 사용 ─────────────────────────────
    if "수집된 신선 난자 수" in df.columns and "시술 당시 나이" in df.columns:
        if "egg_median" not in fe_stats:
            fe_stats["egg_median"] = df["수집된 신선 난자 수"].median()
        egg_median = fe_stats["egg_median"]
        age_num2 = df["시술 당시 나이"].replace(-1, np.nan)
        df["고령저반응"] = (
            (age_num2 >= 3) & (df["수집된 신선 난자 수"] < egg_median)
        ).fillna(False).astype(int)

    df["순수동결_주기"] = (
        (df["동결 배아 사용 여부"] == 1) & (df["신선 배아 사용 여부"] == 0)
    ).astype(int)

    return df, fe_stats


def load_data(data_dir="data"):
    """train / test 로드 및 컬럼 정리"""
    from pathlib import Path
    d = Path(data_dir)
    train = pd.read_csv(d / "train.csv", encoding="utf-8-sig")
    test  = pd.read_csv(d / "test.csv",  encoding="utf-8-sig")
    train.columns = [c.strip() for c in train.columns]
    test.columns  = [c.strip() for c in test.columns]
    return train, test
