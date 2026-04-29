from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from shapely.geometry import shape, Point
except Exception:
    shape = None
    Point = None


# =========================================================
# 0. 기본 설정
# =========================================================

st.set_page_config(
    page_title="생활물가 선행경보 대시보드",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path("data")

PANEL_CANDIDATES = [
    "28_final_living_price_dashboard_panel.csv",
    "final_living_price_dashboard_panel.csv",
    "living_price_dashboard_panel.csv",
]

PRESSURE_CANDIDATES = [
    "27_final_regional_price_pressure_detail_panel.csv",
    "final_regional_price_pressure_detail_panel.csv",
    "regional_price_pressure_detail_panel.csv",
]

GEOJSON_CANDIDATES = [
    "06_region_sido_boundary.geojson",
    "region_sido_boundary.geojson",
    "sido_boundary.geojson",
]

PRESSURE_SCALE = 60.0
PRESSURE_AUTO_TARGET_MEDIAN_ABS = 30.0
PRESSURE_AUTO_MIN_MEDIAN_ABS = 10.0
PRESSURE_CLIP_MIN = -70
PRESSURE_CLIP_MAX = 70


# =========================================================
# 1. CSS
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 2.0rem !important;
        max-width: 1380px !important;
    }

    h1 {
        font-size: 2.05rem !important;
        line-height: 1.25 !important;
        margin-bottom: 0.35rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.04em;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 0.7rem !important;
        margin-bottom: 0.45rem !important;
    }

    h3 {
        font-size: 1.17rem !important;
        margin-top: 0.45rem !important;
        margin-bottom: 0.35rem !important;
    }

    p, li, label, .stMarkdown {
        font-size: 0.96rem !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 0.78rem 0.88rem 0.84rem 0.88rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        min-height: 98px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        color: #4b5563 !important;
        font-weight: 650 !important;
        padding-bottom: 0.10rem !important;
        white-space: normal !important;
        line-height: 1.2 !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.30rem !important;
        line-height: 1.10 !important;
        color: #111827 !important;
        font-weight: 720 !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
    }

    [data-testid="stSidebar"] {
        background: #f3f4f6;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 1.05rem !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        font-size: 0.92rem !important;
        font-weight: 650 !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        font-size: 0.96rem !important;
        min-height: 42px !important;
        background: #ffffff;
    }

    .small-note {
        font-size: 0.90rem !important;
        color: #6b7280 !important;
        margin-top: -0.1rem !important;
        margin-bottom: 0.6rem !important;
    }

    .guide-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.0rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. 상수
# =========================================================

REGION_ALIASES = {
    "서울특별시": "서울", "서울": "서울",
    "부산광역시": "부산", "부산": "부산",
    "대구광역시": "대구", "대구": "대구",
    "인천광역시": "인천", "인천": "인천",
    "광주광역시": "광주", "광주": "광주",
    "대전광역시": "대전", "대전": "대전",
    "울산광역시": "울산", "울산": "울산",
    "세종특별자치시": "세종", "세종": "세종",
    "경기도": "경기", "경기": "경기",
    "강원도": "강원", "강원특별자치도": "강원", "강원": "강원",
    "충청북도": "충북", "충북": "충북",
    "충청남도": "충남", "충남": "충남",
    "전라북도": "전북", "전북특별자치도": "전북", "전북": "전북",
    "전라남도": "전남", "전남": "전남",
    "경상북도": "경북", "경북": "경북",
    "경상남도": "경남", "경남": "경남",
    "제주특별자치도": "제주", "제주도": "제주", "제주": "제주",
}

REGION_ORDER = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]

COLOR_ACTUAL = "#2563eb"
COLOR_FORECAST = "#f59e0b"
COLOR_SELECTED = "#ef4444"
COLOR_BAR = "#f59e0b"
COLOR_MACRO = "#d97706"


# =========================================================
# 3. 유틸 함수
# =========================================================

def clean_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def normalize_region_name(x) -> str:
    s = clean_str(x)
    if not s:
        return ""

    if s in REGION_ALIASES:
        return REGION_ALIASES[s]

    s = s.replace("특별자치도", "")
    s = s.replace("특별자치시", "")
    s = s.replace("특별시", "")
    s = s.replace("광역시", "")
    s = s.replace("도", "")
    s = s.strip()

    extra = {
        "충청북": "충북",
        "충청남": "충남",
        "전라북": "전북",
        "전라남": "전남",
        "경상북": "경북",
        "경상남": "경남",
    }
    return extra.get(s, s)


def region_sort_key(x) -> int:
    return REGION_ORDER.index(x) if x in REGION_ORDER else 999


def parse_quarter_value(q) -> Tuple[int, int]:
    s = clean_str(q).replace(" ", "")
    if not s:
        return 0, 0

    patterns = [
        r"^(\d{4})Q([1-4])$",
        r"^(\d{4})-Q([1-4])$",
        r"^(\d{4})/([1-4])분기$",
        r"^(\d{4})년([1-4])분기$",
        r"^(\d{4})([1-4])$",
    ]

    for p in patterns:
        m = re.match(p, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))

    return 0, 0


def normalize_quarter(q) -> str:
    y, qt = parse_quarter_value(q)
    if y == 0:
        return clean_str(q)
    return f"{y}Q{qt}"


def format_quarter_label(q) -> str:
    y, qt = parse_quarter_value(q)
    if y == 0:
        return clean_str(q)
    return f"{y} / {qt}분기"


def quarter_to_date(q) -> pd.Timestamp:
    y, qt = parse_quarter_value(q)
    if y == 0:
        return pd.Timestamp("2000-01-01")
    month = {1: 1, 2: 4, 3: 7, 4: 10}.get(qt, 1)
    return pd.Timestamp(year=y, month=month, day=1)


def quarter_sort_key(q) -> Tuple[int, int]:
    return parse_quarter_value(q)


def previous_quarter(q: str) -> str:
    y, qt = parse_quarter_value(q)

    if y == 0:
        return ""

    if qt == 1:
        return f"{y - 1}Q4"

    return f"{y}Q{qt - 1}"


def safe_num(v, digits: int = 2) -> str:
    try:
        if pd.isna(v):
            return "-"
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "-"


def safe_int(v) -> str:
    try:
        if pd.isna(v):
            return "-"
        return f"{int(round(float(v))):,}"
    except Exception:
        return "-"


def to_numeric(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clip_score(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(0, 100)


def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)


def find_existing_file(candidates: List[str], suffixes: Tuple[str, ...]) -> Optional[Path]:
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p

    if DATA_DIR.exists():
        files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in suffixes]
        return files[0] if files else None

    return None


def normalize_weights(regional_weight: float, macro_weight: float) -> Dict[str, float]:
    total = regional_weight + macro_weight
    if total <= 0:
        return {"regional": 0.5, "macro": 0.5}
    return {
        "regional": regional_weight / total,
        "macro": macro_weight / total,
    }


def prediction_error_level(abs_err) -> str:
    if pd.isna(abs_err):
        return "-"

    abs_err = float(abs_err)

    if abs_err <= 5:
        return "우수"
    if abs_err <= 10:
        return "보통"
    return "나쁨"


def error_level_summary(df: pd.DataFrame) -> str:
    if "prediction_error_level" not in df.columns:
        return "-"

    counts = df["prediction_error_level"].value_counts(dropna=True).to_dict()

    good = int(counts.get("우수", 0))
    normal = int(counts.get("보통", 0))
    bad = int(counts.get("나쁨", 0))

    return f"우수 {good} · 보통 {normal} · 나쁨 {bad}"


def chart_config(show_modebar=True, fixed=False):
    return {
        "displayModeBar": show_modebar,
        "displaylogo": False,
        "scrollZoom": not fixed,
        "responsive": True,
    }


def fixed_chart_config():
    return {
        "displayModeBar": False,
        "displaylogo": False,
        "scrollZoom": False,
        "responsive": True,
        "doubleClick": False,
    }


def scale_pressure_series(raw_pressure: pd.Series) -> pd.Series:
    s = pd.to_numeric(raw_pressure, errors="coerce").fillna(0)
    scaled = s * PRESSURE_SCALE

    median_abs = scaled.abs().replace(0, np.nan).median(skipna=True)

    if pd.notna(median_abs) and 0 < median_abs < PRESSURE_AUTO_MIN_MEDIAN_ABS:
        auto_factor = PRESSURE_AUTO_TARGET_MEDIAN_ABS / median_abs
        scaled = scaled * auto_factor

    return scaled.clip(PRESSURE_CLIP_MIN, PRESSURE_CLIP_MAX)


# =========================================================
# 4. 데이터 로드
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():
    panel_path = find_existing_file(PANEL_CANDIDATES, (".csv",))
    pressure_path = None

    for name in PRESSURE_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            pressure_path = p
            break

    geo_path = None
    for name in GEOJSON_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            geo_path = p
            break

    if geo_path is None and DATA_DIR.exists():
        geo_files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in [".geojson", ".json"]]
        geo_path = geo_files[0] if geo_files else None

    if panel_path is None:
        return pd.DataFrame(), pd.DataFrame(), None

    panel = read_csv(panel_path)
    pressure = read_csv(pressure_path) if pressure_path else pd.DataFrame()

    geo = None
    if geo_path:
        with open(geo_path, "r", encoding="utf-8") as f:
            geo = json.load(f)

    return panel, pressure, geo


raw_panel, raw_pressure, raw_geo = load_data()

if raw_panel.empty:
    st.error("data 폴더에서 대시보드 패널 CSV를 찾지 못했습니다.")
    st.stop()


# =========================================================
# 5. 패널 표준화
# =========================================================

def standardize_panel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()

    region_col = first_col(d, ["region", "region_short", "시도", "시도명", "지역"])
    quarter_col = first_col(d, ["year_quarter", "quarter", "target_year_quarter", "target_quarter", "분기"])
    data_type_col = first_col(d, ["data_type", "dataset", "type"])

    if region_col is None or quarter_col is None:
        st.error("패널 데이터에 region/quarter 컬럼이 필요합니다.")
        st.stop()

    d["region_short"] = d[region_col].map(normalize_region_name)

    if "target_year_quarter" in d.columns:
        d["target_quarter"] = d["target_year_quarter"].map(normalize_quarter)
    elif "target_quarter" in d.columns:
        d["target_quarter"] = d["target_quarter"].map(normalize_quarter)
    else:
        d["target_quarter"] = d[quarter_col].map(normalize_quarter)

    if "year_quarter" in d.columns:
        d["quarter"] = d["year_quarter"].map(normalize_quarter)
    else:
        d["quarter"] = d[quarter_col].map(normalize_quarter)

    if "x_year_quarter" in d.columns:
        d["base_quarter"] = d["x_year_quarter"].map(normalize_quarter)
    elif "x_quarter" in d.columns:
        d["base_quarter"] = d["x_quarter"].map(normalize_quarter)
    else:
        d["base_quarter"] = np.nan

    if data_type_col is not None:
        d["data_type"] = d[data_type_col].astype(str)
    else:
        d["data_type"] = ""

    actual_score_col = first_col(
        d,
        [
            "actual_living_price_risk_score",
            "actual_target_risk_score",
            "actual_risk_score",
            "living_price_risk_score",
            "risk_score",
        ],
    )

    pred_score_col = first_col(
        d,
        [
            "scenario_forecast_score",
            "pred_score",
            "forecast_score",
            "predicted_risk_score",
            "prediction_score",
        ],
    )

    current_score_col = first_col(
        d,
        [
            "current_living_price_risk_score",
            "current_regional_living_price_risk_score",
            "prev_living_price_risk_score",
            "previous_living_price_risk_score",
            "base_score",
            "current_score",
        ],
    )

    pressure_col = first_col(
        d,
        [
            "scenario_pressure_index",
            "pressure_index",
            "lead_pressure_index",
            "leading_pressure_index",
        ],
    )

    actual_df_local = d[d["data_type"].str.contains("actual", case=False, na=False)].copy()

    if actual_df_local.empty and actual_score_col is not None:
        actual_df_local = d[d[actual_score_col].notna()].copy()

    actual_out = pd.DataFrame()
    if actual_score_col is not None and not actual_df_local.empty:
        actual_out = pd.DataFrame({
            "region_short": actual_df_local["region_short"],
            "quarter": actual_df_local["quarter"],
            "actual_target_risk_score": clip_score(actual_df_local[actual_score_col]),
        })
        actual_out = actual_out.dropna(subset=["region_short", "quarter", "actual_target_risk_score"])
        actual_out = actual_out[actual_out["region_short"] != ""]
        actual_out = actual_out.drop_duplicates(["region_short", "quarter"], keep="last")

    pred_df_local = d[d["data_type"].str.contains("future|backtest|pred", case=False, na=False)].copy()

    if pred_df_local.empty and pred_score_col is not None:
        pred_df_local = d[d[pred_score_col].notna()].copy()

    pred_out = pd.DataFrame()
    if not pred_df_local.empty:
        pred_out = pd.DataFrame({
            "region_short": pred_df_local["region_short"],
            "target_quarter": pred_df_local["target_quarter"],
            "base_quarter": pred_df_local["base_quarter"],
            "data_type": pred_df_local["data_type"],
            "current_living_price_risk_score": clip_score(pred_df_local[current_score_col]) if current_score_col else np.nan,
            "scenario_forecast_score_raw": clip_score(pred_df_local[pred_score_col]) if pred_score_col else np.nan,
            "scenario_pressure_index_raw": to_numeric(pred_df_local[pressure_col]) if pressure_col else np.nan,
        })
        pred_out = pred_out.dropna(subset=["region_short", "target_quarter"])
        pred_out = pred_out[pred_out["region_short"] != ""]
        pred_out = pred_out.drop_duplicates(["region_short", "target_quarter"], keep="last")

    return actual_out, pred_out


actual_df, forecast_df = standardize_panel(raw_panel)


# =========================================================
# 6. 압력 상세 표준화
# =========================================================

def classify_feature_group(feature: str) -> str:
    s = clean_str(feature).lower()

    if any(k in s for k in ["금리", "rate", "cd", "ktb", "loan", "oil", "usd", "환율", "수입"]):
        return "거시 금융환경"

    return "지역 물가 흐름"


def feature_label(feature: str) -> str:
    key = clean_str(feature).lower()

    mapping = {
        # 기준 위험점수
        "current_regional_living_price_risk_score": "이전 분기 위험점수",
        "current_living_price_risk_score": "이전 분기 위험점수",
        "prev_living_price_risk_score": "이전 분기 위험점수",
        "previous_living_price_risk_score": "이전 분기 위험점수",
        "base_score": "이전 분기 위험점수",
        "current_score": "이전 분기 위험점수",

        # 지역 물가 흐름
        "regional_living_food_index_yoy_pct": "식품 생활물가 상승률",
        "regional_fresh_food_index_yoy_pct": "신선식품 가격 체감 변화",
        "regional_living_csi_index_yoy_pct": "생활물가 체감 변화",
        "regional_agri_livestock_fishery_csi_index_yoy_pct": "농축수산물 체감 변화",
        "regional_csi_total_index_yoy_pct": "지역 소비심리 변화",

        # 파일에 CPI 이름으로 남아 있는 변수명 처리
        "regional_cpi_total_index_yoy_pct": "지역 생활물가 종합 변화",
        "regional_living_cpi_index_yoy_pct": "생활물가 관련 지표 변화",
        "regional_fresh_food_cpi_index_yoy_pct": "신선식품 관련 지표 변화",
        "regional_agri_livestock_fishery_cpi_index_yoy_pct": "농축수산물 관련 지표 변화",

        # 거시 금융환경
        "base_rate": "기준금리",
        "cd_91d_rate": "CD 91일물 금리",
        "ktb_3y_rate": "국고채 3년물 금리",
        "ktb_10y_rate": "국고채 10년물 금리",
        "ktb_10y_rate_qoq_diff": "국고채 10년물 금리 변화",
        "bank_loan_rate": "예금은행 대출금리",
        "import_price_index_qoq_diff": "수입물가지수 변화",
        "dubai_oil_price_qoq_diff": "두바이유 가격 변화",
        "usdkrw_qoq_diff": "원/달러 환율 변화",
    }

    if key in mapping:
        return mapping[key]

    # 규칙 기반 fallback: 영어 변수명이 그대로 보이지 않게 정리
    label = key

    replace_rules = [
        ("regional_", "지역 "),
        ("current_", "현재 "),
        ("previous_", "이전 "),
        ("prev_", "이전 "),
        ("base_", "기준 "),
        ("_yoy_pct", " 전년동기비"),
        ("_qoq_diff", " 전분기 변화"),
        ("_index", " 지표"),
        ("_score", " 점수"),
        ("_rate", " 금리"),
        ("_", " "),
        ("cpi", "생활물가"),
        ("csi", "소비심리"),
        ("agri livestock fishery", "농축수산물"),
        ("fresh food", "신선식품"),
        ("living food", "식품 생활물가"),
        ("living", "생활"),
        ("total", "종합"),
        ("ktb", "국고채"),
        ("cd 91d", "CD 91일물"),
        ("bank loan", "예금은행 대출"),
        ("import price", "수입물가"),
        ("dubai oil price", "두바이유 가격"),
        ("usdkrw", "원/달러 환율"),
    ]

    for old, new in replace_rules:
        label = label.replace(old, new)

    label = " ".join(label.split()).strip()

    if not label:
        return "기타 변수"

    return label


def standardize_pressure(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "region_short", "target_quarter", "base_quarter",
                "feature", "feature_label", "group_label",
                "pressure_z", "weight", "contribution"
            ]
        )

    d = df.copy()

    region_col = first_col(d, ["region", "region_short", "시도", "시도명", "지역"])
    target_col = first_col(d, ["target_quarter", "target_year_quarter", "quarter", "year_quarter", "분기"])
    base_col = first_col(d, ["x_quarter", "x_year_quarter", "base_quarter"])
    feature_col = first_col(d, ["feature", "variable", "변수", "변수명"])
    pressure_z_col = first_col(d, ["pressure_z", "z", "z_score"])
    weight_col = first_col(d, ["weight", "importance", "variable_weight"])

    if region_col is None or target_col is None or feature_col is None:
        return pd.DataFrame(
            columns=[
                "region_short", "target_quarter", "base_quarter",
                "feature", "feature_label", "group_label",
                "pressure_z", "weight", "contribution"
            ]
        )

    out = pd.DataFrame({
        "region_short": d[region_col].map(normalize_region_name),
        "target_quarter": d[target_col].map(normalize_quarter),
        "base_quarter": d[base_col].map(normalize_quarter) if base_col else np.nan,
        "feature": d[feature_col].astype(str),
        "pressure_z": to_numeric(d[pressure_z_col]) if pressure_z_col else 0.0,
        "weight": to_numeric(d[weight_col]) if weight_col else 1.0,
    })

    out["feature_label"] = out["feature"].map(feature_label)
    out["group_label"] = out["feature"].map(classify_feature_group)
    out["contribution"] = out["pressure_z"] * out["weight"]

    out = out.dropna(subset=["region_short", "target_quarter", "feature"])
    out = out[out["region_short"] != ""]
    return out


pressure_df = standardize_pressure(raw_pressure)


# =========================================================
# 7. 예측 계산
# =========================================================

def fill_base_score_from_actual(df: pd.DataFrame) -> pd.Series:
    if df.empty or actual_df.empty:
        return pd.Series(np.nan, index=df.index)

    actual_lookup = actual_df.copy()
    actual_lookup["lookup_key"] = (
        actual_lookup["region_short"].astype(str)
        + "__"
        + actual_lookup["quarter"].astype(str)
    )

    score_map = actual_lookup.set_index("lookup_key")["actual_target_risk_score"].to_dict()

    keys = (
        df["region_short"].astype(str)
        + "__"
        + df["target_quarter"].map(previous_quarter).astype(str)
    )

    return keys.map(score_map)


def calculate_forecast(
    forecast: pd.DataFrame,
    pressure: pd.DataFrame,
    lambda_pressure: float,
    weights: Dict[str, float],
) -> pd.DataFrame:
    if forecast.empty:
        return forecast.copy()

    out = forecast.copy()

    if not pressure.empty:
        p = pressure.copy()

        p["group_weight"] = np.where(
            p["group_label"] == "지역 물가 흐름",
            weights["regional"],
            weights["macro"],
        )

        p["weighted_contribution_raw"] = p["contribution"] * p["group_weight"]

        agg = (
            p.groupby(["region_short", "target_quarter"], as_index=False)
            .agg(base_pressure_index=("weighted_contribution_raw", "sum"))
        )

        out = out.merge(
            agg,
            on=["region_short", "target_quarter"],
            how="left",
        )

        raw_pressure = out["base_pressure_index"].fillna(out["scenario_pressure_index_raw"])
    else:
        raw_pressure = out["scenario_pressure_index_raw"]

    out["base_pressure_index"] = scale_pressure_series(raw_pressure)

    out["scenario_pressure_index"] = (
        out["base_pressure_index"] * lambda_pressure
    ).clip(PRESSURE_CLIP_MIN, PRESSURE_CLIP_MAX)

    out["current_living_price_risk_score"] = clip_score(
        out["current_living_price_risk_score"]
    )

    prev_actual_score = fill_base_score_from_actual(out)

    out["current_living_price_risk_score"] = out[
        "current_living_price_risk_score"
    ].fillna(prev_actual_score)

    out["current_living_price_risk_score"] = clip_score(
        out["current_living_price_risk_score"]
    )

    recalculated = (
        out["current_living_price_risk_score"]
        + out["scenario_pressure_index"]
    ).clip(0, 100)

    raw_pred = clip_score(out["scenario_forecast_score_raw"])

    out["scenario_forecast_score"] = recalculated.where(
        out["current_living_price_risk_score"].notna(),
        raw_pred,
    )

    out["scenario_forecast_score"] = clip_score(out["scenario_forecast_score"])

    out["scenario_delta"] = (
        out["scenario_forecast_score"]
        - out["current_living_price_risk_score"]
    )

    return out


def build_display(
    selected_quarter: str,
    lambda_pressure: float,
    weights: Dict[str, float],
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    actual_q = actual_df[actual_df["quarter"] == selected_quarter].copy()

    forecast_calc = calculate_forecast(forecast_df, pressure_df, lambda_pressure, weights)
    forecast_q = forecast_calc[forecast_calc["target_quarter"] == selected_quarter].copy()

    display = pd.merge(
        forecast_q[
            [
                "region_short",
                "target_quarter",
                "base_quarter",
                "current_living_price_risk_score",
                "base_pressure_index",
                "scenario_pressure_index",
                "scenario_forecast_score",
                "scenario_delta",
            ]
        ].rename(columns={"target_quarter": "quarter"}),
        actual_q[["region_short", "quarter", "actual_target_risk_score"]],
        on=["region_short", "quarter"],
        how="outer",
    )

    if display.empty:
        return display, "empty", forecast_calc

    if display["actual_target_risk_score"].notna().any():
        display["actual_target_rank"] = display["actual_target_risk_score"].rank(
            ascending=False,
            method="min",
        )
        n = display["actual_target_risk_score"].notna().sum()
        display["actual_target_percentile"] = ((n - display["actual_target_rank"] + 1) / n) * 100

    if display["scenario_forecast_score"].notna().any():
        display["scenario_rank"] = display["scenario_forecast_score"].rank(
            ascending=False,
            method="min",
        )
        n = display["scenario_forecast_score"].notna().sum()
        display["scenario_percentile"] = ((n - display["scenario_rank"] + 1) / n) * 100

    if {"actual_target_risk_score", "scenario_forecast_score"}.issubset(display.columns):
        display["prediction_error"] = display["scenario_forecast_score"] - display["actual_target_risk_score"]
        display["abs_prediction_error"] = display["prediction_error"].abs()
        display["prediction_error_level"] = display["abs_prediction_error"].map(prediction_error_level)

    has_actual = display["actual_target_risk_score"].notna().any()
    has_pred = display["scenario_forecast_score"].notna().any()

    if has_actual and has_pred:
        status = "actual_with_prediction"
    elif has_actual:
        status = "actual_only"
    elif has_pred:
        status = "forecast_only"
    else:
        status = "empty"

    display = display.sort_values("region_short", key=lambda s: s.map(region_sort_key)).reset_index(drop=True)
    return display, status, forecast_calc


# =========================================================
# 8. GeoJSON
# =========================================================

@st.cache_data(show_spinner=False)
def prepare_geojson(geo):
    if not geo or "features" not in geo:
        return None, None, pd.DataFrame()

    gj = json.loads(json.dumps(geo))
    labels = []

    candidate_keys = [
        "region", "name", "sidonm", "SIDO_NM", "CTP_KOR_NM", "ctp_kor_nm",
        "adm_nm", "NAME_1", "name_1", "시도명", "시도"
    ]

    for feat in gj["features"]:
        props = feat.get("properties", {})
        selected_key = None

        lower_map = {str(k).lower(): k for k in props.keys()}
        for c in candidate_keys:
            if c.lower() in lower_map:
                selected_key = lower_map[c.lower()]
                break

        if selected_key is None and props:
            selected_key = list(props.keys())[0]

        raw_name = props.get(selected_key, "") if selected_key else ""
        region = normalize_region_name(raw_name)
        feat["properties"]["region_short"] = region

        if shape is not None:
            try:
                geom = shape(feat["geometry"])
                pt = geom.representative_point()
                labels.append({"region_short": region, "lat": pt.y, "lon": pt.x})
            except Exception:
                pass

    labels_df = pd.DataFrame(labels).drop_duplicates("region_short") if labels else pd.DataFrame()
    return gj, "properties.region_short", labels_df


# =========================================================
# 9. 차트 - 지도 함수 (Plotly 기반, 빠른 렌더링)
# =========================================================

def build_map(display_df: pd.DataFrame, status: str, selected_region: str):
    """
    Plotly Choroplethmapbox 기반 지도.
    iframe 없이 직접 렌더링 → Folium 대비 훨씬 빠름.
    모든 지역을 점수에 따라 색칠하고, 선택 지역만 경계선 강조.
    """
    gj, featureidkey, _ = prepare_geojson(raw_geo)
    if gj is None:
        return None

    if status == "forecast_only":
        value_col = "scenario_forecast_score"
        label = "예측 위험점수"
    else:
        value_col = "actual_target_risk_score"
        label = "실제 위험점수"

    df = display_df.dropna(subset=["region_short"]).copy()
    if df.empty:
        return None

    all_regions = list(df["region_short"].unique())
    score_map   = {r: float(v) for r, v in zip(df["region_short"], df[value_col]) if pd.notna(v)}

    locs  = all_regions
    zvals = [score_map.get(r, 0) for r in locs]

    # 선택/비선택 경계선 두께·색 분리
    line_widths = [3.5 if r == selected_region else 1.0 for r in locs]
    line_colors = ["#c41e3a" if r == selected_region else "#888888" for r in locs]

    fig = go.Figure()

    fig.add_trace(go.Choroplethmapbox(
        geojson=gj,
        locations=locs,
        z=zvals,
        featureidkey=featureidkey,
        colorscale="YlOrRd",
        zmin=0,
        zmax=100,
        marker=dict(
            line_width=line_widths,
            line_color=line_colors,
            opacity=0.85,
        ),
        colorbar=dict(
            title=dict(text=label, font=dict(size=12)),
            thickness=14,
            len=0.7,
            tickfont=dict(size=11),
        ),
        hovertemplate="<b>%{location}</b><br>" + label + ": %{z:.1f}<extra></extra>",
        name="",
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5.5,
        mapbox_center={"lat": 36.5, "lon": 127.5},
        height=560,
        margin=dict(l=0, r=10, t=0, b=0),
        paper_bgcolor="white",
        clickmode="event+select",
    )

    return fig


def build_rank_bar(display_df: pd.DataFrame, status: str):
    if status == "forecast_only":
        value_col = "scenario_forecast_score"
        title = "지역별 예측 위험점수"
    else:
        value_col = "actual_target_risk_score"
        title = "지역별 실제 위험점수"

    df = display_df[["region_short", value_col]].dropna().copy()
    if df.empty:
        return None

    df = df.sort_values(value_col, ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[value_col],
            y=df["region_short"],
            orientation="h",
            marker=dict(color=COLOR_BAR),
            text=df[value_col].round(1),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>위험점수: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=18)),
        height=max(500, len(df) * 30 + 110),
        margin=dict(l=65, r=45, t=60, b=55),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title="위험점수",
            range=[0, 100],
            fixedrange=True,
            tickfont=dict(size=11),
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12),
            fixedrange=True,
        ),
        showlegend=False,
    )
    return fig


def trend_base(region: str, forecast_calc: pd.DataFrame) -> pd.DataFrame:
    a = actual_df[actual_df["region_short"] == region][["quarter", "actual_target_risk_score"]].copy()

    if not a.empty:
        a["quarter"] = a["quarter"].astype(str)
        a["actual_target_risk_score"] = clip_score(a["actual_target_risk_score"])

    f = forecast_calc[forecast_calc["region_short"] == region][["target_quarter", "scenario_forecast_score"]].copy()

    if not f.empty:
        f = f.rename(columns={"target_quarter": "quarter"})
        f["quarter"] = f["quarter"].astype(str)
        f["scenario_forecast_score"] = clip_score(f["scenario_forecast_score"])

    quarters = sorted(
        set(a["quarter"].dropna().tolist()) | set(f["quarter"].dropna().tolist()),
        key=quarter_sort_key,
    )

    base = pd.DataFrame({"quarter": quarters})

    if base.empty:
        return base

    if not a.empty:
        base = base.merge(a, on="quarter", how="left")

    if not f.empty:
        base = base.merge(f, on="quarter", how="left")

    base["quarter_date"] = base["quarter"].map(quarter_to_date)

    return base.sort_values("quarter_date").reset_index(drop=True)


def filter_window(df: pd.DataFrame, selected_quarter: str, window: str) -> pd.DataFrame:
    if df.empty:
        return df

    selected_date = quarter_to_date(selected_quarter)
    d = df[df["quarter_date"] <= selected_date].copy()

    if window == "1년":
        return d.tail(4)
    if window == "5년":
        return d.tail(20)
    return d


def build_trend_chart(df: pd.DataFrame, selected_quarter: str, region: str):
    if df.empty:
        return None

    fig = go.Figure()

    if "actual_target_risk_score" in df.columns and df["actual_target_risk_score"].notna().any():
        actual_part = df.dropna(subset=["actual_target_risk_score"])

        fig.add_trace(
            go.Scatter(
                x=actual_part["quarter_date"],
                y=actual_part["actual_target_risk_score"],
                mode="lines+markers",
                name="실제 위험점수",
                line=dict(color=COLOR_ACTUAL, width=2.7),
                marker=dict(size=7, color=COLOR_ACTUAL, line=dict(color="white", width=1)),
                hovertemplate="%{x|%Y-%m}<br>실제 위험점수: %{y:.2f}<extra></extra>",
            )
        )

    sel_date = quarter_to_date(selected_quarter)
    sel = df[df["quarter"] == selected_quarter]

    if not sel.empty:
        sel_actual = sel["actual_target_risk_score"].iloc[0] if "actual_target_risk_score" in sel.columns else np.nan
        sel_pred = sel["scenario_forecast_score"].iloc[0] if "scenario_forecast_score" in sel.columns else np.nan

        if not pd.isna(sel_pred):
            fig.add_trace(
                go.Scatter(
                    x=[sel_date],
                    y=[sel_pred],
                    mode="markers",
                    name="해당 분기 예측점",
                    marker=dict(size=16, color=COLOR_SELECTED, line=dict(color="white", width=2)),
                    hovertemplate=format_quarter_label(selected_quarter) + "<br>예측 위험점수: %{y:.2f}<extra></extra>",
                )
            )

        if not pd.isna(sel_actual):
            fig.add_trace(
                go.Scatter(
                    x=[sel_date],
                    y=[sel_actual],
                    mode="markers",
                    name="해당 분기 실제점",
                    marker=dict(size=12, color=COLOR_ACTUAL, line=dict(color="white", width=2)),
                    hovertemplate=format_quarter_label(selected_quarter) + "<br>실제 위험점수: %{y:.2f}<extra></extra>",
                )
            )

        fig.add_vline(
            x=sel_date,
            line_color=COLOR_SELECTED,
            line_width=1.5,
            line_dash="dash",
        )

    fig.update_layout(
        title=dict(text=f"{region} · 실제 위험점수 추이와 해당 분기 예측점", x=0, font=dict(size=18, color="#111827")),
        height=540,
        margin=dict(l=70, r=40, t=95, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            x=1,
            y=1.18,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=12, color="#111827"),
        ),
        xaxis=dict(
            title=dict(text="분기", font=dict(size=12)),
            type="date",
            tickformat="%Y",
            rangeslider=dict(visible=True, thickness=0.10),
            tickfont=dict(size=11),
            gridcolor="rgba(0,0,0,0.08)",
        ),
        yaxis=dict(
            title=dict(text="위험점수", font=dict(size=12)),
            range=[0, 100],
            tickfont=dict(size=11),
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
        ),
        hovermode="x unified",
    )

    return fig


def build_contribution_chart(
    region: str,
    selected_quarter: str,
    weights: Dict[str, float],
    lambda_pressure: float,
):
    if pressure_df.empty:
        return None, None

    d = pressure_df[
        (pressure_df["region_short"] == region)
        & (pressure_df["target_quarter"] == selected_quarter)
    ].copy()

    if d.empty:
        return None, None

    # 영어 변수명이 그대로 노출되지 않도록 강제 변환
    d["feature_label"] = d["feature"].apply(feature_label)

    d["group_weight"] = np.where(
        d["group_label"] == "지역 물가 흐름",
        weights["regional"],
        weights["macro"],
    )

    d["weighted_contribution_raw"] = d["contribution"] * d["group_weight"]
    d["weighted_contribution"] = (
        scale_pressure_series(d["weighted_contribution_raw"]) * lambda_pressure
    )
    d["weighted_contribution"] = d["weighted_contribution"].clip(
        PRESSURE_CLIP_MIN,
        PRESSURE_CLIP_MAX,
    )

    d["abs_val"] = d["weighted_contribution"].abs()
    d = d.sort_values("abs_val", ascending=False).head(12)
    d = d.sort_values("weighted_contribution", ascending=True)

    d["impact_direction"] = np.where(
        d["weighted_contribution"] > 0,
        "예측점수 상승",
        np.where(d["weighted_contribution"] < 0, "예측점수 하락", "영향 거의 없음"),
    )

    fig = go.Figure()

    for group_name, color in [
        ("지역 물가 흐름", COLOR_ACTUAL),
        ("거시 금융환경", COLOR_MACRO),
    ]:
        part = d[d["group_label"] == group_name]

        if part.empty:
            continue

        fig.add_trace(
            go.Bar(
                x=part["weighted_contribution"],
                y=part["feature_label"],
                orientation="h",
                name=group_name,
                marker=dict(color=color),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "예측점수 영향: %{x:.2f}점"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{region} · 예측점수에 영향을 준 요인",
            x=0,
            font=dict(size=18),
        ),
        height=max(450, len(d) * 38 + 130),
        margin=dict(l=220, r=40, t=90, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            x=1,
            y=1.17,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            font=dict(size=12),
        ),
        xaxis=dict(
            title="예측 위험점수에 미친 영향",
            fixedrange=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.45)",
        ),
        yaxis=dict(
            title="",
            fixedrange=True,
            tickfont=dict(size=11),
            automargin=True,
        ),
        barmode="relative",
        dragmode=False,
    )

    table = d[
        [
            "feature_label",
            "group_label",
            "impact_direction",
            "weighted_contribution",
        ]
    ].copy()

    table.columns = [
        "변수명",
        "변수군",
        "영향 방향",
        "예측점수 영향",
    ]

    table["예측점수 영향"] = table["예측점수 영향"].round(2)

    return fig, table

# =========================================================
# 10. 지표 설명
# =========================================================

def render_guide():
    st.markdown("## 지표 설명")

    st.markdown(
        """
<div class="guide-box">

### 1) 산출 구조

이 대시보드는 지역별 생활물가 부담을 **실제 위험점수**와 **예측 위험점수**로 나누어 보여줍니다.

- **실제 위험점수**: 해당 분기의 실제 생활물가 부담 수준
- **기본 선행압력지수**: 지역 물가 흐름과 거시 금융환경을 종합한 사전 신호
- **조정 선행압력지수**: 기본 선행압력지수에 λ를 반영한 값
- **예측 위험점수**: 이전 분기 위험점수에 조정 선행압력지수를 더한 값

</div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        "조정 선행압력지수 = 기본 선행압력지수 × λ\n"
        "예측 위험점수 = 이전 분기 위험점수 + 조정 선행압력지수",
        language="text",
    )

    st.markdown(
        """
<div class="guide-box">

### 2) 위험점수

위험점수는 지역별 생활물가 부담을 **0~100점**으로 환산한 값입니다.  
값이 높을수록 생활물가 부담 위험이 큰 지역입니다.

위험점수는 다음 순서로 산출합니다.

1. **지역별 생활물가 관련 변수 수집**  
   식품 생활물가, 신선식품, 생활 체감, 소비심리 관련 변수를 지역·분기 단위로 정리합니다.

2. **변수 방향 통일**  
   값이 커질수록 생활물가 부담이 커지는 방향으로 변환합니다.

3. **0~100점 표준화**  
   변수마다 단위가 다르기 때문에 지역 간 비교가 가능하도록 0~100점 범위로 변환합니다.

4. **종합 위험점수 계산**  
   표준화된 변수들을 종합하여 해당 분기의 지역별 실제 위험점수를 계산합니다.

5. **이전 분기 위험점수로 연결**  
   예측 대상 분기의 바로 전 분기 위험점수를 예측의 출발점으로 사용합니다.

**지역 물가 흐름 변수**

- 이전 분기 위험점수
- 식품 생활물가 관련 지표 전년동기비
- 농축수산물 체감 관련 지표 전년동기비
- 생활 체감 관련 지표 전년동기비
- 지역 소비심리 관련 지표 전년동기비

**거시 금융환경 변수**

- 기준금리
- CD 91일물 금리
- 국고채 3년물 금리
- 국고채 10년물 금리 변화
- 예금은행 대출금리

</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="guide-box">

### 3) 선행압력지수

선행압력지수는 향후 생활물가 위험이 커질 가능성을 나타내는 보조 지표입니다.

- **양수**: 이전 분기보다 위험이 높아질 가능성
- **음수**: 이전 분기보다 위험이 낮아질 가능성
- **0 근처**: 뚜렷한 상승 또는 하락 신호가 약함

선행압력지수는 변수별 표준화 결과를 종합한 뒤, 지역 간 차이가 드러나도록 스케일을 조정했습니다.  
현재 기본 스케일 계수는 **{PRESSURE_SCALE:.1f}**이며, 값이 지나치게 작을 경우 자동 보정을 적용합니다.

사이드바의 λ는 기본 선행압력지수를 예측에 얼마나 반영할지 정하는 값입니다.  
따라서 λ를 조절하면 화면의 선행압력지수, 예측 위험점수, 과거 예측 오차 수준이 함께 변합니다.

변수별 기여도는 해당 **지역, 분기, λ, 가중치 설정**에 따라 달라집니다.

</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="guide-box">

### 4) 해석 기준

- **0~40점**: 낮음
- **40~70점**: 주의
- **70~100점**: 높음

**오차 수준**

- **우수**: 예측값과 실제값의 차이가 작음
- **보통**: 예측값과 실제값의 차이가 중간 수준
- **나쁨**: 예측값과 실제값의 차이가 큼

</div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 11. 사이드바
# =========================================================

quarter_options = sorted(
    set(actual_df["quarter"].dropna().tolist()) | set(forecast_df["target_quarter"].dropna().tolist()),
    key=quarter_sort_key,
    reverse=True,
)

if not quarter_options:
    st.error("선택 가능한 분기가 없습니다.")
    st.stop()

st.sidebar.markdown("## 생활물가 선행경보")

selected_quarter = st.sidebar.selectbox(
    "해당 분기",
    options=quarter_options,
    index=0,
    format_func=format_quarter_label,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 선행압력 파라미터")

lambda_pressure = st.sidebar.slider(
    "예측 반영 정도 λ",
    min_value=0.00,
    max_value=1.00,
    value=0.30,
    step=0.01,
    key="lambda_pressure_slider",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 선행압력 경로 가중치")

regional_ratio = st.sidebar.slider(
    "지역 물가 흐름 ↔ 거시 금융환경",
    min_value=0.00,
    max_value=1.00,
    value=0.60,
    step=0.01,
    key="regional_weight_slider",
)

st.sidebar.markdown(
    f"""
    <div style="display:flex; justify-content:space-between; 
                margin-top:-0.3rem; font-size:0.88rem; font-weight:600;">
        <span style="color:#2563eb;">지역물가<br>
            <span style="font-size:1.05rem;">{regional_ratio:.0%}</span>
        </span>
        <span style="color:#d97706; text-align:right;">거시금융<br>
            <span style="font-size:1.05rem;">{1 - regional_ratio:.0%}</span>
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

weights = {"regional": regional_ratio, "macro": 1 - regional_ratio}


# =========================================================
# 12. 메인 데이터 준비
# =========================================================

display_df, status, forecast_calc = build_display(selected_quarter, lambda_pressure, weights)

if display_df.empty:
    st.warning("해당 분기의 데이터가 없습니다.")
    st.stop()

regions = sorted(display_df["region_short"].dropna().unique().tolist(), key=region_sort_key)

if "selected_region" not in st.session_state or st.session_state["selected_region"] not in regions:
    st.session_state["selected_region"] = regions[0]


# =========================================================
# 13. 헤더
# =========================================================

st.title("생활물가 선행경보 대시보드")

if status == "forecast_only":
    st.markdown(
        '<div class="small-note">해당 분기는 실제값이 없어 예측값 중심으로 표시됩니다.</div>',
        unsafe_allow_html=True,
    )
elif status == "actual_with_prediction":
    st.markdown(
        '<div class="small-note">해당 분기는 실제값과 예측값을 함께 비교할 수 있습니다.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="small-note">해당 분기는 실제값 기준으로 표시됩니다.</div>',
        unsafe_allow_html=True,
    )

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("해당 분기", format_quarter_label(selected_quarter))

with c2:
    if status == "forecast_only":
        st.metric("평균 예측 위험점수", safe_num(display_df["scenario_forecast_score"].mean()))
    else:
        st.metric("평균 실제 위험점수", safe_num(display_df["actual_target_risk_score"].mean()))

with c3:
    if "scenario_pressure_index" in display_df.columns and display_df["scenario_pressure_index"].notna().any():
        st.metric("평균 조정 선행압력", safe_num(display_df["scenario_pressure_index"].mean()))
    else:
        st.metric("평균 조정 선행압력", "-")

with c4:
    if status == "forecast_only":
        rising = int((display_df["scenario_delta"] > 0).sum()) if "scenario_delta" in display_df.columns else 0
        st.metric("위험 상승 지역 수", safe_int(rising))

    elif status == "actual_with_prediction":
        st.metric(
            "지역별 오차 수준",
            error_level_summary(display_df),
        )

    else:
        high = int((display_df["actual_target_risk_score"] >= 70).sum())
        st.metric("고위험 지역 수", safe_int(high))

# =========================================================
# 14. 탭
# =========================================================

tab_map, tab_rank, tab_detail, tab_guide = st.tabs(
    ["🗺️ 지도", "📊 순위", "🔍 지역 상세", "📘 지표 설명"]
)


# =========================================================
# 15. 지도 탭 (개선됨)
# =========================================================

# ── session_state 초기화 (탭 밖에서 미리) ──────────────────
if "selected_region" not in st.session_state:
    st.session_state["selected_region"] = regions[0]
if st.session_state["selected_region"] not in regions:
    st.session_state["selected_region"] = regions[0]
if "_prev_click" not in st.session_state:
    st.session_state["_prev_click"] = None   # (lat, lng) 튜플 또는 None


def find_region_by_point(lat: float, lng: float) -> Optional[str]:
    """
    클릭 좌표 → 지역명 변환.
    GeoJSON의 'region' 필드를 읽어 normalize_region_name으로 정규화.
    """
    if shape is None or Point is None or raw_geo is None:
        return None
    pt = Point(lng, lat)
    for feat in raw_geo.get("features", []):
        props = feat.get("properties", {})
        # region_short 가 있으면 우선, 없으면 region 필드를 정규화해서 사용
        raw_name = props.get("region_short") or props.get("region") or ""
        rname = normalize_region_name(raw_name)
        if not rname:
            continue
        try:
            geom = shape(feat["geometry"]).buffer(0)
            if geom.contains(pt):
                return rname
        except Exception:
            continue
    return None


with tab_map:

    @st.fragment
    def _map_tab():
        st.subheader("지역별 위험 분포")

        map_col, side_col = st.columns([3.3, 1.7], gap="large")

        with map_col:
            if raw_geo is None:
                st.info("GeoJSON 파일이 없습니다.")
            else:
                fig = build_map(
                    display_df, status, st.session_state["selected_region"]
                )

                if fig is not None:
                    event = st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="choropleth_map",
                        on_select="rerun",
                        selection_mode="points",
                    )

                    # 클릭 이벤트에서 지역명 추출
                    points = (
                        event.selection.get("points", [])
                        if event and hasattr(event, "selection") and event.selection
                        else []
                    )
                    if points:
                        loc = points[0].get("location") or points[0].get("hovertext") or ""
                        rname = normalize_region_name(str(loc))
                        if rname in regions and rname != st.session_state["selected_region"]:
                            st.session_state["selected_region"] = rname
                            st.rerun(scope="fragment")

        with side_col:
            st.markdown("### 지역 선택")

            st.session_state["map_region_selectbox"] = st.session_state["selected_region"]
            sel_box = st.selectbox(
                "목록에서 선택:",
                options=regions,
                key="map_region_selectbox",
                label_visibility="collapsed",
            )

            if sel_box != st.session_state["selected_region"]:
                st.session_state["selected_region"] = sel_box
                st.rerun(scope="fragment")

            current_region = st.session_state["selected_region"]
            row = display_df[display_df["region_short"] == current_region].iloc[0]
            st.markdown(f"### {current_region}")

            if status == "forecast_only":
                a, b = st.columns(2)
                a.metric("예측 점수", safe_num(row.get("scenario_forecast_score")))
                b.metric("이전 점수", safe_num(row.get("current_living_price_risk_score")))
                c, d = st.columns(2)
                c.metric("변화폭", safe_num(row.get("scenario_delta")))
                d.metric("선행압력", safe_num(row.get("scenario_pressure_index")))

            elif status == "actual_with_prediction":
                a, b = st.columns(2)
                a.metric("실제 점수", safe_num(row.get("actual_target_risk_score")))
                b.metric("예측 점수", safe_num(row.get("scenario_forecast_score")))
                c, d = st.columns(2)
                c.metric("예측 차이", safe_num(row.get("prediction_error")))
                d.metric("오차 수준", clean_str(row.get("prediction_error_level")))

            else:
                a, b = st.columns(2)
                a.metric("실제 점수", safe_num(row.get("actual_target_risk_score")))
                b.metric("순위", safe_int(row.get("actual_target_rank")))
                c, d = st.columns(2)
                c.metric("백분위", safe_num(row.get("actual_target_percentile"), 1))
                d.metric("분기", format_quarter_label(selected_quarter))

            st.caption("🎯 지도에서 지역을 클릭하거나, 목록에서 지역을 선택하세요.")

    _map_tab()

# =========================================================
# 16. 순위 탭
# =========================================================

with tab_rank:
    st.subheader("지역 순위")

    rank_fig = build_rank_bar(display_df, status)
    if rank_fig is not None:
        st.plotly_chart(
            rank_fig,
            use_container_width=True,
            config=fixed_chart_config(),
        )

    if status == "forecast_only":
        table_cols = [
            "region_short",
            "scenario_forecast_score",
            "scenario_rank",
            "current_living_price_risk_score",
            "base_pressure_index",
            "scenario_pressure_index",
            "scenario_delta",
        ]
        rename = {
            "region_short": "지역",
            "scenario_forecast_score": "예측 위험점수",
            "scenario_rank": "예측 순위",
            "current_living_price_risk_score": "이전 점수",
            "base_pressure_index": "기본 선행압력",
            "scenario_pressure_index": "조정 선행압력",
            "scenario_delta": "변화폭",
        }

    elif status == "actual_with_prediction":
        table_cols = [
            "region_short",
            "actual_target_risk_score",
            "scenario_forecast_score",
            "prediction_error",
            "abs_prediction_error",
            "prediction_error_level",
            "scenario_pressure_index",
        ]
        rename = {
            "region_short": "지역",
            "actual_target_risk_score": "실제 위험점수",
            "scenario_forecast_score": "예측 위험점수",
            "prediction_error": "예측-실제 차이",
            "abs_prediction_error": "예측 오차",
            "prediction_error_level": "오차 수준",
            "scenario_pressure_index": "조정 선행압력",
        }

    else:
        table_cols = [
            "region_short",
            "actual_target_risk_score",
            "actual_target_rank",
            "actual_target_percentile",
        ]
        rename = {
            "region_short": "지역",
            "actual_target_risk_score": "실제 위험점수",
            "actual_target_rank": "실제 순위",
            "actual_target_percentile": "실제 백분위",
        }

    table_cols = [c for c in table_cols if c in display_df.columns]
    table_df = display_df[table_cols].rename(columns=rename).copy()

    for c in table_df.columns:
        if c not in ["지역", "오차 수준"]:
            table_df[c] = pd.to_numeric(table_df[c], errors="coerce")
            if pd.api.types.is_numeric_dtype(table_df[c]):
                table_df[c] = table_df[c].round(2)

    st.dataframe(table_df, use_container_width=True, hide_index=True)

    if status == "actual_with_prediction":
        st.caption("오차 수준은 현재 선택한 λ와 가중치 기준으로 다시 계산됩니다. 우수는 오차가 작고, 낮음은 오차가 큰 지역입니다.")
    else:
        st.caption("위험점수는 0~100 범위입니다. 값이 높을수록 생활물가 부담 위험이 큰 지역입니다.")


# =========================================================
# 17. 지역 상세 탭
# =========================================================

with tab_detail:
    st.subheader("지역 상세")

    selected_region = st.selectbox(
        "지역 선택",
        options=regions,
        index=regions.index(st.session_state["selected_region"]),
        key="detail_region",
    )
    st.session_state["selected_region"] = selected_region

    row = display_df[display_df["region_short"] == selected_region].iloc[0]

    if status == "forecast_only":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("예측 위험점수", safe_num(row.get("scenario_forecast_score")))
        c2.metric("이전 분기 위험점수", safe_num(row.get("current_living_price_risk_score")))
        c3.metric("조정 선행압력", safe_num(row.get("scenario_pressure_index")))
        c4.metric("변화폭", safe_num(row.get("scenario_delta")))

    elif status == "actual_with_prediction":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("실제 위험점수", safe_num(row.get("actual_target_risk_score")))
        c2.metric("예측 위험점수", safe_num(row.get("scenario_forecast_score")))
        c3.metric("예측-실제 차이", safe_num(row.get("prediction_error")))
        c4.metric("오차 수준", clean_str(row.get("prediction_error_level")))

    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("실제 위험점수", safe_num(row.get("actual_target_risk_score")))
        c2.metric("실제 순위", safe_int(row.get("actual_target_rank")))
        c3.metric("실제 백분위", safe_num(row.get("actual_target_percentile"), 1))
        c4.metric("해당 분기", format_quarter_label(selected_quarter))

    st.markdown("---")

    window = st.radio("기간", ["1년", "5년", "전체"], horizontal=True)

    st.caption("파란선은 실제 위험점수 추이입니다. 빨간 점은 해당 분기의 예측점입니다. 하단 슬라이더로 구간을 이동할 수 있습니다.")

    tb = trend_base(selected_region, forecast_calc)
    tv = filter_window(tb, selected_quarter, window)
    trend_fig = build_trend_chart(tv, selected_quarter, selected_region)

    if trend_fig is not None:
        st.plotly_chart(
            trend_fig,
            use_container_width=True,
            config=chart_config(show_modebar=True),
        )
    else:
        st.info("추이 그래프를 표시할 데이터가 없습니다.")

    st.markdown("---")

    contrib_fig, contrib_table = build_contribution_chart(selected_region, selected_quarter, weights, lambda_pressure)

    if contrib_fig is not None:
        st.plotly_chart(
            contrib_fig,
            use_container_width=True,
            config=fixed_chart_config(),
        )
        if contrib_table is not None:
            st.dataframe(contrib_table, use_container_width=True, hide_index=True)
    else:
        st.info("해당 지역·분기의 변수별 기여도 데이터가 없습니다.")


# =========================================================
# 18. 지표 설명 탭
# =========================================================

with tab_guide:
    render_guide()