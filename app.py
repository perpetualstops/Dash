import math
from functools import lru_cache
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
import calendar

import requests
import pandas as pd
import numpy as np

import yfinance as yf
import datetime as dt
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html as st_html

try:
    from curl_cffi import requests as curl_requests
    HAVE_CURL_CFFI = True
except ImportError:
    curl_requests = None
    HAVE_CURL_CFFI = False

# -----------------------------------------------------------------------------
# Backend: config
# -----------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "macro-constraints-backend/1.0 (+https://localhost)",
        "Accept": "application/json",
    }
)

FRED_API_KEY = "7a1c5818d73952bce4995758997574ce"

WB_INDICATORS: Dict[str, str] = {
    "cpi_yoy": "FP.CPI.TOTL.ZG",
    "debt_gdp": "GC.DOD.TOTL.GD.ZS",
    "fiscal_balance": "GC.BAL.CASH.GD.ZS",
    "real_gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "private_credit_gdp": "FS.AST.PRVT.GD.ZS",
}

FRED_SERIES_POLICY: Dict[str, str] = {
    "United States": "FEDFUNDS",
    "Japan": "IRSTCB01JPM156N",
    "Euro Area": "ECBMRRFR",
}

WB_COUNTRY: Dict[str, str] = {
    "United States": "USA",
    "Japan": "JPN",
    "Euro Area": "EMU",
}

RECESSIONS: Dict[str, list] = {
    "United States": [
        {"start": 1990, "end": 1991},
        {"start": 2001, "end": 2001},
        {"start": 2008, "end": 2009},
        {"start": 2020, "end": 2020},
    ],
    "Japan": [
        {"start": 1992, "end": 1994},
        {"start": 1997, "end": 1998},
        {"start": 2008, "end": 2009},
        {"start": 2020, "end": 2020},
    ],
    "Euro Area": [
        {"start": 2008, "end": 2009},
        {"start": 2011, "end": 2012},
        {"start": 2020, "end": 2020},
    ],
}

NEUTRAL_REAL_RATE: Dict[str, float] = {
    "United States": 1.0,
    "Japan": 0.0,
    "Euro Area": 0.5,
}

FRED_HF_SERIES: Dict[str, Dict[str, str]] = {
    "United States": {
        "cpi": "CPIAUCSL",
        "policy_rate": "FEDFUNDS",
        "unemployment": "UNRATE",
        "wages": "CES0500000003",
        "reer": "",
        "current_account": "",
        "yield_10y": "DGS10",
        "real_yield_10y": "DFII10",
    },
    "Japan": {
        "cpi": "JPNCPIALLMINMEI",
        "policy_rate": "IRSTCB01JPM156N",
        "unemployment": "LRUNTTTJJPM156S",
        "wages": "",
        "reer": "",
        "current_account": "",
        "yield_10y": "IRLTLT01JPM156N",
        "real_yield_10y": "",
    },
    "Euro Area": {
        "cpi": "CP0000EZ19M086NEST",
        "policy_rate": "ECBMRRFR",
        "unemployment": "UNRATE",
        "wages": "",
        "reer": "",
        "current_account": "",
        "yield_10y": "IRLTLT01EZM156N",
        "real_yield_10y": "",
    },
}

MAX_WB_WORKERS: int = 3
MAX_FRED_HF_WORKERS: int = 4

# -----------------------------------------------------------------------------
# Backend: core fetchers
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def fetch_worldbank_series(
    wb_country: str,
    indicator: str,
    start_year: int = 1990,
    end_year: int = 2100,
) -> pd.DataFrame:
    url = (
        f"https://api.worldbank.org/v2/country/{wb_country}/indicator/{indicator}"
        f"?format=json&per_page=2000&date={start_year}:{end_year}"
    )
    r = SESSION.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2 or not data[1]:
        return pd.DataFrame(columns=["year", "value"])

    rows = []
    for item in data[1]:
        year_str = item.get("date")
        val = item.get("value")
        if year_str is None:
            continue
        try:
            year = int(year_str)
        except ValueError:
            continue
        if val is None:
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        rows.append({"year": year, "value": v})

    if not rows:
        return pd.DataFrame(columns=["year", "value"])

    df = pd.DataFrame(rows).sort_values("year")
    return df


@lru_cache(maxsize=None)
def fetch_fred_series(series_id: str, start_date: str = "1990-01-01") -> pd.DataFrame:
    if not series_id:
        return pd.DataFrame(columns=["date", "value"])

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    obs = js.get("observations", [])
    rows = []
    for o in obs:
        d = o.get("date")
        val = o.get("value")
        if val in (".", None, ""):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        rows.append({"date": d, "value": v})

    if not rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def safe_fetch_fred_series(series_id: str, start_date: str = "1990-01-01") -> pd.DataFrame:
    if not series_id:
        return pd.DataFrame(columns=["date", "value"])
    try:
        return fetch_fred_series(series_id, start_date=start_date)
    except Exception:
        return pd.DataFrame(columns=["date", "value"])


def to_annual(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", "value"])
    df = df.copy()
    df["year"] = df["date"].dt.year
    if method == "last":
        ann = df.sort_values("date").groupby("year").tail(1)
        ann = ann[["year", "value"]]
    else:
        ann = df.groupby("year", as_index=False)["value"].mean()
    return ann.sort_values("year")


def _fetch_worldbank_bulk(wb_code: str) -> Dict[str, pd.DataFrame]:
    def worker(name: str, ind: str) -> tuple[str, pd.DataFrame]:
        df = fetch_worldbank_series(wb_code, ind, start_year=1990, end_year=2100)
        df = df.rename(columns={"value": name})
        return name, df

    items = list(WB_INDICATORS.items())
    results: Dict[str, pd.DataFrame] = {}

    if not items:
        return results

    max_workers = min(len(items), MAX_WB_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, name, ind): name for name, ind in items}
        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                key, df = fut.result()
                results[key] = df
            except Exception:
                results[name] = pd.DataFrame(columns=["year", name])

    for name in WB_INDICATORS.keys():
        results.setdefault(name, pd.DataFrame(columns=["year", name]))

    return results


def _fetch_highfreq_bulk(mapping: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    def worker(name: str, sid: str) -> tuple[str, pd.DataFrame]:
        if not sid:
            return name, pd.DataFrame(columns=["date", "value"])
        df = safe_fetch_fred_series(sid, start_date="1990-01-01")
        if df.empty:
            return name, pd.DataFrame(columns=["date", "value"])
        return name, df[["date", "value"]].copy()

    results: Dict[str, pd.DataFrame] = {}

    items = list(mapping.items())
    if not items:
        return {name: pd.DataFrame(columns=["date", "value"]) for name in mapping.keys()}

    max_workers = min(len(items), MAX_FRED_HF_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(worker, name, sid): name for name, sid in items}
        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                key, df = fut.result()
                results[key] = df
            except Exception:
                results[name] = pd.DataFrame(columns=["date", "value"])

    for name in mapping.keys():
        results.setdefault(name, pd.DataFrame(columns=["date", "value"]))

    return results

# -----------------------------------------------------------------------------
# Backend: panel + high-frequency bundle + calendar
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def build_country_panel(country: str) -> pd.DataFrame:
    if country not in WB_COUNTRY:
        raise ValueError(f"Unsupported country: {country}")

    wb_code = WB_COUNTRY[country]
    series_dfs: Dict[str, pd.DataFrame] = _fetch_worldbank_bulk(wb_code)

    policy_id = FRED_SERIES_POLICY.get(country, "")
    if policy_id:
        pol = safe_fetch_fred_series(policy_id, start_date="1990-01-01")
        pol_ann = to_annual(pol, method="mean").rename(columns={"value": "policy_rate"})
    else:
        pol_ann = pd.DataFrame(columns=["year", "policy_rate"])

    years: set[int] = set()
    for df in series_dfs.values():
        if "year" in df.columns:
            years.update(df["year"].tolist())
    years.update(pol_ann["year"].tolist())

    if not years:
        return pd.DataFrame(columns=["year"])

    panel = pd.DataFrame({"year": sorted(y for y in years if y >= 1990)})

    for name, df in series_dfs.items():
        if "year" in df.columns:
            panel = panel.merge(df[["year", name]], on="year", how="left")

    for name in WB_INDICATORS.keys():
        if name not in panel.columns:
            panel[name] = np.nan

    panel = panel.merge(pol_ann, on="year", how="left")

    infl_col = "cpi_yoy"
    if infl_col in panel.columns:
        panel["real_policy_rate"] = panel["policy_rate"] - panel[infl_col]
    else:
        panel["real_policy_rate"] = np.nan

    neutral = NEUTRAL_REAL_RATE.get(country, 0.0)
    panel["neutral_real_rate"] = neutral
    panel["real_rate_minus_neutral"] = panel["real_policy_rate"] - panel["neutral_real_rate"]

    if "real_gdp_growth" in panel.columns:
        panel["growth_minus_real_rate"] = panel["real_gdp_growth"] - panel["real_policy_rate"]
    else:
        panel["growth_minus_real_rate"] = np.nan

    if "debt_gdp" in panel.columns and "private_credit_gdp" in panel.columns:
        panel["total_borrowing_gdp"] = panel["debt_gdp"] + panel["private_credit_gdp"]
    else:
        panel["total_borrowing_gdp"] = np.nan

    panel["inflation_target"] = 2.0

    for col in panel.columns:
        if col == "year":
            continue
        panel[col] = pd.to_numeric(panel[col], errors="ignore")
    panel.replace([np.inf, -np.inf], np.nan, inplace=True)

    metric_cols = [c for c in panel.columns if c != "year"]
    panel = panel.dropna(how="all", subset=metric_cols)

    return panel.sort_values("year").reset_index(drop=True)


@lru_cache(maxsize=None)
def build_highfreq_bundle(country: str) -> Dict[str, pd.DataFrame]:
    if country not in FRED_HF_SERIES:
        raise ValueError(f"Unsupported country: {country}")
    mapping = FRED_HF_SERIES[country]
    return _fetch_highfreq_bulk(mapping)


def build_calendar(country: str) -> Dict:
    if country not in WB_COUNTRY:
        raise ValueError(f"Unsupported country: {country}")

    today = date.today()
    year = today.year

    if country == "Japan":
        events = [
            {"date": f"{year}-01-31", "event": "BoJ Policy Meeting", "importance": "high"},
            {"date": f"{year}-04-30", "event": "BoJ Outlook Report", "importance": "medium"},
            {"date": f"{year}-07-31", "event": "BoJ Policy Meeting", "importance": "high"},
            {"date": f"{year}-10-31", "event": "BoJ Outlook Report", "importance": "medium"},
        ]
    elif country == "United States":
        events = [
            {"date": f"{year}-03-20", "event": "FOMC Meeting", "importance": "high"},
            {"date": f"{year}-06-19", "event": "FOMC Meeting", "importance": "high"},
            {"date": f"{year}-09-18", "event": "FOMC Meeting", "importance": "high"},
            {"date": f"{year}-12-18", "event": "FOMC Meeting", "importance": "high"},
        ]
    else:
        events = [
            {"date": f"{year}-03-15", "event": "ECB Governing Council", "importance": "high"},
            {"date": f"{year}-06-14", "event": "ECB Governing Council", "importance": "high"},
            {"date": f"{year}-09-13", "event": "ECB Governing Council", "importance": "high"},
            {"date": f"{year}-12-13", "event": "ECB Governing Council", "importance": "high"},
        ]

    return {"country": country, "events": events}

# -----------------------------------------------------------------------------
# Streamlit: global config
# -----------------------------------------------------------------------------

PRIMARY_COLOR = "#8B0000"
SECONDARY_COLOR = "#E9A7A7"
AXIS_BOX_COLOR = "#777777"
GRID_COLOR = "#E5E5E5"
RECESSION_SHADE = "#C8C8C8"

COUNTRIES = ["United States", "Japan", "Euro Area"]

METRIC_LABELS = {
    "cpi": "CPI",
    "policy_rate": "Policy rate",
    "unemployment": "Unemployment rate",
    "wages": "Wage index / earnings",
    "reer": "REER proxy",
    "current_account": "Current account",
    "yield_10y": "10Y nominal yield",
    "real_yield_10y": "10Y real yield",
}

COUNTRY_COLORS = {
    "United States": PRIMARY_COLOR,
    "Japan": "#555555",
    "Euro Area": SECONDARY_COLOR,
}

st.set_page_config(
    page_title="ZY's Alpha Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    body {{ font-family: "Helvetica Neue", Arial, sans-serif; }}

    .main-title {{
        font-size: 2.1rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.3rem;
    }}
    .section-title {{
        font-size: 1.4rem;
        font-weight: 700;
        color: #000000;
        margin: 0.3rem 0 0.7rem 0;
    }}
    .chart-heading {{
        font-size: 1.0rem;
        font-weight: 600;
        color: {PRIMARY_COLOR};
        margin-bottom: 0.15rem;
    }}
    .chart-comment {{
        font-size: 0.75rem;
        color: #777777;
        margin-top: 0.0rem;
        margin-bottom: 0.1rem;
    }}

    div[data-testid="stPlotlyChart"] {{
        margin-bottom: 0.02rem;
        background-color: #ffffff;
    }}

    div[data-testid="column"] {{
        padding-right: 0.3rem;
        padding-left: 0.3rem;
    }}

    .chart-placeholder {{
        border: 0px solid transparent;
        border-radius: 12px;
        padding: 0.05rem;
        background-color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 260px;
        font-size: 0.9rem;
        color: {PRIMARY_COLOR};
        text-align: center;
        margin-bottom: 0.6rem;
    }}

    div[data-baseweb="select"] span[data-baseweb="tag"] {{
        background-color: {PRIMARY_COLOR} !important;
        color: #ffffff !important;
        border-radius: 999px !important;
    }}
    div[data-baseweb="select"] span[data-baseweb="tag"] svg {{
        fill: #ffffff !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Streamlit: shared helpers
# -----------------------------------------------------------------------------

def get_yf_session():
    if HAVE_CURL_CFFI and curl_requests is not None:
        return curl_requests.Session(impersonate="chrome")
    return None


def series_to_df(series_json):
    df = pd.DataFrame(series_json)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_yoy(df: pd.DataFrame, col: str = "value", periods: int = 12) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["yoy"] = df[col].pct_change(periods=periods) * 100.0
    return df


def style_figure(fig, height=260, legend=False):
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=30, t=25, b=5),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(size=10),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=AXIS_BOX_COLOR,
        mirror=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(size=11),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=AXIS_BOX_COLOR,
        mirror=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        title_font=dict(size=11),
    )
    if legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.0,
                font=dict(size=10),
            )
        )
    else:
        fig.update_layout(showlegend=False)
    return fig


def metric_axis_label(metric: str, mode: str) -> str:
    if mode.startswith("YoY"):
        return "YoY % change"
    if metric in {
        "policy_rate",
        "unemployment",
        "wages",
        "yield_10y",
        "real_yield_10y",
        "cpi",
    }:
        return "Percent / index"
    if metric == "current_account":
        return "Level / % of GDP"
    return "Index / level"


def z_score(series: pd.Series, dates: pd.Series, mode: str, window_months: int | None) -> pd.Series:
    if series.empty:
        return series

    if mode == "Rolling" and (window_months is None or window_months <= 0):
        mode = "Full sample"

    s = series.astype("float64").copy()
    dts = pd.to_datetime(dates)

    if mode == "Rolling" and window_months is not None:
        result = pd.Series(index=s.index, dtype="float64")
        for i in range(len(s)):
            v = s.iloc[i]
            d = dts.iloc[i]
            if pd.isna(v) or pd.isna(d):
                result.iloc[i] = pd.NA
                continue
            start = d - pd.DateOffset(months=window_months)
            mask = (dts >= start) & (dts <= d) & (~s.isna())
            window_vals = s[mask]
            if window_vals.empty:
                result.iloc[i] = pd.NA
                continue
            mu = window_vals.mean()
            sigma = window_vals.std()
            if sigma is None or sigma == 0:
                result.iloc[i] = 0.0
            else:
                result.iloc[i] = (v - mu) / sigma
        return result

    vals = s.dropna()
    if vals.empty:
        return s * 0.0
    mu = vals.mean()
    sigma = vals.std()
    if sigma is None or sigma == 0:
        return s * 0.0
    return (s - mu) / sigma


def align_to_monthly(df_metric: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp, max_ffill_months: int = 6) -> pd.DataFrame:
    if df_metric.empty:
        return df_metric

    s = df_metric.copy()
    s = s.set_index("date").sort_index()

    idx = pd.date_range(start=start_date, end=end_date, freq="MS")
    s = s.reindex(idx)
    s["y"] = s["y"].ffill(limit=max_ffill_months)
    s = s.dropna(subset=["y"])
    s = s.reset_index().rename(columns={"index": "date"})
    return s


def add_recession_bands(fig, rec_ranges, window_start=None, window_end=None):
    if not rec_ranges:
        return fig

    ws = pd.to_datetime(window_start) if window_start is not None else None
    we = pd.to_datetime(window_end) if window_end is not None else None

    for (start, end) in rec_ranges:
        s = start
        e = end
        if ws is not None and s < ws:
            s = ws
        if we is not None and e > we:
            e = we
        if e is None or s is None or e <= s:
            continue
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=RECESSION_SHADE,
            opacity=0.18,
            line_width=0,
            layer="below",
        )
    return fig


def choose_recession_country(selected_countries):
    if "United States" in selected_countries:
        return "United States"
    return selected_countries[0]

# -----------------------------------------------------------------------------
# Streamlit: backend wrappers (no HTTP)
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=True)
def fetch_constraints_panel(country: str):
    panel_df = build_country_panel(country)
    return {
        "country": country,
        "panel": panel_df.to_dict(orient="records"),
        "recessions": RECESSIONS.get(country, []),
    }


@st.cache_data(ttl=600, show_spinner=True)
def fetch_highfreq(country: str):
    series_out = build_highfreq_bundle(country)
    out = {}
    for name, df in series_out.items():
        out[name] = df.to_dict(orient="records")
    return {
        "country": country,
        "series": out,
        "recessions": RECESSIONS.get(country, []),
    }


@st.cache_data(ttl=600, show_spinner=True)
def fetch_all_calendars():
    rows = []
    for c in COUNTRIES:
        try:
            js = build_calendar(c)
            cc = js.get("country", c)
            for ev in js.get("events", []):
                rows.append(
                    {
                        "date": ev.get("date"),
                        "event": ev.get("event"),
                        "importance": ev.get("importance", "").lower(),
                        "country": cc,
                    }
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "event", "importance", "country", "weekday"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["weekday"] = df["date"].dt.strftime("%a")
    return df.sort_values("date", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def get_recession_ranges_for_country(country: str):
    recs = RECESSIONS.get(country, [])
    ranges = []
    for r in recs:
        y0 = r.get("start")
        y1 = r.get("end")
        if y0 is None or y1 is None:
            continue
        try:
            start = pd.Timestamp(int(y0), 1, 1)
            end = pd.Timestamp(int(y1) + 1, 1, 1)
            ranges.append((start, end))
        except Exception:
            continue
    return ranges

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("Macro dashboard")
country = st.sidebar.selectbox("Focus economy", COUNTRIES, index=0)

st.sidebar.markdown(
    """
**A. Policy & growth constraints**  
Annual constraints for policy, leverage and growth.

**B. High-frequency macro**  
Monthly / higher-frequency series for inflation, labour and rates.
"""
)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_dash, tab_analytics, tab_play, tab_strategy, tab_data = st.tabs(["Dashboard", "Analytics", "Playground", "Strategy", "Data"])

# -----------------------------------------------------------------------------
# DASHBOARD
# -----------------------------------------------------------------------------

with tab_dash:
    st.header("ZYs Alpha Engine")

    st.markdown(
        f'''
        <div class="chart-heading" style="font-size:2.0rem;">
            {country}
        </div>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="font-size:20px; color:#555555; margin-top:0rem; margin-bottom:0.6rem;">
            Macro constraints: policy, balance sheets and growth
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        constraints = fetch_constraints_panel(country)
        panel = pd.DataFrame(constraints.get("panel", []))
        recessions = constraints.get("recessions", [])
    except Exception as e:
        st.error(f"Error fetching constraints panel: {e}")
        st.stop()

    try:
        hf_data = fetch_highfreq(country)
    except Exception as e:
        st.error(f"Error fetching high-frequency data: {e}")
        st.stop()

    cal_df = fetch_all_calendars()

    col_cal, gap_col, col_main = st.columns([0.9, 0.1, 3.1])

    with col_cal:
        st.markdown('<div class="section-title">Economic calendar</div>', unsafe_allow_html=True)

        if cal_df.empty:
            st.info("No calendar data available.")
        else:
            st.markdown('<div class="chart-heading">Economy</div>', unsafe_allow_html=True)

            cal_country = st.selectbox(
                "",
                ["All"] + COUNTRIES,
                index=0,
                key="cal_country",
                label_visibility="collapsed",
            )

            cal_view = cal_df.copy()
            if cal_country != "All":
                cal_view = cal_view[cal_view["country"] == cal_country]

            if cal_view.empty:
                st_html(
                    """
                    <div style="
                        border-radius: 12px;
                        border: 0px solid #B0B0B0;
                        background-color: #ffffff;
                        padding: 0.7rem 0.105rem;
                        height: 650px;
                        overflow-y: auto;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        font-size: 0.85rem;
                        color: #777777;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">
                        No events for this filter.
                    </div>
                    """,
                    height=650,
                    scrolling=False,
                )
            else:
                rows_html = ""
                for _, row in cal_view.iterrows():
                    date_str = row["date"].strftime("%d %b %Y")
                    weekday = row["weekday"]
                    event = row["event"]
                    cty = row["country"]
                    rows_html += f"""
                    <div class="calendar-row">
                        <div>
                            <div class="cal-date">{date_str}</div>
                            <div class="cal-weekday">{weekday}</div>
                        </div>
                        <div>
                            <div class="cal-event">{event}</div>
                            <div class="cal-country">{cty}</div>
                        </div>
                        <div style="text-align:right;">
                            <span class="cal-tag">High</span>
                        </div>
                    </div>
                    """

                calendar_html = f"""
                <html>
                <head>
                    <style>
                        body {{
                            margin: 0;
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        }}
                        .calendar-container {{
                            border-radius: 12px;
                            border: 0px solid #B0B0B0;
                            background-color: #ffffff;
                            padding: 0.7rem 0.205rem;
                            height: 650px;
                            overflow-y: auto;
                        }}
                        .calendar-row {{
                            display: grid;
                            grid-template-columns: 0.9fr 1.8fr 0.9fr;
                            column-gap: 0.45rem;
                            align-items: center;
                            padding: 0.35rem 0.35rem;
                            border-radius: 10px;
                            margin-bottom: 0.5rem;
                        }}
                        .calendar-row:nth-child(odd) {{
                            background-color: #F9F9F9;
                        }}
                        .cal-date {{
                            font-weight: 600;
                            font-size: 0.82rem;
                            color: #333333;
                        }}
                        .cal-weekday {{
                            font-size: 0.75rem;
                            color: #777777;
                        }}
                        .cal-event {{
                            font-size: 0.85rem;
                            color: #222222;
                        }}
                        .cal-country {{
                            font-size: 0.8rem;
                            color: #555555;
                            font-weight: 500;
                        }}
                        .cal-tag {{
                            display: inline-block;
                            padding: 0.2rem 0.6rem;
                            border-radius: 999px;
                            font-size: 0.75rem;
                            font-weight: 600;
                            background-color: #F28E8E;
                            color: #5A0606;
                        }}
                    </style>
                </head>
                <body>
                    <div class="calendar-container">
                        {rows_html}
                    </div>
                </body>
                </html>
                """

                st_html(calendar_html, height=650, scrolling=False)

    with col_main:
        st.markdown('<div class="section-title">A. Policy and growth constraints</div>', unsafe_allow_html=True)

        if panel.empty:
            st.info("No panel data available for this country.")
        else:
            for c in panel.columns:
                if c != "year":
                    panel[c] = pd.to_numeric(panel[c], errors="coerce")

            def add_recessions(fig):
                if not recessions:
                    return fig
                for r in recessions:
                    y0 = r.get("start")
                    y1 = r.get("end")
                    if y0 is None or y1 is None:
                        continue
                    fig.add_vrect(
                        x0=y0,
                        x1=y1,
                        fillcolor=RECESSION_SHADE,
                        opacity=0.45,
                        line_width=0,
                        layer="below",
                    )
                return fig

            r1c1, r1c2, r1c3 = st.columns(3)

            with r1c1:
                st.markdown('<div class="chart-heading">Inflation vs target</div>', unsafe_allow_html=True)
                if "cpi_yoy" in panel.columns:
                    df = panel[["year", "cpi_yoy", "inflation_target"]].dropna(subset=["cpi_yoy"])
                    df_long = df.melt(
                        id_vars="year",
                        value_vars=["cpi_yoy", "inflation_target"],
                        var_name="variable",
                        value_name="value",
                    )
                    color_map = {"cpi_yoy": PRIMARY_COLOR, "inflation_target": SECONDARY_COLOR}
                    fig = px.line(
                        df_long,
                        x="year",
                        y="value",
                        color="variable",
                        color_discrete_map=color_map,
                        labels={"year": "Year", "value": "Percent"},
                    )
                    fig = add_recessions(fig)
                    fig = style_figure(fig, legend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        '<div class="chart-comment">'
                        'Inflation relative to a 2% objective is a clean signal of whether the price regime has been '
                        'persistently too loose or too tight.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="chart-placeholder">Inflation series unavailable.</div>',
                        unsafe_allow_html=True,
                    )

            with r1c2:
                st.markdown('<div class="chart-heading">Public & private leverage</div>', unsafe_allow_html=True)
                if "debt_gdp" in panel.columns and "private_credit_gdp" in panel.columns:
                    df = panel[["year", "debt_gdp", "private_credit_gdp"]].dropna(
                        how="all", subset=["debt_gdp", "private_credit_gdp"]
                    )
                    df_long = df.melt(
                        id_vars="year",
                        value_vars=["debt_gdp", "private_credit_gdp"],
                        var_name="variable",
                        value_name="value",
                    )
                    color_map = {"debt_gdp": PRIMARY_COLOR, "private_credit_gdp": SECONDARY_COLOR}
                    fig = px.line(
                        df_long,
                        x="year",
                        y="value",
                        color="variable",
                        color_discrete_map=color_map,
                        labels={"year": "Year", "value": "% of GDP"},
                    )
                    fig = add_recessions(fig)
                    fig = style_figure(fig, legend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        '<div class="chart-comment">'
                        'Government and private-sector debt-to-GDP together are a proxy for how much the growth model '
                        'leans on balance-sheet expansion rather than productivity.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="chart-placeholder">Debt and credit series unavailable.</div>',
                        unsafe_allow_html=True,
                    )

            with r1c3:
                st.markdown('<div class="chart-heading">Real policy rate vs neutral</div>', unsafe_allow_html=True)
                if "real_policy_rate" in panel.columns and "neutral_real_rate" in panel.columns:
                    df = panel[["year", "real_policy_rate", "neutral_real_rate"]].dropna(
                        how="all", subset=["real_policy_rate"]
                    )
                    df_long = df.melt(
                        id_vars="year",
                        value_vars=["real_policy_rate", "neutral_real_rate"],
                        var_name="variable",
                        value_name="value",
                    )
                    color_map = {"real_policy_rate": PRIMARY_COLOR, "neutral_real_rate": SECONDARY_COLOR}
                    fig = px.line(
                        df_long,
                        x="year",
                        y="value",
                        color="variable",
                        color_discrete_map=color_map,
                        labels={"year": "Year", "value": "Percent"},
                    )
                    fig = add_recessions(fig)
                    fig = style_figure(fig, legend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        '<div class="chart-comment">'
                        'Real policy relative to neutral is a direct signal of whether financial conditions are '
                        'systematically tight or loose through the cycle.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="chart-placeholder">Real policy vs neutral series unavailable.</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)

            r2c1, r2c2, r2c3 = st.columns(3)

            with r2c1:
                st.markdown('<div class="chart-heading">Growth vs real policy rate</div>', unsafe_allow_html=True)
                if "real_gdp_growth" in panel.columns and "real_policy_rate" in panel.columns:
                    df = panel[["year", "real_gdp_growth", "real_policy_rate"]].dropna(
                        how="all", subset=["real_gdp_growth"]
                    )
                    df_long = df.melt(
                        id_vars="year",
                        value_vars=["real_gdp_growth", "real_policy_rate"],
                        var_name="variable",
                        value_name="value",
                    )
                    color_map = {"real_gdp_growth": PRIMARY_COLOR, "real_policy_rate": SECONDARY_COLOR}
                    fig = px.line(
                        df_long,
                        x="year",
                        y="value",
                        color="variable",
                        color_discrete_map=color_map,
                        labels={"year": "Year", "value": "Percent"},
                    )
                    fig = add_recessions(fig)
                    fig = style_figure(fig, legend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        '<div class="chart-comment">'
                        'Real growth relative to the real policy rate is a proxy for whether activity is being '
                        'constrained by policy or by deeper structural forces.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="chart-placeholder">Growth and real-rate series unavailable.</div>',
                        unsafe_allow_html=True,
                    )

            with r2c2:
                st.markdown('<div class="chart-heading">Fiscal balance / GDP</div>', unsafe_allow_html=True)
                if "fiscal_balance" in panel.columns:
                    df = panel[["year", "fiscal_balance"]].dropna(subset=["fiscal_balance"])
                    if df.empty:
                        st.markdown(
                            '<div class="chart-placeholder">Fiscal-balance series unavailable.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        fig = px.line(
                            df,
                            x="year",
                            y="fiscal_balance",
                            labels={"year": "Year", "fiscal_balance": "% of GDP"},
                            color_discrete_sequence=[PRIMARY_COLOR],
                        )
                        fig = add_recessions(fig)
                        fig = style_figure(fig, legend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(
                            '<div class="chart-comment">'
                            'The fiscal balance as a share of GDP is a signal of how far aggregate demand is being '
                            'driven by the public sector versus private spending.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="chart-placeholder">Fiscal-balance series unavailable.</div>',
                        unsafe_allow_html=True,
                    )

            with r2c3:
                st.markdown('<div class="chart-heading">Total borrowing / GDP</div>', unsafe_allow_html=True)
                if "total_borrowing_gdp" in panel.columns:
                    df = panel[["year", "total_borrowing_gdp"]].dropna(subset=["total_borrowing_gdp"])
                    fig = px.line(
                        df,
                        x="year",
                        y="total_borrowing_gdp",
                        labels={"year": "Year", "total_borrowing_gdp": "% of GDP"},
                        color_discrete_sequence=[PRIMARY_COLOR],
                    )
                    fig = add_recessions(fig)
                    fig = style_figure(fig, legend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        '<div class="chart-comment">'
                        'Total public-plus-private debt-to-GDP is a proxy for the system’s sensitivity to shifts in '
                        'funding costs, credit availability, and risk premia.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="chart-placeholder">Total borrowing series unavailable.</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown('<div class="section-title">B. High-frequency macro (FRED)</div>', unsafe_allow_html=True)

    series = hf_data.get("series", {})

    def get_metric_df(name: str):
        s = series.get(name)
        if not s:
            return pd.DataFrame()
        return series_to_df(s)

    df_cpi = get_metric_df("cpi")
    df_policy = get_metric_df("policy_rate")
    df_unemp = get_metric_df("unemployment")
    df_wages = get_metric_df("wages")
    df_reer = get_metric_df("reer")
    df_ca = get_metric_df("current_account")
    df_y10 = get_metric_df("yield_10y")
    df_y10_real = get_metric_df("real_yield_10y")

    b1c1, b1c2 = st.columns(2)
    with b1c1:
        st.markdown('<div class="chart-heading">CPI YoY</div>', unsafe_allow_html=True)
        if not df_cpi.empty:
            df_cpi_yoy = compute_yoy(df_cpi, "value", periods=12)
            fig = px.line(
                df_cpi_yoy,
                x="date",
                y="yoy",
                labels={"date": "Date", "yoy": "%"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig = style_figure(fig, height=260, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="chart-comment">'
                'High-frequency CPI YoY is an early signal of the direction and momentum of underlying inflation '
                'pressure.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">High-frequency inflation series unavailable.</div>',
                unsafe_allow_html=True,
            )

    with b1c2:
        st.markdown('<div class="chart-heading">Policy rate</div>', unsafe_allow_html=True)
        if not df_policy.empty:
            fig = px.line(
                df_policy,
                x="date",
                y="value",
                labels={"date": "Date", "value": "%"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig = style_figure(fig, height=260, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="chart-comment">'
                'The policy-rate path is a real-time signal of how aggressively the central bank has tightened or '
                'eased as conditions evolve.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">Policy-rate series unavailable.</div>',
                unsafe_allow_html=True,
            )

    b2c1, b2c2 = st.columns(2)
    with b2c1:
        st.markdown('<div class="chart-heading">Unemployment rate</div>', unsafe_allow_html=True)
        if not df_unemp.empty:
            fig = px.line(
                df_unemp,
                x="date",
                y="value",
                labels={"date": "Date", "value": "%"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig = style_figure(fig, height=260, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="chart-comment">'
                'The unemployment rate is a summary signal of labour-market slack and the risk of wage and inflation '
                'persistence.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">Unemployment series unavailable.</div>',
                unsafe_allow_html=True,
            )

    with b2c2:
        st.markdown('<div class="chart-heading">Wage growth YoY</div>', unsafe_allow_html=True)
        if not df_wages.empty:
            df_wages_yoy = compute_yoy(df_wages, "value", periods=12)
            fig = px.line(
                df_wages_yoy,
                x="date",
                y="yoy",
                labels={"date": "Date", "yoy": "%"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig = style_figure(fig, height=260, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="chart-comment">'
                'Wage growth YoY is a proxy for the strength of income gains feeding into demand and medium-term '
                'inflation dynamics.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">Wage series unavailable.</div>',
                unsafe_allow_html=True,
            )

    b3c1, b3c2 = st.columns(2)
    with b3c1:
        st.markdown('<div class="chart-heading">REER proxy (YoY)</div>', unsafe_allow_html=True)
        if not df_reer.empty:
            df_reer_yoy = compute_yoy(df_reer, "value", periods=12)
            fig = px.line(
                df_reer_yoy,
                x="date",
                y="yoy",
                labels={"date": "Date", "yoy": "%"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig = style_figure(fig, height=260, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="chart-comment">'
                'Real effective exchange-rate changes are a proxy for shifts in external competitiveness and the '
                'tightness of external financial conditions.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">REER series unavailable.</div>',
                unsafe_allow_html=True,
            )

    with b3c2:
        st.markdown('<div class="chart-heading">Current account</div>', unsafe_allow_html=True)
        if not df_ca.empty:
            fig = px.line(
                df_ca,
                x="date",
                y="value",
                labels={"date": "Date", "value": "Level / % GDP"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig = style_figure(fig, height=260, legend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div class="chart-comment">'
                'The current-account balance is a signal of whether the economy is exporting or importing net savings, '
                'and how exposed it is to external funding swings.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="chart-placeholder">Current-account series unavailable.</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="chart-heading" style="margin-top:0.8rem;">10-year nominal vs real yield</div>', unsafe_allow_html=True)
    if not df_y10.empty or not df_y10_real.empty:
        fig_y = None
        if not df_y10.empty:
            fig_y = px.line(
                df_y10,
                x="date",
                y="value",
                labels={"date": "Date", "value": "%"},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig_y.update_traces(name="10Y nominal", hovertemplate="%{y:.2f}%")

        if not df_y10_real.empty:
            df_y10_real_ = df_y10_real.rename(columns={"value": "value_real"})
            if fig_y is None:
                fig_y = px.line(
                    df_y10_real_,
                    x="date",
                    y="value_real",
                    labels={"date": "Date", "value_real": "%"},
                    color_discrete_sequence=[SECONDARY_COLOR],
                )
                fig_y.update_traces(name="10Y real", hovertemplate="%{y:.2f}%")
            else:
                fig_y.add_scatter(
                    x=df_y10_real_["date"],
                    y=df_y10_real_["value_real"],
                    mode="lines",
                    name="10Y real",
                    line=dict(color=SECONDARY_COLOR),
                )

        fig_y = style_figure(fig_y, height=280, legend=True)
        st.plotly_chart(fig_y, use_container_width=True)
        st.markdown(
            '<div class="chart-comment">'
            'The level of nominal and real 10-year yields is a proxy for the stance of long-horizon financial '
            'conditions and the market’s implied inflation compensation.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="chart-placeholder">10-year nominal / real yield series unavailable.</div>',
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# ANALYTICS
# -----------------------------------------------------------------------------
with tab_analytics:
    import calendar  # local import is fine here

    st.header("Analytics – macro one-pager")

    st.markdown(
        """
        <p style="font-size:13px; color:#555555; margin-top:0rem; margin-bottom:0.55rem;">
        Block-level snapshot of the cycle for the current focus economy, based on rolling,
        out-of-sample z-scores for a compact set of high-frequency indicators.
        </p>
        """,
        unsafe_allow_html=True,
    )

    try:
        constraints = fetch_constraints_panel(country)
        panel = pd.DataFrame(constraints.get("panel", []))
    except Exception as e:
        st.error(f"Error fetching panel data: {e}")
        st.stop()

    try:
        hf_data = fetch_highfreq(country)
        series = hf_data.get("series", {}) or {}
    except Exception as e:
        st.error(f"Error fetching high-frequency data: {e}")
        st.stop()

    if panel.empty and not series:
        st.info("No data available for analytics.")
        st.stop()

    # config per indicator used in analytics
    # source: "hf" (high-frequency) or "panel" (annual)
    # transform: "level" or "yoy"
    # good_when_high: True => high == healthy; False => high == restrictive/negative
    # block: macro block label for heatmap aggregation
    analytics_metrics = {
        "real_gdp_growth": {
            "label": "Real GDP growth (annual)",
            "source": "panel",
            "transform": "level",
            "good_when_high": True,
            "block": "Growth",
        },
        "wages": {
            "label": "Wage growth (YoY)",
            "source": "hf",
            "transform": "yoy",
            "good_when_high": True,
            "block": "Growth",
        },
        "unemployment": {
            "label": "Unemployment rate",
            "source": "hf",
            "transform": "level",
            "good_when_high": False,
            "block": "Labour",
        },
        "cpi": {
            "label": "CPI (YoY)",
            "source": "hf",
            "transform": "yoy",
            "good_when_high": False,
            "block": "Inflation",
        },
        "policy_rate": {
            "label": "Policy rate",
            "source": "hf",
            "transform": "level",
            "good_when_high": False,
            "block": "Policy stance",
        },
        "yield_10y": {
            "label": "10Y nominal yield",
            "source": "hf",
            "transform": "level",
            "good_when_high": False,
            "block": "Financial conditions",
        },
        "real_yield_10y": {
            "label": "10Y real yield",
            "source": "hf",
            "transform": "level",
            "good_when_high": False,
            "block": "Financial conditions",
        },
        "current_account": {
            "label": "Current account",
            "source": "hf",
            "transform": "level",
            "good_when_high": True,
            "block": "External",
        },
        "reer": {
            "label": "REER proxy (YoY)",
            "source": "hf",
            "transform": "yoy",
            "good_when_high": False,
            "block": "External",
        },
    }

    window_months = 120  # 10y rolling for "out-of-sample" normalisation
    max_monthly_mom_lag = 12
    max_annual_mom_lag_years = 3

    metric_rows = []
    block_scores_level = {}
    block_scores_mom = {}

    # helper: small percentile function (no scipy)
    def percentile_rank(vals: pd.Series, latest: float) -> float:
        vals = vals.dropna()
        if vals.empty:
            return np.nan
        return float((vals <= latest).mean())

    for key, cfg in analytics_metrics.items():
        src = cfg["source"]

        if src == "hf":
            raw = series.get(key)
            if not raw:
                continue
            df = series_to_df(raw)
            if df.empty:
                continue
            df = df.sort_values("date")
        else:
            if panel.empty or key not in panel.columns:
                continue
            df = panel[["year", key]].dropna()
            if df.empty:
                continue
            df = df.rename(columns={key: "value"}).copy()
            df["date"] = pd.to_datetime(df["year"].astype(int).astype(str) + "-12-31")
            df = df[["date", "value"]].sort_values("date")

        if cfg["transform"] == "yoy":
            if src == "hf":
                df = compute_yoy(df, "value", periods=12)
                col = "yoy"
            else:
                df["yoy"] = df["value"].pct_change(1) * 100.0
                col = "yoy"
        else:
            col = "value"

        df = df.dropna(subset=[col])
        if df.empty:
            continue

        # level z-score (rolling)
        df["z_level"] = z_score(
            df[col].astype("float64"),
            df["date"],
            mode="Rolling",
            window_months=window_months,
        )

        if df["z_level"].isna().all():
            continue

        # momentum: change over lag; monthly vs annual
        if src == "hf":
            lag = min(max_monthly_mom_lag, len(df) - 1) if len(df) > 1 else 1
        else:
            lag = min(max_annual_mom_lag_years, len(df) - 1) if len(df) > 1 else 1

        df["delta"] = df[col] - df[col].shift(lag)
        df["z_mom"] = z_score(
            df["delta"].astype("float64"),
            df["date"],
            mode="Rolling",
            window_months=window_months,
        )

        latest = df.iloc[-1]
        val_latest = float(latest[col])
        z_level_latest = float(latest["z_level"]) if not pd.isna(latest["z_level"]) else np.nan
        z_mom_latest = float(latest["z_mom"]) if not pd.isna(latest["z_mom"]) else np.nan

        if np.isnan(z_level_latest):
            continue

        direction_sign = 1.0 if cfg["good_when_high"] else -1.0
        level_score = direction_sign * z_level_latest
        mom_score = direction_sign * (z_mom_latest if not np.isnan(z_mom_latest) else 0.0)

        # beat / miss bias: extreme percentile given sign convention
        pct = percentile_rank(df[col].astype("float64"), val_latest)
        bias = "Neutral"
        if not np.isnan(pct):
            if cfg["good_when_high"]:
                if pct > 0.85:
                    bias = "Downside surprise risk"
                elif pct < 0.15:
                    bias = "Upside surprise risk"
            else:
                if pct > 0.85:
                    bias = "Upside surprise risk"
                elif pct < 0.15:
                    bias = "Downside surprise risk"

        metric_rows.append(
            {
                "Metric": cfg["label"],
                "Block": cfg["block"],
                "Latest": val_latest,
                "Transform": cfg["transform"],
                "z_level": z_level_latest,
                "z_mom": z_mom_latest,
                "Level score": level_score,
                "Momentum score": mom_score,
                "Direction": "High is good" if cfg["good_when_high"] else "High is bad",
                "Beat/miss bias": bias,
                "Raw_key": key,
                "Source": src,
            }
        )

        block = cfg["block"]
        block_scores_level.setdefault(block, []).append(level_score)
        block_scores_mom.setdefault(block, []).append(mom_score)

    if not metric_rows:
        st.info("No usable indicators for analytics in this economy.")
        st.stop()

    metric_df = pd.DataFrame(metric_rows)

    # block-level aggregate scores
    blocks = sorted(block_scores_level.keys())
    block_level = []
    block_mom = []
    for b in blocks:
        lvl = np.nanmean(block_scores_level.get(b, [np.nan]))
        mm = np.nanmean(block_scores_mom.get(b, [np.nan]))
        block_level.append(lvl)
        block_mom.append(mm)

    block_matrix = np.vstack([block_level, block_mom])  # 2 x N

    # --- Block heatmap (level vs momentum) ------------------------------------
    st.markdown("### Block snapshot – level vs momentum")

    # 2 x N matrix: row 0 = level, row 1 = momentum
    block_matrix = np.vstack([block_level, block_mom])

    # symmetric colour range around 0
    if np.isfinite(np.nanmax(np.abs(block_matrix))):
        max_abs = float(np.nanmax(np.abs(block_matrix)))
        if max_abs == 0:
            max_abs = 1.0
    else:
        max_abs = 1.0

    heat_fig = px.imshow(
        block_matrix,
        x=blocks,
        y=["Level", "Momentum"],
        color_continuous_scale=[
            [0.0, "#B22222"],   # red = weak / tight
            [0.5, "#FFF7D9"],   # neutral
            [1.0, "#006400"],   # green = strong / easy
        ],
        labels={"x": "Block", "y": "", "color": "Score"},
        aspect="auto",
    )
    # enforce symmetric range around zero so colours are comparable
    heat_fig.update_coloraxes(cmin=-max_abs, cmax=max_abs)

    heat_fig.update_layout(
        height=220,
        margin=dict(l=0, r=30, t=30, b=10),
        font=dict(size=10),
    )

    st.plotly_chart(heat_fig, width="stretch")

    # simple text read-through
    strongest_block = blocks[int(np.argmax(block_level))]
    weakest_block = blocks[int(np.argmin(block_level))]

    st.markdown(
        f"""
        **Read-through**

        - Strongest block: **{strongest_block}**  
        - Most vulnerable: **{weakest_block}**  

        Level scores capture where the block sits relative to its own history (rolling z-scores).  
        Momentum scores capture the direction and speed of change over the last year (or few years for annual data),
        adjusted for whether “high” is good or bad.
        """
    )

    st.markdown("---")

    # --- GIP-style quadrant using wage vs CPI signals -------------------------
    st.markdown("### Growth–inflation quadrant (proxy)")

    cpi_raw = series.get("cpi")
    wages_raw = series.get("wages")

    gip_text = "Insufficient CPI / wage data to compute GIP quadrant."
    gip_quad = None

    if cpi_raw and wages_raw:
        cpi_df = series_to_df(cpi_raw)
        wages_df = series_to_df(wages_raw)
        cpi_df = compute_yoy(cpi_df, "value", 12).dropna(subset=["yoy"])
        wages_df = compute_yoy(wages_df, "value", 12).dropna(subset=["yoy"])

        merged = pd.merge(
            cpi_df[["date", "yoy"]].rename(columns={"yoy": "cpi_yoy"}),
            wages_df[["date", "yoy"]].rename(columns={"yoy": "wage_yoy"}),
            on="date",
            how="inner",
        )
        merged = merged.sort_values("date")
        if len(merged) >= 24:
            merged["real_income"] = merged["wage_yoy"] - merged["cpi_yoy"]

            lookback = min(12, len(merged) - 1)
            latest = merged.iloc[-1]
            prev = merged.iloc[-1 - lookback]

            growth_accel = latest["real_income"] - prev["real_income"]
            infl_accel = latest["cpi_yoy"] - prev["cpi_yoy"]

            if growth_accel >= 0 and infl_accel >= 0:
                gip_quad = "G1 – reflation (growth ↑, inflation ↑)"
            elif growth_accel >= 0 and infl_accel < 0:
                gip_quad = "G2 – Goldilocks (growth ↑, inflation ↓)"
            elif growth_accel < 0 and infl_accel >= 0:
                gip_quad = "G3 – stagflation risk (growth ↓, inflation ↑)"
            else:
                gip_quad = "G4 – disinflation / slowdown (growth ↓, inflation ↓)"

            gip_text = (
                f"Latest proxy suggests: **{gip_quad}** "
                f"(real income change over last {lookback} months vs inflation acceleration)."
            )

    st.markdown(
        f"""
        <div style="font-size:13px; color:#555555; margin-bottom:0.5rem;">
        {gip_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- Indicator table: level + momentum + bias -----------------------------
    st.markdown("### Indicator table")

    display_df = metric_df.copy()
    display_df["Latest"] = display_df["Latest"].map(lambda x: f"{x:.2f}")
    display_df["z_level"] = display_df["z_level"].map(lambda x: f"{x:.2f}")
    display_df["z_mom"] = display_df["z_mom"].map(
        lambda x: "" if np.isnan(x) else f"{x:.2f}"
    )

    display_df = display_df.sort_values(
        ["Block", "Level score"], ascending=[True, False]
    )

    st.dataframe(
        display_df[
            [
                "Block",
                "Metric",
                "Latest",
                "z_level",
                "z_mom",
                "Direction",
                "Beat/miss bias",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="chart-comment">
        Level z-score is rolling, out-of-sample within-indicator. Momentum z-score is the z-score
        of the last 12M (or few-year) change. Beat/miss bias flags where prints are most likely
        to surprise consensus given where we are in the historical distribution.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Drill-down: indicator detail")

    metric_choice = st.selectbox(
        "Indicator",
        options=list(display_df["Metric"]),
        index=0,
    )
    row = display_df[display_df["Metric"] == metric_choice].iloc[0]
    raw_key = row["Raw_key"]
    src = row["Source"]

    if src == "hf":
        df_full = series_to_df(series.get(raw_key, []))
        if df_full.empty:
            st.info("No data for this indicator.")
            st.stop()
        df_full = df_full.sort_values("date")
    else:
        if panel.empty or raw_key not in panel.columns:
            st.info("No data for this indicator.")
            st.stop()
        df_full = panel[["year", raw_key]].dropna()
        df_full = df_full.rename(columns={raw_key: "value"}).copy()
        df_full["date"] = pd.to_datetime(
            df_full["year"].astype(int).astype(str) + "-12-31"
        )
        df_full = df_full[["date", "value"]].sort_values("date")

    df_full["yoy"] = df_full["value"].pct_change(12 if src == "hf" else 1) * 100.0
    df_full["mom"] = df_full["value"].pct_change(1) * 100.0

    chart_mode = st.radio(
        "Chart basis",
        ["Level", "YoY %", "MoM %"],
        index=0,
        horizontal=True,
        key="analytics_chart_mode",
    )

    if chart_mode == "Level":
        y_col = "value"
        y_label = "Level / index"
    elif chart_mode == "YoY %":
        y_col = "yoy"
        y_label = "YoY %"
    else:
        y_col = "mom"
        y_label = "MoM %"

    df_chart = df_full.dropna(subset=[y_col])
    if df_chart.empty:
        st.info("Not enough data under this transformation.")
        st.stop()

    fig_line = px.line(
        df_chart,
        x="date",
        y=y_col,
        labels={"date": "Date", y_col: y_label},
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig_line = style_figure(fig_line, height=280, legend=False)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown(
        '<div class="chart-comment">'
        'Line chart lets you sanity-check the block and table read-through against the raw series.'
        '</div>',
        unsafe_allow_html=True,
    )

    # bar chart on last N years for higher-frequency indicators
    if src == "hf":
        bar_years = st.slider(
            "Bar window (years)",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            key="analytics_bar_years_v2",
        )
        cutoff = df_chart["date"].max() - pd.DateOffset(years=bar_years)
        df_bar = df_chart[df_chart["date"] >= cutoff].copy()

        if not df_bar.empty:
            fig_bar = px.bar(
                df_bar,
                x="date",
                y=y_col,
                labels={"date": "Date", y_col: y_label},
                color_discrete_sequence=[PRIMARY_COLOR],
            )
            fig_bar = style_figure(fig_bar, height=220, legend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown(
                '<div class="chart-comment">'
                'Recent bars show whether the series is accelerating or rolling over into the latest prints.'
                '</div>',
                unsafe_allow_html=True,
            )

    # seasonality heatmap on YoY
    st.markdown("#### Seasonality heatmap (YoY %)")

    df_season = df_full.dropna(subset=["yoy"]).copy()
    if df_season.empty:
        st.info("Not enough YoY history to compute seasonality.")
    else:
        df_season["year"] = df_season["date"].dt.year
        df_season["month"] = df_season["date"].dt.month

        pivot = df_season.pivot_table(
            index="year", columns="month", values="yoy", aggfunc="mean"
        ).sort_index(ascending=False)

        month_labels = [calendar.month_abbr[m] for m in pivot.columns]

        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            labels={"x": "Month", "y": "Year", "color": "YoY %"},
            x=month_labels,
            color_continuous_scale="RdBu_r",
            zmid=0.0,
        )
        fig_heat.update_layout(
            height=260,
            margin=dict(l=0, r=30, t=25, b=5),
            font=dict(size=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown(
            '<div class="chart-comment">'
            'Seasonality heatmap highlights whether the latest year is behaving like a typical year or breaking the usual pattern, '
            'which feeds into the beat/miss bias on upcoming prints.'
            '</div>',
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# PLAYGROUND
# -----------------------------------------------------------------------------

with tab_play:
    st.header("Macro Playground")

    selected_countries = st.multiselect(
        "Economies",
        COUNTRIES,
        default=COUNTRIES,
        help="Select one or more economies to overlay.",
    )

    if not selected_countries:
        st.info("Select at least one economy to start.")
    else:
        selected_metrics = st.multiselect(
            "Metrics",
            options=list(METRIC_LABELS.keys()),
            default=["policy_rate", "yield_10y", "cpi"],
            format_func=lambda k: METRIC_LABELS.get(k, k),
            help="Choose one or more metrics; each will be plotted as a separate chart.",
        )

        if not selected_metrics:
            st.info("Select at least one metric to plot.")
        else:
            hf_by_country = {}
            for c in selected_countries:
                try:
                    js = fetch_highfreq(c)
                    hf_by_country[c] = js.get("series", {}) or {}
                except Exception:
                    hf_by_country[c] = {}

            all_dates = []
            for c in selected_countries:
                series_dict = hf_by_country.get(c, {})
                for m in selected_metrics:
                    raw = series_dict.get(m)
                    if not raw:
                        continue
                    df_tmp = series_to_df(raw)
                    if df_tmp.empty:
                        continue
                    all_dates.append(df_tmp["date"].min())
                    all_dates.append(df_tmp["date"].max())

            all_dates = [d for d in all_dates if pd.notnull(d)]

            if not all_dates:
                st.info("No overlapping high-frequency data for the current selection.")
            else:
                min_date_raw = min(all_dates)
                max_date_raw = max(all_dates)
                default_min = max(min_date_raw, max_date_raw - pd.DateOffset(years=15))

                start_date, end_date = st.slider(
                    "Sample window",
                    min_value=min_date_raw.date(),
                    max_value=max_date_raw.date(),
                    value=(default_min.date(), max_date_raw.date()),
                    help="Apply the same date window to all charts below.",
                )

                for metric in selected_metrics:
                    st.markdown(
                        f'<div class="chart-heading" style="font-size:1.5rem;">'
                        f'{METRIC_LABELS.get(metric, metric)}</div>',
                        unsafe_allow_html=True,
                    )

                    ctl1, ctl2, ctl3 = st.columns([1.5, 1.7, 2.0])

                    with ctl1:
                        value_mode_metric = st.radio(
                            "Value mode",
                            ["Level", "YoY % change (12m)"],
                            index=0,
                            horizontal=True,
                            key=f"value_mode_{metric}",
                        )

                    with ctl2:
                        use_z = st.checkbox(
                            "Z-normalise series",
                            key=f"z_{metric}",
                            help="Standardise each line to mean 0, stdev 1 within this chart.",
                        )
                        z_mode = "Full sample"
                        z_months = None
                        if use_z:
                            z_mode = st.radio(
                                "Z-window",
                                ["Full sample", "Rolling"],
                                index=0,
                                horizontal=True,
                                key=f"z_mode_{metric}",
                            )
                            if z_mode == "Rolling":
                                z_months = st.slider(
                                    "Rolling window (months)",
                                    min_value=0,
                                    max_value=240,
                                    value=60,
                                    step=6,
                                    key=f"z_months_{metric}",
                                )

                    with ctl3:
                        overlay_options = ["None"] + [
                            m for m in METRIC_LABELS.keys() if m != metric
                        ]
                        overlay_metric = st.selectbox(
                            "Overlay metric (optional)",
                            options=overlay_options,
                            index=0,
                            key=f"overlay_{metric}",
                            format_func=lambda k: "None"
                            if k == "None"
                            else METRIC_LABELS.get(k, k),
                            help="Add a second metric on the same chart (e.g. CPI over policy rate).",
                        )

                        show_composite = st.checkbox(
                            "Show aggregate composite",
                            key=f"agg_{metric}",
                            help="Plot an aggregate composite across economies as a separate line.",
                        )

                        composite_weights = {}
                        if show_composite:
                            st.caption(
                                "Composite weights by economy (% of total, will be normalised)."
                            )
                            num_cols = min(3, len(selected_countries))
                            weight_cols = st.columns(num_cols)
                            for idx, c in enumerate(selected_countries):
                                col = weight_cols[idx % num_cols]
                                if c == "United States":
                                    default_w = 150.0
                                elif c == "Japan":
                                    default_w = 80.0
                                elif c == "Euro Area":
                                    default_w = 100.0
                                else:
                                    default_w = 100.0
                                with col:
                                    composite_weights[c] = st.number_input(
                                        c,
                                        min_value=0.0,
                                        value=default_w,
                                        step=10.0,
                                        key=f"agg_w_{metric}_{c}",
                                    )

                    fig = px.line()
                    any_trace = False

                    country_series_plot = {}
                    country_series_aligned = {}

                    for c in selected_countries:
                        series_dict = hf_by_country.get(c, {})
                        raw = series_dict.get(metric)
                        if not raw:
                            continue

                        df = series_to_df(raw)
                        if df.empty:
                            continue

                        mask = (df["date"].dt.date >= start_date) & (
                            df["date"].dt.date <= end_date
                        )
                        df = df.loc[mask]
                        if df.empty:
                            continue

                        if value_mode_metric.startswith("YoY"):
                            df = compute_yoy(df, col="value", periods=12)
                            y_col = "yoy"
                        else:
                            y_col = "value"

                        if y_col not in df.columns:
                            continue

                        df = df.dropna(subset=[y_col])
                        if df.empty:
                            continue

                        df_metric = df[["date", y_col]].rename(columns={y_col: "y"})
                        country_series_plot[c] = df_metric
                        country_series_aligned[c] = align_to_monthly(
                            df_metric, start_date, end_date, max_ffill_months=6
                        )

                        series_vals = df_metric["y"]
                        if use_z:
                            series_vals = z_score(
                                series_vals, df_metric["date"], z_mode, z_months
                            )

                        fig.add_scatter(
                            x=df_metric["date"],
                            y=series_vals,
                            mode="lines",
                            name=f"{c} – {METRIC_LABELS.get(metric, metric)}",
                            line=dict(
                                color=COUNTRY_COLORS.get(c, None),
                                width=1.7,
                            ),
                        )
                        any_trace = True

                    if show_composite and country_series_aligned:
                        agg_df = None
                        for c, df_c in country_series_aligned.items():
                            if df_c.empty:
                                continue
                            tmp = df_c.rename(columns={"y": c})
                            if agg_df is None:
                                agg_df = tmp
                            else:
                                agg_df = pd.merge(agg_df, tmp, on="date", how="outer")

                        if agg_df is not None and not agg_df.empty:
                            if composite_weights:
                                w_series = pd.Series(composite_weights, dtype="float64")
                            else:
                                w_series = pd.Series(
                                    {c: 1.0 for c in country_series_aligned.keys()},
                                    dtype="float64",
                                )

                            if (w_series > 0).sum() == 0:
                                w_series = pd.Series(
                                    {c: 1.0 for c in country_series_aligned.keys()},
                                    dtype="float64",
                                )

                            w_series = w_series / w_series.sum()

                            cols_present = [
                                c for c in w_series.index if c in agg_df.columns
                            ]
                            if cols_present:
                                vals = agg_df[cols_present]
                                w_eff = w_series[cols_present]

                                valid_counts = (~vals.isna()).sum(axis=1)
                                mask_enough = valid_counts >= 2
                                vals = vals[mask_enough]
                                agg_df = agg_df.loc[mask_enough]

                                if not vals.empty:
                                    total_w = w_eff.sum()
                                    weighted = vals.mul(w_eff, axis=1)
                                    agg_df["agg"] = weighted.sum(axis=1) / total_w
                                    agg_df = agg_df[["date", "agg"]].dropna()

                                    if not agg_df.empty:
                                        agg_series_vals = agg_df["agg"]
                                        if use_z:
                                            agg_series_vals = z_score(
                                                agg_series_vals,
                                                agg_df["date"],
                                                z_mode,
                                                z_months,
                                            )

                                        fig.add_scatter(
                                            x=agg_df["date"],
                                            y=agg_series_vals,
                                            mode="lines",
                                            name="Composite",
                                            line=dict(color="#000000", width=2.0),
                                        )
                                        any_trace = True

                    if overlay_metric != "None":
                        for c in selected_countries:
                            series_dict = hf_by_country.get(c, {})
                            raw_ov = series_dict.get(overlay_metric)
                            if not raw_ov:
                                continue

                            df_ov = series_to_df(raw_ov)
                            if df_ov.empty:
                                continue

                            mask_ov = (df_ov["date"].dt.date >= start_date) & (
                                df_ov["date"].dt.date <= end_date
                            )
                            df_ov = df_ov.loc[mask_ov]
                            if df_ov.empty:
                                continue

                            if value_mode_metric.startswith("YoY"):
                                df_ov = compute_yoy(df_ov, col="value", periods=12)
                                y_col_ov = "yoy"
                            else:
                                y_col_ov = "value"

                            if y_col_ov not in df_ov.columns:
                                continue

                            df_ov = df_ov.dropna(subset=[y_col_ov])
                            if df_ov.empty:
                                continue

                            series_vals_ov = df_ov[y_col_ov]
                            if use_z:
                                series_vals_ov = z_score(
                                    series_vals_ov,
                                    df_ov["date"],
                                    z_mode,
                                    z_months,
                                )

                            fig.add_scatter(
                                x=df_ov["date"],
                                y=series_vals_ov,
                                mode="lines",
                                name=f"{c} – {METRIC_LABELS.get(overlay_metric, overlay_metric)}",
                                line=dict(
                                    color=COUNTRY_COLORS.get(c, None),
                                    width=1.4,
                                    dash="dash",
                                ),
                            )
                            any_trace = True

                    if not any_trace:
                        st.markdown(
                            '<div class="chart-placeholder">No usable data for this metric and date window.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title=metric_axis_label(metric, value_mode_metric)
                            if not use_z
                            else "Z-score (within series)",
                        )
                        fig = style_figure(fig, height=320, legend=True)
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown(
                        '<div class="chart-comment">'
                        'Overlay by economy, optional composite weighting and, where used, z-normalisation provide a '
                        'compact view of relative cycles and policy stances. Rolling z-scores focus on regime shifts '
                        'within a chosen window rather than over the entire sample.'
                        '</div>'
                        "<div style='height:0.8rem;'></div>",
                        unsafe_allow_html=True,
                    )

# -----------------------------------------------------------------------------
# Helper: Yahoo history
# -----------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def fetch_ticker_history(symbol: str, start_date, end_date, interval: str):
    symbol = symbol.upper().strip()
    if not symbol:
        return pd.DataFrame(columns=["date", "ticker", "Close"])

    session = get_yf_session()
    if session is not None:
        ticker_obj = yf.Ticker(symbol, session=session)
    else:
        ticker_obj = yf.Ticker(symbol)

    hist = ticker_obj.history(start=start_date, end=end_date, interval=interval)
    if hist.empty:
        return pd.DataFrame(columns=["date", "ticker", "Close"])

    hist = hist.reset_index()
    date_col = "Date" if "Date" in hist.columns else hist.columns[0]
    out = hist[[date_col, "Close"]].rename(columns={date_col: "date"})
    out["ticker"] = symbol
    return out

# -----------------------------------------------------------------------------
# STRATEGY
# -----------------------------------------------------------------------------

with tab_strategy:
    st.header("Strategy sandbox")

    st.markdown(
        """
        <p style="font-size:13px; color:#555555; margin-top:0rem; margin-bottom:0.6rem;">
            Build a macro signal using the same normalisation and composite logic as the playground, 
            then backtest a simple rule-based strategy on a chosen Yahoo ticker.
        </p>
        """,
        unsafe_allow_html=True,
    )

    sel_col1, sel_col2 = st.columns([2.0, 2.0])

    with sel_col1:
        selected_countries = st.multiselect(
            "Economies (signal inputs)",
            COUNTRIES,
            default=COUNTRIES,
            help="Select one or more economies whose macro data will form the signal.",
        )

    if not selected_countries:
        st.info("Select at least one economy to start.")
    else:
        with sel_col2:
            metric = st.selectbox(
                "Macro signal metric",
                options=list(METRIC_LABELS.keys()),
                index=list(METRIC_LABELS.keys()).index("yield_10y")
                if "yield_10y" in METRIC_LABELS
                else 0,
                format_func=lambda k: METRIC_LABELS.get(k, k),
            )

        hf_by_country = {}
        for c in selected_countries:
            try:
                js = fetch_highfreq(c)
                hf_by_country[c] = js.get("series", {}) or {}
            except Exception:
                hf_by_country[c] = {}

        all_dates = []
        for c in selected_countries:
            series_dict = hf_by_country.get(c, {})
            raw = series_dict.get(metric)
            if not raw:
                continue
            df_tmp = series_to_df(raw)
            if df_tmp.empty:
                continue
            all_dates.append(df_tmp["date"].min())
            all_dates.append(df_tmp["date"].max())

        all_dates = [d for d in all_dates if pd.notnull(d)]

        if not all_dates:
            st.info("No overlapping data for this metric across selected economies.")
        else:
            min_date_raw = min(all_dates)
            max_date_raw = max(all_dates)
            default_min = max(min_date_raw, max_date_raw - pd.DateOffset(years=15))

            start_date, end_date = st.slider(
                "Sample / backtest window",
                min_value=min_date_raw.date(),
                max_value=max_date_raw.date(),
                value=(default_min.date(), max_date_raw.date()),
                help="Signal is built over this window; ticker backtest will use the same dates.",
            )

            st.markdown(
                f'<div class="chart-heading" style="font-size:1.5rem;">'
                f'{METRIC_LABELS.get(metric, metric)}</div>',
                unsafe_allow_html=True,
            )

            ctl1, ctl2, ctl3 = st.columns([1.5, 1.7, 2.0])

            with ctl1:
                value_mode_metric = st.radio(
                    "Value mode",
                    ["Level", "YoY % change (12m)"],
                    index=0,
                    horizontal=True,
                    key=f"value_mode_strategy_{metric}",
                )

            with ctl2:
                use_z = st.checkbox(
                    "Z-normalise series",
                    key=f"z_strategy_{metric}",
                    help="Standardise each line to mean 0, stdev 1 within this chart.",
                )
                z_mode = "Full sample"
                z_months = None
                if use_z:
                    z_mode = st.radio(
                        "Z-window",
                        ["Full sample", "Rolling"],
                        index=0,
                        horizontal=True,
                        key=f"z_mode_strategy_{metric}",
                    )
                    if z_mode == "Rolling":
                        z_months = st.slider(
                            "Rolling window (months)",
                            min_value=0,
                            max_value=240,
                            value=60,
                            step=6,
                            key=f"z_months_strategy_{metric}",
                        )

            with ctl3:
                overlay_options = ["None"] + [
                    m for m in METRIC_LABELS.keys() if m != metric
                ]
                overlay_metric = st.selectbox(
                    "Overlay metric (optional)",
                    options=overlay_options,
                    index=0,
                    key=f"overlay_strategy_{metric}",
                    format_func=lambda k: "None"
                    if k == "None"
                    else METRIC_LABELS.get(k, k),
                    help="Add a second macro metric on the same chart (visual only).",
                )

                show_composite = st.checkbox(
                    "Show aggregate composite",
                    key=f"agg_strategy_{metric}",
                    help="Plot an aggregate composite across economies as a separate line.",
                )

                composite_weights = {}
                if show_composite:
                    st.caption(
                        "Composite weights by economy (% of total, will be normalised)."
                    )
                    num_cols = min(3, len(selected_countries))
                    weight_cols = st.columns(num_cols)
                    for idx, c in enumerate(selected_countries):
                        col = weight_cols[idx % num_cols]
                        if c == "United States":
                            default_w = 150.0
                        elif c == "Japan":
                            default_w = 80.0
                        elif c == "Euro Area":
                            default_w = 100.0
                        else:
                            default_w = 100.0
                        with col:
                            composite_weights[c] = st.number_input(
                                c,
                                min_value=0.0,
                                value=default_w,
                                step=10.0,
                                key=f"agg_w_strategy_{metric}_{c}",
                            )

            vis_cols = st.columns(min(3, len(selected_countries)))
            show_country = {}
            for idx, c in enumerate(selected_countries):
                col = vis_cols[idx % len(vis_cols)]
                with col:
                    show_country[c] = st.checkbox(
                        f"Show {c}",
                        value=True,
                        key=f"show_strategy_{metric}_{c}",
                    )

            fig = px.line()
            any_trace = False

            country_series_plot = {}
            country_series_aligned = {}
            signal_sources = {}

            for c in selected_countries:
                series_dict = hf_by_country.get(c, {})
                raw = series_dict.get(metric)
                if not raw:
                    continue

                df = series_to_df(raw)
                if df.empty:
                    continue

                mask = (df["date"].dt.date >= start_date) & (
                    df["date"].dt.date <= end_date
                )
                df = df.loc[mask]
                if df.empty:
                    continue

                if value_mode_metric.startswith("YoY"):
                    df = compute_yoy(df, col="value", periods=12)
                    y_col = "yoy"
                else:
                    y_col = "value"

                if y_col not in df.columns:
                    continue

                df = df.dropna(subset=[y_col])
                if df.empty:
                    continue

                df_metric = df[["date", y_col]].rename(columns={y_col: "y"})
                country_series_plot[c] = df_metric
                country_series_aligned[c] = align_to_monthly(
                    df_metric, start_date, end_date, max_ffill_months=6
                )

                series_vals = df_metric["y"]
                if use_z:
                    series_vals = z_score(
                        series_vals, df_metric["date"], z_mode, z_months
                    )

                signal_sources[c] = pd.DataFrame(
                    {"date": df_metric["date"], "signal": series_vals.values}
                )

                if show_country.get(c, True):
                    fig.add_scatter(
                        x=df_metric["date"],
                        y=series_vals,
                        mode="lines",
                        name=f"{c} – {METRIC_LABELS.get(metric, metric)}",
                        line=dict(
                            color=COUNTRY_COLORS.get(c, None),
                            width=1.7,
                        ),
                    )
                    any_trace = True

            if show_composite and country_series_aligned:
                agg_df = None
                for c, df_c in country_series_aligned.items():
                    if df_c.empty:
                        continue
                    tmp = df_c.rename(columns={"y": c})
                    if agg_df is None:
                        agg_df = tmp
                    else:
                        agg_df = pd.merge(agg_df, tmp, on="date", how="outer")

                if agg_df is not None and not agg_df.empty:
                    if composite_weights:
                        w_series = pd.Series(composite_weights, dtype="float64")
                    else:
                        w_series = pd.Series(
                            {c: 1.0 for c in country_series_aligned.keys()},
                            dtype="float64",
                        )

                    if (w_series > 0).sum() == 0:
                        w_series = pd.Series(
                            {c: 1.0 for c in country_series_aligned.keys()},
                            dtype="float64",
                        )

                    w_series = w_series / w_series.sum()

                    cols_present = [c for c in w_series.index if c in agg_df.columns]
                    if cols_present:
                        vals = agg_df[cols_present]
                        w_eff = w_series[cols_present]

                        valid_counts = (~vals.isna()).sum(axis=1)
                        mask_enough = valid_counts >= 2
                        vals = vals[mask_enough]
                        agg_df = agg_df.loc[mask_enough]

                        if not vals.empty:
                            total_w = w_eff.sum()
                            weighted = vals.mul(w_eff, axis=1)
                            agg_df["agg"] = weighted.sum(axis=1) / total_w
                            agg_df = agg_df[["date", "agg"]].dropna()

                            if not agg_df.empty:
                                agg_series_vals = agg_df["agg"]
                                if use_z:
                                    agg_series_vals = z_score(
                                        agg_series_vals,
                                        agg_df["date"],
                                        z_mode,
                                        z_months,
                                    )

                                fig.add_scatter(
                                    x=agg_df["date"],
                                    y=agg_series_vals,
                                    mode="lines",
                                    name="Composite",
                                    line=dict(color="#000000", width=2.0),
                                )
                                signal_sources["Composite"] = pd.DataFrame(
                                    {
                                        "date": agg_df["date"],
                                        "signal": agg_series_vals.values,
                                    }
                                )
                                any_trace = True

            if overlay_metric != "None":
                for c in selected_countries:
                    series_dict = hf_by_country.get(c, {})
                    raw_ov = series_dict.get(overlay_metric)
                    if not raw_ov:
                        continue

                    df_ov = series_to_df(raw_ov)
                    if df_ov.empty:
                        continue

                    mask_ov = (df_ov["date"].dt.date >= start_date) & (
                        df_ov["date"].dt.date <= end_date
                    )
                    df_ov = df_ov.loc[mask_ov]
                    if df_ov.empty:
                        continue

                    if value_mode_metric.startswith("YoY"):
                        df_ov = compute_yoy(df_ov, col="value", periods=12)
                        y_col_ov = "yoy"
                    else:
                        y_col_ov = "value"

                    if y_col_ov not in df_ov.columns:
                        continue

                    df_ov = df_ov.dropna(subset=[y_col_ov])
                    if df_ov.empty:
                        continue

                    series_vals_ov = df_ov[y_col_ov]
                    if use_z:
                        series_vals_ov = z_score(
                            series_vals_ov,
                            df_ov["date"],
                            z_mode,
                            z_months,
                        )

                    fig.add_scatter(
                        x=df_ov["date"],
                        y=series_vals_ov,
                        mode="lines",
                        name=f"{c} – {METRIC_LABELS.get(overlay_metric, overlay_metric)}",
                        line=dict(
                            color=COUNTRY_COLORS.get(c, None),
                            width=1.4,
                            dash="dash",
                        ),
                    )
                    any_trace = True

            recession_country = choose_recession_country(selected_countries)
            rec_ranges = get_recession_ranges_for_country(recession_country)
            if any_trace and rec_ranges:
                fig = add_recession_bands(
                    fig, rec_ranges, window_start=start_date, window_end=end_date
                )

            if not any_trace:
                st.markdown(
                    '<div class="chart-placeholder">No usable data for this metric and date window.</div>',
                    unsafe_allow_html=True,
                )
                signal_sources = {}
            else:
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=metric_axis_label(metric, value_mode_metric)
                    if not use_z
                    else "Z-score (within series)",
                )
                fig = style_figure(fig, height=320, legend=True)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"strategy_macro_{metric}",
                )

                st.markdown(
                    '<div class="chart-comment">'
                    'Overlay by economy, optional composite weighting and, where used, z-normalisation provide a '
                    'compact view of relative cycles and policy stances. Rolling z-scores focus on regime shifts '
                    'within a chosen window rather than over the entire sample.'
                    '</div>'
                    "<div style='height:0.8rem;'></div>",
                    unsafe_allow_html=True,
                )

            if signal_sources:
                st.subheader("Backtest on Yahoo ticker")

                sig_names = list(signal_sources.keys())
                default_idx = sig_names.index("Composite") if "Composite" in sig_names else 0

                bt_col1, bt_col2 = st.columns([2.0, 2.0])
                with bt_col1:
                    signal_choice = st.selectbox(
                        "Signal series used for trading",
                        options=sig_names,
                        index=default_idx,
                        help="Choose which line above becomes the trading signal.",
                    )

                with bt_col2:
                    ticker_bt = st.text_input(
                        "Backtest ticker (Yahoo symbol)",
                        "SPY",
                        help="Example: SPY, QQQ, ^GSPC, 2800.HK, 9988.HK, etc.",
                    )

                mode_col1, mode_col2 = st.columns([2.0, 2.0])
                with mode_col1:
                    bt_mode = st.radio(
                        "Trading mode",
                        ["Long-only", "Long/Short"],
                        index=0,
                        horizontal=True,
                    )

                if bt_mode == "Long-only":
                    thr_c1, thr_c2 = st.columns([1.5, 2.5])
                    with thr_c1:
                        threshold = st.number_input(
                            "Signal threshold",
                            value=2.0,
                            step=0.1,
                            help="Go long (or stay flat) depending on the rule below.",
                        )
                    with thr_c2:
                        rule = st.radio(
                            "Trading rule",
                            ["Long when signal > threshold", "Long when signal < threshold"],
                            index=0,
                            horizontal=True,
                        )
                    upper_threshold = threshold
                    lower_threshold = None
                else:
                    thr_c1, thr_c2 = st.columns(2)
                    with thr_c1:
                        lower_threshold = st.number_input(
                            "Lower threshold (go long)",
                            value=-2.0,
                            step=0.1,
                        )
                    with thr_c2:
                        upper_threshold = st.number_input(
                            "Upper threshold (go short)",
                            value=2.0,
                            step=0.1,
                        )
                    rule = None

                run_bt = st.button("Run backtest")

                if run_bt:
                    if not ticker_bt.strip():
                        st.error("Please provide a valid Yahoo ticker.")
                    else:
                        sig_df = signal_sources.get(signal_choice)
                        if sig_df is None or sig_df.empty:
                            st.error("Chosen signal series has no data.")
                        else:
                            sig_daily = sig_df.copy()
                            sig_daily["date"] = pd.to_datetime(sig_daily["date"])
                            sig_daily = sig_daily.sort_values("date")
                            idx = pd.date_range(
                                start=start_date, end=end_date, freq="B"
                            )
                            sig_daily = (
                                sig_daily.set_index("date")
                                .reindex(idx)
                                .rename_axis("date")
                                .reset_index()
                            )
                            sig_daily["signal"] = sig_daily["signal"].ffill()
                            sig_daily = sig_daily.dropna(subset=["signal"])

                            try:
                                session = get_yf_session()
                                if session is not None:
                                    tk = yf.Ticker(ticker_bt.strip(), session=session)
                                else:
                                    tk = yf.Ticker(ticker_bt.strip())

                                px_df = tk.history(
                                    start=start_date, end=end_date, interval="1d"
                                )

                                if px_df.empty:
                                    st.error(
                                        "No price data returned for this ticker and window."
                                    )
                                else:
                                    px_df = px_df.reset_index()
                                    date_col = (
                                        "Date"
                                        if "Date" in px_df.columns
                                        else px_df.columns[0]
                                    )
                                    px_df = px_df[[date_col, "Close"]].rename(
                                        columns={date_col: "date"}
                                    )
                                    px_df["date"] = pd.to_datetime(px_df["date"])
                                    if getattr(px_df["date"].dtype, "tz", None) is not None:
                                        px_df["date"] = px_df["date"].dt.tz_localize(None)
                                    px_df = px_df.sort_values("date")

                                    merged = pd.merge(
                                        px_df, sig_daily, on="date", how="inner"
                                    )
                                    if merged.empty:
                                        st.error(
                                            "No overlapping dates between signal and price data."
                                        )
                                    else:
                                        merged = merged.sort_values("date")
                                        merged["ret"] = (
                                            merged["Close"].pct_change().fillna(0.0)
                                        )

                                        if bt_mode == "Long-only":
                                            if rule.startswith("Long when signal >"):
                                                merged["position"] = (
                                                    merged["signal"] > upper_threshold
                                                ).astype(float)
                                            else:
                                                merged["position"] = (
                                                    merged["signal"] < upper_threshold
                                                ).astype(float)
                                        else:
                                            merged["position"] = np.where(
                                                merged["signal"] < lower_threshold,
                                                1.0,
                                                np.where(
                                                    merged["signal"] > upper_threshold,
                                                    -1.0,
                                                    0.0,
                                                ),
                                            )

                                        merged["position_lag"] = (
                                            merged["position"].shift(1).fillna(0.0)
                                        )
                                        merged["strategy_ret"] = (
                                            merged["position_lag"] * merged["ret"]
                                        )

                                        merged["eq_strategy"] = (
                                            1.0 + merged["strategy_ret"]
                                        ).cumprod()
                                        merged["eq_buyhold"] = (
                                            1.0 + merged["ret"]
                                        ).cumprod()

                                        merged["pos_prev"] = (
                                            merged["position_lag"].shift(1).fillna(0.0)
                                        )

                                        if bt_mode == "Long-only":
                                            long_entries = merged[
                                                (merged["position_lag"] > 0)
                                                & (merged["pos_prev"] <= 0)
                                            ].copy()
                                            short_entries = merged.iloc[0:0].copy()
                                        else:
                                            long_entries = merged[
                                                (merged["position_lag"] > 0)
                                                & (merged["pos_prev"] <= 0)
                                            ].copy()
                                            short_entries = merged[
                                                (merged["position_lag"] < 0)
                                                & (merged["pos_prev"] >= 0)
                                            ].copy()

                                        import math

                                        n_days = len(merged)
                                        if n_days <= 1:
                                            st.error(
                                                "Not enough data points for a meaningful backtest."
                                            )
                                        else:
                                            total_strat = (
                                                merged["eq_strategy"].iloc[-1] - 1.0
                                            )
                                            total_bh = (
                                                merged["eq_buyhold"].iloc[-1] - 1.0
                                            )

                                            ann_factor = 252 / max(n_days, 1)
                                            ann_strat = (1.0 + total_strat) ** ann_factor - 1.0
                                            ann_bh = (1.0 + total_bh) ** ann_factor - 1.0

                                            vol_strat = (
                                                merged["strategy_ret"].std()
                                                * math.sqrt(252)
                                            )
                                            sharpe_strat = (
                                                ann_strat / vol_strat
                                                if vol_strat and vol_strat != 0
                                                else float("nan")
                                            )

                                            st.markdown("#### Trade Log")

                                            fig_sig_bt = px.line(
                                                merged,
                                                x="date",
                                                y="signal",
                                                labels={"date": "Date", "signal": "Signal"},
                                            )

                                            sig_vals = merged["signal"].dropna()
                                            if not sig_vals.empty:
                                                y_min = float(sig_vals.min())
                                                y_max = float(sig_vals.max())

                                                if bt_mode == "Long-only":
                                                    if rule.startswith("Long when signal >"):
                                                        y0 = upper_threshold
                                                        y1 = y_max
                                                    else:
                                                        y0 = y_min
                                                        y1 = upper_threshold
                                                    fig_sig_bt.add_hrect(
                                                        y0=y0,
                                                        y1=y1,
                                                        fillcolor="rgba(0,180,0,0.10)",
                                                        line_width=0,
                                                        layer="below",
                                                    )
                                                else:
                                                    fig_sig_bt.add_hrect(
                                                        y0=y_min,
                                                        y1=lower_threshold,
                                                        fillcolor="rgba(0,180,0,0.10)",
                                                        line_width=0,
                                                        layer="below",
                                                    )
                                                    fig_sig_bt.add_hrect(
                                                        y0=upper_threshold,
                                                        y1=y_max,
                                                        fillcolor="rgba(200,0,0,0.10)",
                                                        line_width=0,
                                                        layer="below",
                                                    )

                                            if bt_mode == "Long-only":
                                                fig_sig_bt.add_hline(
                                                    y=upper_threshold,
                                                    line=dict(color="green", width=1, dash="dot"),
                                                    annotation_text="Threshold",
                                                    annotation_position="top left",
                                                )
                                            else:
                                                fig_sig_bt.add_hline(
                                                    y=lower_threshold,
                                                    line=dict(color="green", width=1, dash="dot"),
                                                    annotation_text="Lower (long)",
                                                    annotation_position="bottom left",
                                                )
                                                fig_sig_bt.add_hline(
                                                    y=upper_threshold,
                                                    line=dict(color="red", width=1, dash="dot"),
                                                    annotation_text="Upper (short)",
                                                    annotation_position="top left",
                                                )

                                            if not long_entries.empty:
                                                fig_sig_bt.add_scatter(
                                                    x=long_entries["date"],
                                                    y=long_entries["signal"],
                                                    mode="markers",
                                                    name="Long entries",
                                                    marker=dict(
                                                        symbol="triangle-up",
                                                        size=9,
                                                        color="green",
                                                    ),
                                                )
                                            if not short_entries.empty:
                                                fig_sig_bt.add_scatter(
                                                    x=short_entries["date"],
                                                    y=short_entries["signal"],
                                                    mode="markers",
                                                    name="Short entries",
                                                    marker=dict(
                                                        symbol="triangle-down",
                                                        size=9,
                                                        color="red",
                                                    ),
                                                )

                                            if rec_ranges:
                                                fig_sig_bt = add_recession_bands(
                                                    fig_sig_bt,
                                                    rec_ranges,
                                                    window_start=merged["date"].min(),
                                                    window_end=merged["date"].max(),
                                                )

                                            fig_sig_bt.update_layout(
                                                yaxis_title="Signal"
                                                if not use_z
                                                else "Z-score (within series)",
                                            )
                                            fig_sig_bt = style_figure(
                                                fig_sig_bt, height=320, legend=True
                                            )
                                            st.plotly_chart(
                                                fig_sig_bt,
                                                use_container_width=True,
                                                key=f"strategy_signal_bt_{ticker_bt}_{signal_choice}",
                                            )

                                            st.markdown("#### PnL profile")

                                            fig_bt = px.line()
                                            fig_bt.add_scatter(
                                                x=merged["date"],
                                                y=merged["eq_strategy"],
                                                mode="lines",
                                                name="Strategy",
                                                line=dict(color=PRIMARY_COLOR, width=2.0),
                                            )
                                            fig_bt.add_scatter(
                                                x=merged["date"],
                                                y=merged["eq_buyhold"],
                                                mode="lines",
                                                name="Buy & hold",
                                                line=dict(
                                                    color=SECONDARY_COLOR, width=1.8
                                                ),
                                            )

                                            if not long_entries.empty:
                                                fig_bt.add_scatter(
                                                    x=long_entries["date"],
                                                    y=long_entries["eq_strategy"],
                                                    mode="markers",
                                                    name="Long entries (PnL)",
                                                    marker=dict(
                                                        symbol="triangle-up",
                                                        size=9,
                                                        color="green",
                                                    ),
                                                )
                                            if not short_entries.empty:
                                                fig_bt.add_scatter(
                                                    x=short_entries["date"],
                                                    y=short_entries["eq_strategy"],
                                                    mode="markers",
                                                    name="Short entries (PnL)",
                                                    marker=dict(
                                                        symbol="triangle-down",
                                                        size=9,
                                                        color="red",
                                                    ),
                                                )

                                            if rec_ranges:
                                                fig_bt = add_recession_bands(
                                                    fig_bt,
                                                    rec_ranges,
                                                    window_start=merged["date"].min(),
                                                    window_end=merged["date"].max(),
                                                )

                                            fig_bt.update_layout(
                                                xaxis_title="Date",
                                                yaxis_title="Equity (normalised to 1)",
                                            )
                                            fig_bt = style_figure(
                                                fig_bt, height=320, legend=True
                                            )
                                            st.plotly_chart(
                                                fig_bt,
                                                use_container_width=True,
                                                key=f"strategy_pnl_{ticker_bt}_{signal_choice}",
                                            )

                                            stats_df = pd.DataFrame(
                                                {
                                                    "Metric": [
                                                        "Total return (strategy)",
                                                        "Total return (buy & hold)",
                                                        "Annualised return (strategy)",
                                                        "Annualised return (buy & hold)",
                                                        "Annualised vol (strategy)",
                                                        "Sharpe (strategy, rf=0)",
                                                    ],
                                                    "Value": [
                                                        f"{total_strat*100:.2f}%",
                                                        f"{total_bh*100:.2f}%",
                                                        f"{ann_strat*100:.2f}%",
                                                        f"{ann_bh*100:.2f}%",
                                                        f"{vol_strat*100:.2f}%",
                                                        f"{sharpe_strat:.2f}",
                                                    ],
                                                }
                                            )
                                            st.dataframe(
                                                stats_df,
                                                use_container_width=True,
                                                hide_index=True,
                                            )
                            except Exception as e:
                                st.error(f"Error during backtest: {e}")

# -----------------------------------------------------------------------------
# DATA TAB
# -----------------------------------------------------------------------------

with tab_data:
    st.header("Inspect Data")
    st.write("**Panel data (A. Policy & growth constraints)**")
    try:
        constraints = fetch_constraints_panel(country)
        panel = pd.DataFrame(constraints.get("panel", []))
    except Exception:
        panel = pd.DataFrame()
    if not panel.empty:
        st.dataframe(panel, use_container_width=True)
    else:
        st.info("No panel data available.")

    st.write("**High-frequency series available (B. FRED)**")
    try:
        hf_data = fetch_highfreq(country)
        series_meta = list(hf_data.get("series", {}).keys())
        st.write(series_meta)
    except Exception:
        st.info("No high-frequency metadata available.")
