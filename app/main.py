"""
DMAP-AI API (demo-locked)
- SPI endpoints:
    • /indices/spi — z-score SPI from provided P sum/mean/std (kept for quick tests)
    • /demo/spi_historical_auto — z-score SPI computed from NASA POWER daily (baseline window)
    • /demo/spi_gamma_historical_auto — Gamma-fit SPI with zero-precip adjustment, mapped to standard normal
    • /demo/spi_gamma_series — NEW: SPI series (monthly or yearly), supports custom anchor and yearly aggregation
"""

import os
import datetime as dt
from statistics import mean, stdev
from typing import Optional, Tuple, Dict, List, Literal, DefaultDict
from collections import defaultdict

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware 

# SciPy for Gamma CDF and Normal inverse CDF
from scipy.stats import gamma as gamma_dist, norm

app = FastAPI(title="DMAP-AI API", version="0.6.0")

# ----------------------------- CORS ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later e.g. ["https://dmap.agrimetsoft.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- DEMO LOCK ------------------------------
DEMO_LOCK = os.getenv("DEMO_LOCK", "1") == "1"
ALLOWED_MODE = os.getenv("DEMO_MODE", "historical").lower()
ALLOWED_AOI = os.getenv("DEMO_AOI", "point").lower()
ALLOWED_DATASOURCE = os.getenv("DEMO_DATASOURCE", "era5").lower()


def _enforce_demo(mode: Optional[str], aoi: Optional[str], datasource: Optional[str]):
    if not DEMO_LOCK:
        return
    if mode is not None and mode.lower() != ALLOWED_MODE:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: only 'historical' is enabled.",
            "allowed": {"mode": ALLOWED_MODE, "aoi": ALLOWED_AOI, "datasource": ALLOWED_DATASOURCE}
        })
    if aoi is not None and aoi.lower() != ALLOWED_AOI:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: only 'point' AOI is enabled.",
            "allowed": {"mode": ALLOWED_MODE, "aoi": ALLOWED_AOI, "datasource": ALLOWED_DATASOURCE}
        })
    if datasource is not None and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: only 'ERA5' datasource is enabled.",
            "allowed": {"mode": ALLOWED_MODE, "aoi": ALLOWED_AOI, "datasource": ALLOWED_DATASOURCE}
        })


@app.get("/")
def root():
    return {"ok": True, "message": "DMAP-AI API. See /health and /docs", "demo_lock": DEMO_LOCK}


@app.get("/health")
def health():
    return {"ok": True, "demo_lock": DEMO_LOCK}


# -------------------------- SPI (z-score) ---------------------------
@app.get("/indices/spi")
def spi(
    sum_rain_mm: float = Query(..., description="Aggregated precipitation for the window (mm)"),
    clim_mean_mm: float = Query(..., description="Climatological mean for the same window (mm)"),
    clim_std_mm: float = Query(..., description="Climatological std for the same window (mm)", gt=0),
    window_days: int = Query(30, ge=1, le=365, description="Window length (days)"),
    mode: Optional[str] = Query(None, description="historical|prediction"),
    aoi: Optional[str] = Query(None, description="point|box"),
    datasource: Optional[str] = Query(None, description="era5|gridmet|prism|gpm|user"),
):
    _enforce_demo(mode, aoi, datasource)
    spi_value = (sum_rain_mm - clim_mean_mm) / clim_std_mm
    return {
        "window_days": window_days,
        "sum_mm": sum_rain_mm,
        "clim_mean_mm": clim_mean_mm,
        "clim_std_mm": clim_std_mm,
        "spi": round(spi_value, 3),
        "demo_lock": DEMO_LOCK,
        "note": "Gamma-fit SPI also available at /demo/spi_gamma_historical_auto.",
    }


# ------------------- Common helpers for auto SPI --------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAM = "PRECTOTCORR"  # daily precipitation (mm/day)


def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _parse_baseline(baseline: str) -> Tuple[int, int]:
    try:
        a, b = baseline.split("-")
        return int(a), int(b)
    except Exception:
        return (1981, 2010)


def _fetch_power_precip(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, float]:
    params = {
        "parameters": POWER_PARAM,
        "community": "AG",
        "latitude": f"{lat}",
        "longitude": f"{lon}",
        "start": _yyyymmdd(start),
        "end": _yyyymmdd(end),
        "format": "JSON",
    }
    r = requests.get(POWER_BASE, params=params, timeout=45)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"message": "NASA POWER error", "status": r.status_code, "text": r.text[:300]})
    j = r.json()
    param = (j.get("properties", {}).get("parameter", {}).get(POWER_PARAM)
             or j.get("parameters", {}).get(POWER_PARAM))
    if not isinstance(param, dict):
        raise HTTPException(status_code=502, detail={"message": "Unexpected NASA POWER payload", "sample_keys": list(j.keys())[:5]})
    # keys may be strings of YYYYMMDD; ensure str keys -> float values
    out: Dict[str, float] = {}
    for k, v in param.items():
        kk = str(k)
        try:
            out[kk] = float(v)
        except Exception:
            out[kk] = 0.0
    return out


def _sum_window(s: dt.date, e: dt.date, dct: Dict[str, float]) -> float:
    total = 0.0
    cur = s
    while cur <= e:
        total += dct.get(_yyyymmdd(cur), 0.0)
        cur += dt.timedelta(days=1)
    return total


def _aligned_baseline_window_sums(end: dt.date, win: int, base_all: Dict[str, float], by0: int, by1: int) -> List[float]:
    """One window-sum per baseline year, aligned to end.month/end.day (clamp Feb 29)."""
    sums: List[float] = []
    for yr in range(by0, by1 + 1):
        try:
            last_day = (dt.date(yr, end.month, 1) + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(days=1)
            end_y = dt.date(yr, end.month, min(end.day, last_day.day))
        except Exception:
            end_y = dt.date(yr, end.month, 28) if end.month == 2 else dt.date(yr, end.month, 1)
        start_y = end_y - dt.timedelta(days=win - 1)
        sums.append(_sum_window(start_y, end_y, base_all))
    return sums


def _last_day_of_month(y: int, m: int) -> dt.date:
    if m == 12:
        return dt.date(y, 12, 31)
    return dt.date(y, m + 1, 1) - dt.timedelta(days=1)


def _month_end_dates(start: dt.date, end: dt.date) -> List[dt.date]:
    """All month-end dates between start and end inclusive."""
    dates: List[dt.date] = []
    y, m = start.year, start.month
    first = _last_day_of_month(y, m)
    if first < start:
        # move to next month
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
        first = _last_day_of_month(y, m)
    cur = first
    while cur <= end:
        dates.append(cur)
        if cur.month == 12:
            y, m = cur.year + 1, 1
        else:
            y, m = cur.year, cur.month + 1
        cur = _last_day_of_month(y, m)
    return dates


def _anchor_for_year(anchor_mmdd: Tuple[int, int], year: int) -> dt.date:
    m, d = anchor_mmdd
    # clamp day to month length (handle Feb 29)
    last = _last_day_of_month(year, m)
    d2 = min(d, last.day)
    return dt.date(year, m, d2)


# ------------------- SPI (auto, z-score / legacy) -------------------
@app.get("/demo/spi_historical_auto")
def spi_historical_auto(
    lat: float = Query(...),
    lon: float = Query(...),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    window_days: int = Query(30, ge=7, le=120),
    baseline: str = Query("1981-2010", description="e.g., 1981-2010"),
    datasource: str = Query("era5", description="locked to era5 for demo"),
):
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: ERA5 only for auto SPI."})
    try:
        end = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid end_date. Use YYYY-MM-DD."})

    win = max(1, min(window_days, 120))
    start = end - dt.timedelta(days=win - 1)

    # Analysis window precip
    obs = _fetch_power_precip(lat, lon, start, end)
    sum_obs = _sum_window(start, end, obs)

    # Baseline per-year window sums
    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_power_precip(lat, lon, dt.date(by0, 1, 1), dt.date(by1, 12, 31))
    baseline_sums = _aligned_baseline_window_sums(end, win, base_all, by0, by1)

    if len(baseline_sums) < 2:
        raise HTTPException(status_code=502, detail={"message": "Insufficient baseline length for SPI."})
    try:
        sd = stdev(baseline_sums)
    except Exception:
        sd = 0.0
    if sd == 0:
        raise HTTPException(status_code=502, detail={"message": "Insufficient baseline variance for SPI z-score."})

    mu = mean(baseline_sums)
    spi_z = (sum_obs - mu) / sd

    return {
        "method": "zscore",
        "datasource": datasource,
        "lat": lat, "lon": lon,
        "window_days": win,
        "end_date": end.isoformat(),
        "baseline": f"{by0}-{by1}",
        "obs_sum_mm": round(sum_obs, 3),
        "baseline_mean_mm": round(mu, 3),
        "baseline_std_mm": round(sd, 3),
        "spi": round(spi_z, 3),
        "note": "Legacy demo using z-score SPI. Prefer /demo/spi_gamma_historical_auto for Gamma SPI.",
    }


# --------- SPI (auto, Gamma fit with zero-precip adjustment) --------
@app.get("/demo/spi_gamma_historical_auto")
def spi_gamma_historical_auto(
    lat: float = Query(...),
    lon: float = Query(...),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    window_days: int = Query(30, ge=7, le=120),
    baseline: str = Query("1981-2010", description="e.g., 1981-2010"),
    datasource: str = Query("era5", description="locked to era5 for demo"),
):
    # Demo lock: only ERA5 path is allowed
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: ERA5 only for auto SPI (Gamma)."})

    try:
        end = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid end_date. Use YYYY-MM-DD."})

    win = max(1, min(window_days, 120))
    start = end - dt.timedelta(days=win - 1)

    # Analysis window precip
    obs = _fetch_power_precip(lat, lon, start, end)
    sum_obs = _sum_window(start, end, obs)

    # Baseline
    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_power_precip(lat, lon, dt.date(by0, 1, 1), dt.date(by1, 12, 31))
    baseline_sums = _aligned_baseline_window_sums(end, win, base_all, by0, by1)

    if len(baseline_sums) < 5:
        raise HTTPException(status_code=502, detail={"message": "Insufficient baseline length for Gamma fit."})

    # Zero handling (mixed distribution)
    zeros = [s for s in baseline_sums if s <= 0.0]
    pos = [s for s in baseline_sums if s > 0.0]
    n = len(baseline_sums)
    p0 = len(zeros) / n

    if len(pos) < 3:
        # Fallback to z-score
        try:
            sd = stdev(baseline_sums)
        except Exception:
            sd = 0.0
        if sd == 0:
            raise HTTPException(status_code=502, detail={"message": "Baseline variance is zero; cannot compute SPI."})
        mu = mean(baseline_sums)
        spi_z = (sum_obs - mu) / sd
        return {
            "method": "zscore-fallback",
            "datasource": datasource,
            "lat": lat, "lon": lon,
            "window_days": win,
            "end_date": end.isoformat(),
            "baseline": f"{by0}-{by1}",
            "obs_sum_mm": round(sum_obs, 3),
            "baseline_mean_mm": round(mu, 3),
            "baseline_std_mm": round(sd, 3),
            "p_zero": round(p0, 3),
            "spi": round(spi_z, 3),
            "note": "Fallback to z-score due to insufficient positive samples for Gamma fit.",
        }

    # Fit Gamma to positive sums (loc fixed at 0)
    shape, loc, scale = gamma_dist.fit(pos, floc=0)

    # CDF for observed sum
    Gx = float(gamma_dist.cdf(max(sum_obs, 0.0), a=shape, loc=0, scale=scale))

    # Mixed CDF with zeros
    H = p0 + (1.0 - p0) * Gx
    H = min(max(H, 1e-10), 1.0 - 1e-10)  # clip
    spi_val = float(norm.ppf(H))

    return {
        "method": "gamma",
        "datasource": datasource,
        "lat": lat, "lon": lon,
        "window_days": win,
        "end_date": end.isoformat(),
        "baseline": f"{by0}-{by1}",
        "obs_sum_mm": round(sum_obs, 3),
        "n_baseline": n,
        "p_zero": round(p0, 3),
        "gamma_shape": round(shape, 6),
        "gamma_scale": round(scale, 6),
        "cdf_gamma": round(Gx, 6),
        "cdf_mixed": round(H, 6),
        "spi": round(spi_val, 3),
        "note": "SPI via 2-parameter Gamma fit on baseline window sums (zeros handled).",
    }


# ---------------- SPI SERIES (monthly/yearly, anchor & aggregate) ----------------
@app.get("/demo/spi_gamma_series")
def spi_gamma_series(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    window_days: int = Query(30, ge=7, le=120),
    baseline: str = Query("1981-2010", description="e.g., 1981-2010"),
    datasource: str = Query("era5", description="locked to era5 for demo"),
    step: Literal["month", "year"] = Query("month", description="Series step: 'month' or 'year'"),
    yearly_method: Literal["dec31", "aggregate"] = Query("dec31", description="Yearly mode: 'dec31' window or 'aggregate' from monthly SPI"),
    anchor: Optional[str] = Query(None, description="Optional custom anchor date 'YYYY-MM-DD' (uses only MM-DD) for yearly dec31 method"),
):
    """
    Return SPI series using Gamma-fit (with zero handling) at chosen step:
      - step='month': value per month (window ends on month-end) between start_date and end_date.
      - step='year', yearly_method='dec31': one value per year with window ending at Dec-31 (or custom anchor MM-DD if provided).
      - step='year', yearly_method='aggregate': compute monthly SPI first, then aggregate by year (mean/min and drought counts).
    """
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: ERA5 only for SPI series."})

    try:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid start_date/end_date. Use YYYY-MM-DD."})
    if end < start:
        raise HTTPException(status_code=400, detail={"message": "end_date must be >= start_date."})

    win = int(window_days)

    # Fetch observations covering all needed windows
    fetch_start = start - dt.timedelta(days=win - 1)
    obs_all = _fetch_power_precip(lat, lon, fetch_start, end)

    # Fetch baseline once (full years)
    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_power_precip(lat, lon, dt.date(by0, 1, 1), dt.date(by1, 12, 31))

    def _spi_for_end_date(ed: dt.date) -> Dict[str, Optional[float]]:
        s = ed - dt.timedelta(days=win - 1)
        obs_sum = _sum_window(s, ed, obs_all)
        bl_sums = _aligned_baseline_window_sums(ed, win, base_all, by0, by1)
        # Gamma with zero handling, fallback to z-score when needed
        note = "gamma"
        p0 = shape = scale = Gx = H = None
        if len(bl_sums) < 5:
            try:
                sd = stdev(bl_sums)
            except Exception:
                sd = 0.0
            if sd == 0:
                return {
                    "obs_sum_mm": round(obs_sum, 3),
                    "spi": None,
                    "method": "undefined-baseline",
                    "p_zero": None,
                    "gamma_shape": None,
                    "gamma_scale": None,
                    "cdf_gamma": None,
                    "cdf_mixed": None,
                }
            mu = mean(bl_sums)
            spi_val = (obs_sum - mu) / sd
            return {
                "obs_sum_mm": round(obs_sum, 3),
                "spi": round(spi_val, 3),
                "method": "zscore-fallback",
                "p_zero": None,
                "gamma_shape": None,
                "gamma_scale": None,
                "cdf_gamma": None,
                "cdf_mixed": None,
            }
        zeros = [x for x in bl_sums if x <= 0.0]
        pos = [x for x in bl_sums if x > 0.0]
        n = len(bl_sums)
        p0 = len(zeros) / n
        if len(pos) < 3:
            try:
                sd = stdev(bl_sums)
            except Exception:
                sd = 0.0
            if sd == 0:
                return {
                    "obs_sum_mm": round(obs_sum, 3),
                    "spi": None,
                    "method": "undefined-baseline",
                    "p_zero": round(p0, 3),
                    "gamma_shape": None,
                    "gamma_scale": None,
                    "cdf_gamma": None,
                    "cdf_mixed": None,
                }
            mu = mean(bl_sums)
            spi_val = (obs_sum - mu) / sd
            return {
                "obs_sum_mm": round(obs_sum, 3),
                "spi": round(spi_val, 3),
                "method": "zscore-fallback",
                "p_zero": round(p0, 3),
                "gamma_shape": None,
                "gamma_scale": None,
                "cdf_gamma": None,
                "cdf_mixed": None,
            }
        # Gamma fit
        shape, loc, scale = gamma_dist.fit(pos, floc=0)
        Gx = float(gamma_dist.cdf(max(obs_sum, 0.0), a=shape, loc=0, scale=scale))
        H = p0 + (1.0 - p0) * Gx
        H = min(max(H, 1e-10), 1.0 - 1e-10)
        spi_val = float(norm.ppf(H))
        return {
            "obs_sum_mm": round(obs_sum, 3),
            "spi": round(spi_val, 3),
            "method": note,
            "p_zero": round(p0, 3),
            "gamma_shape": round(shape, 6),
            "gamma_scale": round(scale, 6),
            "cdf_gamma": round(Gx, 6),
            "cdf_mixed": round(H, 6),
        }

    if step == "month":
        month_ends = _month_end_dates(start, end)
        series: List[Dict] = []
        for ed in month_ends:
            out = _spi_for_end_date(ed)
            series.append({
                "year": ed.year,
                "month": ed.month,
                "end_date": ed.isoformat(),
                **out,
            })
        return {
            "method": "gamma-series-month",
            "datasource": datasource,
            "lat": lat, "lon": lon,
            "window_days": win,
            "baseline": f"{by0}-{by1}",
            "step": step,
            "count": len(series),
            "series": series,
            "note": "SPI per month using window ending at each month-end between start and end.",
        }

    # step == 'year'
    if yearly_method == "aggregate":
        # 1) compute monthly series across range
        month_ends = _month_end_dates(start, end)
        per_year: DefaultDict[int, List[float]] = defaultdict(list)
        per_year_all: DefaultDict[int, List[Dict]] = defaultdict(list)
        for ed in month_ends:
            out = _spi_for_end_date(ed)
            if out["spi"] is not None:
                per_year[ed.year].append(out["spi"])  # SPI value only
            per_year_all[ed.year].append({"end_date": ed.isoformat(), **out})
        # 2) aggregate by year
        annual: List[Dict] = []
        for y in range(start.year, end.year + 1):
            months = per_year_all.get(y, [])
            vals = per_year.get(y, [])
            if len(months) == 0:
                continue
            mean_spi = round(sum(vals) / len(vals), 3) if len(vals) else None
            min_spi = round(min(vals), 3) if len(vals) else None
            counts = {
                "lt_-1.0": sum(1 for v in vals if v < -1.0),
                "lt_-1.5": sum(1 for v in vals if v < -1.5),
                "lt_-2.0": sum(1 for v in vals if v < -2.0),
            }
            annual.append({
                "year": y,
                "months": months,
                "mean_spi": mean_spi,
                "min_spi": min_spi,
                "drought_counts": counts,
            })
        return {
            "method": "gamma-series-year-aggregate",
            "datasource": datasource,
            "lat": lat, "lon": lon,
            "window_days": win,
            "baseline": f"{by0}-{by1}",
            "step": "year",
            "yearly_method": yearly_method,
            "count": len(annual),
            "series": annual,
            "note": "Yearly aggregation from monthly SPI: mean/min and drought-month counts.",
        }

    # yearly_method == 'dec31' (or custom anchor)
    if anchor:
        try:
            anc = dt.date.fromisoformat(anchor)
            anchor_mmdd = (anc.month, anc.day)
        except Exception:
            raise HTTPException(status_code=400, detail={"message": "Invalid anchor. Use YYYY-MM-DD (MM-DD part used)."})
    else:
        anchor_mmdd = (12, 31)

    years = list(range(start.year, end.year + 1))
    year_ends: List[dt.date] = []
    for y in years:
        ed = _anchor_for_year(anchor_mmdd, y)
        if ed < start or ed > end:
            continue
        year_ends.append(ed)
    if not year_ends:
        raise HTTPException(status_code=400, detail={"message": "No anchor dates inside the range for yearly series."})

    series: List[Dict] = []
    for ed in year_ends:
        out = _spi_for_end_date(ed)
        series.append({
            "year": ed.year,
            "end_date": ed.isoformat(),
            **out,
        })

    return {
        "method": "gamma-series-year-anchor",
        "datasource": datasource,
        "lat": lat, "lon": lon,
        "window_days": win,
        "baseline": f"{by0}-{by1}",
        "step": "year",
        "yearly_method": yearly_method,
        "anchor": f"{anchor_mmdd[0]:02d}-{anchor_mmdd[1]:02d}",
        "count": len(series),
        "series": series,
        "note": "One SPI per year; window ends on custom anchor (default Dec-31).",
    }


# ---------------------- Disabled (demo) routes ---------------------
@app.get("/indices/spei")
def spei_disabled():
    if DEMO_LOCK:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: SPEI endpoint is disabled. SPI-only is enabled for historical+point+ERA5.",
            "enable_hint": "Set DEMO_LOCK=0 in environment to enable full endpoints (once implemented)."
        })
    raise HTTPException(status_code=501, detail="SPEI not implemented yet.")


@app.get("/forecast/next7")
def fcst_next7_disabled():
    if DEMO_LOCK:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: forecast endpoints are disabled.",
            "enable_hint": "Set DEMO_LOCK=0 to enable once forecasts are ready."
        })
    raise HTTPException(status_code=501, detail="Forecasts not implemented yet.")
