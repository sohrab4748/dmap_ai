"""
DMAP-AI API (demo-locked)
Adds SPI monthly & yearly Gamma series and convenience endpoints.

Endpoints
---------
• GET /                               — info
• GET /health                         — health
• GET /indices/spi                    — z-score SPI from user-provided sum/mean/std
• GET /demo/spi_historical_auto       — legacy z-score SPI using NASA POWER
• GET /demo/spi_gamma_historical_auto — single Gamma SPI on trailing window
• GET /demo/spi_gamma_series          — SPI series (step=month|year; yearly_method=total|window)
• GET /demo/spi_gamma_series_monthly  — Convenience (calls series with step=month)
• GET /demo/spi_gamma_series_yearly   — Convenience (calls series with step=year)
• GET /demo/spi_gamma_both            — Returns {monthly, yearly}
• GET /indices/spei, /forecast/next7  — disabled in demo

Notes
-----
• For series items we now emit **window_sum_mm**, and also include aliases
  **total_mm** and **obs_sum_mm** for frontend compatibility.
• Yearly “total” means Jan–Dec annual precipitation totals (no window_days sent).
• Yearly “window” uses a rolling N-day trailing window ending on an anchor date
  in each year (N is capped at 120 for demo performance).
"""

import os
import calendar
import datetime as dt
from statistics import mean, stdev
from typing import Optional, Tuple, Dict, List, Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# SciPy for Gamma CDF and Normal inverse CDF
from scipy.stats import gamma as gamma_dist, norm

app = FastAPI(title="DMAP-AI API", version="0.8.0")

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

# --------------------------- Root/Health ----------------------------
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
    # NOTE: For demo we use NASA POWER daily precip. UI may show 'ERA5' but demo fetches POWER.
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
    return {k: float(v) for k, v in param.items()}

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

# --------------- Extra helpers for monthly & yearly SPI --------------

def _last_day_of_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def _month_iter(start: dt.date, end: dt.date):
    """Yield (year, month) pairs between two dates inclusive."""
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        yield (y, m)
        if m == 12:
            y += 1; m = 1
        else:
            m += 1

def _annual_sum(year: int, daily_map: Dict[str, float]) -> float:
    s = dt.date(year, 1, 1)
    e = dt.date(year, 12, 31)
    return _sum_window(s, e, daily_map)

def _fit_gamma_from_samples(samples: List[float]):
    """Return (p_zero, shape, scale, pos_samples, zero_samples). floc=0 for shape/scale."""
    zeros = [x for x in samples if x <= 0.0]
    pos   = [x for x in samples if x > 0.0]
    n = len(samples)
    p0 = (len(zeros) / n) if n else 0.0
    if len(pos) >= 3:
        shape, loc, scale = gamma_dist.fit(pos, floc=0)
        return p0, float(shape), float(scale), pos, zeros
    else:
        return p0, None, None, pos, zeros

def _gamma_spi_for_value(x: float, p0: float, shape: Optional[float], scale: Optional[float]):
    """Return (spi, Gx, H). If insufficient shape/scale, returns (None, None, None)."""
    if shape is None or scale is None:
        return None, None, None
    Gx = float(gamma_dist.cdf(max(x, 0.0), a=shape, loc=0, scale=scale))
    H  = p0 + (1.0 - p0) * Gx
    H  = min(max(H, 1e-10), 1.0 - 1e-10)
    spi_val = float(norm.ppf(H))
    return spi_val, Gx, H

def _zscore(x: float, samples: List[float]) -> Optional[float]:
    if len(samples) < 2:
        return None
    try:
        mu = mean(samples)
        sd = stdev(samples)
    except Exception:
        return None
    if sd == 0:
        return None
    return (x - mu) / sd

def _spi_category(spi: Optional[float]) -> str:
    if spi is None:
        return "Undefined"
    if spi <= -2.0: return "Extreme drought"
    if spi <= -1.5: return "Severe drought"
    if spi <= -1.0: return "Moderate drought"
    if spi <  1.0:  return "Near normal"
    if spi <  1.5:  return "Moderately wet"
    if spi <  2.0:  return "Very wet"
    return "Extremely wet"

def _add_sum_aliases(d: Dict, val: float):
    """Ensure all expected sum keys are present for frontend compatibility."""
    d["window_sum_mm"] = round(val, 3)
    d["obs_sum_mm"] = round(val, 3)
    d["total_mm"] = round(val, 3)

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
    base_all = _fetch_power_precip(lat, lon, dt.date(by0,1,1), dt.date(by1,12,31))
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
    base_all = _fetch_power_precip(lat, lon, dt.date(by0,1,1), dt.date(by1,12,31))
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

# ============== SPI SERIES (MONTHLY or YEARLY) ======================
@app.get("/demo/spi_gamma_series")
def spi_gamma_series(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    window_days: int = Query(30, ge=7, le=120, description="Used only for yearly_method='window'"),
    baseline: str = Query("1981-2010", description="e.g., 1981-2010"),
    datasource: str = Query("era5", description="demo-locked to ERA5"),
    step: Literal["month","year"] = Query("year", description="Aggregate by 'month' or 'year'"),
    yearly_method: Literal["total","window"] = Query("total", description="'total' = Jan–Dec totals, 'window' = trailing window ending on anchor"),
    anchor_mm: int = Query(12, ge=1, le=12, description="If yearly_method='window', end-month of window (default Dec)"),
    anchor_dd: int = Query(31, ge=1, le=31, description="If yearly_method='window', end-day of window (clamped)"),
):
    # Demo lock: only ERA5 in demo
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: ERA5 only for SPI series."})

    try:
        start = dt.date.fromisoformat(start_date)
        end   = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid start_date/end_date. Use YYYY-MM-DD."})
    if end < start:
        raise HTTPException(status_code=400, detail={"message": "end_date must be >= start_date."})

    # Fetch observations & baseline once
    obs_all  = _fetch_power_precip(lat, lon, dt.date(start.year, 1, 1), dt.date(end.year, 12, 31))
    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_power_precip(lat, lon, dt.date(by0,1,1), dt.date(by1,12,31))

    if step == "month":
        # Baseline monthly totals per calendar month
        baseline_month_samples = {m: [] for m in range(1,13)}
        for y in range(by0, by1+1):
            for m in range(1,13):
                s = dt.date(y, m, 1)
                e = dt.date(y, m, _last_day_of_month(y,m))
                baseline_month_samples[m].append(_sum_window(s, e, base_all))

        # Fit Gamma per month
        month_params = {}
        for m in range(1,13):
            p0, shape, scale, pos, zeros = _fit_gamma_from_samples(baseline_month_samples[m])
            month_params[m] = {"p_zero": p0, "shape": shape, "scale": scale, "n": len(baseline_month_samples[m])}

        # Build series within requested range (full months only)
        series = []
        for (y, m) in _month_iter(start, end):
            s = dt.date(y, m, 1)
            e = dt.date(y, m, _last_day_of_month(y,m))
            if (y == start.year and m == start.month and start != s) or (y == end.year and m == end.month and end != e):
                continue
            total = _sum_window(s, e, obs_all)
            params = month_params[m]
            spi_val, Gx, H = _gamma_spi_for_value(total, params["p_zero"], params["shape"], params["scale"])
            if spi_val is None:
                z = _zscore(total, baseline_month_samples[m])
                spi_val = None if z is None else float(z)
                Gx = H = None
            item = {
                "year": y,
                "month": m,
                "end_date": e.isoformat(),
                "spi": None if spi_val is None else round(spi_val, 3),
                "category": _spi_category(spi_val),
                "gamma_shape": None if params["shape"] is None else round(params["shape"], 6),
                "gamma_scale": None if params["scale"] is None else round(params["scale"], 6),
                "p_zero": round(params["p_zero"], 3),
                "cdf_gamma": None if Gx is None else round(Gx, 6),
                "cdf_mixed": None if H  is None else round(H, 6),
            }
            _add_sum_aliases(item, total)
            series.append(item)

        return {
            "method": "gamma-monthly",
            "datasource": datasource,
            "lat": lat, "lon": lon,
            "baseline": f"{by0}-{by1}",
            "count": len(series),
            "series": series,
            "note": "SPI per calendar month; gamma parameters estimated per month from baseline monthly totals.",
        }

    # YEARLY
    years = list(range(start.year, end.year + 1))

    if yearly_method == "total":
        # Annual Jan–Dec totals
        baseline_annual = [_annual_sum(y, base_all) for y in range(by0, by1+1)]
        p0, shape, scale, pos, zeros = _fit_gamma_from_samples(baseline_annual)
        yearly_params = {
            "p_zero": round(p0, 3),
            "gamma_shape": None if shape is None else round(shape, 6),
            "gamma_scale": None if scale is None else round(scale, 6),
            "n_baseline": len(baseline_annual),
        }
        series = []
        for y in years:
            total = _annual_sum(y, obs_all)
            spi_val, Gx, H = _gamma_spi_for_value(total, p0, shape, scale)
            if spi_val is None:
                z = _zscore(total, baseline_annual)
                spi_val = None if z is None else float(z)
                Gx = H = None
            item = {
                "year": y,
                "end_date": dt.date(y,12,31).isoformat(),
                "spi": None if spi_val is None else round(spi_val, 3),
                "category": _spi_category(spi_val),
                "cdf_gamma": None if Gx is None else round(Gx, 6),
                "cdf_mixed": None if H  is None else round(H, 6),
            }
            _add_sum_aliases(item, total)
            series.append(item)
        return {
            "method": "gamma-yearly-total",
            "datasource": datasource,
            "lat": lat, "lon": lon,
            "baseline": f"{by0}-{by1}",
            "yearly_params": yearly_params,
            "count": len(series),
            "series": series,
            "note": "SPI per calendar year using annual (Jan–Dec) totals; one gamma fit on baseline annual totals.",
        }

    # Yearly trailing window (legacy-compatible)
    win = int(window_days)
    series = []
    for y in years:
        last_day = calendar.monthrange(y, anchor_mm)[1]
        dd = min(anchor_dd, last_day)
        ed = dt.date(y, anchor_mm, dd)
        st = ed - dt.timedelta(days=win - 1)
        total = _sum_window(st, ed, obs_all)
        bl_sums = _aligned_baseline_window_sums(ed, win, base_all, by0, by1)
        p0, shape, scale, pos, zeros = _fit_gamma_from_samples(bl_sums)
        spi_val, Gx, H = _gamma_spi_for_value(total, p0, shape, scale)
        if spi_val is None:
            z = _zscore(total, bl_sums)
            spi_val = None if z is None else float(z)
            Gx = H = None
        item = {
            "year": y,
            "end_date": ed.isoformat(),
            "window_days": win,
            "spi": None if spi_val is None else round(spi_val, 3),
            "category": _spi_category(spi_val),
            "gamma_shape": None if shape is None else round(shape, 6),
            "gamma_scale": None if scale is None else round(scale, 6),
            "p_zero": round(p0, 3),
            "cdf_gamma": None if Gx is None else round(Gx, 6),
            "cdf_mixed": None if H  is None else round(H, 6),
        }
        _add_sum_aliases(item, total)
        series.append(item)

    return {
        "method": "gamma-yearly-window",
        "datasource": datasource,
        "lat": lat, "lon": lon,
        "baseline": f"{by0}-{by1}",
        "anchor_mm": anchor_mm,
        "anchor_dd": anchor_dd,
        "window_days": win,
        "count": len(series),
        "series": series,
        "note": "SPI per year using trailing window ending on anchor; gamma fit computed per year from baseline-aligned windows.",
    }

# ============== Convenience endpoints (avoid 404s) ==================
@app.get("/demo/spi_gamma_series_monthly")
def spi_gamma_series_monthly(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
):
    return spi_gamma_series(lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                            window_days=30, baseline=baseline, datasource=datasource,
                            step="month", yearly_method="total")

@app.get("/demo/spi_gamma_series_yearly")
def spi_gamma_series_yearly(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
    yearly_method: Literal["total","window"] = Query("total"),
    window_days: int = Query(30, ge=7, le=120),
    anchor: str = Query("12-31"),
):
    # Parse anchor "MM-DD"
    try:
        mm, dd = anchor.split("-")
        anchor_mm = int(mm); anchor_dd = int(dd)
    except Exception:
        anchor_mm, anchor_dd = 12, 31
    return spi_gamma_series(lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                            window_days=window_days, baseline=baseline, datasource=datasource,
                            step="year", yearly_method=yearly_method,
                            anchor_mm=anchor_mm, anchor_dd=anchor_dd)

# ============== Convenience: get BOTH monthly & yearly ==============
@app.get("/demo/spi_gamma_both")
def spi_gamma_both(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
):
    monthly = spi_gamma_series(lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                               window_days=30, baseline=baseline, datasource=datasource,
                               step="month", yearly_method="total")  # yearly_method ignored for month
    yearly  = spi_gamma_series(lat=lat, lon=lon, start_date=start_date, end_date=end_date,
                               window_days=30, baseline=baseline, datasource=datasource,
                               step="year",  yearly_method="total")
    return {"monthly": monthly, "yearly": yearly}

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
