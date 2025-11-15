import os
import calendar
import datetime as dt
from statistics import mean, stdev
from typing import Optional, Tuple, Dict, List, Literal

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pywt
from pydantic import BaseModel
from scipy.stats import gamma as gamma_dist, norm

app = FastAPI(title="DMAP-AI API", version="0.9.2")

# ----------------------------- CORS ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Demo mode: only 'historical' is enabled.",
                "allowed": {
                    "mode": ALLOWED_MODE,
                    "aoi": ALLOWED_AOI,
                    "datasource": ALLOWED_DATASOURCE,
                },
            },
        )
    if aoi is not None and aoi.lower() != ALLOWED_AOI:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Demo mode: only 'point' AOI is enabled.",
                "allowed": {
                    "mode": ALLOWED_MODE,
                    "aoi": ALLOWED_AOI,
                    "datasource": ALLOWED_DATASOURCE,
                },
            },
        )
    if datasource is not None and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Demo mode: only one datasource is enabled.",
                "allowed": {
                    "mode": ALLOWED_MODE,
                    "aoi": ALLOWED_AOI,
                    "datasource": ALLOWED_DATASOURCE,
                },
            },
        )


# --------------------------- Root/Health ----------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "message": "DMAP-AI API. See /health and /docs",
        "demo_lock": DEMO_LOCK,
    }


@app.get("/health")
def health():
    return {"ok": True, "demo_lock": DEMO_LOCK}


# -------------------------- SPI (simple) ----------------------------
@app.get("/indices/spi")
def spi(
    sum_rain_mm: float = Query(...),
    clim_mean_mm: float = Query(...),
    clim_std_mm: float = Query(..., gt=0),
    window_days: int = Query(30, ge=1, le=365),
    mode: Optional[str] = Query(None),
    aoi: Optional[str] = Query(None),
    datasource: Optional[str] = Query(None),
):
    _enforce_demo(mode, aoi, datasource)
    spi_value = (sum_rain_mm - clim_mean_mm) / clim_std_mm
    return {
        "window_days": window_days,
        "sum_mm": sum_rain_mm,
        "clim_mean_mm": clim_mean_mm,
        "clim_std_mm": clim_std_mm,
        "spi": round(spi_value, 3),
        "datasource": datasource,
    }


# ------------------- Common helpers / POWER backend -----------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAM = "PRECTOTCORR"  # mm/day


def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _parse_baseline(baseline: str) -> Tuple[int, int]:
    try:
        a, b = baseline.split("-")
        return int(a), int(b)
    except Exception:
        return 1981, 2010


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
        raise HTTPException(
            status_code=502,
            detail={"message": "NASA POWER error", "status": r.status_code, "text": r.text[:300]},
        )
    j = r.json()
    param = (
        j.get("properties", {})
        .get("parameter", {})
        .get(POWER_PARAM)
        or j.get("parameters", {}).get(POWER_PARAM)
    )
    if not isinstance(param, dict):
        raise HTTPException(
            status_code=502,
            detail={"message": "Unexpected NASA POWER payload", "sample_keys": list(j.keys())[:5]},
        )
    return {k: float(v) for k, v in param.items()}


# ---------------------- Datasource routing --------------------------
SUPPORTED_GRID_DATASOURCES = {"era5", "gridmet", "prism", "gpm"}
USER_DATASOURCE = "user"


def _resolve_datasource(datasource: Optional[str]) -> str:
    ds = (datasource or "era5").lower()
    if ds in SUPPORTED_GRID_DATASOURCES:
        return ds
    if ds == USER_DATASOURCE:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "User-upload ('user') datasource is not implemented on the server yet. "
                "The current backend only supports gridded sources. You must upload / preprocess "
                "station CSVs separately and feed them via a dedicated endpoint."
            },
        )
    raise HTTPException(
        status_code=400,
        detail={
            "message": f"Unknown datasource '{datasource}'. "
            "Supported: era5, gridmet, prism, gpm, user."
        },
    )


def _fetch_precip_era5(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, float]:
    """Placeholder ERA5 implementation.

    Right now this just calls NASA POWER so that the API keeps working.
    To use real ERA5, replace this with something that reads your own
    preprocessed ERA5 daily precip (e.g., from NetCDF on disk).
    """
    return _fetch_power_precip(lat, lon, start, end)


def _fetch_precip_prism(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, float]:
    """PRISM placeholder.

    IMPLEMENT ME: This should read PRISM daily precip from your local files or
    your own PRISM server. For now, we raise a clear error so you notice it.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "message": "PRISM datasource is not wired yet. "
            "Implement _fetch_precip_prism() to read your PRISM data.",
        },
    )


def _fetch_precip_gridmet(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, float]:
    """GridMET placeholder.

    IMPLEMENT ME: This should read GridMET precip from local NetCDF/THREDDS etc.
    """
    raise HTTPException(
        status_code=501,
        detail={
            "message": "GridMET datasource is not wired yet. "
            "Implement _fetch_precip_gridmet() to read your GridMET data.",
        },
    )


def _fetch_precip_gpm(lat: float, lon: float, start: dt.date, end: dt.date) -> Dict[str, float]:
    """GPM placeholder.

    IMPLEMENT ME: This should read GPM IMERG precip (e.g., daily accumulations).
    """
    raise HTTPException(
        status_code=501,
        detail={
            "message": "GPM datasource is not wired yet. "
            "Implement _fetch_precip_gpm() to read your GPM data.",
        },
    )


def _fetch_precip_by_source(
    lat: float,
    lon: float,
    start: dt.date,
    end: dt.date,
    datasource: str,
) -> Dict[str, float]:
    ds = _resolve_datasource(datasource)
    if ds == "era5":
        return _fetch_precip_era5(lat, lon, start, end)
    if ds == "prism":
        return _fetch_precip_prism(lat, lon, start, end)
    if ds == "gridmet":
        return _fetch_precip_gridmet(lat, lon, start, end)
    if ds == "gpm":
        return _fetch_precip_gpm(lat, lon, start, end)
    raise HTTPException(status_code=500, detail={"message": "Unhandled datasource routing."})


# ---------------- SPI helpers (gamma / zscore) ----------------------
def _sum_window(s: dt.date, e: dt.date, dct: Dict[str, float]) -> float:
    total = 0.0
    cur = s
    while cur <= e:
        total += dct.get(_yyyymmdd(cur), 0.0)
        cur += dt.timedelta(days=1)
    return total


def _aligned_baseline_window_sums(
    end: dt.date, win: int, base_all: Dict[str, float], by0: int, by1: int
) -> List[float]:
    sums: List[float] = []
    for yr in range(by0, by1 + 1):
        try:
            last_day = (dt.date(yr, end.month, 1) + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(
                days=1
            )
            end_y = dt.date(yr, end.month, min(end.day, last_day.day))
        except Exception:
            end_y = dt.date(yr, end.month, 28) if end.month == 2 else dt.date(yr, end.month, 1)
        start_y = end_y - dt.timedelta(days=win - 1)
        sums.append(_sum_window(start_y, end_y, base_all))
    return sums


def _last_day_of_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def _month_iter(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1


def _annual_sum(year: int, daily_map: Dict[str, float]) -> float:
    s = dt.date(year, 1, 1)
    e = dt.date(year, 12, 31)
    return _sum_window(s, e, daily_map)


def _fit_gamma_from_samples(samples: List[float]):
    zeros = [x for x in samples if x <= 0.0]
    pos = [x for x in samples if x > 0.0]
    n = len(samples)
    p0 = (len(zeros) / n) if n else 0.0
    if len(pos) >= 3:
        shape, loc, scale = gamma_dist.fit(pos, floc=0)
        return p0, float(shape), float(scale), pos, zeros
    else:
        return p0, None, None, pos, zeros


def _gamma_spi_for_value(x: float, p0: float, shape: Optional[float], scale: Optional[float]):
    if shape is None or scale is None:
        return None, None, None
    Gx = float(gamma_dist.cdf(max(x, 0.0), a=shape, loc=0, scale=scale))
    H = p0 + (1.0 - p0) * Gx
    H = min(max(H, 1e-10), 1.0 - 1e-10)
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
    if spi <= -2.0:
        return "Extreme drought"
    if spi <= -1.5:
        return "Severe drought"
    if spi <= -1.0:
        return "Moderate drought"
    if spi < 1.0:
        return "Near normal"
    if spi < 1.5:
        return "Moderately wet"
    if spi < 2.0:
        return "Very wet"
    return "Extremely wet"


# ----------------- Wavelet helpers & request model ------------------
def _interp_nan_to_mean(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(float)
    n = a.size
    idx = np.arange(n)
    mask = np.isfinite(a)
    if mask.sum() < 2:
        return np.zeros_like(a)
    a[~mask] = np.interp(idx[~mask], idx[mask], a[mask])
    return a


class WaveletRequest(BaseModel):
    spi: List[float]
    precip: Optional[List[float]] = None
    dt: float = 1.0
    min_period: int = 1
    max_period: int = 24
    n_scales: int = 10


# ------------------- SPI (auto, z-score / legacy) -------------------
@app.get("/demo/spi_historical_auto")
def spi_historical_auto(
    lat: float = Query(...),
    lon: float = Query(...),
    end_date: str = Query(...),
    window_days: int = Query(30, ge=7, le=120),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
):
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: only one datasource is enabled."})

    try:
        end = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid end_date. Use YYYY-MM-DD."})

    win = max(1, min(window_days, 120))
    start = end - dt.timedelta(days=win - 1)

    obs = _fetch_precip_by_source(lat, lon, start, end, datasource)
    sum_obs = _sum_window(start, end, obs)

    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_precip_by_source(lat, lon, dt.date(by0, 1, 1), dt.date(by1, 12, 31), datasource)
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
        "lat": lat,
        "lon": lon,
        "window_days": win,
        "end_date": end.isoformat(),
        "baseline": f"{by0}-{by1}",
        "obs_sum_mm": round(sum_obs, 3),
        "baseline_mean_mm": round(mu, 3),
        "baseline_std_mm": round(sd, 3),
        "spi": round(spi_z, 3),
    }


# --------- SPI (auto, Gamma fit with zero-precip adjustment) --------
@app.get("/demo/spi_gamma_historical_auto")
def spi_gamma_historical_auto(
    lat: float = Query(...),
    lon: float = Query(...),
    end_date: str = Query(...),
    window_days: int = Query(30, ge=7, le=120),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
):
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: only one datasource is enabled."})

    try:
        end = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid end_date. Use YYYY-MM-DD."})

    win = max(1, min(window_days, 120))
    start = end - dt.timedelta(days=win - 1)

    obs = _fetch_precip_by_source(lat, lon, start, end, datasource)
    sum_obs = _sum_window(start, end, obs)

    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_precip_by_source(lat, lon, dt.date(by0, 1, 1), dt.date(by1, 12, 31), datasource)
    baseline_sums = _aligned_baseline_window_sums(end, win, base_all, by0, by1)

    if len(baseline_sums) < 5:
        raise HTTPException(status_code=502, detail={"message": "Insufficient baseline length for Gamma fit."})

    zeros = [s for s in baseline_sums if s <= 0.0]
    pos = [s for s in baseline_sums if s > 0.0]
    n = len(baseline_sums)
    p0 = len(zeros) / n

    if len(pos) < 3:
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
            "lat": lat,
            "lon": lon,
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

    shape, loc, scale = gamma_dist.fit(pos, floc=0)
    Gx = float(gamma_dist.cdf(max(sum_obs, 0.0), a=shape, loc=0, scale=scale))
    H = p0 + (1.0 - p0) * Gx
    H = min(max(H, 1e-10), 1.0 - 1e-10)
    spi_val = float(norm.ppf(H))

    return {
        "method": "gamma",
        "datasource": datasource,
        "lat": lat,
        "lon": lon,
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
    }


# ================= SPI SERIES (MONTHLY or YEARLY) ===================
@app.get("/demo/spi_gamma_series")
def spi_gamma_series(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
    window_days: Optional[int] = Query(None),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
    step: Literal["month", "year"] = Query("year"),
    yearly_method: Literal["total", "window"] = Query("total"),
    anchor_mm: int = Query(12, ge=1, le=12),
    anchor_dd: int = Query(31, ge=1, le=31),
    anchor: Optional[str] = Query(None),
):
    if DEMO_LOCK and datasource.lower() != ALLOWED_DATASOURCE:
        raise HTTPException(status_code=403, detail={"message": "Demo mode: only one datasource is enabled."})

    try:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)
    except Exception:
        raise HTTPException(status_code=400, detail={"message": "Invalid start_date/end_date. Use YYYY-MM-DD."})
    if end < start:
        raise HTTPException(status_code=400, detail={"message": "end_date must be >= start_date."})

    if anchor:
        try:
            mm_s, dd_s = anchor.split("-")
            anchor_mm = max(1, min(12, int(mm_s)))
            anchor_dd = max(1, min(31, int(dd_s)))
        except Exception:
            pass

    # Pull full daily series for analysis and baseline spans
    obs_all = _fetch_precip_by_source(lat, lon, dt.date(start.year, 1, 1), dt.date(end.year, 12, 31), datasource)
    by0, by1 = _parse_baseline(baseline)
    base_all = _fetch_precip_by_source(lat, lon, dt.date(by0, 1, 1), dt.date(by1, 12, 31), datasource)

    # ---------- MONTHLY ----------
    if step == "month":
        baseline_month_samples: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
        for y in range(by0, by1 + 1):
            for m in range(1, 13):
                s = dt.date(y, m, 1)
                e = dt.date(y, m, _last_day_of_month(y, m))
                baseline_month_samples[m].append(_sum_window(s, e, base_all))

        month_params: Dict[int, Dict] = {}
        for m in range(1, 13):
            p0, shape, scale, pos, zeros = _fit_gamma_from_samples(baseline_month_samples[m])
            month_params[m] = {
                "p_zero": p0,
                "shape": shape,
                "scale": scale,
                "n": len(baseline_month_samples[m]),
            }

        series: List[Dict] = []
        for y, m in _month_iter(start, end):
            s = dt.date(y, m, 1)
            e = dt.date(y, m, _last_day_of_month(y, m))
            if (y == start.year and m == start.month and start != s) or (y == end.year and m == end.month and end != e):
                continue
            tot = _sum_window(s, e, obs_all)
            params = month_params[m]
            spi_val, Gx, H = _gamma_spi_for_value(tot, params["p_zero"], params["shape"], params["scale"])
            if spi_val is None:
                z = _zscore(tot, baseline_month_samples[m])
                spi_val = None if z is None else float(z)
                Gx = H = None

            ws = round(tot, 3)
            series.append({
                "year": y,
                "month": m,
                "end_date": e.isoformat(),
                "window_sum_mm": ws,
                "total_mm": ws,
                "obs_sum_mm": ws,
                "spi": None if spi_val is None else round(spi_val, 3),
                "category": _spi_category(spi_val),
                "gamma_shape": None if params["shape"] is None else round(params["shape"], 6),
                "gamma_scale": None if params["scale"] is None else round(params["scale"], 6),
                "p_zero": round(params["p_zero"], 3),
                "cdf_gamma": None if Gx is None else round(Gx, 6),
                "cdf_mixed": None if H is None else round(H, 6),
            })

        return {
            "method": "gamma-monthly",
            "datasource": datasource,
            "lat": lat,
            "lon": lon,
            "baseline": f"{by0}-{by1}",
            "count": len(series),
            "series": series,
        }

    # ---------- YEARLY ----------
    years = list(range(start.year, end.year + 1))

    if yearly_method == "total":
        baseline_annual = [_annual_sum(y, base_all) for y in range(by0, by1 + 1)]
        p0, shape, scale, pos, zeros = _fit_gamma_from_samples(baseline_annual)

        series: List[Dict] = []
        for y in years:
            tot = _annual_sum(y, obs_all)
            spi_val, Gx, H = _gamma_spi_for_value(tot, p0, shape, scale)
            if spi_val is None:
                z = _zscore(tot, baseline_annual)
                spi_val = None if z is None else float(z)
                Gx = H = None

            ws = round(tot, 3)
            series.append({
                "year": y,
                "end_date": dt.date(y, 12, 31).isoformat(),
                "window_sum_mm": ws,
                "total_mm": ws,
                "obs_sum_mm": ws,
                "spi": None if spi_val is None else round(spi_val, 3),
                "category": _spi_category(spi_val),
                "cdf_gamma": None if Gx is None else round(Gx, 6),
                "cdf_mixed": None if H is None else round(H, 6),
            })

        return {
            "method": "gamma-yearly-total",
            "datasource": datasource,
            "lat": lat,
            "lon": lon,
            "baseline": f"{by0}-{by1}",
            "count": len(series),
            "series": series,
        }

    # yearly_method == "window"
    win = window_days if window_days is not None else 30
    if win < 7 or win > 120:
        raise HTTPException(
            status_code=422,
            detail=[{
                "type": "less_than_equal",
                "loc": ["query", "window_days"],
                "msg": "For yearly_method='window', window_days must be between 7 and 120",
                "input": str(window_days),
                "ctx": {"ge": 7, "le": 120},
            }],
        )

    series: List[Dict] = []
    for y in years:
        last_day = calendar.monthrange(y, anchor_mm)[1]
        dd = min(anchor_dd, last_day)
        ed = dt.date(y, anchor_mm, dd)
        st = ed - dt.timedelta(days=win - 1)
        tot = _sum_window(st, ed, obs_all)

        bl_sums = _aligned_baseline_window_sums(ed, win, base_all, by0, by1)
        p0, shape, scale, pos, zeros = _fit_gamma_from_samples(bl_sums)
        spi_val, Gx, H = _gamma_spi_for_value(tot, p0, shape, scale)
        if spi_val is None:
            z = _zscore(tot, bl_sums)
            spi_val = None if z is None else float(z)
            Gx = H = None

        ws = round(tot, 3)
        series.append({
            "year": y,
            "end_date": ed.isoformat(),
            "window_days": win,
            "window_sum_mm": ws,
            "total_mm": ws,
            "obs_sum_mm": ws,
            "spi": None if spi_val is None else round(spi_val, 3),
            "category": _spi_category(spi_val),
            "gamma_shape": None if shape is None else round(shape, 6),
            "gamma_scale": None if scale is None else round(scale, 6),
            "p_zero": round(p0, 3),
            "cdf_gamma": None if Gx is None else round(Gx, 6),
            "cdf_mixed": None if H is None else round(H, 6),
        })

    return {
        "method": "gamma-yearly-window",
        "datasource": datasource,
        "lat": lat,
        "lon": lon,
        "baseline": f"{by0}-{by1}",
        "anchor_mm": anchor_mm,
        "anchor_dd": anchor_dd,
        "window_days": win,
        "count": len(series),
        "series": series,
    }


# --------------- Wavelet analysis endpoint --------------------------
@app.post("/analysis/spi_wavelet")
def spi_wavelet(req: WaveletRequest):
    if not req.spi or len(req.spi) < 8:
        raise HTTPException(status_code=400, detail={"message": "Need at least 8 SPI values for wavelet analysis."})

    spi_arr = np.asarray(req.spi, dtype=float)
    spi_arr = _interp_nan_to_mean(spi_arr)

    n_scales = max(1, int(req.n_scales))
    min_p = max(1, int(req.min_period))
    max_p = max(min_p, int(req.max_period))
    scales = np.linspace(min_p, max_p, n_scales)

    coeff_spi, freqs = pywt.cwt(spi_arr, scales, "morl")
    power = (coeff_spi * np.conj(coeff_spi)).real

    scalogram_data = power.T.tolist()
    global_power = power.mean(axis=1).tolist()

    coherence = None
    if req.precip is not None:
        if len(req.precip) != len(req.spi):
            raise HTTPException(
                status_code=400,
                detail={"message": "If provided, 'precip' must have same length as 'spi'."},
            )
        prec_arr = np.asarray(req.precip, dtype=float)
        prec_arr = _interp_nan_to_mean(prec_arr)
        coeff_prec, _ = pywt.cwt(prec_arr, scales, "morl")

        Wxy = coeff_spi * np.conj(coeff_prec)
        Sxx = (coeff_spi * np.conj(coeff_spi)).real
        Syy = (coeff_prec * np.conj(coeff_prec)).real

        num = np.abs(Wxy.mean(axis=1)) ** 2
        den = Sxx.mean(axis=1) * Syy.mean(axis=1)
        coh = np.zeros_like(num)
        valid = den > 0
        coh[valid] = (num[valid] / den[valid]).real
        coh = np.clip(coh, 0.0, 1.0)
        coherence = coh.tolist()

    return {
        "ok": True,
        "n_points": int(spi_arr.size),
        "periods": scales.tolist(),
        "global_power": global_power,
        "scalogram": scalogram_data,
        "coherence": coherence,
    }


# ------------- Convenience endpoints (avoid 404s) -------------------
@app.get("/demo/spi_gamma_series_monthly")
def spi_gamma_series_monthly(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
):
    return spi_gamma_series(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        window_days=30,
        baseline=baseline,
        datasource=datasource,
        step="month",
        yearly_method="total",
    )


@app.get("/demo/spi_gamma_series_yearly")
def spi_gamma_series_yearly(
    lat: float = Query(...),
    lon: float = Query(...),
    start_date: str = Query(...),
    end_date: str = Query(...),
    baseline: str = Query("1981-2010"),
    datasource: str = Query("era5"),
    yearly_method: Literal["total", "window"] = Query("total"),
    window_days: int = Query(30, ge=7, le=120),
    anchor: str = Query("12-31"),
):
    try:
        mm, dd = anchor.split("-")
        anchor_mm = int(mm)
        anchor_dd = int(dd)
    except Exception:
        anchor_mm, anchor_dd = 12, 31

    return spi_gamma_series(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        baseline=baseline,
        datasource=datasource,
        step="year",
        yearly_method=yearly_method,
        anchor_mm=anchor_mm,
        anchor_dd=anchor_dd,
    )


# ---------------------- Disabled placeholders -----------------------
@app.get("/indices/spei")
def spei_disabled():
    if DEMO_LOCK:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Demo mode: SPEI endpoint is disabled.",
                "enable_hint": "Set DEMO_LOCK=0 when SPEI is implemented.",
            },
        )
    raise HTTPException(status_code=501, detail="SPEI not implemented yet.")


@app.get("/forecast/next7")
def fcst_next7_disabled():
    if DEMO_LOCK:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Demo mode: forecast endpoints are disabled.",
                "enable_hint": "Set DEMO_LOCK=0 when forecasts are implemented.",
            },
        )
    raise HTTPException(status_code=501, detail="Forecasts not implemented yet.")
