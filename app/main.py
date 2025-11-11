"""
DMAP‑AI API (demo‑locked)
- CORS enabled for browser calls
- SPI endpoint (/indices/spi): expects P sum/mean/std and returns z‑score SPI
- Auto SPI endpoint (/demo/spi_historical_auto): fetches daily precip from NASA POWER (PRECTOTCORR),
  builds baseline window sums, and returns z‑score SPI (gamma fit can be added later)
- Demo lock restricts to historical + point + ERA5 path

Requirements (requirements.txt):
fastapi
uvicorn[standard]
requests
"""

import os
import datetime as dt
from statistics import mean, stdev
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DMAP-AI API", version="0.4.0")

# ----------------------------- CORS ---------------------------------
# Allow your web origins; keep "*" for demo, then tighten to your domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # e.g., ["https://dmap.agrimetsoft.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- DEMO LOCK ------------------------------
DEMO_LOCK = os.getenv("DEMO_LOCK", "1") == "1"  # set DEMO_LOCK=0 to disable restrictions
ALLOWED_MODE = os.getenv("DEMO_MODE", "historical").lower()
ALLOWED_AOI = os.getenv("DEMO_AOI", "point").lower()
ALLOWED_DATASOURCE = os.getenv("DEMO_DATASOURCE", "era5").lower()


def _enforce_demo(mode: Optional[str], aoi: Optional[str], datasource: Optional[str]):
    """Raise 403 if any provided flag violates the demo path."""
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
    # Optional flags so the UI can pass what the user selected (we enforce them in demo)
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
        "note": "Demo mode enforces historical+point+ERA5. Gamma-fit SPI can be added next.",
    }


# ------------------- SPI (auto from NASA POWER) --------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAM = "PRECTOTCORR"  # daily precipitation (mm/day)


def _yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def _parse_baseline(baseline: str) -> tuple[int, int]:
    """Parse '1981-2010' → (1981, 2010); fallback if malformed."""
    try:
        a, b = baseline.split("-")
        return int(a), int(b)
    except Exception:
        return (1981, 2010)


def _fetch_power_precip(lat: float, lon: float, start: dt.date, end: dt.date) -> dict[str, float]:
    """Return dict YYYYMMDD -> precip(mm/day) from NASA POWER."""
    params = {
        "parameters": POWER_PARAM,
        "community": "AG",
        "latitude": f"{lat}",
        "longitude": f"{lon}",
        "start": _yyyymmdd(start),
        "end": _yyyymmdd(end),
        "format": "JSON",
    }
    r = requests.get(POWER_BASE, params=params, timeout=30)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"message": "NASA POWER error", "status": r.status_code, "text": r.text[:300]})
    j = r.json()
    param = (
        j.get("properties", {}).get("parameter", {}).get(POWER_PARAM)
        or j.get("parameters", {}).get(POWER_PARAM)
    )
    if not isinstance(param, dict):
        raise HTTPException(status_code=502, detail={"message": "Unexpected NASA POWER payload", "sample_keys": list(j.keys())[:5]})
    return {k: float(v) for k, v in param.items()}


def _sum_window(s: dt.date, e: dt.date, dct: dict[str, float]) -> float:
    total = 0.0
    cur = s
    while cur <= e:
        total += dct.get(_yyyymmdd(cur), 0.0)
        cur += dt.timedelta(days=1)
    return total


@app.get("/demo/spi_historical_auto")
def spi_historical_auto(
    lat: float = Query(...),
    lon: float = Query(...),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    window_days: int = Query(30, ge=7, le=120),
    baseline: str = Query("1981-2010", description="e.g., 1981-2010"),
    datasource: str = Query("era5", description="locked to era5 for demo"),
):
    # Demo lock: only ERA5 path is allowed (implicitly historical+point)
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

    # Baseline per-year window sums aligned to the same month/day
    by0, by1 = _parse_baseline(baseline)
    base_start = dt.date(by0, 1, 1)
    base_end = dt.date(by1, 12, 31)
    base_all = _fetch_power_precip(lat, lon, base_start, base_end)

    baseline_sums = []
    for yr in range(by0, by1 + 1):
        # align end to same month/day (clamp Feb 29 → Feb 28 if needed)
        try:
            last_day = (dt.date(yr, end.month, 1) + dt.timedelta(days=31)).replace(day=1) - dt.timedelta(days=1)
            end_y = dt.date(yr, end.month, min(end.day, last_day.day))
        except Exception:
            end_y = dt.date(yr, end.month, 28) if end.month == 2 else dt.date(yr, end.month, 1)
        start_y = end_y - dt.timedelta(days=win - 1)
        baseline_sums.append(_sum_window(start_y, end_y, base_all))

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
        "datasource": datasource,
        "lat": lat,
        "lon": lon,
        "window_days": win,
        "end_date": end.isoformat(),
        "baseline": f"{by0}-{by1}",
        "obs_sum_mm": round(sum_obs, 3),
        "baseline_mean_mm": round(mu, 3),
        "baseline_std_mm": round(sd, 3),
        "spi_zscore": round(spi_z, 3),
        "note": "Demo uses NASA POWER daily precipitation (PRECTOTCORR) and z-score SPI. Gamma-fit SPI can be added next.",
    }


# ---------------------- Disabled (demo) routes ---------------------
@app.get("/indices/spei")
def spei_disabled():
    if DEMO_LOCK:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: SPEI endpoint is disabled. SPI-only is enabled for historical+point+ERA5.",
            "enable_hint": "Set DEMO_LOCK=0 in environment to enable full endpoints (once implemented).",
        })
    raise HTTPException(status_code=501, detail="SPEI not implemented yet.")


@app.get("/forecast/next7")
def fcst_next7_disabled():
    if DEMO_LOCK:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: forecast endpoints are disabled.",
            "enable_hint": "Set DEMO_LOCK=0 to enable once forecasts are ready.",
        })
    raise HTTPException(status_code=501, detail="Forecasts not implemented yet.")
