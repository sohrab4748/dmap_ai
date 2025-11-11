# app/main.py
import os
from fastapi import FastAPI, HTTPException, Query
from typing import Optional

app = FastAPI(title="DMAP-AI API", version="0.2.0")
# allow your web origins (use "*" for now, tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # e.g. ["https://dmap.agrimetsoft.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- DEMO LOCK CONFIG ---
DEMO_LOCK = os.getenv("DEMO_LOCK", "1") == "1"          # set to "0" to disable restrictions
ALLOWED_MODE = os.getenv("DEMO_MODE", "historical").lower()
ALLOWED_AOI = os.getenv("DEMO_AOI", "point").lower()
ALLOWED_DATASOURCE = os.getenv("DEMO_DATASOURCE", "era5").lower()


@app.get("/")
def root():
    return {"ok": True, "message": "DMAP-AI API. See /health and /docs", "demo_lock": DEMO_LOCK}


@app.get("/health")
def health():
    return {"ok": True, "demo_lock": DEMO_LOCK}


# ---------- SPI (ALLOWED IN DEMO) ----------
@app.get("/indices/spi")
def spi(
    sum_rain_mm: float = Query(..., description="Aggregated precipitation for the window (mm)"),
    clim_mean_mm: float = Query(..., description="Climatological mean for the same window (mm)"),
    clim_std_mm: float = Query(..., description="Climatological std for the same window (mm)", gt=0),
    window_days: int = Query(30, ge=1, le=365, description="Window length (days)"),
    # Optional flags so the UI can pass what the user selected (we enforce them in demo)
    mode: Optional[str] = Query(None, description="historical|prediction"),
    aoi: Optional[str] = Query(None, description="point|box"),
    datasource: Optional[str] = Query(None, description="era5|gridmet|prism|gpm|user")
):
    if DEMO_LOCK:
        # Enforce: ONLY historical + point + era5
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

    # Minimal SPI (z-score) using mean/std — good enough for demo wiring.
    spi_value = (sum_rain_mm - clim_mean_mm) / clim_std_mm

    return {
        "window_days": window_days,
        "sum_mm": sum_rain_mm,
        "clim_mean_mm": clim_mean_mm,
        "clim_std_mm": clim_std_mm,
        "spi": round(spi_value, 3),
        "demo_lock": DEMO_LOCK,
        "note": "Demo mode is enforcing historical+point+ERA5. Full gamma-fit SPI will be in the production build."
    }


# ---------- SPEI (DISABLED IN DEMO) ----------
@app.get("/indices/spei")
def spei_disabled():
    if DEMO_LOCK:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: SPEI endpoint is disabled. SPI-only is enabled for historical+point+ERA5.",
            "enable_hint": "Set DEMO_LOCK=0 in environment to enable full endpoints (once implemented)."
        })
    # (Your real SPEI implementation would go here when demo lock is off)
    raise HTTPException(status_code=501, detail="SPEI not implemented yet.")


# ---------- FORECASTS (DISABLED IN DEMO) ----------
@app.get("/forecast/next7")
def fcst_next7_disabled():
    if DEMO_LOCK:
        raise HTTPException(status_code=403, detail={
            "message": "Demo mode: forecast endpoints are disabled.",
            "enable_hint": "Set DEMO_LOCK=0 to enable once forecasts are ready."
        })
    raise HTTPException(status_code=501, detail="Forecasts not implemented yet.")
