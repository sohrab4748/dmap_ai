# app/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="DMAP-AI API", version="0.1.0")

# CORS: allow your sites (set ALLOW_ORIGINS in Render → Environment)
origins = [o.strip() for o in os.getenv("ALLOW_ORIGINS","").split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

@app.get("/", tags=["meta"])
def home():
    return {"service": "DMAP-AI API", "status": "live"}

@app.get("/health", tags=["meta"])
def health():
    return {"ok": True}

@app.get("/indices/spi", tags=["indices"])
def spi(
    sum_rain_mm: float,
    clim_mean_mm: float,
    clim_std_mm: float,
    window_days: int = Query(30, enum=[7, 30, 90], description="Length of window")
):
    std = clim_std_mm or 1e-6
    spi = (sum_rain_mm - clim_mean_mm) / std
    return {"window_days": window_days, "sum_mm": sum_rain_mm, "spi": spi}

@app.get("/indices/spei", tags=["indices"])
def spei(
    p_sum: float,
    pet_sum: float,
    clim_mean: float,   # climatology mean of (P - PET) for this window
    clim_std: float,    # climatology std of (P - PET) for this window
    window_days: int = Query(30, enum=[7, 30, 90], description="Length of window")
):
    deficit = p_sum - pet_sum
    z = (deficit - clim_mean) / (clim_std or 1e-6)
    return {"window_days": window_days, "deficit": deficit, "spei": z}

@app.get("/forecast/next7", tags=["forecast"])
def next7(prob: float = 0.42, severity: float = -1.1):
    return {"horizon_days": 7, "prob": prob, "severity": severity, "why": ["low rain", "high PET"]}
