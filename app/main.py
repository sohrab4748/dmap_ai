from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="DMAP-AI API")

# CORS: allow your sites
origins = [o.strip() for o in os.getenv("ALLOW_ORIGINS","").split(",") if o.strip()] or ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"service": "DMAP-AI API", "status": "live"}
    
@app.get("/health")
def health():
    return {"ok": True}

# SPI from 30-day precip sum (z-score vs baseline)
@app.get("/indices/spi")
def spi_30d(sum_rain_mm: float, clim_mean_mm: float, clim_std_mm: float):
    std = clim_std_mm if clim_std_mm else 1e-6
    spi = (sum_rain_mm - clim_mean_mm) / std
    return {"window_days": 30, "sum_mm": sum_rain_mm, "spi": spi}

# SPEI from 30-day (P - PET)
@app.get("/indices/spei")
def spei_30d(p_sum: float, pet_sum: float, clim_mean: float, clim_std: float):
    deficit = p_sum - pet_sum
    std = clim_std if clim_std else 1e-6
    z = (deficit - clim_mean) / std
    return {"window_days": 30, "deficit": deficit, "spei": z}

# Demo forecast stub (replace later)
@app.get("/forecast/next7")
def next7(prob: float = 0.42, severity: float = -1.1):
    return {"horizon_days": 7, "prob": prob, "severity": severity, "why": ["low rain","high PET"]}
