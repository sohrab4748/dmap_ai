from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os

origins_env = os.getenv("ALLOW_ORIGINS", "")
origins = [o.strip() for o in origins_env.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app = FastAPI(title="DMAP-AI API (demo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/hello")
def hello(name: str = "world"):
    return {"message": f"Hello, {name}!"}

@app.get("/indices/spi")
def spi_30d(sum_rain_mm: float = Query(..., description="Sum of rain over 30 days"),
            clim_mean_mm: float = Query(..., description="30-day climatology mean"),
            clim_std_mm: float = Query(..., description="30-day climatology std dev")):
    std = clim_std_mm if clim_std_mm != 0 else 1e-6
    spi = (sum_rain_mm - clim_mean_mm) / std
    return {"window_days": 30, "sum_mm": sum_rain_mm, "spi": spi}
