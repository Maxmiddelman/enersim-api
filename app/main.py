from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.forecast_service import (
    run_forecast_for_site,
    train_models_for_site,
)
from app.services.optimizer_service import run_optimizer_for_site

app = FastAPI(title="EnerSim API")

# Tijdelijk breed open zetten voor testen.
# Later kun je dit beperken tot je echte domeinen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SiteRequest(BaseModel):
    site_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.options("/{full_path:path}")
def preflight_handler(full_path: str):
    return {"ok": True}


@app.post("/train")
def train(req: SiteRequest):
    return train_models_for_site(req.site_id)


@app.post("/forecast")
def forecast(req: SiteRequest):
    return run_forecast_for_site(req.site_id)


@app.post("/optimize")
def optimize(req: SiteRequest):
    return run_optimizer_for_site(req.site_id)


@app.post("/forecast-and-optimize")
def forecast_and_optimize(req: SiteRequest):
    forecast = run_forecast_for_site(req.site_id)
    optimize = run_optimizer_for_site(req.site_id)

    return {
        "forecast": forecast,
        "optimization": optimize
    }
