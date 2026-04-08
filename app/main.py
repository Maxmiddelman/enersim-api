from fastapi import FastAPI
from pydantic import BaseModel

from app.services.forecast_service import (
    run_forecast_for_site,
    train_models_for_site,
)

from app.services.optimizer_service import run_optimizer_for_site

app = FastAPI(title="EnerSim API")


class SiteRequest(BaseModel):
    site_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


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
