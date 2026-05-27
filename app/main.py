from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.forecast_service import (
    run_forecast_for_site,
    train_models_for_site,
)
from app.services.optimizer_service import run_optimizer_for_site

app = FastAPI(title="EnerSim API")

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
    try:
        return train_models_for_site(req.site_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/forecast")
def forecast(req: SiteRequest):
    try:
        result = run_forecast_for_site(req.site_id)
        if "error" in result:
            return JSONResponse(status_code=422, content=result)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/optimize")
def optimize(req: SiteRequest):
    try:
        return run_optimizer_for_site(req.site_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/forecast-and-optimize")
def forecast_and_optimize(req: SiteRequest):
    try:
        fc = run_forecast_for_site(req.site_id)
        opt = run_optimizer_for_site(req.site_id)
        return {
            "forecast": fc,
            "optimization": opt
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )
