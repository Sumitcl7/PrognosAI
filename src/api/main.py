from __future__ import annotations
import numpy as np
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.prediction_service import PrognosAIPredictionService


service: PrognosAIPredictionService | None = None
fleet_cache: dict[str, Any] | None = None


class EngineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle: int = Field(ge=1)

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[EngineObservation] = Field(
        min_length=1,
        max_length=1000,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service

    service = PrognosAIPredictionService()
    yield
    service = None


app = FastAPI(
    title="PrognosAI API",
    description=(
        "RUL prediction, uncertainty estimation, "
        "sensor drift monitoring, and maintenance decision support."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": "PrognosAI",
        "description": "Predictive maintenance decision support API",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not initialized.",
        )

    return {
        "status": "healthy",
        "model": type(service.model).__name__,
        "features": len(service.features),
        "drift_monitor": "loaded",
    }



@app.get("/fleet/evaluation")
def fleet_evaluation() -> dict[str, Any]:
    global fleet_cache

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not initialized.",
        )

    # Reuse the cached fleet result on subsequent requests.
    if fleet_cache is not None:
        return fleet_cache

    root = Path(__file__).resolve().parents[2]
    dataset_path = (
        root / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"
    )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Official FD001 test dataset not found.",
        )

    columns = (
        ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    try:
        test_data = pd.read_csv(
            dataset_path,
            sep=r"\s+",
            header=None,
            names=columns,
        )

        engines = []

        for engine_id, history in test_data.groupby("engine_id"):
            history = history.sort_values("cycle").reset_index(drop=True)

            result = service.predict(history)

            prediction = result["prediction"]
            engine = result["engine"]
            maintenance = result["maintenance"]
            drift = result["drift"]

            interval = prediction["engine_level_interval"]

            engines.append({
                "engine_id": int(engine_id),
                "latest_cycle": engine["latest_cycle"],
                "history_observations": engine["history_observations"],
                "predicted_stage": engine["predicted_stage"],
                "predicted_rul": prediction["predicted_rul"],
                "lower_bound": interval["lower_bound"],
                "upper_bound": interval["upper_bound"],
                "interval_width": interval["width"],
                "confidence_level": prediction["confidence_level"],
                "risk_level": maintenance["risk_level"],
                "confidence_category": maintenance["confidence_level"],
                "maintenance_action": maintenance["maintenance_action"],
                "model_review_required": maintenance["model_review_required"],
                "drift_status": drift["status"],
                "drift_evaluated": drift["evaluated"],
            })

        if len(engines) != 100:
            raise ValueError(
                f"Expected 100 test engines, found {len(engines)}."
            )

        risk_counts = {
            level: sum(e["risk_level"] == level for e in engines)
            for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        }

        drift_counts = {
            status: sum(e["drift_status"] == status for e in engines)
            for status in [
                "stable",
                "warning",
                "drift_detected",
                "not_evaluated",
            ]
        }

        maintenance_counts = {
            action: sum(
                e["maintenance_action"] == action for e in engines
            )
            for action in sorted({
                e["maintenance_action"] for e in engines
            })
        }

        fleet_cache = {
            "evaluation": {
                "dataset": "NASA C-MAPSS FD001",
                "cohort": "Official unseen test engines",
                "mode": "Offline evaluation",
                "total_engines": len(engines),
                "labels_used_for_prediction": False,
            },
            "summary": {
                "risk_counts": risk_counts,
                "drift_counts": drift_counts,
                "maintenance_counts": maintenance_counts,
                "maintenance_alerts": sum(
                    e["risk_level"] in ["HIGH", "CRITICAL"]
                    for e in engines
                ),
                "model_reviews_required": sum(
                    e["model_review_required"] for e in engines
                ),
                "average_predicted_rul": round(
                    sum(e["predicted_rul"] for e in engines)
                    / len(engines), 2
                ),
            },
            "engines": engines,
        }

        return fleet_cache

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Fleet evaluation failed.",
        ) from exc


@app.get("/engines/{engine_id}/history")
def engine_prediction_history(engine_id: int) -> dict[str, Any]:
    """
    Return cycle-wise RUL predictions and uncertainty intervals
    for one engine from the official FD001 test cohort.

    Ground-truth RUL labels are never used for prediction.
    """

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not initialized.",
        )

    if not 1 <= engine_id <= 100:
        raise HTTPException(
            status_code=404,
            detail="Engine ID must be between 1 and 100.",
        )

    root = Path(__file__).resolve().parents[2]
    dataset_path = (
        root / "data" / "raw" / "CMAPSSData" / "test_FD001.txt"
    )

    if not dataset_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Official FD001 test dataset not found.",
        )

    columns = (
        ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    try:
        test_data = pd.read_csv(
            dataset_path,
            sep=r"\s+",
            header=None,
            names=columns,
        )

        history = (
            test_data[test_data["engine_id"] == engine_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        if history.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Engine {engine_id} not found.",
            )

        features = history[service.features].apply(
            pd.to_numeric,
            errors="coerce",
        )

        feature_values = features.to_numpy(dtype=float)

        if not pd.notna(feature_values).all():
            raise ValueError("Engine sensor history contains invalid values.")

        if not np.isfinite(feature_values).all():
            raise ValueError("Engine sensor history contains non-finite values.")

        # Predict every observation using the frozen baseline model.
        raw_predictions = service.model.predict(features)

        predicted_rul = raw_predictions.clip(0, 125)

        stages = service._assign_stages(
            pd.Series(predicted_rul)
        )

        predictions = []

        for cycle, rul, stage in zip(
            history["cycle"],
            predicted_rul,
            stages,
        ):
            rul = float(rul)

            observation_interval = service._make_interval(
                rul,
                service.observation_quantile,
            )

            engine_interval = service._make_interval(
                rul,
                service.engine_quantile,
            )

            predictions.append({
                "cycle": int(cycle),
                "predicted_rul": rul,
                "predicted_stage": str(stage),
                "observation_interval": observation_interval,
                "engine_interval": engine_interval,
            })

        return {
            "engine_id": engine_id,
            "dataset": "NASA C-MAPSS FD001",
            "evaluation_mode": "Offline evaluation",
            "model": type(service.model).__name__,
            "history_observations": len(predictions),
            "confidence_level": 1.0 - service.engine_alpha,
            "labels_used_for_prediction": False,
            "predictions": predictions,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Engine prediction history could not be generated.",
        ) from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not initialized.",
        )

    try:
        history = pd.DataFrame(
            [observation.model_dump() for observation in request.observations]
        )

        return service.predict(history)

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal service error.",
        ) from exc
