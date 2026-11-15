from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.drift.combined_drift_monitor import CombinedDriftMonitor
from src.decision.maintenance_engine_v2 import evaluate_maintenance_v2


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts" / "baseline" / "random_forest_fd001.joblib"
BASELINE_METRICS_PATH = ROOT / "artifacts" / "baseline" / "metrics.json"

OBSERVATION_METRICS_PATH = (
    ROOT / "artifacts" / "uncertainty" / "conformal_metrics.json"
)
ENGINE_METRICS_PATH = (
    ROOT / "artifacts" / "uncertainty" / "engine_level_conformal_metrics.json"
)

RUL_CAP = 125.0
WINDOW_SIZE = 20

STAGE_BINS = [0, 30, 60, 90, 125]
STAGE_LABELS = ["Critical", "Late-middle", "Middle", "Early"]


class PrognosAIPredictionService:
    """Loads PrognosAI artifacts once and serves engine-level predictions."""

    def __init__(self) -> None:
        required_files = [
            MODEL_PATH,
            BASELINE_METRICS_PATH,
            OBSERVATION_METRICS_PATH,
            ENGINE_METRICS_PATH,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(f"Required artifact not found: {path}")

        self.model = joblib.load(MODEL_PATH)

        with BASELINE_METRICS_PATH.open("r", encoding="utf-8") as file:
            baseline_metrics = json.load(file)

        with OBSERVATION_METRICS_PATH.open("r", encoding="utf-8") as file:
            observation_metrics = json.load(file)

        with ENGINE_METRICS_PATH.open("r", encoding="utf-8") as file:
            engine_metrics = json.load(file)

        self.features = list(self.model.feature_names_in_)

        if self.features != baseline_metrics["features"]:
            raise ValueError(
                "Model feature order does not match baseline metadata."
            )

        if len(self.features) != 18:
            raise ValueError(
                f"Expected 18 model features, found {len(self.features)}."
            )

        if self.model.n_features_in_ != len(self.features):
            raise ValueError("Model feature count is inconsistent.")

        self.observation_quantile = float(
            observation_metrics["conformal_quantile"]
        )

        self.engine_quantile = float(
            engine_metrics["conformal_quantile"]
        )

        self.observation_alpha = float(observation_metrics["alpha"])
        self.engine_alpha = float(engine_metrics["alpha"])

        if not np.isclose(self.observation_alpha, self.engine_alpha):
            raise ValueError("Conformal alpha values do not match.")

        if not np.isclose(
            float(observation_metrics["rul_cap"]), RUL_CAP
        ) or not np.isclose(
            float(engine_metrics["rul_cap"]), RUL_CAP
        ):
            raise ValueError("Conformal RUL caps do not match the service.")

        self.drift_monitor = CombinedDriftMonitor()

    @staticmethod
    def _assign_stages(predictions: pd.Series) -> pd.Series:
        clipped = predictions.clip(0, RUL_CAP)

        return pd.cut(
            clipped,
            bins=STAGE_BINS,
            labels=STAGE_LABELS,
            include_lowest=True,
        ).astype(str)

    @staticmethod
    def _make_interval(
        predicted_rul: float,
        quantile: float,
    ) -> dict[str, float]:
        lower = max(0.0, predicted_rul - quantile)
        upper = min(RUL_CAP, predicted_rul + quantile)

        return {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "width": float(upper - lower),
        }

    def predict(self, history: pd.DataFrame) -> dict[str, Any]:
        """
        Predict RUL and maintenance status for one engine.

        Input:
            DataFrame containing cycle and all 18 model features.

        At least one observation is required. Drift is evaluated only
        when the latest 20 observations have consecutive cycles and
        belong to one predicted-RUL stage.
        """

        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame.")

        if history.empty:
            raise ValueError("Engine history cannot be empty.")

        required = {"cycle", *self.features}
        missing = sorted(required.difference(history.columns))

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        data = history.copy()

        data["cycle"] = pd.to_numeric(data["cycle"], errors="coerce")

        if data["cycle"].isna().any():
            raise ValueError("Cycle values must be numeric.")

        if not np.isfinite(data["cycle"].to_numpy(dtype=float)).all():
            raise ValueError("Cycle values must be finite.")

        if not np.equal(
            data["cycle"].to_numpy(dtype=float),
            np.floor(data["cycle"].to_numpy(dtype=float)),
        ).all():
            raise ValueError("Cycle values must be integers.")

        data["cycle"] = data["cycle"].astype(int)
        data = data.sort_values("cycle").reset_index(drop=True)

        if data["cycle"].duplicated().any():
            raise ValueError("Duplicate cycle values are not allowed.")

        feature_data = data[self.features].apply(
            pd.to_numeric, errors="coerce"
        )

        feature_values = feature_data.to_numpy(dtype=float)

        if not np.isfinite(feature_values).all():
            raise ValueError(
                "Model features contain missing or non-finite values."
            )

        raw_predictions = self.model.predict(feature_data)

        data["predicted_RUL"] = np.clip(
            raw_predictions, 0, RUL_CAP
        )

        data["predicted_stage"] = self._assign_stages(
            pd.Series(data["predicted_RUL"], index=data.index)
        )

        latest = data.iloc[-1]
        predicted_rul = float(latest["predicted_RUL"])

        observation_interval = self._make_interval(
            predicted_rul, self.observation_quantile
        )

        engine_interval = self._make_interval(
            predicted_rul, self.engine_quantile
        )

        drift_result: dict[str, Any]

        if len(data) < WINDOW_SIZE:
            drift_result = {
                "evaluated": False,
                "status": "not_evaluated",
                "reason": (
                    f"At least {WINDOW_SIZE} observations are required; "
                    f"received {len(data)}."
                ),
                "maintenance_drift_status": "not_evaluated",
            }

        else:
            window = data.tail(WINDOW_SIZE).copy()
            cycles = window["cycle"].to_numpy(dtype=int)

            if not np.all(np.diff(cycles) == 1):
                drift_result = {
                    "evaluated": False,
                    "status": "not_evaluated",
                    "reason": (
                        "Latest 20 observations do not have consecutive cycles."
                    ),
                    "maintenance_drift_status": "not_evaluated",
                }
            else:
                drift_result = self.drift_monitor.score_window(window)

        drift_status = drift_result["maintenance_drift_status"]

        maintenance_decision = evaluate_maintenance_v2(
            predicted_rul=predicted_rul,
            lower_bound=engine_interval["lower_bound"],
            upper_bound=engine_interval["upper_bound"],
            drift_status=drift_status,
        )

        return {
            "engine": {
                "latest_cycle": int(latest["cycle"]),
                "history_observations": int(len(data)),
                "predicted_stage": str(latest["predicted_stage"]),
            },
            "prediction": {
                "predicted_rul": predicted_rul,
                "rul_cap": RUL_CAP,
                "observation_level_interval": observation_interval,
                "engine_level_interval": engine_interval,
                "confidence_level": 1.0 - self.engine_alpha,
            },
            "drift": drift_result,
            "maintenance": maintenance_decision.to_dict(),
            "model": {
                "name": type(self.model).__name__,
                "feature_count": len(self.features),
            },
        }
