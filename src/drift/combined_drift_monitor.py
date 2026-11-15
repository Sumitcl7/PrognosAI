"""
PrognosAI - Combined calibrated sensor drift monitor.

Loads the already-calibrated mean and variance detector artifacts and scores
one complete, stage-homogeneous 20-observation window.

This is an operational fusion layer, not a newly calibrated statistical test.
It does not retrain the RUL model or modify any existing artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MEAN_DIR = ROOT / "artifacts" / "drift" / "calibrated_sensor"
VARIANCE_DIR = ROOT / "artifacts" / "drift" / "sensor_variance_drift"

WINDOW_SIZE = 20
TOP_K = 5
EPSILON = 1e-8

STAGE_LABELS = ("Critical", "Late-middle", "Middle", "Early")


@dataclass(frozen=True)
class DetectorResult:
    score: float
    threshold: float
    drift_detected: bool
    sensors_scored: int


class CombinedDriftMonitor:
    """Score mean and variance drift for one stage-homogeneous window."""

    def __init__(
        self,
        mean_dir: Path = MEAN_DIR,
        variance_dir: Path = VARIANCE_DIR,
    ) -> None:
        self.mean_dir = Path(mean_dir)
        self.variance_dir = Path(variance_dir)

        mean_metrics = self._read_json(
            self.mean_dir / "calibrated_sensor_drift_metrics.json"
        )
        variance_metrics = self._read_json(
            self.variance_dir / "sensor_variance_drift_metrics.json"
        )

        self.mean_threshold = self._mean_threshold(mean_metrics)
        self.variance_threshold = float(variance_metrics["threshold"])

        self.mean_profiles = pd.read_csv(
            self.mean_dir / "sensor_profiles_floor_0.05.csv"
        )
        self.variance_profiles = pd.read_csv(
            self.variance_dir / "variance_reference_profiles.csv"
        )

        self.mean_sensors = sorted(self.mean_profiles["sensor"].unique().tolist())
        self.variance_sensors = list(variance_metrics["selected_sensors"])

        self._validate_artifacts()

        self.mean_lookup = {
            (str(row.predicted_stage), str(row.sensor)): {
                "median": float(row.reference_median),
                "scale": float(row.effective_scale),
            }
            for row in self.mean_profiles.itertuples(index=False)
        }
        self.variance_lookup = {
            (str(row.predicted_stage), str(row.sensor)): float(row.reference_mad)
            for row in self.variance_profiles.itertuples(index=False)
        }

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        import json

        if not path.exists():
            raise FileNotFoundError(f"Required calibration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _mean_threshold(metrics: dict[str, Any]) -> float:
        candidates = metrics.get("threshold_sensitivity", [])
        for item in candidates:
            if np.isclose(float(item.get("variability_floor", -1)), 0.05):
                return float(item["threshold"])
        raise ValueError(
            "Could not find mean-detector threshold for variability_floor=0.05 "
            "in calibrated_sensor_drift_metrics.json"
        )

    def _validate_artifacts(self) -> None:
        required_mean = {
            "predicted_stage", "sensor", "reference_median", "effective_scale"
        }
        required_variance = {"predicted_stage", "sensor", "reference_mad"}

        if not required_mean.issubset(self.mean_profiles.columns):
            raise ValueError("Mean profile CSV is missing required columns.")
        if not required_variance.issubset(self.variance_profiles.columns):
            raise ValueError("Variance profile CSV is missing required columns.")

        if len(self.variance_sensors) != TOP_K:
            raise ValueError(
                f"Expected {TOP_K} selected variance sensors, "
                f"found {len(self.variance_sensors)}."
            )

        for stage in STAGE_LABELS:
            for sensor in self.mean_sensors:
                if (stage, sensor) not in {
                    (str(r.predicted_stage), str(r.sensor))
                    for r in self.mean_profiles.itertuples(index=False)
                }:
                    raise ValueError(f"Missing mean profile: {stage}/{sensor}")
            for sensor in self.variance_sensors:
                if (stage, sensor) not in {
                    (str(r.predicted_stage), str(r.sensor))
                    for r in self.variance_profiles.itertuples(index=False)
                }:
                    raise ValueError(f"Missing variance profile: {stage}/{sensor}")

        if not np.isfinite(self.mean_threshold) or not np.isfinite(self.variance_threshold):
            raise ValueError("Calibration thresholds must be finite.")

    @staticmethod
    def _mad(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("Cannot calculate MAD from empty/non-finite values.")
        center = float(np.median(values))
        return float(np.median(np.abs(values - center)))

    def score_window(self, window: pd.DataFrame) -> dict[str, Any]:
        """
        Score one complete 20-row window.

        Required columns:
          - cycle
          - predicted_stage
          - all sensors present in the mean profile

        predicted_stage must come from the frozen baseline predicted RUL stage
        assignment. The monitor intentionally does not infer/recompute it.
        """
        if not isinstance(window, pd.DataFrame):
            raise TypeError("window must be a pandas DataFrame.")

        if len(window) != WINDOW_SIZE:
            return self._not_evaluated(
                f"Expected exactly {WINDOW_SIZE} observations; received {len(window)}."
            )

        required = {"cycle", "predicted_stage", *self.mean_sensors}
        missing = sorted(required.difference(window.columns))
        if missing:
            return self._not_evaluated(f"Missing required columns: {missing}")

        data = window.sort_values("cycle").reset_index(drop=True).copy()

        if data["cycle"].isna().any() or data["predicted_stage"].isna().any():
            return self._not_evaluated("cycle/predicted_stage contains missing values.")

        stages = data["predicted_stage"].astype(str).unique().tolist()
        if len(stages) != 1:
            return self._not_evaluated(
                "Window crosses predicted-RUL stages; provide a stage-homogeneous window."
            )

        stage = stages[0]
        if stage not in STAGE_LABELS:
            return self._not_evaluated(f"Unknown predicted_stage: {stage}")

        sensor_values = data[self.mean_sensors].apply(
            pd.to_numeric, errors="coerce"
        )
        if sensor_values.isna().any().any() or not np.isfinite(
            sensor_values.to_numpy(dtype=float)
        ).all():
            return self._not_evaluated("Sensor window contains missing/non-finite values.")

        # Mean/location detector: standardized absolute window-median deviations;
        # aggregate the five largest sensor scores, matching the calibrated method.
        mean_sensor_scores: dict[str, float] = {}
        for sensor in self.mean_sensors:
            profile = self.mean_lookup[(stage, sensor)]
            current_median = float(sensor_values[sensor].median())
            scale = max(float(profile["scale"]), EPSILON)
            mean_sensor_scores[sensor] = abs(
                current_median - float(profile["median"])
            ) / scale

        top_mean = sorted(mean_sensor_scores.values(), reverse=True)[:TOP_K]
        mean_score = float(np.mean(top_mean))
        mean_result = DetectorResult(
            score=mean_score,
            threshold=self.mean_threshold,
            drift_detected=mean_score > self.mean_threshold,
            sensors_scored=len(mean_sensor_scores),
        )

        # Variance detector: one-sided log ratio of current window MAD
        # to the stage/sensor reference MAD, averaged across five sensors.
        variance_sensor_scores: dict[str, float] = {}
        for sensor in self.variance_sensors:
            reference_mad = self.variance_lookup[(stage, sensor)]
            if reference_mad <= EPSILON:
                return self._not_evaluated(
                    f"Reference MAD is zero/near-zero for {stage}/{sensor}."
                )
            current_mad = self._mad(
                data[sensor].to_numpy(dtype=float)
            )
            variance_sensor_scores[sensor] = max(
                0.0,
                float(np.log(max(current_mad, EPSILON) / reference_mad)),
            )

        variance_score = float(np.mean(list(variance_sensor_scores.values())))
        variance_result = DetectorResult(
            score=variance_score,
            threshold=self.variance_threshold,
            drift_detected=variance_score > self.variance_threshold,
            sensors_scored=len(variance_sensor_scores),
        )

        mean_flag = mean_result.drift_detected
        variance_flag = variance_result.drift_detected

        if mean_flag and variance_flag:
            combined_status = "drift_detected"
        elif mean_flag or variance_flag:
            combined_status = "warning"
        else:
            combined_status = "stable"

        return {
            "evaluated": True,
            "status": combined_status,
            "predicted_stage": stage,
            "window_start": int(data["cycle"].min()),
            "window_end": int(data["cycle"].max()),
            "mean_detector": {
                "score": mean_result.score,
                "threshold": mean_result.threshold,
                "drift_detected": mean_result.drift_detected,
                "sensors_scored": mean_result.sensors_scored,
                "top_sensor_scores": dict(
                    sorted(
                        mean_sensor_scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:TOP_K]
                ),
            },
            "variance_detector": {
                "score": variance_result.score,
                "threshold": variance_result.threshold,
                "drift_detected": variance_result.drift_detected,
                "sensors_scored": variance_result.sensors_scored,
                "sensor_scores": variance_sensor_scores,
            },
            "fusion_rule": "both=drift_detected; one=warning; neither=stable",
            "maintenance_drift_status": combined_status,
        }

    @staticmethod
    def _not_evaluated(reason: str) -> dict[str, Any]:
        return {
            "evaluated": False,
            "status": "not_evaluated",
            "reason": reason,
            "mean_detector": None,
            "variance_detector": None,
            "maintenance_drift_status": "not_evaluated",
        }


def evaluate_window(window: pd.DataFrame) -> dict[str, Any]:
    """Convenience function for API/service integration."""
    monitor = CombinedDriftMonitor()
    return monitor.score_window(window)
