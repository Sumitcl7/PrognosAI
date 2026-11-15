
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MaintenanceV2Config:
    critical_rul: float = 15.0
    high_rul: float = 30.0
    moderate_rul: float = 60.0

    high_confidence_width: float = 40.0
    medium_confidence_width: float = 80.0


@dataclass
class MaintenanceV2Decision:
    predicted_rul: float
    lower_bound: float
    upper_bound: float
    interval_width: float

    risk_level: str
    confidence_level: str

    maintenance_action: str
    model_review_required: bool
    drift_status: str

    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_maintenance_v2(
    predicted_rul: float,
    lower_bound: float,
    upper_bound: float,
    drift_status: str = "not_evaluated",
    config: MaintenanceV2Config | None = None,
) -> MaintenanceV2Decision:

    config = config or MaintenanceV2Config()

    values = [predicted_rul, lower_bound, upper_bound]

    if any(not isinstance(v, (int, float)) for v in values):
        raise TypeError("RUL values must be numeric.")

    if predicted_rul < 0 or upper_bound < 0:
        raise ValueError("Predicted RUL and upper bound must be non-negative.")

    if lower_bound > upper_bound:
        raise ValueError("lower_bound cannot exceed upper_bound.")

    if not lower_bound <= predicted_rul <= upper_bound:
        raise ValueError("Prediction interval must contain predicted_rul.")

    valid_drift = {
        "stable",
        "warning",
        "drift_detected",
        "not_evaluated",
    }

    if drift_status not in valid_drift:
        raise ValueError(f"Invalid drift status: {drift_status}")

    if not (
        0 <= config.critical_rul
        <= config.high_rul
        <= config.moderate_rul
    ):
        raise ValueError("RUL thresholds must be in ascending order.")

    if not (
        0 <= config.high_confidence_width
        <= config.medium_confidence_width
    ):
        raise ValueError("Confidence thresholds must be in ascending order.")

    interval_width = float(upper_bound - lower_bound)

    # Dimension A: predicted failure/degradation risk.
    if predicted_rul <= config.critical_rul:
        risk_level = "CRITICAL"
        action = "URGENT_MAINTENANCE_REVIEW"

    elif predicted_rul <= config.high_rul:
        risk_level = "HIGH"
        action = "SCHEDULE_MAINTENANCE_SOON"

    elif predicted_rul <= config.moderate_rul:
        risk_level = "MODERATE"
        action = "PLAN_MAINTENANCE"

    else:
        risk_level = "LOW"
        action = "CONTINUE_MONITORING"

    # Dimension B: model confidence.
    if interval_width <= config.high_confidence_width:
        confidence_level = "HIGH"

    elif interval_width <= config.medium_confidence_width:
        confidence_level = "MEDIUM"

    else:
        confidence_level = "LOW"

    reasons = [
        f"Predicted RUL indicates {risk_level.lower()} risk.",
        f"Interval width indicates {confidence_level.lower()} confidence.",
    ]

    model_review_required = confidence_level == "LOW"

    if model_review_required:
        reasons.append(
            "Wide prediction interval requires model or inspection review."
        )

    # Drift changes model-review status, not failure risk.
    if drift_status == "warning":
        model_review_required = True
        reasons.append("Drift warning requires model review.")

    elif drift_status == "drift_detected":
        model_review_required = True
        reasons.append("Detected drift requires model review.")

    elif drift_status == "not_evaluated":
        reasons.append("Drift has not been evaluated for this engine.")

    return MaintenanceV2Decision(
        predicted_rul=float(predicted_rul),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        interval_width=interval_width,
        risk_level=risk_level,
        confidence_level=confidence_level,
        maintenance_action=action,
        model_review_required=model_review_required,
        drift_status=drift_status,
        reasons=reasons,
    )


if __name__ == "__main__":
    import json

    example = evaluate_maintenance_v2(
        predicted_rul=67.13,
        lower_bound=0,
        upper_bound=125,
        drift_status="not_evaluated",
    )

    print(json.dumps(example.to_dict(), indent=4))
