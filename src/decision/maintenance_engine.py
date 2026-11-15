from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MaintenanceConfig:
    critical_rul: float = 15.0
    urgent_rul: float = 30.0
    planning_rul: float = 60.0
    wide_interval: float = 40.0


@dataclass
class MaintenanceDecision:
    predicted_rul: float
    lower_bound: float
    upper_bound: float
    effective_lower_bound: float
    interval_width: float
    drift_status: str
    priority: str
    action: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIORITY_LEVELS = {
    0: ("MONITOR", "Continue normal monitoring."),
    1: ("PLAN", "Include this engine in maintenance planning."),
    2: ("SCHEDULE_SOON", "Arrange a maintenance inspection soon."),
    3: ("URGENT", "Escalate for urgent maintenance review."),
}


def evaluate_maintenance(
    predicted_rul: float,
    lower_bound: float,
    upper_bound: float,
    drift_status: str = "stable",
    config: MaintenanceConfig | None = None,
) -> MaintenanceDecision:
    """
    Produce a transparent maintenance recommendation.

    The lower confidence bound determines the base risk.
    Drift and interval width can raise, but never lower, priority.

    These thresholds are configurable research assumptions,
    not certified industrial maintenance limits.
    """

    config = config or MaintenanceConfig()

    values = [predicted_rul, lower_bound, upper_bound]
    if any(not isinstance(v, (int, float)) for v in values):
        raise TypeError("RUL values must be numeric.")

    if predicted_rul < 0 or upper_bound < 0:
        raise ValueError("Predicted RUL and upper bound must be non-negative.")

    if lower_bound > upper_bound:
        raise ValueError("lower_bound cannot exceed upper_bound.")

    if not lower_bound <= predicted_rul <= upper_bound:
        raise ValueError(
            "The prediction interval must contain predicted_rul."
        )

    valid_drift_statuses = {"stable", "warning", "drift_detected"}
    if drift_status not in valid_drift_statuses:
        raise ValueError(
            f"drift_status must be one of {sorted(valid_drift_statuses)}."
        )

    if not (
        0 <= config.critical_rul
        <= config.urgent_rul
        <= config.planning_rul
    ):
        raise ValueError("RUL thresholds must be in ascending order.")

    if config.wide_interval < 0:
        raise ValueError("wide_interval cannot be negative.")

    # Negative lower confidence bounds are possible mathematically.
    # For maintenance decisions, treat them as zero remaining cycles.
    effective_lower = max(0.0, float(lower_bound))
    interval_width = float(upper_bound - lower_bound)

    # Base priority from conservative RUL estimate.
    if effective_lower <= config.critical_rul:
        level = 3
        reasons = ["Conservative RUL estimate is in the critical range."]
    elif effective_lower <= config.urgent_rul:
        level = 2
        reasons = ["Conservative RUL estimate indicates near-term risk."]
    elif effective_lower <= config.planning_rul:
        level = 1
        reasons = ["Conservative RUL estimate requires maintenance planning."]
    else:
        level = 0
        reasons = ["Conservative RUL estimate is above planning threshold."]

    # Uncertainty safeguard.
    if interval_width >= config.wide_interval:
        if level < 1:
            level = 1
        reasons.append("Wide prediction interval increases uncertainty.")

    # Drift safeguards.
    if drift_status == "warning":
        level = max(level, 1)
        reasons.append("Drift warning requires additional maintenance attention.")

    elif drift_status == "drift_detected":
        level = max(level, 2)
        reasons.append("Detected drift requires an earlier maintenance review.")

    priority, action = PRIORITY_LEVELS[level]

    return MaintenanceDecision(
        predicted_rul=float(predicted_rul),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        effective_lower_bound=effective_lower,
        interval_width=interval_width,
        drift_status=drift_status,
        priority=priority,
        action=action,
        reasons=reasons,
    )


if __name__ == "__main__":
    example = evaluate_maintenance(
        predicted_rul=45,
        lower_bound=10,
        upper_bound=80,
        drift_status="stable",
    )

    import json
    print(json.dumps(example.to_dict(), indent=4))