
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api.prediction_service import PrognosAIPredictionService


FEATURES = [
    "setting_1", "setting_2", "setting_3",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]


class DummyModel:
    """Deterministic model used only for API integration tests."""

    feature_names_in_ = np.array(FEATURES)

    def predict(self, features):
        return np.full(len(features), 50.0)


class DummyDriftMonitor:
    """Controlled drift result without calibration artifacts."""

    def score_window(self, window):
        return {
            "evaluated": True,
            "status": "stable",
            "maintenance_drift_status": "stable",
        }


@pytest.fixture
def client():
    # Create the real service without loading external model artifacts.
    service = PrognosAIPredictionService.__new__(
        PrognosAIPredictionService
    )

    service.model = DummyModel()
    service.features = FEATURES
    service.observation_quantile = 10.0
    service.engine_quantile = 15.0
    service.observation_alpha = 0.10
    service.engine_alpha = 0.10
    service.drift_monitor = DummyDriftMonitor()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            api_main,
            "PrognosAIPredictionService",
            lambda: service,
        )

        with TestClient(api_main.app) as test_client:
            yield test_client


@pytest.fixture
def valid_observations():
    observations = []

    for cycle in range(1, 21):
        row = {"cycle": cycle}

        for index, feature in enumerate(FEATURES):
            row[feature] = 0.1 * (index + 1) + 0.001 * cycle

        observations.append(row)

    return observations


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "PrognosAI"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["features"] == 18


def test_valid_prediction(client, valid_observations):
    response = client.post(
        "/predict",
        json={"observations": valid_observations},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["engine"]["history_observations"] == 20
    assert 0 <= result["prediction"]["predicted_rul"] <= 125

    assert "observation_level_interval" in result["prediction"]
    assert "engine_level_interval" in result["prediction"]

    assert result["drift"]["status"] in {
        "stable",
        "warning",
        "drift_detected",
        "not_evaluated",
    }

    assert "maintenance_action" in result["maintenance"]


def test_missing_sensor_rejected(client, valid_observations):
    observations = [row.copy() for row in valid_observations]

    del observations[0]["sensor_2"]

    response = client.post(
        "/predict",
        json={"observations": observations},
    )

    assert response.status_code == 422


def test_duplicate_cycles_rejected(client, valid_observations):
    observations = [row.copy() for row in valid_observations]

    observations[1]["cycle"] = observations[0]["cycle"]

    response = client.post(
        "/predict",
        json={"observations": observations},
    )

    assert response.status_code == 422
    assert "Duplicate cycle" in response.json()["detail"]


def test_non_consecutive_cycles_not_evaluated(
    client, valid_observations
):
    observations = [row.copy() for row in valid_observations]

    observations[10]["cycle"] = 100

    response = client.post(
        "/predict",
        json={"observations": observations},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["drift"]["status"] == "not_evaluated"
    assert "consecutive cycles" in result["drift"]["reason"]


def test_insufficient_history_not_evaluated(
    client, valid_observations
):
    response = client.post(
        "/predict",
        json={"observations": valid_observations[:10]},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["engine"]["history_observations"] == 10
    assert result["drift"]["status"] == "not_evaluated"
    assert result["maintenance"]["drift_status"] == "not_evaluated"


def test_empty_history_rejected(client):
    response = client.post(
        "/predict",
        json={"observations": []},
    )

    assert response.status_code == 422


def test_extra_field_rejected(client, valid_observations):
    observations = [row.copy() for row in valid_observations]

    observations[0]["unexpected_field"] = 123

    response = client.post(
        "/predict",
        json={"observations": observations},
    )

    assert response.status_code == 422
