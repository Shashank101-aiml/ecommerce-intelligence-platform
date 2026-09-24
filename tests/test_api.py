import pytest
from fastapi.testclient import TestClient

import api.main as main
from ml.config import FEATURE_COLUMNS

VALID = {c: 1.0 for c in FEATURE_COLUMNS} | {"is_uk": 1}


class FakeModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == FEATURE_COLUMNS
        return [[0.2, 0.8]]


@pytest.fixture
def client(monkeypatch):
    logged = []
    monkeypatch.setattr(main, "log_prediction", lambda **kw: logged.append(kw))
    main.app.state.model, main.app.state.model_version = FakeModel(), "7"
    c = TestClient(main.app)  # no context manager: skip the MLflow-loading lifespan
    c.logged = logged
    yield c
    main.app.state.model, main.app.state.model_version = None, None


def test_health_reports_model_version(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_version": "7"}


def test_health_degraded_without_model(client):
    main.app.state.model = None
    assert client.get("/health").json()["status"] == "degraded"


def test_predict_returns_probability_and_logs(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json() == {"churn_probability": 0.8, "predicted_label": 1, "model_version": "7"}
    assert client.logged[0]["status"] == "ok" and client.logged[0]["latency_ms"] >= 0


def test_predict_rejects_missing_negative_and_extra_fields(client):
    assert client.post("/predict", json={"recency_days": 1}).status_code == 422
    assert client.post("/predict", json=VALID | {"frequency": -1}).status_code == 422
    assert client.post("/predict", json=VALID | {"surprise": 1}).status_code == 422
    assert client.post("/predict", json=VALID | {"is_uk": 2}).status_code == 422


def test_predict_503_when_model_missing(client):
    main.app.state.model = None
    assert client.post("/predict", json=VALID).status_code == 503


def test_model_failure_returns_500_and_is_logged(client):
    class Broken:
        def predict_proba(self, frame):
            raise RuntimeError("boom")
    main.app.state.model = Broken()
    assert client.post("/predict", json=VALID).status_code == 500
    assert client.logged[-1]["status"] == "error" and "boom" in client.logged[-1]["error"]
