"""
tests/test_api.py
-----------------
Automated integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_get_datasets_status(client):
    res = client.get("/api/datasets/status")
    assert res.status_code == 200
    data = res.json()
    assert "datasets" in data
    assert "ASL" in data["datasets"]
    assert "ISL" in data["datasets"]
    assert "BSL" in data["datasets"]
    assert "CUSTOM" in data["datasets"]
    assert data["total_samples"] >= 0


def test_custom_signs_api_lifecycle(client):
    res_start = client.post("/api/custom-sign/start")
    assert res_start.status_code == 200
    assert "min_samples_required" in res_start.json()

    res_sample = client.post(
        "/api/custom-sign/sample",
        json={"features": [0.5] * 126},
    )
    assert res_sample.status_code == 200
    assert res_sample.json()["sample_type"] == "static"

    bad_save = client.post(
        "/api/custom-sign/save",
        json={"user_id": "api_user", "label": "Coffee", "samples": [{"sample_type": "static", "features": [0.5] * 126}]},
    )
    assert bad_save.status_code == 400

    valid_samples = [{"sample_type": "static", "features": [0.5] * 126} for _ in range(3)]
    good_save = client.post(
        "/api/custom-sign/save",
        json={"user_id": "api_user", "label": "Coffee Please", "samples": valid_samples},
    )
    assert good_save.status_code == 201
    created_sign = good_save.json()
    sign_id = created_sign["id"]
    assert created_sign["label"] == "Coffee Please"

    list_res = client.get("/api/custom-signs?user_id=api_user")
    assert list_res.status_code == 200
    assert any(s["id"] == sign_id for s in list_res.json())

    get_res = client.get(f"/api/custom-sign/{sign_id}")
    assert get_res.status_code == 200
    assert get_res.json()["label"] == "Coffee Please"

    del_res = client.delete(f"/api/custom-sign/{sign_id}")
    assert del_res.status_code == 200

    get_after = client.get(f"/api/custom-sign/{sign_id}")
    assert get_after.status_code == 404


def test_recognition_endpoints(client):
    res_frame = client.post(
        "/api/recognition/frame",
        json={"language": "ASL", "features": [0.5] * 126},
    )
    assert res_frame.status_code == 200
    assert "label" in res_frame.json()
    assert "confidence" in res_frame.json()

    res_seq = client.post(
        "/api/recognition/sequence",
        json={"language": "ISL", "sequence": [[0.5] * 126 for _ in range(30)]},
    )
    assert res_seq.status_code == 200
    assert "label" in res_seq.json()

    res_trans = client.get("/api/recognition/transcript")
    assert res_trans.status_code == 200

    res_clear = client.post("/api/recognition/transcript/clear")
    assert res_clear.status_code == 200
