import base64
import io
from unittest.mock import AsyncMock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.inference import Prediction
from app.main import app


def image_bytes(fmt="PNG"):
    data = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(data, fmt)
    return data.getvalue()


def test_health_does_not_load_model():
    app.state.inference = type("Stub", (), {"loaded": False})()
    response = TestClient(app).get("/api/health")
    assert response.json() == {"status": "ok"}
    assert app.state.inference.loaded is False


def test_prediction_contract_and_original_size():
    stub = type("Stub", (), {})()
    stub.predict_async = AsyncMock(return_value=Prediction(np.ones((256, 256), dtype=np.float32), np.ones((8, 12, 3), dtype=np.uint8), 12.5))
    app.state.inference = stub
    response = TestClient(app).post("/api/predict", files={"image": ("input.png", image_bytes(), "image/png")}, data={"resolution": "256"})
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["original_width"] == 12
    assert body["metadata"]["original_height"] == 8
    raw = np.load(io.BytesIO(base64.b64decode(body["prediction_npy_base64"])), allow_pickle=False)
    assert raw.shape == (256, 256) and raw.dtype == np.float32
    stub.predict_async.assert_awaited_once()


def test_rejects_bad_resolution_and_checkpoint():
    client = TestClient(app)
    files = {"image": ("input.png", image_bytes(), "image/png")}
    assert client.post("/api/predict", files=files, data={"resolution": "1024"}).status_code == 422
    assert client.post("/api/predict", files={"image": ("input.png", image_bytes(), "image/png")}, data={"checkpoint": "../../bad"}).status_code == 422


def test_rejects_malformed_and_wrong_type():
    client = TestClient(app)
    assert client.post("/api/predict", files={"image": ("x.png", b"bad", "image/png")}).status_code == 400
    assert client.post("/api/predict", files={"image": ("x.gif", b"bad", "image/gif")}).status_code == 415


def test_rejects_seed_outside_torch_range():
    response = TestClient(app).post(
        "/api/predict",
        files={"image": ("input.png", image_bytes(), "image/png")},
        data={"seed": "9223372036854775808"},
    )
    assert response.status_code == 422
