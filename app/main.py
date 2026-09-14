"""FastAPI HTTP and static frontend entry point."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.inference import (
    DEFAULT_CHECKPOINT,
    SUPPORTED_CHECKPOINTS,
    InferenceConfigurationError,
    InferenceOutOfMemoryError,
    MarigoldInference,
    encode_npy,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", Path(__file__).parent / "static"))


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictionMetadata(BaseModel):
    checkpoint: str
    resolution: int
    original_width: int
    original_height: int
    raw_width: int
    raw_height: int
    inference_duration_ms: float = Field(ge=0)


class PredictionResponse(BaseModel):
    depth_png_base64: str = Field(description="Display PNG resized to the input dimensions")
    prediction_npy_base64: str = Field(description="Raw float32 NumPy array at inference resolution")
    metadata: PredictionMetadata


app = FastAPI(
    title="Marigold V2 Inference",
    version="2.0.0",
    description="GPU-backed monocular depth estimation using the repository's Marigold V2 graph.",
)
app.state.inference = MarigoldInference()


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Cheap liveness check. It does not load the model or run inference."""
    return HealthResponse()


@app.post(
    "/api/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={400: {"description": "Invalid image or parameters"}, 413: {"description": "Upload too large"}, 503: {"description": "GPU or model unavailable"}},
)
async def predict(
    image: Annotated[UploadFile, File(description="JPEG or PNG input image")],
    resolution: Annotated[int, Form(description="Square inference resolution: 256 or 512")] = 256,
    checkpoint: Annotated[str, Form(description="Allow-listed Marigold checkpoint")] = DEFAULT_CHECKPOINT,
    seed: Annotated[
        int,
        Form(
            description="VAE encoder random seed",
            ge=0,
            le=9_223_372_036_854_775_807,
        ),
    ] = 2025,
) -> PredictionResponse:
    if resolution not in (256, 512):
        raise HTTPException(422, "resolution must be 256 or 512")
    if checkpoint not in SUPPORTED_CHECKPOINTS:
        raise HTTPException(422, f"checkpoint must be one of: {', '.join(SUPPORTED_CHECKPOINTS)}")
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(415, "Only JPEG and PNG uploads are supported")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    await image.close()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(data)) as opened:
            source = opened.convert("RGB")
            source.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, "Malformed or unsafe image") from exc

    width, height = source.size
    try:
        result = await app.state.inference.predict_async(
            source, resolution, checkpoint, seed
        )
    except InferenceOutOfMemoryError as exc:
        raise HTTPException(503, str(exc)) from exc
    except InferenceConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        source.close()

    png = io.BytesIO()
    Image.fromarray(result.display).save(png, format="PNG", optimize=True)
    return PredictionResponse(
        depth_png_base64=base64.b64encode(png.getvalue()).decode("ascii"),
        prediction_npy_base64=base64.b64encode(encode_npy(result.raw)).decode("ascii"),
        metadata=PredictionMetadata(
            checkpoint=checkpoint,
            resolution=resolution,
            original_width=width,
            original_height=height,
            raw_width=int(result.raw.shape[1]),
            raw_height=int(result.raw.shape[0]),
            inference_duration_ms=round(result.duration_ms, 2),
        ),
    )


if FRONTEND_DIR.is_dir():
    assets = FRONTEND_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        candidate = (FRONTEND_DIR / path).resolve()
        if path and candidate.is_file() and FRONTEND_DIR.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIR / "index.html")
