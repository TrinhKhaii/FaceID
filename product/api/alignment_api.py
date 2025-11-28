from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
from contextlib import asynccontextmanager
import torch
from product.core.alignment import FaceAlignment
import base64
import time

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ml_models["alignment"] = FaceAlignment(
        landmark_model_path="./product/models/weights/1k3d68.onnx",
        output_size=112
    )
    yield
    ml_models.clear()

class AlignmentRequest(BaseModel):
    cropped_face_base64: str

app = FastAPI(title="Face Alignment API", lifespan=lifespan)

@app.post("/align")
async def alignment(request: AlignmentRequest):
    try:
        img_bytes = base64.b64decode(request.cropped_face_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        alignment = ml_models["alignment"]
        aligned_face = alignment.align(img)
        if aligned_face is None:
            raise HTTPException(400, "Cannot align image")
        
        if isinstance(aligned_face, np.ndarray):
            aligned_face_pil = Image.fromarray(aligned_face.astype('uint8'))
        else:
            aligned_face_pil = aligned_face

        buffered = io.BytesIO()
        aligned_face_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return {
            "success": True,
            "aligned_face_base64": img_base64,
            "width": aligned_face_pil.width,
            "height": aligned_face_pil.height
        }
    except Exception as e:
        raise HTTPException(500, str(e))