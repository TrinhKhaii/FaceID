from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
from contextlib import asynccontextmanager
import torch
import base64
from product.core.recognition import FaceRecognition
import time


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ml_models["recognition"] = FaceRecognition(
        weight_path="./product/models/ms1mv3_arcface_r100_fp16/backbone.pth", 
        device=device
    )
    yield
    ml_models.clear()

class RecognitionRequest(BaseModel):
    aligned_face_base64: str

app = FastAPI(title="Face Recognition API", lifespan=lifespan)

@app.post("/recognize")
async def recognition(request: RecognitionRequest):
    try:
        img_bytes = base64.b64decode(request.aligned_face_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img)

        recognition = ml_models["recognition"]
        emb = recognition.get_embedding(img_array)

        if emb is None:
            raise HTTPException(400, "Cannot extract embedding vector")
        
        emb_list = emb.flatten().tolist()
        return {
            "success": True,
            "embedding": emb_list,
            "embedding_size": len(emb_list)
        }
    except Exception as e:
        raise HTTPException(500, str(e))


