from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
from contextlib import asynccontextmanager
import torch
from product.core.MediaPipeFaceDetector import MediaPipeFaceDetector
import base64
import time
from fastapi.middleware.cors import CORSMiddleware

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ml_models["detector"] = MediaPipeFaceDetector(
        model_selection=1,
        min_detection_confidence=0.7,
        margin=20
    )
    yield
    ml_models.clear()

app = FastAPI(title="Face Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/detect")
async def detector(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        detector = ml_models["detector"]
        
        results = detector.detect_all(img)

        if not results:
            return {"success": False, "message": "No face detected"}

        cropped_face, bbox = results[0]

        buffered = io.BytesIO()
        cropped_face.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "detections": [
                {
                    "cropped_face_base64": img_base64,
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                    "width": cropped_face.width,
                    "height": cropped_face.height
                }
            ]
        }

    except Exception as e:
        print(f"Server error: {str(e)}")
        raise HTTPException(500, str(e))
