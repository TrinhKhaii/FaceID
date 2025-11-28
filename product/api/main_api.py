import io
import time
from contextlib import asynccontextmanager
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from product.main import FaceIDPipelineAPIClient
from product.core.qdrant import QdrantFaceBank
from fastapi.middleware.cors import CORSMiddleware
import datetime


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    ml_models.clear()


app = FastAPI(title="FaceID Full Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline_client = FaceIDPipelineAPIClient(
    detector_url="http://localhost:8001",
    alignment_url="http://localhost:8002",
    recognition_urls=[
        "http://localhost:8003",
        "http://localhost:8005"
    ],
)

facebank = QdrantFaceBank(
    collection_name="FaceID",
    host="localhost",
    port=6333,
    embedding_size=512,
)

class CropInput(BaseModel):
    cropped_face_base64: str


class BenchmarkResponse(BaseModel):
    name: str
    score: float
    updated: bool
    detector_time: float
    alignment_time: float
    recognition_time: float
    search_time: float
    total_time: float
    top_scores: dict


def normalize(v):
    return v / np.linalg.norm(v) if v is not None else None

@app.post("/benchmark")
async def full_benchmark(image: UploadFile = File(...)):
    try:
        total_start = time.time()

        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # 1. Detector
        detect_start = time.time()
        cropped_base64 = pipeline_client.call_detector(img)
        detect_time = time.time() - detect_start
        print(f"detect_time: {detect_time:.4f}s")

        if cropped_base64 is None:
            print("Detection failed")
            return JSONResponse(
                status_code=400,
                content={"error": "Detection failed"},
            )

        # 2. Alignment
        align_start = time.time()
        aligned_face = pipeline_client.call_alignment(cropped_base64)
        align_time = time.time() - align_start
        print(f"align_time: {align_time:.4f}s")

        if aligned_face is None:
            print("Alignment failed")
            return JSONResponse(
                status_code=400,
                content={"error": "Alignment failed"},
            )

        # 3. Recognition
        recognize_start = time.time()
        emb = pipeline_client.call_recognition(aligned_face)
        recognition_time = time.time() - recognize_start
        print(f"recognition_time: {recognition_time:.4f}s")

        if emb is None:
            print("Recognition failed")
            return JSONResponse(
                status_code=400,
                content={"error": "Recognition failed"},
            )

        emb = normalize(emb)

        search_start = time.time()
        best_name, max_score, updated, scores = facebank.recognize_and_adaptive_update(
            emb,
            high_threshold=0.70,
            low_threshold=0.40,
        )
        search_time = time.time() - search_start
        print(f"search_time: {search_time:.4f}s")

        total_time = time.time() - total_start

        top_scores: dict[str, list[float]] = {}
        for name, sc_list in scores.items():
            top_scores[name] = sorted(sc_list, reverse=True)[:3]

        return BenchmarkResponse(
            name=best_name or "Unknown",
            score=float(max_score),
            updated=updated,
            detector_time=detect_time,
            alignment_time=align_time,
            recognition_time=recognition_time,
            search_time=search_time,
            total_time=total_time,
            top_scores=top_scores,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Internal server error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal error: {str(e)}"},
        )



@app.post("/identify_crop")
async def identify_crop(data: CropInput):
    try:
        start_time = time.time()
        cropped_b64 = data.cropped_face_base64

        aligned_face_b64 = pipeline_client.call_alignment(cropped_b64)
        if aligned_face_b64 is None:
            return {"name": "Align Error", "score": 0.0}

        emb = pipeline_client.call_recognition(aligned_face_b64)
        
        if emb is None:
            return {"name": "Recog Error", "score": 0.0}

        emb_np = normalize(emb)

        name, score, updated, _ = facebank.recognize_and_adaptive_update(
            emb_np, 
            high_threshold=0.70, 
            low_threshold=0.40
        )
        
        process_time = (time.time() - start_time) * 1000
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Identified: {name} ({score:.2f}) in {process_time:.0f}ms")

        return {
            "name": name if name else "Unknown",
            "score": float(score)
        }
        
    except Exception as e:
        print(f"Error in identify_crop: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/add_user_form")
async def add_user_form(
    full_name: str = Form(...),
    face_image_1: UploadFile = File(...),
    face_image_2: UploadFile = File(...),
):
    try:
        embeddings_added = 0
        errors: list[str] = []

        async def process_one_image(file: UploadFile, image_id: int):
            nonlocal embeddings_added
            try:
                img_bytes = await file.read()
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                emb = pipeline_client.forward(img)

                if emb is not None:
                    emb = normalize(emb)
                    facebank.add_user(
                        full_name,
                        emb,
                        metadata={"image_type": "original", "image_id": image_id},
                    )
                    embeddings_added += 1

                img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                emb_flipped = pipeline_client.forward(img_flipped)
                if emb_flipped is not None:
                    emb_flipped = normalize(emb_flipped)
                    facebank.add_user(
                        full_name,
                        emb_flipped,
                        metadata={"image_type": "flipped", "image_id": image_id},
                    )
                    embeddings_added += 1
                print(f"embeddings_added: {embeddings_added}")

            except Exception as e:
                errors.append(f"Image {image_id} error: {str(e)}")

        await process_one_image(face_image_1, image_id=1)
        await process_one_image(face_image_2, image_id=2)

        if embeddings_added == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No embeddings can be extracted for the user {full_name}. Error: {errors}",
            )

        return {
            "success": True,
            "name": full_name,
            "embeddings_added": embeddings_added,
            "errors": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /add_user_form: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
