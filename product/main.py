import numpy as np
from product.core.detection import *
from product.core.alignment import *
from product.core.recognition import *
from product.core.MediaPipeFaceDetector import *
from PIL import Image
import time
from product.core.qdrant import *
import io
import requests
from fastapi import HTTPException
import random
from qdrant_client import QdrantClient
from qdrant_client import models 


def init_qdrant_collection(
    collection_name: str = "FaceID",
    host: str = "localhost",
    port: int = 6333,
    embedding_size: int = 512
):
    client = QdrantClient(host=host, port=port)
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=Distance.COSINE
            )
        )
        print(f"Create Qdrant collection {collection_name} with size={embedding_size}")
    else:
        print(f"Collection {collection_name} existed.")
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="name",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                lowercase=True
            )
        )
    except Exception as e:
        print("Cannot create qdrant field 'nane'")
    return client


def cosine_similarity(a, b):
    return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))


class FaceIDPipeline:
    def __init__(self, landmark_model_path, recognition_weight_path, device):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # self.detector = FaceDetection(image_size=160, margin=20, device=self.device)
        self.detector = MediaPipeFaceDetector(
            model_selection=1,
            min_detection_confidence=0.7,
            margin=20
        )
        self.alignment = FaceAlignment(landmark_model_path, output_size=112)
        self.recognition = FaceRecognition(recognition_weight_path, device=self.device)

    
    def forward(self, portrait_image):
        # start = time.time()
        cropped_face = self.detector.detect_and_crop(portrait_image)
        # end = time.time()
        # print("Detector time:", end - start)

        if cropped_face is None:
            return None
        # start = time.time()
        aligned_face = self.alignment.align(cropped_face)
        # end = time.time()
        # print("alignment time:", end - start)
        # start = time.time()
        emb = self.recognition.get_embedding(aligned_face)
        # end = time.time()
        # print("recognition time:", end - start)
        return emb
    

class FaceIDPipelineAPIClient:
    def __init__(self, 
                 detector_url="http://localhost:8001",
                 alignment_url="http://localhost:8002",
                 recognition_urls=None,
                 max_retries=3,
                 retry_delay=1.0):
        self.detector_url = detector_url
        self.alignment_url = alignment_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if recognition_urls is None:
            self.recognition_urls = ["http://localhost:8003"]
        elif isinstance(recognition_urls, str):
            self.recognition_urls = [recognition_urls]
        else:
            self.recognition_urls = recognition_urls
        
    def call_api_with_retry(self, url, files=None, json=None, method="post"):
        for attempt in range(self.max_retries):
            try:
                if method == "post":
                    response = requests.post(url, files=files, json=json, timeout=30)
                    if response.status_code == 200:
                        return response.json()
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: Status {response.status_code} at {url}")    
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1}/{self.max_retries} connection error: {e} at {url}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        print(f"All {self.max_retries} attempts failed for {url}")
        return None

    def forward(self, portrait_image):
        try:
            # start = time.time()
            detections = self.call_detector(portrait_image)
            # print(f"Detector time: {time.time() - start}")
            if not detections or len(detections) == 0:
                print("No face detected")
                return None
            
            first_face = detections[0]
            cropped_base64 = first_face.get("cropped_face_base64")
            
            # start = time.time()
            aligned_face = self.call_alignment(cropped_base64)
            # print(f"Alignment time: {time.time() - start}")

            if aligned_face is None:
                print("hello")
                return None
            
            # start = time.time()
            emb = self.call_recognition(aligned_face)
            # print(f"Recognition time: {time.time() - start}")

            return emb
        except Exception as e:
            print(f"error: {str(e)}")
            return None

    def call_detector(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)
        result = self.call_api_with_retry(
            f"{self.detector_url}/detect",
            files={"image": ("image.jpg", buffered, "image/jpeg")}
        )

        if result and result.get("success"):
            return result.get("detections", [])
        return None 
        
    
    def call_alignment(self, cropped_face_base64):
        result = self.call_api_with_retry(
            f"{self.alignment_url}/align",
            json={"cropped_face_base64": cropped_face_base64},
        )
        
        if result:
            return result.get("aligned_face_base64")
        return None
        
        
    def call_recognition(self, aligned_face_base64):
        for attempt in range(self.max_retries):
            current_url = random.choice(self.recognition_urls)

            try:
            
                response = requests.post(
                    f"{current_url}/recognize",
                    json={"aligned_face_base64": aligned_face_base64},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return np.array(result["embedding"], dtype=np.float32)
                print(f"Recog attempt {attempt + 1} failed on {current_url}: Status {response.status_code}")    

            except requests.RequestException as e:
                print(f"Recog attempt {attempt + 1} error on {current_url}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        print("All recognition attempts failed.")
        return None

    
class FaceBank:
    def __init__(self):
        self.users = {}

    def add_user(self, name, embedding):
        if name not in self.users:
            self.users[name] = []
        self.users[name].append(embedding)

    def recognize_and_adaptive_update(self, test_emb, high_threshold = 0.7, low_threshold = 0.4):
        best_name = None
        max_score = -1
        updated = False
        scores = {}
       
        start_search = time.time()
        for name, embeddings in self.users.items():
            for emb in embeddings:
                score = np.dot(test_emb, emb)
                if name not in scores:
                    scores[name] = []
                scores[name].append(score)
                if score > max_score:
                    max_score = score
                    best_name = name
        end_search = time.time()
        search_time = end_search - start_search
        print(f"Search time: {search_time}"  )
        if max_score >= high_threshold:
            print(f"score: {max_score:.3f} > high_threshold")
            updated = False
        elif max_score >= low_threshold:
            self.add_user(best_name, test_emb)
            print(f"low_threshold < score: {max_score:.3f} < high_threshold")
            updated = True
        else:
            print(f"Unknown score: {max_score:.3f}")
            updated = False
        return best_name, max_score, updated, scores
    
    def get_user_embedding_count(self, name):
        return len(self.users.get(name, []))
    


def add_user(name, img_path_1, img_path_2, facebank, pipeline):
    embeddings_added = 0

    for i, img_path in enumerate([img_path_1, img_path_2], 1):
        
        img = Image.open(img_path)

        emb = pipeline.forward(img)
        if emb is not None:
            emb = emb / np.linalg.norm(emb)
            facebank.add_user(name, emb, metadata={"image_type": "original", "image_id": i})
            embeddings_added += 1
        
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        emb_flipped = pipeline.forward(img_flipped)
        if emb_flipped is not None:
            emb_flipped = emb_flipped / np.linalg.norm(emb_flipped)
            facebank.add_user(name, emb_flipped, metadata={"image_type": "flipped", "image_id": i})
            embeddings_added += 1

    return embeddings_added


    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_init = time.time()
    main_url="http://localhost:8004"
    pipeline = FaceIDPipelineAPIClient(
        detector_url="http://localhost:8001",
        alignment_url="http://localhost:8002",
        recognition_urls=[
            "http://localhost:8003",
            "http://localhost:8005"
    ],
    )
    end_init = time.time()
    print("Init model time:", end_init - start_init)

    qdrant_client = init_qdrant_collection(
        collection_name="FaceID",
        host="localhost",
        port=6333,
        embedding_size=512
    )

    start_init = time.time()
    facebank = QdrantFaceBank(
        collection_name="FaceID",
        host="localhost",
        port=6333,
        embedding_size=512
    )

    # add_user(
    #     "Trinh Khai", 
    #     "dev/tests/data/khai.JPG", 
    #     "dev/tests/data/khai_2.jpg",
    #     facebank, 
    #     pipeline
    # )

    # add_user(
    #     "Phuong Anh", 
    #     "dev/tests/data/phuonganh.JPG", 
    #     "dev/tests/data/phuonganh_test_2.JPG", 
    #     facebank, 
    #     pipeline
    # )
    # add_user(
    #     "Hoang Anh", 
    #     "dev/tests/data/hoanganh_test_2.jpg", 
    #     "dev/tests/data/hoanganh_test_4.jpg", 
    #     facebank, 
    #     pipeline
    # )
    # add_user(
    #     "Ronaldo", 
    #     "dev/tests/data/ronaldo.jpg", 
    #     "dev/tests/data/ronaldo_2.jpg", 
    #     facebank, 
    #     pipeline
    # )
    # add_user(
    #     "Messi", 
    #     "dev/tests/data/messi.jpg", 
    #     "dev/tests/data/messi_2.jpg", 
    #     facebank, 
    #     pipeline
    # )
    # add_user(
    #     "Bao Thach", 
    #     "dev/tests/data/thach.jpg", 
    #     "dev/tests/data/thach-2.jpg", 
    #     facebank, 
    #     pipeline
    # )
    
    end_init = time.time()
    print("Init database time:", end_init - start_init)
    print("----------------------")

    # test_img = Image.open('dev/tests/data/neymar.jpg')
    # buffered = io.BytesIO()
    # test_img.save(buffered, format="JPEG")
    # buffered.seek(0)
    # response = requests.post(
    #             f"{main_url}",
    #             files={"image": ("image.jpg", buffered, "image/jpeg")},
    #             timeout=30
    #         )
    # test_emb = pipeline.forward(test_img)
    
    # if test_emb is not None:
    #     test_emb = test_emb / np.linalg.norm(test_emb)
        
    #     best_name, max_score, updated, scores = facebank.recognize_and_adaptive_update(
    #         test_emb,
    #         high_threshold=0.70,
    #         low_threshold=0.50
    #     )
        
        
    #     print("\nTop scores :")
    #     for name in sorted(scores.keys()):
    #         top_scores = sorted(scores[name], reverse=True)[:3]
    #         scores_str = ", ".join([f"{s:.3f}" for s in top_scores])
    #         print(f"  {name}: {scores_str}")
    



