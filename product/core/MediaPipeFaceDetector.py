import mediapipe as mp
import cv2 
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any

class MediaPipeFaceDetector:
    def __init__(self, model_selection=1, min_detection_confidence=0.5, margin=20):
        self.margin = margin
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection = model_selection,
            min_detection_confidence = min_detection_confidence
        )

    def detect_and_crop(self, image):
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        if len(image_np.shape) == 2:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_np
            
        results = self.face_detection.process(image_rgb)
        if not results.detections:
            return None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w = image_rgb.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        x = max(0, x - self.margin)
        y = max(0, y - self.margin)
        width = min(w - x, width + 2 * self.margin)
        height = min(h - y, height + 2 * self.margin)
        
        cropped_face = image_rgb[y:y+height, x:x+width]
        cropped_face_pil = Image.fromarray(cropped_face)
        
        return cropped_face_pil
    
    def detect_all(self, image: Image.Image) -> List[Tuple[Image.Image, List[int]]]:
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        if len(image_np.shape) == 2:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif len(image_np.shape) == 4:
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
            image_rgb = image_np

        results = self.face_detection.process(image_rgb)
        if not results.detections:
            return []
            
        detections_list = []
        h, w = image_rgb.shape[:2]

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x_new = max(0, x - self.margin)
            y_new = max(0, y - self.margin)
            
            width_new = min(w - x_new, width + 2 * self.margin)
            height_new = min(h - y_new, height + 2 * self.margin)

            cropped_face = image_rgb[y_new : y_new+height_new, x_new : x_new+width_new]
            cropped_face_pil = Image.fromarray(cropped_face)

            bbox_final = [x_new, y_new, width_new, height_new]

            detections_list.append((cropped_face_pil, bbox_final))

        return detections_list