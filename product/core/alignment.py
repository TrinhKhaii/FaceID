from PIL import Image
import numpy as np
from product.core.scrfd import ScrfdLocal
import product.core.align as align
import cv2


class FaceAlignment:
    def __init__(self, landmark_model_path, output_size=112):
        self.scrfd = ScrfdLocal(model_path=landmark_model_path)
        self.output_size = output_size

    def align(self, cropped_face):
        """
        Input: cropped_face - numpy array (160, 160, 3)
        Output: aligned_face - numpy array (112, 112, 3)
        """
        if isinstance(cropped_face, Image.Image):
            face_np = np.array(cropped_face)
            face_pil = cropped_face
        elif isinstance(cropped_face, np.ndarray):
            face_np = cropped_face
            face_pil = Image.fromarray(cropped_face.astype('uint8'))
        else:
            print(f"Error: Unsupported input type: {type(cropped_face)}")
            return None

        
        landmarks, _ = self.scrfd.forward(face_pil)

        if len(landmarks) == 0:
            print("Warning: No landmarks detected, using simple resize")
            return cv2.resize(cropped_face, (self.output_size, self.output_size))
        

        if len(landmarks) < 68:
            print(f"Warning: Only {len(landmarks)} landmarks detected")
            return cv2.resize(cropped_face, (self.output_size, self.output_size))
            
        key_points_idx = [36, 45, 30, 48, 54]
        landmarks_5pts = []
        
        for i in key_points_idx:
            x = landmarks[i].x * face_pil.width
            y = landmarks[i].y * face_pil.height
            landmarks_5pts.append([x, y])

        aligned_face = align.norm_crop(np.array(face_pil), np.array(landmarks_5pts, dtype=np.float32))
        
        return aligned_face  