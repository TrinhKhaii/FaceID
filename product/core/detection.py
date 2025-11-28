from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import torch


class FaceDetection:
    def __init__(self, image_size=160, margin=20, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            post_process=False,
            device=self.device
        )
    
    def detect_and_crop(self, portrait_image):
        """
        Input: portrait_image - PIL Image 
        Output: cropped_face - numpy array (160, 160, 3) 
        """
        cropped_face_tensor = self.mtcnn(portrait_image)
        if cropped_face_tensor is None:
            return None
        cropped_face = (cropped_face_tensor.permute(1, 2, 0) * 255).byte().numpy()
        return cropped_face
        


