import numpy as np
import torch
from product.core.iresnet import IResNet, IBasicBlock

class FaceRecognition:
    def __init__(self, weight_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = IResNet(IBasicBlock, [3, 13, 30, 3])
        self.model.load_state_dict(torch.load(weight_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, aligned_face):
        """
        Input: aligned_face - numpy array (112, 112, 3)
        Output: embedding - numpy array (1, 512)
        """
        img = np.transpose(aligned_face, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5

        embedding = self.model(img_tensor).cpu().numpy().flatten()

        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    