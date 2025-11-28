import os
import numpy as np
import onnxruntime
import cv2
from PIL import Image
from skimage import transform as trans
from typing import List, NamedTuple


class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1


class ScrfdLocal(object):
    def __init__(
            self, 
            model_path: str,
            providers=None,
            verbose=True
        ):
        assert os.path.exists(model_path), f"Model file does not exist: {model_path}"
        
        self.model_path = model_path
        
        if providers is None:
            if onnxruntime.get_device() == 'GPU':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        self.session = onnxruntime.InferenceSession(
            self.model_path, 
            providers=providers
        )

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_shape = input_cfg.shape
        self.input_size = tuple(self.input_shape[2:4][::-1])  # (W, H)
        
        output_cfg = self.session.get_outputs()[0]
        self.output_name = output_cfg.name
        
        if verbose:
            print(f"Model loaded: {model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Providers: {self.session.get_providers()}")

    def transform(self, data, center, output_size, scale, rotation):
        scale_ratio = scale
        rot = float(rotation) * np.pi / 180.0
        
        t1 = trans.SimilarityTransform(scale=scale_ratio)
        cx = center[0] * scale_ratio
        cy = center[1] * scale_ratio
        t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
        t3 = trans.SimilarityTransform(rotation=rot)
        t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
        t = t1 + t2 + t3 + t4
        M = t.params[0:2]
        
        cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
        return cropped, M

    def trans_points2d(self, pts, M):
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            new_pts[i] = new_pt[0:2]
        return new_pts

    def trans_points3d(self, pts, M):
        scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
        new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M, new_pt)
            new_pts[i][0:2] = new_pt[0:2]
            new_pts[i][2] = pts[i][2] * scale
        return new_pts
    
    def trans_points(self, pts, M):
        if pts.shape[1] == 2:
            return self.trans_points2d(pts, M)
        else:
            return self.trans_points3d(pts, M)
        
    def run_inference(self, input_blob: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            [self.output_name], 
            {self.input_name: input_blob}
        )
        return outputs[0][0] 
    
    def pre_process(self, input_image, input_std=1.0, input_mean=0.0):
        img = input_image[:,:,::-1]  # RGB to BGR
        bbox = [0, 0, img.shape[0], img.shape[1]]
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = 192 / (max(w, h) * 1.5)
        
        aimg, M = self.transform(img, center, 192, _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0/input_std, input_size, 
            (input_mean, input_mean, input_mean), 
            swapRB=True
        )
        return blob.astype(np.float32), M
    
    def post_process(self, pred, M, lmk_num=68, input_size=(192, 192)):
        pred = pred.reshape(-1)
        
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
            
        if lmk_num < pred.shape[0]:
            pred = pred[lmk_num*-1:,:]
        
        pred = pred.copy()
        pred[:, 0:2] = pred[:, 0:2] + 1
        pred[:, 0:2] = pred[:, 0:2] * (input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (input_size[0] // 2)
        
        IM = cv2.invertAffineTransform(M)
        pred = self.trans_points(pred, IM)
        
        pred = [Keypoint(x=xy[0], y=xy[1]) for xy in pred]
        return pred
    
    def forward(self, input_image) -> tuple:
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
        
        blob, M = self.pre_process(input_image)

        pred = self.run_inference(blob)

        poses = self.post_process(pred, M)
        
        poses = [
            Keypoint(
                x=item.x / input_image.shape[1], 
                y=item.y / input_image.shape[0]
            ) 
            for item in poses
        ]
        
        return poses, blob

    def draw_poses(self, poses, H, W):
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        eps = 0.01
        
        for keypoint in poses:
            x, y = keypoint.x, keypoint.y
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
        
        return canvas

