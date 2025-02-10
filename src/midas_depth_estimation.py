# src/midas_depth_estimation.py
import torch
import cv2
import numpy as np
from midas.model_loader import default_models, load_model

class MidasDepthEstimator:
    def __init__(self, model_type="dpt_hybrid"):
        self.model_path = default_models[model_type]
        self.transform = default_models[model_type]["transform"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas_model = load_model(
            self.model_path, self.transform,
            optimize=False, device=self.device
        )
        self.midas_model.eval()

    def estimate_depth(self, image_path):
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        input_batch = self.midas_model.transforms(image_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return depth_map
