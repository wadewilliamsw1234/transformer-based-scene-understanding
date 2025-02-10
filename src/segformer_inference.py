# src/segformer_inference.py
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

class SegformerInference:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()

    def predict(self, image):
        """
        image can be a PIL image or file path.
        Returns a 2D numpy array with class IDs.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        # upsample
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        return pred_seg
