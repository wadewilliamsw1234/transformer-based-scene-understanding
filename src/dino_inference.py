# src/dino_inference.py
import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

class DinoInference:
    def __init__(self, model_name="facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, image):
        """
        image can be a PIL image or file path string.
        Returns the last hidden state from DINOv2.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
