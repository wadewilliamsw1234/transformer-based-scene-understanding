# src/data_loading.py
import os
import glob
from PIL import Image

class ImageFolderLoader:
    """
    A simple class to load images from a folder.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                           glob.glob(os.path.join(folder_path, "*.png")) + \
                           glob.glob(os.path.join(folder_path, "*.jpeg"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return img, img_path
