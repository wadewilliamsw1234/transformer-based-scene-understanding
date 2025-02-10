# src/visualization.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def overlay_segmentation(image_pil, segmentation_map, alpha=0.5):
    """
    Creates a color overlay of segmentation on the original image.
    image_pil: PIL image
    segmentation_map: 2D array with class IDs
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    num_classes = segmentation_map.max() + 1
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    seg_color = colors[segmentation_map]
    seg_color = Image.fromarray(seg_color).convert("RGBA")

    image_rgba = image_pil.convert("RGBA")
    overlayed = Image.blend(image_rgba, seg_color, alpha)
    return overlayed

def draw_yolo_detections(image_bgr, results):
    """
    Takes an OpenCV BGR image and YOLO results, draws bounding boxes.
    results: YOLO results object
    """
    # Actually YOLO's .plot() already does this, but if you want custom:
    annotated = image_bgr.copy()
    # parse results[0].boxes, etc.
    return annotated

def colorize_depth(depth_map, cmap='inferno'):
    """
    Convert a depth_map (numpy 2D) to a color heatmap using a matplotlib colormap.
    Returns a 3-channel BGR or RGB image.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    depth_min, depth_max = depth_map.min(), depth_map.max()
    norm_depth = (depth_map - depth_min)/(depth_max - depth_min + 1e-8)
    colormap = plt.get_cmap(cmap)
    colored = colormap(norm_depth)[:, :, :3]  # RGBA → RGB
    colored = (colored*255).astype(np.uint8)
    return colored
