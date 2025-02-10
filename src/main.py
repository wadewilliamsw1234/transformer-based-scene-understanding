# src/main.py
import os
import sys
from PIL import Image
import cv2
import argparse
import glob

from dino_inference import DinoInference
from segformer_inference import SegformerInference
from yolo_inference import YOLOInference
from midas_depth_estimation import MidasDepthEstimator
from visualization import overlay_segmentation, colorize_depth
import torch

def process_single_image(image_path, output_dir, yolo, segformer, midas_estimator, dino):
    """
    Runs YOLO detection, SegFormer segmentation, MiDaS depth, and DINO features
    on a single image, then saves the outputs to 'output_dir'.
    """
    # 1) YOLO for detection
    results, annotated_frame = yolo.predict(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path_yolo = os.path.join(output_dir, f"{base_name}_yolo.jpg")
    cv2.imwrite(out_path_yolo, annotated_frame)
    print(f"YOLO detection saved to {out_path_yolo}")

    # 2) SegFormer for segmentation
    pred_seg = segformer.predict(image_path)
    image_pil = Image.open(image_path).convert("RGB")
    seg_overlay = overlay_segmentation(image_pil, pred_seg)
    out_path_seg = os.path.join(output_dir, f"{base_name}_segformer.png")
    seg_overlay.save(out_path_seg)
    print(f"SegFormer overlay saved to {out_path_seg}")

    # 3) MiDaS depth
    depth_map = midas_estimator.estimate_depth(image_path)
    depth_color = colorize_depth(depth_map, cmap='inferno')
    out_path_depth = os.path.join(output_dir, f"{base_name}_midas.png")
    cv2.imwrite(out_path_depth, depth_color)
    print(f"Depth map saved to {out_path_depth}")

    # 4) DINO for feature extraction (example)
    features = dino.extract_features(image_path)
    print(f"DINO features shape for {image_path}:", features.shape)

def run_demo(args):
    # Initialize all models once
    yolo = YOLOInference(model_path="yolov8n.pt")
    segformer = SegformerInference()
    midas_estimator = MidasDepthEstimator("dpt_hybrid_384")
    dino = DinoInference()

    # Check if the user passed a folder or a single image
    if os.path.isdir(args.image_path):
        # It's a directory: loop over all .jpg, .png, etc.
        image_files = glob.glob(os.path.join(args.image_path, "*.jpg")) + \
                      glob.glob(os.path.join(args.image_path, "*.jpeg")) + \
                      glob.glob(os.path.join(args.image_path, "*.png"))

        if not image_files:
            print(f"No image files found in folder: {args.image_path}")
            return

        print(f"Processing {len(image_files)} images in folder: {args.image_path}")
        for img_path in image_files:
            process_single_image(img_path, args.output_dir, yolo, segformer, midas_estimator, dino)
    else:
        # It's a single file
        if not os.path.isfile(args.image_path):
            print(f"File not found: {args.image_path}")
            return
        process_single_image(args.image_path, args.output_dir, yolo, segformer, midas_estimator, dino)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="bus.jpg",
                        help="Path to a single test image or a folder of images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Where to store results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_demo(args)

if __name__ == "__main__":
    main()
