# Transformer-Based Scene Understanding

## Project Overview
This project demonstrates how to use pre-trained Transformer-based models (DINOv2, SegFormer, YOLO, and MiDaS) for:
- Object detection
- Semantic segmentation
- Depth estimation
- Feature extraction (for classification or other tasks)

## Setup
1. **Clone** this repo and **cd** into it.
2. Create/activate a conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate scene_env
    ```
   or install from `requirements.txt`.
3. **Run**:
    ```bash
    python src/main.py --image_path path/to/image.jpg --output_dir outputs
    ```

## Folder Layout
- `src/` : Contains all Python modules and scripts
- `data/`: (Optional) place small test images
- `notebooks/`: (Optional) Jupyter notebooks

## Models
- **DINOv2**: Feature extraction
- **SegFormer**: Semantic segmentation
- **YOLOv8**: Object detection
- **MiDaS**: Depth estimation

