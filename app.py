import gradio as gr
import torch
import mmcv
from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os

# Load config and checkpoint
CONFIG_FILE = 'configs/plane/faster-rcnn_r50_fpn_plane.py'
CHECKPOINT_FILE = 'work_dirs/faster-rcnn_r50_fpn_plane/epoch_1.pth'  # change if needed

# Initialize model once
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0' if torch.cuda.is_available() else 'cpu')

def detect_planes(image):
    # Convert image to BGR (MMDetection uses OpenCV)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Run inference
    result = inference_detector(model, img_bgr)

    # Visualize results on a copy
    vis_img = model.show_result(img_bgr, result, score_thr=0.5, show=False)
    
    # Convert BGR back to RGB for display
    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    return vis_rgb

# Launch Gradio app
gr.Interface(
    fn=detect_planes,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(type="numpy", label="Detected Planes"),
    title="Plane Detector",
    description="Upload an aerial image to detect planes using a trained Faster R-CNN model."
).launch()
