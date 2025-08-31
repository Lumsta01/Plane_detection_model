# debug_gradio_plane_detection.py - Let's see what's happening

import gradio as gr
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
from PIL import Image, ImageDraw, ImageFont
import torch

# Model paths
CONFIG_FILE = 'configs/plane/faster-rcnn_r50_fpn_plane.py'
CHECKPOINT_FILE = 'work_dirs/faster-rcnn_r50_fpn_plane/epoch_12.pth'

# Load model
print("Loading your trained model...")
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')
print("Model loaded successfully!")

def debug_and_detect(uploaded_image):
    """Debug version to see what's happening"""
    
    try:
        # Convert PIL Image to numpy array
        if isinstance(uploaded_image, Image.Image):
            image_array = np.array(uploaded_image)
        else:
            image_array = uploaded_image
        
        print(f"Input image shape: {image_array.shape}")
        
        # Run detection
        detection_results = inference_detector(model, image_array)
        print(f"Detection results type: {type(detection_results)}")
        
        # Let's examine the results structure
        if hasattr(detection_results, 'pred_instances'):
            pred_instances = detection_results.pred_instances
            print(f"Number of detections: {len(pred_instances)}")
            
            if len(pred_instances) > 0:
                print(f"Bboxes shape: {pred_instances.bboxes.shape}")
                print(f"Scores: {pred_instances.scores}")
                print(f"Max score: {pred_instances.scores.max()}")
                
                # Manual drawing for debugging
                result_img = Image.fromarray(image_array)
                draw = ImageDraw.Draw(result_img)
                
                # Draw bounding boxes manually
                for i, (bbox, score) in enumerate(zip(pred_instances.bboxes, pred_instances.scores)):
                    if score > 0.1:  # Very low threshold for debugging
                        x1, y1, x2, y2 = bbox.cpu().numpy()
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                        draw.text((x1, y1-20), f'Plane: {score:.2f}', fill='red')
                        print(f"Drew bbox {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score: {score:.3f}")
                
                return result_img, f"Found {len(pred_instances)} detections, max score: {pred_instances.scores.max():.3f}"
            else:
                return uploaded_image, "No detections found"
        
        # If different result format
        elif isinstance(detection_results, list):
            print(f"Results is list with {len(detection_results)} elements")
            if len(detection_results) > 0:
                first_result = detection_results[0]
                print(f"First result shape: {first_result.shape if hasattr(first_result, 'shape') else 'no shape'}")
                print(f"First result type: {type(first_result)}")
                
                if len(first_result) > 0:
                    # Manual drawing for list format
                    result_img = Image.fromarray(image_array)
                    draw = ImageDraw.Draw(result_img)
                    
                    count = 0
                    for detection in first_result:
                        if len(detection) >= 5:
                            x1, y1, x2, y2, score = detection[:5]
                            if score > 0.1:  # Low threshold for debugging
                                count += 1
                                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                                draw.text((x1, y1-20), f'Plane: {score:.2f}', fill='red')
                                print(f"Drew detection: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score: {score:.3f}")
                    
                    return result_img, f"Found {count} detections in list format"
                else:
                    return uploaded_image, "No detections in list format"
        
        return uploaded_image, f"Unknown result format: {type(detection_results)}"
        
    except Exception as error:
        print(f"Error: {error}")
        import traceback
        traceback.print_exc()
        return uploaded_image, f"Error: {str(error)}"

# Create Gradio interface
interface = gr.Interface(
    fn=debug_and_detect,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Detection Results"),
        gr.Textbox(label="Debug Info")
    ],
    title="🔍 Debug Plane Detection",
    description="Upload an image to debug the detection process"
)

if __name__ == "__main__":
    interface.launch(share=True, debug=False)