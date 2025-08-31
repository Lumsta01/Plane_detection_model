# streamlit_plane_detection.py
import streamlit as st
import numpy as np
from mmdet.apis import init_detector, inference_detector
from PIL import Image, ImageDraw
import torch
import torchvision

# Configure page
st.set_page_config(
    page_title="Plane Detection",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    CONFIG_FILE = 'configs/plane/faster-rcnn_r50_fpn_plane.py'
    CHECKPOINT_FILE = 'work_dirs/faster-rcnn_r50_fpn_plane/epoch_12.pth'
    return init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cpu')

def detect_planes(image, model, confidence_threshold, nms_threshold):
    """Run plane detection"""
    # Convert to numpy array
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    # Run detection
    results = inference_detector(model, image_array)
    
    # Process results
    if hasattr(results, 'pred_instances'):
        boxes = results.pred_instances.bboxes
        scores = results.pred_instances.scores
        
        # Filter by confidence
        confident_mask = scores > confidence_threshold
        final_boxes = boxes[confident_mask]
        final_scores = scores[confident_mask]
        
        # Apply NMS
        if len(final_boxes) > 1:
            keep_indices = torchvision.ops.nms(final_boxes, final_scores, nms_threshold)
            final_boxes = final_boxes[keep_indices]
            final_scores = final_scores[keep_indices]
    else:
        final_boxes = []
        final_scores = []
    
    # Draw results
    result_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(result_img)
    
    for bbox, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = bbox.cpu().numpy()
        score_val = score.cpu().numpy()
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        draw.text((x1, max(0, y1-20)), f'Plane: {score_val:.2f}', fill='red')
    
    return result_img, len(final_boxes)

# Main UI
st.title("✈️ Plane Detection System")
st.markdown("Upload an image to detect aircraft using your trained model")

# Load model
with st.spinner('Loading model...'):
    model = load_model()
st.success('Model loaded successfully!')

# Sidebar controls
st.sidebar.header("Detection Parameters")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.01,
    max_value=0.5,
    value=0.1,
    step=0.01
)

nms_threshold = st.sidebar.slider(
    "NMS Threshold",
    min_value=0.1,
    max_value=0.8,
    value=0.3,
    step=0.1
)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload an image containing aircraft"
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.subheader("Detection Results")
        
        # Run detection
        with st.spinner('Running detection...'):
            result_img, plane_count = detect_planes(
                image, model, confidence_threshold, nms_threshold
            )
        
        st.image(result_img, caption=f"Detected {plane_count} planes", use_column_width=True)
        
        # Show results
        if plane_count > 0:
            st.success(f"Found {plane_count} aircraft in the image!")
        else:
            st.info("No aircraft detected. Try lowering the confidence threshold.")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. **Upload an image** using the file uploader above
    2. **Adjust detection parameters** in the sidebar:
       - **Confidence Threshold**: Lower values detect more objects but may include false positives
       - **NMS Threshold**: Controls removal of overlapping detections
    3. **View results** in the right column
    
    **Tips:**
    - Start with confidence threshold around 0.1
    - Use NMS threshold between 0.2-0.4
    - Works best with aerial images of airports
    """)

# Run with: streamlit run streamlit_plane_detection.py