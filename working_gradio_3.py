# plane_gradio_fixed.py
import gradio as gr
import cv2
import torch
import numpy as np
from mmcv.visualization import imshow_det_bboxes
from mmdet.apis import init_detector, inference_detector

# --- Load your trained model ---
CONFIG_FILE = 'configs/plane/faster-rcnn_r50_fpn_plane.py'
CHECKPOINT_FILE = 'work_dirs/faster-rcnn_r50_fpn_plane/epoch_12.pth'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=device)

CLASS_NAMES = ('plane',)  # Your dataset classes

# --- Helper function to convert DetDataSample to bboxes + scores ---
# Helper function to convert DetDataSample to numpy arrays for imshow_det_bboxes
def extract_bboxes_labels(result):
    """
    Converts MMDet DetDataSample to bboxes (N,4), scores (N,) and labels (N,)
    """
    if hasattr(result, 'pred_instances'):
        det_instances = result.pred_instances
        if det_instances is None or len(det_instances) == 0:
            return np.empty((0,4)), np.empty((0,)), np.empty((0,))
        bboxes = det_instances.bboxes.cpu().numpy()             # (N,4)
        scores = det_instances.scores.cpu().numpy()            # (N,)
        labels = det_instances.labels.cpu().numpy()            # (N,)
        return bboxes, scores, labels
    else:
        # fallback for old MMDet versions
        return result[0][:,:4], result[0][:,4], np.zeros(result[0].shape[0], dtype=int)


# --- Gradio inference ---
def plane_detection(image_path, confidence_threshold=0.05, nms_threshold=0.5):
    if image_path is None:
        return None, "Please upload an image."

    # Run detection
    result = inference_detector(model, image_path)

    # Extract numpy arrays
    bboxes, scores, labels = extract_bboxes_labels(result)

    # Apply confidence threshold
    keep = scores >= confidence_threshold
    bboxes, scores, labels = bboxes[keep], scores[keep], labels[keep]

    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Visualize
# Combine bboxes and scores (N,5)
    bboxes_with_scores = np.hstack([bboxes, scores[:, None]])

    # Visualize — pass labels inside bboxes_with_scores if using older version, 
    # or pass them separately for newer versions carefully
    vis_img = imshow_det_bboxes(
        img.copy(),
        bboxes=bboxes_with_scores,  # (N,5)
        labels=labels,              # (N,)
        class_names=CLASS_NAMES,    # tuple/list of class names
        score_thr=confidence_threshold,
        show=False
    )


    return vis_img, f"Detected {len(bboxes)} planes."


# --- Build Gradio UI ---
def create_interface():
    with gr.Blocks(title="Plane Detection Demo") as demo:
        gr.Markdown("""
        ## Plane Detection with Your Trained Faster R-CNN
        Upload an image containing planes, adjust confidence threshold and NMS IoU threshold.
        """)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Image", type="filepath")
                confidence_slider = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.05, step=0.01, label="Confidence Threshold"
                )
                nms_slider = gr.Slider(
                    minimum=0.1, maximum=0.8, value=0.5, step=0.05, label="NMS IoU Threshold (currently ignored)"
                )
                run_button = gr.Button("Detect Planes")

            with gr.Column():
                output_image = gr.Image(label="Detection Results", interactive=False)
                results_text = gr.Textbox(label="Detection Summary", lines=2, interactive=False)

        run_button.click(
            plane_detection,
            inputs=[image_input, confidence_slider, nms_slider],
            outputs=[output_image, results_text]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=5002)