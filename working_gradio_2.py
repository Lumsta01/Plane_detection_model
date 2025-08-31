# plane_detection_gradio_mmdet.py - Using MMDetection's official DetInferencer

import gradio as gr
import torch
from mmengine.logging import MMLogger
from mmdet.apis import DetInferencer

logger = MMLogger('mmdetection', logger_name='mmdet')


def get_free_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


class PlaneDetectionTab:
    def __init__(self) -> None:
        # Your model configuration
        self.model_config = 'configs/plane/faster-rcnn_r50_fpn_plane.py'
        self.model_weights = 'work_dirs/faster-rcnn_r50_fpn_plane/epoch_12.pth'
        self.create_ui()

    def create_ui(self):
        with gr.Row():
            with gr.Column():
                # Configuration controls
                confidence_slider = gr.Slider(
                    minimum=0.01,
                    maximum=0.5,
                    value=0.1,
                    step=0.01,
                    label="Confidence Threshold"
                )

                nms_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.8,
                    value=0.3,
                    step=0.05,
                    label="NMS IoU Threshold"
                )

            with gr.Column():
                image_input = gr.Image(
                    label='Upload Image',
                    type='filepath',
                    interactive=True,
                )

                output = gr.Image(
                    label='Plane Detection Results',
                    interactive=False,
                )

                results_text = gr.Textbox(
                    label='Detection Summary',
                    lines=2,
                    interactive=False,
                )

                run_button = gr.Button(
                    'Detect Planes',
                    variant="primary"
                )

                run_button.click(
                    self.inference,
                    inputs=[image_input, confidence_slider, nms_slider],
                    outputs=[output, results_text],
                )

        # Instructions
        with gr.Row():
            gr.Markdown("### Instructions:")
            gr.Markdown("""
            1. Upload an image containing aircraft.
            2. Adjust confidence threshold (start with 0.1).
            3. Adjust NMS IoU threshold to control overlapping boxes.
            4. Click 'Detect Planes' to run detection.
            """)

    def inference(self, image_path, confidence_threshold, nms_threshold):
        try:
            if image_path is None:
                return None, "Please upload an image first."

            # Initialize the DetInferencer
            det_inferencer = DetInferencer(
                model=self.model_config,
                weights=self.model_weights,
                scope='mmdet',
                device=get_free_device()
            )

            # Dynamically override NMS IoU threshold
            det_inferencer.model.test_cfg.rcnn.nms.iou_threshold = nms_threshold

            # Run inference
            results_dict = det_inferencer(
                image_path,
                pred_score_thr=confidence_threshold,
                return_vis=True,
                no_save_vis=True,
                return_datasamples=True
            )

            # Visualization
            vis_image = results_dict['visualization'][0]

            # Detection summary
            predictions = results_dict['predictions'][0]
            if hasattr(predictions, 'pred_instances'):
                num_planes = len(predictions.pred_instances)
                if num_planes > 0:
                    scores = predictions.pred_instances.scores.cpu().numpy()
                    summary = f"Detected {num_planes} planes. Confidence range: {scores.min():.3f}-{scores.max():.3f}"
                else:
                    summary = "No planes detected."
            else:
                summary = "Detection completed."

            return vis_image, summary

        except Exception as error:
            return None, f"Error during inference: {str(error)}"


# Standalone interface
def create_plane_detection_interface():
    title = 'Plane Detection with Your Trained Model'

    description = '''
    ## Plane Detection Demo
    
    This demo uses your trained Faster R-CNN model to detect aircraft in uploaded images.
    
    **Tips for best results:**
    - Start with confidence threshold around 0.1
    - Adjust NMS IoU threshold between 0.2-0.4 to control overlapping boxes
    - Works best with aerial/satellite images of airports and aircraft
    '''

    with gr.Blocks(analytics_enabled=False, title=title) as demo:
        gr.Markdown(description)
        # Create the plane detection interface
        plane_tab = PlaneDetectionTab()

    return demo


if __name__ == '__main__':
    # Launch the interface with automatic free port selection
    demo = create_plane_detection_interface()
    demo.queue().launch()
