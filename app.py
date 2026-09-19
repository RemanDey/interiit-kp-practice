"""Zero-shot object detection + segmentation: YOLO-World + SAM 2, Gradio UI.

Deploy: Hugging Face Spaces, SDK = Gradio, hardware = ZeroGPU, app_file = app.py.
GPU is allocated per-request via @spaces.GPU; models load on CPU at startup,
move to CUDA inside the decorated call, and weights auto-download via
Ultralytics on first boot, so no manual weight upload is needed.

CPU defaults are the lightweight variants (override with env vars):
    YOLO_WEIGHTS=yolov8x-worldv2.pt SAM_WEIGHTS=sam2_b.pt python app.py
"""

import os
import time

import cv2
import gradio as gr
import numpy as np
import spaces
import supervision as sv
import torch
from PIL import Image
from ultralytics import SAM, YOLOWorld

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8s-worldv2.pt")
SAM_WEIGHTS = os.environ.get("SAM_WEIGHTS", "sam2_t.pt")
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1024"))  # downscale for CPU speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[boot] device={DEVICE} cuda={torch.cuda.is_available()}")

detector = YOLOWorld(YOLO_WEIGHTS)
sam_model = SAM(SAM_WEIGHTS)
mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)
box_annotator = sv.BoxAnnotator()
print(f"[boot] models loaded on {DEVICE}")


@spaces.GPU(duration=120)  # ZeroGPU: attach an A100 only for this call
def segment(image: Image.Image, prompt: str, conf: float):
    t0 = time.perf_counter()
    if image is None:
        raise gr.Error("Please upload an image first.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a target object prompt.")
    if len(prompt) > 200:
        raise gr.Error("Prompt too long (max 200 characters).")
    classes = [c.strip() for c in prompt.split(",") if c.strip()]
    if not classes or len(classes) > 10:
        raise gr.Error("Enter 1–10 classes (comma-separated).")

    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:  # keeps ZeroGPU time quota + RAM in check
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
        h, w = cv_img.shape[:2]

    # ZeroGPU attaches CUDA only inside this decorated call, so (re)move
    # models here every request instead of relying on the startup device.
    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        detector.to(runtime_device)
        sam_model.to(runtime_device)
    except Exception as e:
        print(f"[warn] device move failed, staying on CPU: {e}")
        runtime_device = "cpu"

    detector.set_classes(classes)
    det_results = detector.predict(cv_img, conf=float(conf),
                                   device=runtime_device, verbose=False)
    r = det_results[0]
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
    cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)

    annotated = cv_img.copy()
    mask_canvas = np.zeros((h, w), dtype=np.uint8)
    rows = []

    if len(boxes):
        try:
            sam_results = sam_model(cv_img, bboxes=np.array(boxes, dtype=np.float32),
                                      device=runtime_device, verbose=False)
            dets = sv.Detections.from_ultralytics(sam_results[0])
            annotated = mask_annotator.annotate(scene=cv_img.copy(), detections=dets)
            annotated = box_annotator.annotate(scene=annotated, detections=dets)
            if dets.mask is not None:
                for m in dets.mask:
                    mask_canvas = np.maximum(mask_canvas, (m > 0).astype(np.uint8) * 255)
        except Exception as e:
            print(f"[warn] SAM failed, box-only fallback: {e}")
            dets = sv.Detections(xyxy=np.array(boxes, dtype=np.float32),
                                 confidence=confs, class_id=cls)
            annotated = box_annotator.annotate(scene=cv_img.copy(), detections=dets)
        for i, (b, c, k) in enumerate(zip(boxes, confs, cls)):
            x1, y1, x2, y2 = (round(float(v), 1) for v in b)
            rows.append([i + 1, classes[int(k)] if int(k) < len(classes) else classes[0],
                         round(float(c), 4), f"[{x1}, {y1}, {x2}, {y2}]"])

    annotated_rgb = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    mask_rgb = Image.fromarray(np.stack([np.zeros_like(mask_canvas), mask_canvas,
                                         np.zeros_like(mask_canvas)], axis=-1))
    elapsed = round(time.perf_counter() - t0, 3)
    status = (f"Found {len(rows)} object(s) in {elapsed}s on {runtime_device}."
              if rows else
              f"No objects detected for '{prompt}' in {elapsed}s. Try lowering confidence.")
    return annotated_rgb, mask_rgb, rows, status


with gr.Blocks(title="YOLO-World + SAM 2 — Zero-Shot Segmentation") as demo:
    gr.Markdown("# YOLO-World + SAM 2 — Zero-Shot Detection + Segmentation")
    gr.Markdown("Describe any object, detect all instances, segment each. "
                "GPU is allocated per request via ZeroGPU.")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Input image")
            prompt_in = gr.Textbox(value="blue A4 sheet box", max_lines=1,
                                   label="Target object (comma = multi-class)")
            conf_in = gr.Slider(0.01, 1.0, value=0.15, step=0.01, label="Confidence")
            btn = gr.Button("RUN SEGMENTATION", variant="primary")
        with gr.Column():
            img_out = gr.Image(label="Segmented")
            mask_out = gr.Image(label="Mask")
            table = gr.Dataframe(headers=["#", "Class", "Conf", "Box [x1,y1,x2,y2]"],
                                 label="Detections")
            status = gr.Textbox(label="Status")
    btn.click(segment, inputs=[img_in, prompt_in, conf_in],
              outputs=[img_out, mask_out, table, status])

if __name__ == "__main__":
    demo.launch()
