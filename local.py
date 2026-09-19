"""
Local Zero-Shot Object Detection + Segmentation
YOLO-World + SAM 2

Designed for:
    NVIDIA RTX 4050
    CUDA
    Local Gradio server

Models:
    YOLO-World: yolov8s-worldv2.pt
    SAM 2:     sam2_t.pt
"""

import os
import time

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch

from PIL import Image
from ultralytics import SAM, YOLOWorld


# ============================================================
# CONFIGURATION
# ============================================================

YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS",
    "yolov8s-worldv2.pt"
)

SAM_WEIGHTS = os.environ.get(
    "SAM_WEIGHTS",
    "sam2_t.pt"
)

# RTX 4050 laptop GPU has limited VRAM, so keep this reasonable.
MAX_SIDE = int(
    os.environ.get("MAX_SIDE", "1024")
)

CONF_DEFAULT = float(
    os.environ.get("CONF_DEFAULT", "0.15")
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Use FP16 on NVIDIA GPU to reduce VRAM usage.
USE_HALF = torch.cuda.is_available()


# ============================================================
# STARTUP
# ============================================================

print("=" * 60)
print("YOLO-World + SAM 2")
print("LOCAL GPU VERSION")
print("=" * 60)

print(f"[boot] PyTorch: {torch.__version__}")
print(f"[boot] CUDA available: {torch.cuda.is_available()}")
print(f"[boot] Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"[boot] GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"[boot] VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

else:

    print(
        "[warning] CUDA is not available."
    )

    print(
        "[warning] The application will run on CPU."
    )


# ============================================================
# LOAD MODELS ONCE
# ============================================================

print()
print("[boot] Loading YOLO-World...")

detector = YOLOWorld(YOLO_WEIGHTS)

print("[boot] Loading SAM 2...")

sam_model = SAM(SAM_WEIGHTS)


# ============================================================
# MOVE MODELS TO GPU
# ============================================================

if torch.cuda.is_available():

    print("[boot] Moving YOLO-World to GPU...")

    detector.to(DEVICE)

    print("[boot] Moving SAM 2 to GPU...")

    sam_model.to(DEVICE)

    print("[boot] GPU models ready.")

else:

    print("[boot] Using CPU.")


print("[boot] Models loaded successfully.")
print("=" * 60)


# ============================================================
# ANNOTATORS
# ============================================================

mask_annotator = sv.MaskAnnotator(
    color=sv.Color.GREEN
)

box_annotator = sv.BoxAnnotator()


# ============================================================
# SEGMENTATION
# ============================================================

def segment(
    image: Image.Image,
    prompt: str,
    conf: float
):

    """
    Detect objects using YOLO-World
    and segment them using SAM 2.
    """

    t0 = time.perf_counter()

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if image is None:

        raise gr.Error(
            "Please upload an image first."
        )


    # --------------------------------------------------------
    # Validate prompt
    # --------------------------------------------------------

    prompt = (prompt or "").strip()

    if not prompt:

        raise gr.Error(
            "Please enter a target object prompt."
        )

    if len(prompt) > 200:

        raise gr.Error(
            "Prompt too long. Maximum 200 characters."
        )


    classes = [
        c.strip()
        for c in prompt.split(",")
        if c.strip()
    ]


    if not classes:

        raise gr.Error(
            "Please enter at least one object class."
        )


    if len(classes) > 10:

        raise gr.Error(
            "Maximum 10 classes are allowed."
        )


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    cv_img = cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2BGR
    )


    original_h, original_w = cv_img.shape[:2]


    # --------------------------------------------------------
    # Resize large images
    # --------------------------------------------------------

    scale = min(
        1.0,
        MAX_SIDE / max(original_h, original_w)
    )


    if scale < 1.0:

        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        cv_img = cv2.resize(
            cv_img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )


    h, w = cv_img.shape[:2]


    # --------------------------------------------------------
    # GPU information
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = DEVICE

        print(
            f"[inference] Using GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = "cpu"

        print(
            "[inference] CUDA unavailable. Using CPU."
        )


    # ========================================================
    # YOLO-WORLD
    # ========================================================

    print(
        f"[inference] Detecting: {classes}"
    )


    detector.set_classes(classes)


    # --------------------------------------------------------
    # Run YOLO-World
    # --------------------------------------------------------

    with torch.inference_mode():

        det_results = detector.predict(
        source=cv_img,
        conf=float(conf),
        device=device,
        verbose=False,
        half=False
    )


    result = det_results[0]


    # ========================================================
    # EXTRACT DETECTIONS
    # ========================================================

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        cls = (
            result.boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

    else:

        boxes = np.zeros(
            (0, 4),
            dtype=np.float32
        )

        confs = np.zeros(
            (0,),
            dtype=np.float32
        )

        cls = np.zeros(
            (0,),
            dtype=int
        )


    # ========================================================
    # PREPARE VISUALIZATION
    # ========================================================

    annotated = cv_img.copy()

    mask_canvas = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    rows = []


    # ========================================================
    # SAM 2 SEGMENTATION
    # ========================================================

    if len(boxes) > 0:

        try:

            print(
                f"[inference] "
                f"Segmenting {len(boxes)} object(s)..."
            )


            # ------------------------------------------------
            # SAM
            # ------------------------------------------------

            with torch.inference_mode():

                sam_results = sam_model(
                    source=cv_img,
                    bboxes=np.array(
                        boxes,
                        dtype=np.float32
                    ),
                    device=device,
                    verbose=False
                )


            # ------------------------------------------------
            # Convert SAM result
            # ------------------------------------------------

            dets = sv.Detections.from_ultralytics(
                sam_results[0]
            )


            # ------------------------------------------------
            # Draw masks
            # ------------------------------------------------

            annotated = mask_annotator.annotate(
                scene=annotated,
                detections=dets
            )


            # ------------------------------------------------
            # Draw bounding boxes
            # ------------------------------------------------

            annotated = box_annotator.annotate(
                scene=annotated,
                detections=dets
            )


            # ------------------------------------------------
            # Combined mask
            # ------------------------------------------------

            if dets.mask is not None:

                for mask in dets.mask:

                    mask_canvas = np.maximum(
                        mask_canvas,
                        (
                            mask > 0
                        ).astype(np.uint8) * 255
                    )


        except Exception as e:

            print(
                f"[warn] SAM failed: {e}"
            )

            print(
                "[warn] Falling back to bounding boxes."
            )


            # ------------------------------------------------
            # Bounding-box fallback
            # ------------------------------------------------

            dets = sv.Detections(
                xyxy=np.array(
                    boxes,
                    dtype=np.float32
                ),
                confidence=confs,
                class_id=cls
            )


            annotated = box_annotator.annotate(
                scene=annotated,
                detections=dets
            )


    # ========================================================
    # DETECTION TABLE
    # ========================================================

    for i, (
        box,
        confidence,
        class_id
    ) in enumerate(
        zip(boxes, confs, cls)
    ):

        x1, y1, x2, y2 = (
            round(float(v), 1)
            for v in box
        )


        class_index = int(class_id)


        if (
            0 <= class_index < len(classes)
        ):

            class_name = classes[class_index]

        else:

            class_name = str(class_index)


        rows.append(
            [
                i + 1,
                class_name,
                round(float(confidence), 4),
                f"[{x1}, {y1}, {x2}, {y2}]"
            ]
        )


    # ========================================================
    # OUTPUT IMAGE
    # ========================================================

    annotated_rgb = Image.fromarray(
        cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )
    )


    # ========================================================
    # MASK IMAGE
    # ========================================================

    mask_rgb_array = np.stack(
        [
            np.zeros_like(mask_canvas),
            mask_canvas,
            np.zeros_like(mask_canvas)
        ],
        axis=-1
    )


    mask_rgb = Image.fromarray(
        mask_rgb_array
    )


    # ========================================================
    # STATUS
    # ========================================================

    elapsed = round(
        time.perf_counter() - t0,
        3
    )


    if rows:

        status = (
            f"Found {len(rows)} object(s) "
            f"in {elapsed}s "
            f"using {torch.cuda.get_device_name(0)}"
            if torch.cuda.is_available()
            else f"Found {len(rows)} object(s) "
                 f"in {elapsed}s using CPU."
        )

        if torch.cuda.is_available():

            status = (
                f"Found {len(rows)} object(s) "
                f"in {elapsed}s using "
                f"{torch.cuda.get_device_name(0)}."
            )

    else:

        status = (
            f"No objects detected for "
            f"'{prompt}' in {elapsed}s. "
            f"Try lowering confidence."
        )


    print(
        f"[inference] {status}"
    )


    # ========================================================
    # CUDA MEMORY CLEANUP
    # ========================================================

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return (
        annotated_rgb,
        mask_rgb,
        rows,
        status
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Segmentation"
) as demo:


    gr.Markdown(
        """
"""
    )


    gr.Markdown(
        """
"""
    )


    # --------------------------------------------------------
    # MAIN LAYOUT
    # --------------------------------------------------------

    with gr.Row():

        # ====================================================
        # INPUT
        # ====================================================

        with gr.Column():

            img_in = gr.Image(
                type="pil",
                label="Input Image"
            )


            prompt_in = gr.Textbox(
                value="blue A4 sheet box",
                max_lines=1,
                label="Target Object",
                placeholder=(
                    "e.g. blue carton, chair, "
                    "person, red toolbox"
                )
            )


            gr.Markdown(
                "Use commas to detect multiple classes."
            )


            conf_in = gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=CONF_DEFAULT,
                step=0.01,
                label="Confidence Threshold"
            )


            btn = gr.Button(
                "RUN SEGMENTATION",
                variant="primary"
            )


        # ====================================================
        # OUTPUT
        # ====================================================

        with gr.Column():

            img_out = gr.Image(
                label="Segmented Image"
            )


            mask_out = gr.Image(
                label="Combined Mask"
            )


            table = gr.Dataframe(
                headers=[
                    "#",
                    "Class",
                    "Conf",
                    "Box [x1,y1,x2,y2]"
                ],
                datatype=[
                    "number",
                    "str",
                    "number",
                    "str"
                ],
                label="Detections",
                interactive=False
            )


            status = gr.Textbox(
                label="Status",
                interactive=False
            )


    # ========================================================
    # BUTTON EVENT
    # ========================================================

    btn.click(
        fn=segment,
        inputs=[
            img_in,
            prompt_in,
            conf_in
        ],
        outputs=[
            img_out,
            mask_out,
            table,
            status
        ],
        api_name="segment"
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    print()
    print("[boot] Launching Gradio...")
    print("[boot] Open the URL shown below.")
    print()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )