import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLOWorld, SAM

# 1. Load Detector & SAM 2
detector = YOLOWorld("yolov8x-worldv2.pt")
sam2_model = SAM("sam2_b.pt")

# Set target class
text_prompt = "blue A4 sheet box"
detector.set_classes([text_prompt])

# 2. Read Image
image_path = "test1.png"
image_cv = cv2.imread(image_path)
h, w, _ = image_cv.shape

# 3. Detect with YOLO-World
detection_results = detector.predict(image_cv, conf=0.15)
boxes = detection_results[0].boxes.xyxy.cpu().numpy()

if len(boxes) == 0:
    print("No target detected.")
    exit()

# Fix Warning: Convert bounding box array cleanly to float32 NumPy matrix
target_box = np.array([boxes[0]], dtype=np.float32)

# 4. Segment with SAM 2
sam_results = sam2_model(image_cv, bboxes=target_box)

# 5. Extract Detections for Supervision
detections = sv.Detections.from_ultralytics(sam_results[0])

# Extract binary mask array (H, W)
binary_mask = detections.mask[0]

# Verification check for shape alignment
print(f"Image Shape: ({h}, {w}), Mask Shape: {binary_mask.shape}")

# 6. Annotate using fixed Supervision API
mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)
box_annotator = sv.BoxAnnotator()

annotated_frame = mask_annotator.annotate(scene=image_cv.copy(), detections=detections)
annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

cv2.imwrite("segmented_carton_output.jpg", annotated_frame)
print("Saved output to segmented_carton_output.jpg")