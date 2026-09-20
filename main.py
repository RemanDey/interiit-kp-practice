import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLOWorld, SAM

detector = YOLOWorld("yolov8x-worldv2.pt")
sam2_model = SAM("sam2_b.pt")

text_prompt = "blue A4 sheet box"
detector.set_classes([text_prompt])

image_path = "test1.png"
image_cv = cv2.imread(image_path)
h, w, _ = image_cv.shape


detection_results = detector.predict(image_cv, conf=0.15)
boxes = detection_results[0].boxes.xyxy.cpu().numpy()

if len(boxes) == 0:
    print("No target detected.")
    exit()


target_box = np.array([boxes[0]], dtype=np.float32)


sam_results = sam2_model(image_cv, bboxes=target_box)

detections = sv.Detections.from_ultralytics(sam_results[0])


binary_mask = detections.mask[0]

print(f"Image Shape: ({h}, {w}), Mask Shape: {binary_mask.shape}")

mask_annotator = sv.MaskAnnotator(color=sv.Color.GREEN)
box_annotator = sv.BoxAnnotator()

annotated_frame = mask_annotator.annotate(scene=image_cv.copy(), detections=detections)
annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

cv2.imwrite("segmented_carton_output.jpg", annotated_frame)
print("Saved output to segmented_carton_output.jpg")