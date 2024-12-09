# src/headPoseDetection/detect.py

import torch
import cv2
from pathlib import Path
from src.faceDetection.yolov9FaceDetection.yolov9.models.common import DetectMultiBackend
from src.faceDetection.yolov9FaceDetection.yolov9.utils.general import (check_file, check_img_size, non_max_suppression, scale_boxes)
from src.faceDetection.yolov9FaceDetection.yolov9.utils.torch_utils import select_device

def load_yolo_model(weights, device):
    # Load the YOLO model
    model = DetectMultiBackend(weights, device=device)
    return model

def detect_faces(model, frame, device, imgsz=(640, 640), conf_thres=0.50, iou_thres=0.45):
    # Preprocess the input frame
    frame_resized = cv2.resize(frame, imgsz)
    img = torch.from_numpy(frame_resized).to(device)
    img = img.half() if model.fp16 else img.float()
    img /= 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Run detection
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    return pred

def get_boxes(pred, frame_shape):
    # Extract bounding boxes and scale to the original frame size
    boxes = []
    for det in pred:  # Per image
        if len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_boxes(frame_shape, det[:, :4], frame_shape).round()
            for *xyxy, conf, cls in det:
                boxes.append(xyxy)
    return boxes

def save_detected_face(im0, boxes, save_dir, file_name):
    for i, *xyxy in enumerate(boxes):
        # Crop the detected face area
        x_min, y_min, x_max, y_max = map(int, xyxy)  # Convert coordinates to integers
        face_crop = im0[y_min:y_max, x_min:x_max]  # Crop the face from the image
        # Save the cropped face
        save_path = save_dir / f"{file_name}_face_{i}.jpg"
        cv2.imwrite(str(save_path), face_crop)
