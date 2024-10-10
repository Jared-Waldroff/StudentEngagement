import cv2
import torch
from pathlib import Path
from src.models.hopenet import load_hopenet_model, estimate_head_pose, preprocess_face, get_head_pose_direction
from src.detectors.yolov9FaceDetectionMain.yolov9.utils.general import non_max_suppression, scale_boxes
from src.detectors.yolov9FaceDetectionMain.yolov9.utils.torch_utils import select_device
from src.detectors.yolov9FaceDetectionMain.yolov9.models.common import DetectMultiBackend

# Configuration
yolo_weights = Path('src/detectors/yolov9FaceDetectionMain/preTrainedModel/best.pt')  # Adjust the path accordingly
hopenet_weights = Path('src/models/shuff_epoch_120k.pth')  # Adjust the path accordingly
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
video_source = 0  # Webcam source or path to video file

# Load YOLOv9 model
yolo_model = DetectMultiBackend(yolo_weights, device=device)
yolo_model.warmup(imgsz=(1, 3, 640, 640))  # Warmup with example input size

# Load HopeNet model
hopenet_model = load_hopenet_model(hopenet_weights, device=device)

# Video capture
cap = cv2.VideoCapture(video_source)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for YOLO
    img = cv2.resize(frame, (640, 640))
    img = torch.from_numpy(img).to(device)
    img = img.half() if yolo_model.fp16 else img.float()
    img /= 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Face detection with YOLOv9
    pred = yolo_model(img)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Extract and scale bounding boxes
    boxes = []
    for det in pred:  # Per image
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                boxes.append(xyxy)

    # Annotate the frame
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        face_crop = frame[y_min:y_max, x_min:x_max]

        # Head pose estimation
        if face_crop.size != 0:  # Ensure the face crop is valid
            face_tensor = preprocess_face(face_crop)
            yaw, pitch, roll = estimate_head_pose(hopenet_model, face_tensor, device)
            head_pose = get_head_pose_direction(yaw, pitch, roll)

            # Draw bounding box and head pose
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'Head Pose: {head_pose}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=(0, 255, 0), thickness=2)

    # Display the frame
    cv2.imshow('Face and Head Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()