import cv2
import torch
from pathlib import Path
import numpy as np
from detectors import stable_hopenetlite
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors


# Load YOLO Model
def load_yolo_model(weights_path, device='cpu'):
    model = DetectMultiBackend(weights_path, device=device)
    return model


# Load Hopenet Model
def load_hopenet_model(model_path, device='cpu'):
    model = stable_hopenetlite.shufflenet_v2_x1_0()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# Preprocess face image for Hopenet
def preprocess_face(face_image):
    # Assuming face_image is in BGR format (as in OpenCV)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    resized_face = cv2.resize(face_image, (224, 224))
    normalized_face = (resized_face / 255.0 - mean) / std
    tensor_face = torch.tensor(normalized_face).permute(2, 0, 1).unsqueeze(
        0).float()  # Convert to CHW and add batch dimension
    return tensor_face


# Estimate head pose
def estimate_head_pose(model, face_tensor, device='cpu'):
    face_tensor = face_tensor.to(device)
    with torch.no_grad():
        yaw, pitch, roll = model(face_tensor)
    return yaw.item(), pitch.item(), roll.item()


# Detect faces using YOLO
def detect_faces(yolo_model, frame, device='cpu', img_size=(640, 640)):
    img_size = check_img_size(img_size, s=yolo_model.stride)  # Ensure img_size is compatible
    img = cv2.resize(frame, img_size)
    img = img.transpose((2, 0, 1))  # Convert to CHW format
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize
    if len(img.shape) == 3:
        img = img[None]  # Add batch dimension
    with torch.no_grad():
        pred = yolo_model(img)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)
    return pred


# Draw results on frame
def draw_results(frame, det, head_pose, annotator, label_color=(0, 255, 0)):
    for (xyxy, pose) in zip(det, head_pose):
        x_min, y_min, x_max, y_max = map(int, xyxy[:4])
        annotator.box_label(xyxy, pose, color=label_color)


# Main function
def main():
    # Paths and device configuration
    yolo_weights = 'yolov9-faceDetection/preTrainedModel/best.pt'
    hopenet_weights = 'hopenet-head-pose-detection/model/shuff_epoch_120.pkl'
    video_source = 0  # Change to file path for video file
    device = select_device('')  # Change to 'cuda' if GPU is available

    # Load models
    yolo_model = load_yolo_model(yolo_weights, device)
    hopenet_model = load_hopenet_model(hopenet_weights, device)

    # Video capture
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        detections = detect_faces(yolo_model, frame, device)

        # Initialize annotator for drawing
        annotator = Annotator(frame, line_width=3, example=str('Face'))
        head_poses = []

        for det in detections:
            if len(det):
                # Rescale boxes from model size to frame size
                det[:, :4] = scale_boxes((640, 640), det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    # Crop face from frame
                    x_min, y_min, x_max, y_max = map(int, xyxy[:4])
                    face_crop = frame[y_min:y_max, x_min:x_max]

                    # Preprocess and estimate head pose
                    face_tensor = preprocess_face(face_crop)
                    yaw, pitch, roll = estimate_head_pose(hopenet_model, face_tensor, device)

                    # Determine head pose direction
                    if yaw < -15:
                        pose_label = "Looking Left"
                    elif yaw > 15:
                        pose_label = "Looking Right"
                    elif pitch < -10:
                        pose_label = "Looking Down"
                    elif pitch > 10:
                        pose_label = "Looking Up"
                    else:
                        pose_label = "Looking Forward"

                    head_poses.append(pose_label)

        # Draw results
        draw_results(frame, det, head_poses, annotator)

        # Display the frame
        cv2.imshow('YOLO + Hopenet Head Pose', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
