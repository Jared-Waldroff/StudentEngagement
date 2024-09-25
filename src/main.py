import cv2
import torch
from pathlib import Path
from src.models.detect import load_yolo_model, detect_faces, get_boxes
from src.models.hopenet import load_hopenet_model, preprocess_face, estimate_head_pose, get_head_pose_direction

# Function to select device (CPU or GPU)
def select_device(device=''):
    """ Selects the appropriate device based on input ('cpu' or 'cuda'). """
    if device.lower() == 'cpu':
        return torch.device('cpu')
    elif device.lower() == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        print("Warning: CUDA not available, using CPU instead.")
        return torch.device('cpu')

def main():
    # Configuration
    yolo_weights = Path('src/detectors/yolov9faceDetection/preTrainedModel/best.pt')  # Adjust the path as needed
    hopenet_weights = Path('src/models/shuff_epoch_120.pkl')
    device = select_device('/Users/Jared.Waldroff/PycharmProjects/StudentEngagement/assets/chiro2.mov')  # Use '' for auto, or 'cuda' for GPU, or 'cpu' for CPU
    video_source = 0  # Webcam source or path to video file

    # Load models
    yolo_model = load_yolo_model(yolo_weights, device=device)
    hopenet_model = load_hopenet_model(hopenet_weights, device=device)

    # Video capture
    cap = cv2.VideoCapture(video_source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face detection
        predictions = detect_faces(yolo_model, frame, device)
        boxes = get_boxes(predictions, frame.shape)

        # Annotate the frame
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            face_crop = frame[y_min:y_max, x_min:x_max]

            # Head pose estimation
            face_tensor = preprocess_face(face_crop)
            yaw, pitch, roll = estimate_head_pose(hopenet_model, face_tensor, device)
            head_pose = get_head_pose_direction(yaw, pitch, roll)

            # Draw bounding box and head pose
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{head_pose}', (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face and Head Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
