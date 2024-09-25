# src/models/hopenet.py

import torch
import torch.nn.functional as F
import cv2
from src.stable_hopenetlite import shufflenet_v2_x1_0

def load_hopenet_model(weights_path, device):
    # Load Hopenet model
    model = shufflenet_v2_x1_0()
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

def preprocess_face(face_crop, input_size=224):
    # Resize, normalize, and transform to tensor
    face_resized = cv2.resize(face_crop, (input_size, input_size))
    face_tensor = torch.from_numpy(face_resized).float().permute(2, 0, 1) / 255.0  # Normalize
    face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
    return face_tensor

def estimate_head_pose(model, face_tensor, device):
    face_tensor = face_tensor.to(device)
    yaw, pitch, roll = model(face_tensor)
    yaw = F.softmax(yaw, dim=1)
    pitch = F.softmax(pitch, dim=1)
    roll = F.softmax(roll, dim=1)

    # Convert output to angle values
    yaw = torch.sum(yaw * torch.arange(-99, 102, 3).float()).item()
    pitch = torch.sum(pitch * torch.arange(-99, 102, 3).float()).item()
    roll = torch.sum(roll * torch.arange(-99, 102, 3).float()).item()

    return yaw, pitch, roll

def get_head_pose_direction(yaw, pitch, roll):
    # Define head pose direction based on thresholds
    if abs(yaw) < 15 and abs(pitch) < 15:
        return "Forward"
    elif yaw > 15:
        return "Looking Left"
    elif yaw < -15:
        return "Looking Right"
    elif pitch > 15:
        return "Looking Up"
    elif pitch < -15:
        return "Looking Down"
    else:
        return "Unknown"
