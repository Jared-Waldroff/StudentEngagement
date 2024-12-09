import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from src.drawer.drawer import Drawer  # Importing the Drawer class
import os
import stable_hopenetlite

# Load the pre-trained stable HopeNetLite model using ShuffleNetV2
model_path = 'model/shuff_epoch_120.pkl'
model = stable_hopenetlite.shufflenet_v2_x1_0()
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Initialize the Drawer class
drawer = Drawer()

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Directory containing sample images
images_dir = '../../assets/photo'

# Loop through all images in the directory and run head pose estimation
for image_name in os.listdir(images_dir):
    image_path = os.path.join(images_dir, image_name)

    # Open the image and preprocess it
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the model to get yaw, pitch, and roll
    with torch.no_grad():
        yaw, pitch, roll = model(img_tensor)
        yaw_predicted = torch.argmax(yaw, dim=1).item()
        pitch_predicted = torch.argmax(pitch, dim=1).item()
        roll_predicted = torch.argmax(roll, dim=1).item()

    # Convert yaw, pitch, and roll to degrees
    yaw_predicted = (yaw_predicted - 33) * 3  # Adjusting scale
    pitch_predicted = (pitch_predicted - 33) * 3
    roll_predicted = (roll_predicted - 33) * 3

    # Convert yaw, pitch, and roll to radians for Rodrigues
    yaw_radians = np.radians(yaw_predicted)
    pitch_radians = np.radians(pitch_predicted)
    roll_radians = np.radians(roll_predicted)

    # Prepare the image for drawing
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw the axes using the Drawer class
    landmarks = [[(image_cv.shape[1] // 2, image_cv.shape[0] // 2)]]  # Center of the face as a landmark
    poses = [(yaw_radians, pitch_radians, roll_radians)]
    bbox = {0: {"face": [0, 0, image_cv.shape[1], image_cv.shape[0]]}}  # Entire image as face bounding box

    # Draw the pose estimation
    image_with_pose = drawer.draw_axis(image_cv, poses, landmarks, bbox)

    # Display the image with the pose estimation
    cv2.imshow(f'Head Pose Estimation - {image_name}', image_with_pose)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print the predicted yaw, pitch, and roll
    print(f"Image: {image_name}")
    print(f"Predicted Yaw: {yaw_predicted}, Pitch: {pitch_predicted}, Roll: {roll_predicted}\n")