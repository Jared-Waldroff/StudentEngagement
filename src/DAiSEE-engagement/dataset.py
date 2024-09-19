import os
import cv2
import pandas as pd
from torch.utils.data import Dataset


class DaiseeDataset(Dataset):
    def __init__(self, labels_file, root_dir, transform=None):
        """
        Args:
            labels_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the video data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(labels_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get video file path
        video_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])

        # Read the video frames
        video_capture = cv2.VideoCapture(video_name)
        frames = []
        success, frame = video_capture.read()
        while success:
            frames.append(frame)
            success, frame = video_capture.read()
        video_capture.release()

        # Convert frames to a numpy array (or any other pre-processing step)
        video_frames = self.preprocess_frames(frames)

        # Get labels for the video
        labels = self.labels.iloc[idx, 1:].to_numpy()

        # Create sample
        sample = {'video': video_frames, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def preprocess_frames(self, frames):
        # Perform any pre-processing you need here
        # For example, resizing frames or converting to grayscale
        processed_frames = [cv2.resize(frame, (224, 224)) for frame in frames]  # Example resize
        return processed_frames
