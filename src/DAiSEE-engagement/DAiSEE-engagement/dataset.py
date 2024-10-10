import os
import cv2
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DaiseeDataset(Dataset):
    def __init__(self, labels_file, root_dir, transform=None):
        self.labels = pd.read_csv(labels_file)
        self.root_dir = root_dir
        self.transform = transform
        self.video_paths = self._map_clipid_to_path()

    def _map_clipid_to_path(self):
        """
        Create a dictionary mapping ClipID to actual video file paths.
        This function searches through the nested directories to find each video file.
        Handles both .avi and .mp4 extensions.
        """
        clipid_to_path = {}
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.avi') or file.endswith('.mp4'):
                    clip_id_base = os.path.splitext(file)[0]
                    avi_clip_id = f"{clip_id_base}.avi"
                    mp4_clip_id = f"{clip_id_base}.mp4"
                    clip_path = os.path.normpath(os.path.join(root, file))
                    clipid_to_path[avi_clip_id] = clip_path
                    clipid_to_path[mp4_clip_id] = clip_path

        # Debug: Print out a few mappings to check
        print("Mapping of ClipID to video paths (sample):")
        for key, value in list(clipid_to_path.items())[:10]:
            print(f"ClipID: {key} -> Path: {value}")
        return clipid_to_path

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get video path using ClipID
        clip_id = self.labels.iloc[idx, 0]
        video_path = self.video_paths.get(clip_id, None)

        if video_path is None:
            print(f"Warning: Video file not found for ClipID {clip_id}: {video_path}")
            return None

        # Read the video frames
        video_capture = cv2.VideoCapture(video_path)
        frames = []
        success, frame = video_capture.read()
        while success:
            frames.append(frame)
            success, frame = video_capture.read()
        video_capture.release()

        # Check if frames are read successfully
        if len(frames) == 0:
            print(f"Warning: Video at index {idx} is empty or corrupted.")
            return None

        # Convert frames to a numpy array and preprocess them
        video_frames = self.preprocess_frames(frames)

        # Convert to tensor
        video_frames = torch.tensor(video_frames, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)

        # Get labels and convert them to a tensor
        labels = self.labels.iloc[idx, 1:].to_numpy()

        # Convert all label elements to float (in case of mixed types or None)
        try:
            labels = labels.astype(np.float32)  # Convert all elements to float32
        except ValueError as e:
            print(f"Error converting labels for ClipID {clip_id}: {labels}")
            labels = np.array([0.0] * len(labels), dtype=np.float32)  # Fallback to zeros if conversion fails

        labels = torch.tensor(labels, dtype=torch.float32)

        # Create sample
        sample = {'video': video_frames, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def preprocess_frames(self, frames):
        # Resize frames to (224, 224) and normalize (example preprocessing)
        processed_frames = [cv2.resize(frame, (224, 224)) / 255.0 for frame in frames]  # Normalizing the frames
        return np.array(processed_frames)  # Convert to numpy array


def load_daisee_datasets(batch_size=4, num_workers=0):
    # Define paths
    train_labels_file = '/src/DAiSEE-engagement/DAiSEE/Labels/TrainLabels.csv'
    train_data_dir = '/src/DAiSEE-engagement/DAiSEE/DataSet/Train'

    val_labels_file = '/src/DAiSEE-engagement/DAiSEE/Labels/ValidationLabels.csv'
    val_data_dir = '/src/DAiSEE-engagement/DAiSEE/DataSet/Validation'

    test_labels_file = '/src/DAiSEE-engagement/DAiSEE/Labels/TestLabels.csv'
    test_data_dir = '/src/DAiSEE-engagement/DAiSEE/DataSet/Test'

    # Create datasets
    train_dataset = DaiseeDataset(labels_file=train_labels_file, root_dir=train_data_dir)
    val_dataset = DaiseeDataset(labels_file=val_labels_file, root_dir=val_data_dir)
    test_dataset = DaiseeDataset(labels_file=test_labels_file, root_dir=test_data_dir)

    # Filter out None samples
    train_dataset = [data for data in train_dataset if data is not None]
    val_dataset = [data for data in val_dataset if data is not None]
    test_dataset = [data for data in test_dataset if data is not None]

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
