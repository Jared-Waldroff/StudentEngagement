from dataset import DaiseeDataset

# Replace these paths with the actual paths to your CSV and data directory
labels_file = 'C:/Users/jared/PycharmProjects/StudentEngagement/src/DAiSEE-engagement/DAiSEE/Labels/TrainLabels.csv'
data_dir = 'C:/Users/jared/PycharmProjects/StudentEngagement/src/DAiSEE-engagement/DAiSEE/DataSet/Train'

# Create an instance of the dataset
train_dataset = DaiseeDataset(labels_file=labels_file, root_dir=data_dir)

# Fetch a sample to verify
sample = train_dataset[0]
video, labels = sample['video'], sample['labels']

# Print details about the sample
print(f"Number of frames in video: {len(video)}")
print(f"Labels: {labels}")
