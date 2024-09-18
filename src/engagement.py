import deeplake
import torch
import torch.nn as nn
import torch.optim as optim

# Load the training, validation, and testing subsets
ds_train = deeplake.load("hub://activeloop/daisee-train")
ds_validation = deeplake.load("hub://activeloop/daisee-validation")
ds_test = deeplake.load("hub://activeloop/daisee-test")

dataloader = ds_train.pytorch(num_workers=0, batch_size=4, shuffle=True)
for batch in dataloader:
    videos, labels = batch['video'], batch['engagement']  # example with engagement labels


class EngagementNet(nn.Module):
    def __init__(self):
        super(EngagementNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 4)  # Assuming output size corresponds to 4 engagement levels

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = EngagementNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
