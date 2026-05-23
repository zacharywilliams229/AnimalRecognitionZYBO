import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimalCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Pooling Layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(
            64 * 16 * 16,
            128
        )

        self.fc2 = nn.Linear(
            128,
            num_classes
        )

        self.dropout = nn.Dropout(0.3)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):

        # Conv Block 1
        x = self.pool(
            F.relu(self.bn1(self.conv1(x)))
        )

        # Conv Block 2
        x = self.pool(
            F.relu(self.bn2(self.conv2(x)))
        )

        # Conv Block 3
        x = self.pool(
            F.relu(self.bn3(self.conv3(x)))
        )

        # Flatten
        x = torch.flatten(x, 1)

        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x