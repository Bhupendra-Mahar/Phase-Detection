import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import pandas as pd
import os
# from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import pandas as pd
import os

# Define a custom dataset for Cholec80
class CholecDataset(Dataset):
    def __init__(self, video_path, phase_file_path, transform=None, frames_per_phase=1000):
        self.video_path = video_path
        self.transform = transform
        self.phase_data = self.load_phase_data(phase_file_path)
        self.frames_per_phase = frames_per_phase
        self.frames = self.extract_frames()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        phase = self.phase_data.loc[idx, 'phase']
        if self.transform is not None:
            frame = self.transform(frame)
        print(phase,frame)
        return frame, phase

    def load_phase_data(self, phase_file_path):
        phase_data = pd.read_csv(phase_file_path, sep='\t', header=None, skiprows=1)
        phase_data.columns = ['frame', 'phase']
        phase_data['frame'] = phase_data['frame'].astype(int)
        return phase_data

    def extract_frames(self):
        frames = []
        video = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_idx % self.frames_per_phase == 0:
                frames.append(frame)
            frame_idx += 1
        video.release()
        print(len(frames))
        print(len(frames[0]))
        print(len(frames[0][0]))
        return frames

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set paths and parameters
video_path = "Z:\\New folder (6)\\cholec80\\Single_Video\\video01.mp4"
phase_file_path = "Z:\\New folder (6)\\cholec80\\Single_Video\\video01-phase.txt"
num_classes = 5
batch_size = 32

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for ImageNet
])

# Create dataset and dataloader
dataset = CholecDataset(video_path, phase_file_path, transform)
print(dataset.frames,dataset.phase_data)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
import matplotlib.pyplot as plt
import numpy as np

# Iterate over the data loader and display images


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Set random seed for reproducibility
# torch.manual_seed(42)
#
# # Set device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Set paths and parameters
# video_path = "Z:\\New folder (6)\\cholec80\\Single_Video\\video01.mp4"
# phase_file_path = "Z:\\New folder (6)\\cholec80\\Single_Video\\video01-phase.txt"
# num_classes = 5
# batch_size = 32
num_epochs = 10
learning_rate = 0.001

# Define transformations for the dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for ImageNet
# ])

# Create dataset and dataloader
# dataset = CholecDataset(video_path, phase_file_path, transform)

# train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Instantiate the model
model = CNNModel(num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Set model in training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        print(images,labels)
        images = images.to(device)
        labels=list(labels)
        labels=torch.Tensor(labels)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total

    # Set model in evaluation mode
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / total

    # Print progress
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
