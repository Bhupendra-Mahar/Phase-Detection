import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
        frame_idx = (idx * self.frames_per_phase) + 1  # Calculate the frame number
        frame = self.frames[idx]
        phase = self.phase_data.loc[frame_idx, 'phase']
        if self.transform is not None:
            frame = self.transform(frame)
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
        return frames

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set paths and parameters
video_path = "Z:\\New folder (6)\\cholec80\\Single_Video\\video01.mp4"
phase_file_path = "Z:\\New folder (6)\\cholec80\\Single_Video\\video01-phase.txt"
num_classes = 5
batch_size = 4

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization values for ImageNet
])

# Create dataset and dataloader
dataset = CholecDataset(video_path, phase_file_path, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Visualize images from the dataloader with their corresponding phase
for images, phases in dataloader:
    # Convert images to numpy arrays
    images = images.numpy()

    # Denormalize pixel values to [0, 255]
    images = np.transpose(images, (0, 2, 3, 1))  # Transpose image tensor from (N, C, H, W) to (N, H, W, C)
    images = (images * 255).astype(np.uint8)  # Rescale and convert to uint8

    # Display images and corresponding phases
    for i in range(len(images)):
        image = images[i]
        phase = phases[i]

        plt.imshow(image)
        plt.title(f'Phase: {phase}')
        plt.axis('off')
        plt.show()








