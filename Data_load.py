# import os
# import cv2
# import torch
# import torch.utils.data as data
#
# class Cholec80Dataset(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform
#
#         self.video_files, self.phase_labels = self._load_data()
#
#     def __getitem__(self, index):
#         video_path = self.video_files[index]
#         phase_labels = self.phase_labels[index]
#
#         # Read video frames
#         frames = self._read_video_frames(video_path)
#
#         # Apply transformations to each frame
#         if self.transform:
#             frames = [self.transform(frame) for frame in frames]
#
#         return frames, phase_labels
#
#     def __len__(self):
#         return len(self.video_files)
#
#     def _load_data(self):
#         video_files = []
#         phase_labels = []
#
#         annotation_dir = os.path.join(self.data_dir, 'phase_annotations')
#         video_dir = os.path.join(self.data_dir, 'videos')
#
#         for file_name in os.listdir(annotation_dir):
#             video_name = file_name.replace('.txt', '.mp4')
#             video_path = os.path.join(video_dir, video_name)
#             video_files.append(video_path)
#
#             annotation_path = os.path.join(annotation_dir, file_name)
#             with open(annotation_path, 'r') as f:
#                 lines = f.readlines()
#                 phase_labels_video = [line.strip() for line in lines]
#                 phase_labels.append(phase_labels_video)
#         print(len(video_files),len(phase_labels[0]))
#         return video_files, phase_labels
#
#     def _read_video_frames(self, video_path):
#         frames = []
#
#         cap = cv2.VideoCapture(video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frames.append(frame)
#
#         return frames
#     def _read_phase_labels(self, annotation_path):
#         phase_labels = []
#
#         with open(annotation_path, 'r') as f:
#             for line in f:
#                 frame_number, phase_name = line.strip().split('\t')
#                 phase_labels.append(phase_name)
#         print(phase_labels)
#         return phase_labels
#
#
#
import torchvision.transforms as transforms
#
# # # Define the path to the Cholec80 dataset directory
# data_dir = 'Z:\\New folder (6)\\cholec80'
# #
# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # Create an instance of the Cholec80Dataset
# dataset = Cholec80Dataset(data_dir, transform=transform)
#
# # Use the dataset in a DataLoader for batch processing
# batch_size = 4
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#
# import matplotlib.pyplot as plt
#
# def visualize_dataset(dataset):
#     num_samples = len(dataset)
#     fig, axs = plt.subplots(num_samples, figsize=(10, 8*num_samples))
#
#     for i in range(num_samples):
#         frames, phase_labels = dataset[i]
#
#         for j, frame in enumerate(frames):
#             axs[i].imshow(frame)
#             axs[i].set_title(phase_labels[j])  # Use phase label as title
#             axs[i].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# # Create an instance of the Cholec80Dataset
# dataset = Cholec80Dataset(data_dir, transform=transform)
#
# # Define the phase names
# phase_names = [
#     'Preparation',
#     'CalotTriangleDissection',
#     'ClippingCutting',
#     'GallbladderDissection',
#     'GallbladderPackaging',
#     'CleaningCoagulation',
#     'GallbladderRetraction',
#     'GallbladderResection'
# ]
#
# # Visualize a subset of samples
# subset_indices = [0, 1, 2]  # Indices of samples to visualize
# subset_dataset = torch.utils.data.Subset(dataset, subset_indices)
# subset_dataset.phase_names = phase_names  # Assign phase_names to subset_dataset
#
# # Call the visualize_dataset function
# visualize_dataset(subset_dataset)
#


import os
import cv2
import torch
import torch.utils.data as data

class Cholec80Dataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.video_files, self.phase_labels = self._load_data()

    def __getitem__(self, index):
        video_path = self.video_files[index]
        phase_labels = self.phase_labels[index]

        # Read video frames
        frames = self._read_video_frames(video_path)
        # print("frames:",frames)

        # Apply transformations to each frame
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return frames, phase_labels

    def __len__(self):
        return len(self.video_files)

    def _load_data(self):
        video_files = []
        phase_labels = []

        annotation_dir = os.path.join(self.data_dir, 'phase_annotations')
        video_dir = os.path.join(self.data_dir, 'videos')

        for file_name in os.listdir(annotation_dir):
            video_name = file_name.replace('.txt', '.mp4')
            video_path = os.path.join(video_dir, video_name)
            video_files.append(video_path)

            annotation_path = os.path.join(annotation_dir, file_name)
            phase_labels_video = self._read_phase_labels(annotation_path)
            phase_labels.append(phase_labels_video)

        return video_files, phase_labels

    def _read_video_frames(self, video_path):
        print(video_path)
        frames = []

        cap = cv2.VideoCapture('Z:\\New folder (6)\\cholec80\\videos\\video01.mp4')
        while cap.isOpened():
            # print('123455666')
            ret, frame = cap.read()
            # print("frame:",frame)
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def _read_phase_labels(self, annotation_path):
        phase_labels = []

        with open(annotation_path, 'r') as f:
            for line in f:
                frame_number, phase_name = line.strip().split('\t')
                phase_labels.append(phase_name)
        # print(phase_labels)
        return phase_labels
import matplotlib.pyplot as plt

# # Define the path to the Cholec80 dataset directory
data_dir = 'Z:\\New folder (6)\\cholec80'
#
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def visualize_dataset(dataset):
    num_samples = len(dataset)
    fig, axs = plt.subplots(num_samples, figsize=(10, 8*num_samples))

    for i in range(num_samples):
        frames, phase_labels = dataset[i]

        for j, frame in enumerate(frames):
            axs[i].imshow(frame)
            axs[i].set_title(phase_labels[j])
            axs[i].axis('off')

    plt.tight_layout()
    plt.show()
# Create an instance of the Cholec80Dataset
dataset = Cholec80Dataset(data_dir, transform=transform)

# Visualize a subset of samples
subset_indices = [0, 1, 2]  # Indices of samples to visualize
subset_dataset = torch.utils.data.Subset(dataset, subset_indices)

# Call the visualize_dataset function
visualize_dataset(subset_dataset)
