import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
# Define the path to the Cholec80 dataset directory
data_dir = 'Z:\\New folder (6)\\cholec80'

# Define the mapping from original phase labels to new phase labels
phase_mapping = {
    0: 0,  # Phase 1 -> Phase 1
    1: 0,  # Phase 2 -> Phase 1
    2: 1,  # Phase 3 -> Phase 2
    3: 1,  # Phase 4 -> Phase 2
    4: 2,  # Phase 5 -> Phase 3
    5: 3,  # Phase 6 -> Phase 4
    6: 4   # Phase 7 -> Phase 5
}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the Cholec80 dataset and apply phase mapping
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataset.targets = [phase_mapping[phase] for phase in dataset.targets]

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define dataloaders for training and validation
batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

phase_labels = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']

# Visualize a sample of the dataset with annotations
fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Cholec80 Dataset Samples')

for i in range(10):
    img, label = dataset[i]
    phase = phase_labels[label]

    img = img.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    axs[i // 5, i % 5].imshow(img)
    axs[i // 5, i % 5].set_title(phase)
    axs[i // 5, i % 5].axis('off')

plt.tight_layout()
plt.show()