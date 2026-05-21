from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def stratified_split(dataset, train_ratio=0.9):

    targets = np.array(dataset.targets)

    train_indices = []
    val_indices = []

    for class_id in np.unique(targets):

        class_indices = np.where(targets == class_id)[0]
        np.random.shuffle(class_indices)

        split = int(len(class_indices) * train_ratio)

        train_indices.extend(class_indices[:split])
        val_indices.extend(class_indices[split:])

    return train_indices, val_indices

def get_dataloaders(data_dir="raw-img", batch_size=32, train_ratio=0.9):

    # 1. Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. Load datasets from folders
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    # Compute split sizes
    # Split stratified
    train_indices, val_indices = stratified_split(full_dataset, train_ratio)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # 3. Create DataLoaders (batching + shuffling)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader