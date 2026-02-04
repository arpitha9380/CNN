"""
Data loading and preprocessing for CIFAR-10 dataset
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_data_transforms(augment=True):
    """
    Get data transformation pipelines for training and testing
    
    Args:
        augment: Whether to apply data augmentation for training
    
    Returns:
        Tuple of (train_transform, test_transform)
    """
    # Normalization values for CIFAR-10
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return train_transform, test_transform


def get_data_loaders(data_dir='./data', batch_size=128, num_workers=2, 
                     validation_split=0.1, augment=True):
    """
    Create data loaders for CIFAR-10 dataset
    
    Args:
        data_dir: Directory to download/load CIFAR-10 data
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        validation_split: Fraction of training data to use for validation
        augment: Whether to apply data augmentation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_transform, test_transform = get_data_transforms(augment=augment)
    
    # Download and load training data
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load test data
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training data into train and validation sets
    if validation_split > 0:
        train_size = int((1 - validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_dataset = full_train_dataset
        val_loader = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(train_loader, val_loader, test_loader):
    """
    Print information about the datasets
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    print("=" * 60)
    print("CIFAR-10 Dataset Information")
    print("=" * 60)
    print(f"Number of classes: {len(CIFAR10_CLASSES)}")
    print(f"Classes: {', '.join(CIFAR10_CLASSES)}")
    print(f"\nTraining samples: {len(train_loader.dataset):,}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"\nBatch size: {train_loader.batch_size}")
    print(f"Number of batches (train): {len(train_loader)}")
    if val_loader:
        print(f"Number of batches (val): {len(val_loader)}")
    print(f"Number of batches (test): {len(test_loader)}")
    print("=" * 60)


if __name__ == "__main__":
    # Test data loading
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=128,
        num_workers=0,  # Use 0 for testing
        validation_split=0.1
    )
    
    get_dataset_info(train_loader, val_loader, test_loader)
    
    # Display a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape: {images.shape}")
    print(f"Sample labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
