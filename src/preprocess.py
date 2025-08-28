import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def setup_data_loaders(dataset_name='mnist', batch_size=128, num_train=10000, num_test=2000, data_dir='./data'):
    """
    Setup data loaders for MNIST or Fashion-MNIST datasets.
    
    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        batch_size: Batch size for data loaders
        num_train: Number of training samples to use (for quick testing)
        num_test: Number of test samples to use
        data_dir: Directory to store/load data
    
    Returns:
        train_loader, test_loader, dataset_info
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if num_train < len(train_dataset):
        train_indices = torch.randperm(len(train_dataset))[:num_train]
        train_dataset = Subset(train_dataset, train_indices)
    
    if num_test < len(test_dataset):
        test_indices = torch.randperm(len(test_dataset))[:num_test]
        test_dataset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=min(4, os.cpu_count() or 4),
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=min(4, os.cpu_count() or 4),
        pin_memory=torch.cuda.is_available()
    )
    
    dataset_info = {
        'name': dataset_name,
        'input_shape': (1, 28, 28),
        'num_classes': 10,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset)
    }
    
    return train_loader, test_loader, dataset_info

def corrupt_images(batch, dropout_rate=0.3, salt_pepper_rate=0.1):
    """
    Add corruption to images for denoising experiments.
    
    Args:
        batch: Tensor of images [B, C, H, W]
        dropout_rate: Probability of setting pixels to 0
        salt_pepper_rate: Probability of salt and pepper noise
    
    Returns:
        Corrupted images
    """
    corrupted = batch.clone()
    
    dropout_mask = torch.rand_like(batch) > dropout_rate
    corrupted = corrupted * dropout_mask.float()
    
    salt_mask = torch.rand_like(batch) < salt_pepper_rate / 2
    pepper_mask = torch.rand_like(batch) > (1 - salt_pepper_rate / 2)
    
    corrupted[salt_mask] = 1.0
    corrupted[pepper_mask] = -1.0
    
    return corrupted

def preprocess_for_rbm(images):
    """
    Preprocess images for RBM training.
    Convert from [-1, 1] to [0, 1] and flatten.
    
    Args:
        images: Tensor of images [B, C, H, W]
    
    Returns:
        Flattened and normalized images [B, H*W]
    """
    images = (images + 1.0) / 2.0
    
    images = images.view(images.size(0), -1)
    
    return images

if __name__ == "__main__":
    print("Testing data preprocessing...")
    
    train_loader, test_loader, info = setup_data_loaders(
        dataset_name='mnist',
        batch_size=64,
        num_train=1000,
        num_test=200
    )
    
    print(f"Dataset info: {info}")
    
    for batch, labels in train_loader:
        print(f"Original batch shape: {batch.shape}")
        print(f"Original range: [{batch.min():.3f}, {batch.max():.3f}]")
        
        corrupted = corrupt_images(batch)
        print(f"Corrupted range: [{corrupted.min():.3f}, {corrupted.max():.3f}]")
        
        rbm_input = preprocess_for_rbm(batch)
        print(f"RBM input shape: {rbm_input.shape}")
        print(f"RBM input range: [{rbm_input.min():.3f}, {rbm_input.max():.3f}]")
        
        break
    
    print("Preprocessing test completed successfully!")
