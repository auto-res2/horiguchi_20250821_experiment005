import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Optional

from mu_sh_rbm import MUSHRBM
from preprocess import setup_data_loaders

try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False

class MUSHRBMTrainer:
    """
    Trainer class for Multi-Scale Unitary Spectral Hopfield RBM.
    """
    
    def __init__(self, 
                 model: MUSHRBM,
                 device: str = 'cuda',
                 learning_rate: float = 3e-3,
                 beta1: float = 0.5,
                 beta2: float = 0.99):
        self.model = model.to(device)
        self.device = device
        
        if GEOOPT_AVAILABLE:
            self.optimizer = geoopt.optim.RiemannianAdam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2)
            )
        
        self.history = {
            'epoch': [],
            'loss': [],
            'k_used': [],
            'tau': [],
            'lr': []
        }
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   epoch: int,
                   max_k: int = 10) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            max_k: Maximum number of Gibbs steps
        
        Returns:
            Dictionary with epoch statistics
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_k = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(self.device)
            
            v_pos = data.view(data.size(0), -1)
            v_pos = (v_pos + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            
            loss, k_used = self.model.contrastive_divergence(v_pos, epoch=epoch)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_k += k_used
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'k': k_used,
                'τ': f'{self.model.criticality_tracker.tau.item():.4f}'
            })
        
        avg_loss = epoch_loss / num_batches
        avg_k = epoch_k / num_batches
        current_tau = self.model.criticality_tracker.tau.item()
        current_lr = self.optimizer.param_groups[0]['lr']
        
        self.history['epoch'].append(epoch)
        self.history['loss'].append(avg_loss)
        self.history['k_used'].append(avg_k)
        self.history['tau'].append(current_tau)
        self.history['lr'].append(current_lr)
        
        return {
            'loss': avg_loss,
            'k_used': avg_k,
            'tau': current_tau,
            'lr': current_lr
        }
    
    def train(self, 
              train_loader: DataLoader,
              num_epochs: int = 50,
              save_dir: str = './models',
              plot_dir: str = './.research/iteration1/images') -> Dict[str, List]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            plot_dir: Directory to save training plots
        
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        print(f"Starting MU-SH-RBM training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            epoch_stats = self.train_epoch(train_loader, epoch)
            
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {epoch_stats['loss']:.4f} | "
                  f"k̄: {epoch_stats['k_used']:.1f} | "
                  f"τ: {epoch_stats['tau']:.4f}")
            
            if epoch_stats['loss'] < best_loss:
                best_loss = epoch_stats['loss']
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch, epoch_stats['loss']
                )
            
            if epoch % 10 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'),
                    epoch, epoch_stats['loss']
                )
        
        self.save_checkpoint(
            os.path.join(save_dir, 'final_model.pth'),
            num_epochs, self.history['loss'][-1]
        )
        
        self.plot_training_curves(plot_dir)
        
        print(f"Training completed! Best loss: {best_loss:.4f}")
        
        return self.history
    
    def save_checkpoint(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history,
            'model_config': {
                'input_shape': self.model.input_shape,
                'hidden_per_scale': self.model.hidden_per_scale,
                'scales': self.model.scales,
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint['epoch'], checkpoint['loss']
    
    def plot_training_curves(self, save_dir: str):
        """Generate and save training curve plots."""
        
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MU-SH-RBM Training Curves', fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        
        axes[0, 0].plot(epochs, self.history['loss'], 'b-', linewidth=2, label='CD Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Contrastive Divergence Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(epochs, self.history['k_used'], 'r-', linewidth=2, label='k̄')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Gibbs Steps')
        axes[0, 1].set_title('Adaptive k Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].plot(epochs, self.history['tau'], 'g-', linewidth=2, label='τ')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Criticality Measure τ')
        axes[1, 0].set_title('Differentiable Criticality Tracker')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(epochs, self.history['lr'], 'm-', linewidth=2, label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.pdf'), 
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['loss'], 'b-', linewidth=2, alpha=0.8)
        plt.fill_between(epochs, self.history['loss'], alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Contrastive Divergence Loss', fontsize=12)
        plt.title('MU-SH-RBM Training Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'loss_curve.pdf'), 
                   dpi=300, bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"Training curves saved to {save_dir}")

def train_mushrbm(config: Dict) -> MUSHRBM:
    """
    Main training function for MU-SH-RBM.
    
    Args:
        config: Training configuration dictionary
    
    Returns:
        Trained MU-SH-RBM model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    train_loader, _, dataset_info = setup_data_loaders(
        dataset_name=config.get('dataset', 'mnist'),
        batch_size=config.get('batch_size', 128),
        num_train=config.get('num_train', 10000),
        num_test=config.get('num_test', 2000),
        data_dir=config.get('data_dir', './data')
    )
    
    print(f"Dataset: {dataset_info}")
    
    model = MUSHRBM(
        input_shape=dataset_info['input_shape'],
        hidden_per_scale=config.get('hidden_per_scale', 64),
        scales=config.get('scales', 3),
        attention_dim=config.get('attention_dim', 32),
        wavelet=config.get('wavelet', 'haar')
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    trainer = MUSHRBMTrainer(
        model=model,
        device=str(device),
        learning_rate=config.get('learning_rate', 3e-3),
        beta1=config.get('beta1', 0.5),
        beta2=config.get('beta2', 0.99)
    )
    
    history = trainer.train(
        train_loader=train_loader,
        num_epochs=config.get('num_epochs', 20),
        save_dir=config.get('save_dir', './models'),
        plot_dir=config.get('plot_dir', './.research/iteration1/images')
    )
    
    return model, history

if __name__ == "__main__":
    config = {
        'dataset': 'mnist',
        'batch_size': 64,
        'num_train': 2000,
        'num_test': 500,
        'num_epochs': 5,
        'hidden_per_scale': 32,
        'scales': 2,
        'learning_rate': 1e-3,
        'data_dir': './data',
        'save_dir': './models',
        'plot_dir': './.research/iteration1/images'
    }
    
    print("Testing MU-SH-RBM training...")
    model, history = train_mushrbm(config)
    print("Training test completed successfully!")
