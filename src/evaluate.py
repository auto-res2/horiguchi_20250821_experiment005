import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from mu_sh_rbm import MUSHRBM
from preprocess import setup_data_loaders, corrupt_images, preprocess_for_rbm

class MUSHRBMEvaluator:
    """
    Evaluator class for Multi-Scale Unitary Spectral Hopfield RBM.
    """
    
    def __init__(self, model: MUSHRBM, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        self.fid_metric = FrechetInceptionDistance(feature=64).to(device)
        self.psnr_metric = PeakSignalNoiseRatio().to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    def evaluate_generation(self, 
                           test_loader,
                           num_samples: int = 1000,
                           k_sampling: int = 25) -> Dict[str, float]:
        """
        Evaluate generative quality using FID and Inception Score.
        
        Args:
            test_loader: Test data loader for real samples
            num_samples: Number of samples to generate
            k_sampling: Number of Gibbs steps for sampling
        
        Returns:
            Dictionary with generation metrics
        """
        print(f"Evaluating generation quality with {num_samples} samples...")
        
        with torch.no_grad():
            generated_samples = self.model.sample(num_samples, k=k_sampling, device=self.device)
            
            generated_images = generated_samples.view(-1, *self.model.input_shape)
            generated_images = generated_images.clamp(0, 1)
            
            if generated_images.shape[1] == 1:  # Grayscale to RGB
                generated_images = generated_images.repeat(1, 3, 1, 1)
            generated_images_uint8 = (generated_images * 255).to(torch.uint8)
            
            real_samples = []
            for batch, _ in test_loader:
                batch = batch.to(self.device)
                batch = (batch + 1.0) / 2.0
                if batch.shape[1] == 1:  # Grayscale to RGB
                    batch = batch.repeat(1, 3, 1, 1)
                batch_uint8 = (batch * 255).to(torch.uint8)
                real_samples.append(batch_uint8)
                if len(real_samples) * batch.size(0) >= num_samples:
                    break
            
            real_images = torch.cat(real_samples, dim=0)[:num_samples]
            
            self.fid_metric.update(generated_images_uint8, real=False)
            self.fid_metric.update(real_images, real=True)
            fid_score = self.fid_metric.compute().item()
            self.fid_metric.reset()
            
            sample_diversity = torch.std(generated_images).item()
            
            reconstruction_error = 0.0
            num_batches = 0
            
            for batch, _ in test_loader:
                batch = batch.to(self.device)
                batch_flat = preprocess_for_rbm(batch)
                
                reconstructed = self.model.one_shot_recall(batch_flat)
                
                mse = F.mse_loss(reconstructed, batch_flat)
                reconstruction_error += mse.item()
                num_batches += 1
                
                if num_batches >= 10:  # Limit for speed
                    break
            
            avg_reconstruction_error = reconstruction_error / num_batches
        
        metrics = {
            'fid_score': fid_score,
            'sample_diversity': sample_diversity,
            'reconstruction_mse': avg_reconstruction_error,
            'num_samples': num_samples,
            'k_sampling': k_sampling
        }
        
        print(f"Generation metrics: FID={fid_score:.2f}, Diversity={sample_diversity:.4f}")
        
        return metrics
    
    def evaluate_denoising(self, 
                          test_loader,
                          corruption_rates: List[float] = [0.1, 0.3, 0.5],
                          num_test_images: int = 500) -> Dict[str, Dict[str, float]]:
        """
        Evaluate one-shot denoising performance.
        
        Args:
            test_loader: Test data loader
            corruption_rates: List of corruption rates to test
            num_test_images: Number of test images to use
        
        Returns:
            Dictionary with denoising metrics for each corruption rate
        """
        print(f"Evaluating denoising performance on {num_test_images} images...")
        
        results = {}
        
        with torch.no_grad():
            test_images = []
            for batch, _ in test_loader:
                test_images.append(batch)
                if len(test_images) * batch.size(0) >= num_test_images:
                    break
            
            test_images = torch.cat(test_images, dim=0)[:num_test_images].to(self.device)
            
            for corruption_rate in corruption_rates:
                print(f"Testing corruption rate: {corruption_rate}")
                
                corrupted_images = corrupt_images(test_images, dropout_rate=corruption_rate)
                
                clean_flat = preprocess_for_rbm(test_images)
                corrupted_flat = preprocess_for_rbm(corrupted_images)
                
                reconstructed_flat = self.model.one_shot_recall(corrupted_flat)
                
                reconstructed_images = reconstructed_flat.view_as(test_images)
                clean_images_01 = (test_images + 1.0) / 2.0  # Convert to [0,1]
                
                self.psnr_metric.reset()
                self.ssim_metric.reset()
                
                psnr_values = []
                ssim_values = []
                
                for i in range(min(100, num_test_images)):  # Limit for speed
                    psnr = self.psnr_metric(reconstructed_images[i:i+1], clean_images_01[i:i+1])
                    ssim = self.ssim_metric(reconstructed_images[i:i+1], clean_images_01[i:i+1])
                    
                    psnr_values.append(psnr.item())
                    ssim_values.append(ssim.item())
                
                mse = F.mse_loss(reconstructed_images, clean_images_01).item()
                
                results[f'corruption_{corruption_rate}'] = {
                    'psnr_mean': np.mean(psnr_values),
                    'psnr_std': np.std(psnr_values),
                    'ssim_mean': np.mean(ssim_values),
                    'ssim_std': np.std(ssim_values),
                    'mse': mse
                }
                
                print(f"  PSNR: {np.mean(psnr_values):.2f}±{np.std(psnr_values):.2f}")
                print(f"  SSIM: {np.mean(ssim_values):.3f}±{np.std(ssim_values):.3f}")
        
        return results
    
    def evaluate_associative_recall(self, 
                                   test_loader,
                                   num_patterns: int = 100) -> Dict:
        """
        Evaluate associative recall capability.
        
        Args:
            test_loader: Test data loader
            num_patterns: Number of patterns to test
        
        Returns:
            Dictionary with recall metrics
        """
        print(f"Evaluating associative recall on {num_patterns} patterns...")
        
        with torch.no_grad():
            patterns = []
            for batch, _ in test_loader:
                batch = batch.to(self.device)
                patterns.append(preprocess_for_rbm(batch))
                if len(patterns) * batch.size(0) >= num_patterns:
                    break
            
            patterns = torch.cat(patterns, dim=0)[:num_patterns]
            
            recall_accuracies = []
            corruption_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            for corruption in corruption_levels:
                correct_recalls = 0
                
                for i in range(min(50, num_patterns)):  # Test subset for speed
                    original = patterns[i:i+1]
                    
                    mask = torch.rand_like(original) > corruption
                    corrupted = original * mask.float()
                    
                    recalled = self.model.one_shot_recall(corrupted)
                    
                    original_dist = F.mse_loss(recalled, original)
                    random_pattern = torch.rand_like(original)
                    random_dist = F.mse_loss(recalled, random_pattern)
                    
                    if original_dist < random_dist:
                        correct_recalls += 1
                
                accuracy = correct_recalls / min(50, num_patterns)
                recall_accuracies.append(accuracy)
                print(f"  Corruption {corruption}: {accuracy:.3f} recall accuracy")
        
        return {
            'recall_accuracies': recall_accuracies,
            'corruption_levels': corruption_levels,
            'mean_recall_accuracy': np.mean(recall_accuracies)
        }
    
    def generate_evaluation_plots(self, 
                                 test_loader,
                                 save_dir: str = './.research/iteration1/images',
                                 num_samples: int = 64):
        """
        Generate comprehensive evaluation plots.
        
        Args:
            test_loader: Test data loader
            save_dir: Directory to save plots
            num_samples: Number of samples for visualization
        """
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            print("Generating sample visualization...")
            generated_samples = self.model.sample(num_samples, k=25, device=self.device)
            generated_images = generated_samples.view(-1, *self.model.input_shape)
            generated_images = generated_images.clamp(0, 1)
            
            fig, axes = plt.subplots(8, 8, figsize=(12, 12))
            fig.suptitle('MU-SH-RBM Generated Samples', fontsize=16, fontweight='bold')
            
            for i in range(64):
                row, col = i // 8, i % 8
                axes[row, col].imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'generated_samples.pdf'), 
                       dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print("Generating denoising comparison...")
            test_batch = next(iter(test_loader))[0][:16].to(self.device)
            
            corruption_rates = [0.2, 0.4, 0.6]
            
            fig, axes = plt.subplots(len(corruption_rates) + 1, 16, figsize=(20, 8))
            
            for i in range(16):
                axes[0, i].imshow((test_batch[i, 0].cpu().numpy() + 1) / 2, cmap='gray')
                axes[0, i].axis('off')
                if i == 0:
                    axes[0, i].set_ylabel('Original', rotation=90, fontsize=12)
            
            for row, corruption in enumerate(corruption_rates, 1):
                corrupted = corrupt_images(test_batch, dropout_rate=corruption)
                corrupted_flat = preprocess_for_rbm(corrupted)
                reconstructed_flat = self.model.one_shot_recall(corrupted_flat)
                reconstructed = reconstructed_flat.view_as(test_batch)
                
                for i in range(16):
                    if i < 8:
                        axes[row, i].imshow((corrupted[i, 0].cpu().numpy() + 1) / 2, cmap='gray')
                    else:
                        axes[row, i].imshow(reconstructed[i-8, 0].cpu().numpy(), cmap='gray')
                    axes[row, i].axis('off')
                
                axes[row, 0].set_ylabel(f'Corrupt {corruption}', rotation=90, fontsize=12)
                axes[row, 8].set_ylabel(f'Recon {corruption}', rotation=90, fontsize=12)
            
            plt.suptitle('MU-SH-RBM One-Shot Denoising', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'denoising_comparison.pdf'), 
                       dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
            
            print("Generating energy landscape...")
            patterns = preprocess_for_rbm(test_batch[:32])
            energies = []
            
            for pattern in patterns:
                h_prob = self.model.visible_to_hidden(pattern.unsqueeze(0))
                energy = self.model.energy(pattern.unsqueeze(0), h_prob)
                energies.append(energy.item())
            
            plt.figure(figsize=(10, 6))
            plt.hist(energies, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Energy', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Energy Distribution of Test Patterns', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'energy_distribution.pdf'), 
                       dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
        
        print(f"Evaluation plots saved to {save_dir}")

def evaluate_mushrbm(model_path: str, config: Dict) -> Dict:
    """
    Main evaluation function for MU-SH-RBM.
    
    Args:
        model_path: Path to trained model checkpoint
        config: Evaluation configuration
    
    Returns:
        Dictionary with all evaluation results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    model = MUSHRBM(
        input_shape=model_config['input_shape'],
        hidden_per_scale=model_config['hidden_per_scale'],
        scales=model_config['scales']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    _, test_loader, dataset_info = setup_data_loaders(
        dataset_name=config.get('dataset', 'mnist'),
        batch_size=config.get('batch_size', 128),
        num_train=1000,  # Not used for evaluation
        num_test=config.get('num_test', 2000),
        data_dir=config.get('data_dir', './data')
    )
    
    evaluator = MUSHRBMEvaluator(model, str(device))
    
    results = {}
    
    print("\n=== EVALUATION RESULTS ===")
    
    if config.get('evaluate_generation', True):
        gen_results = evaluator.evaluate_generation(
            test_loader, 
            num_samples=config.get('num_gen_samples', 1000)
        )
        results['generation'] = gen_results
    
    if config.get('evaluate_denoising', True):
        denoise_results = evaluator.evaluate_denoising(
            test_loader,
            num_test_images=config.get('num_denoise_images', 500)
        )
        results['denoising'] = denoise_results
    
    if config.get('evaluate_recall', True):
        recall_results = evaluator.evaluate_associative_recall(
            test_loader,
            num_patterns=config.get('num_recall_patterns', 100)
        )
        results['associative_recall'] = recall_results
    
    if config.get('generate_plots', True):
        evaluator.generate_evaluation_plots(
            test_loader,
            save_dir=config.get('plot_dir', './.research/iteration1/images')
        )
    
    return results

if __name__ == "__main__":
    config = {
        'dataset': 'mnist',
        'batch_size': 64,
        'num_test': 1000,
        'num_gen_samples': 500,
        'num_denoise_images': 200,
        'num_recall_patterns': 50,
        'data_dir': './data',
        'plot_dir': './.research/iteration1/images',
        'evaluate_generation': True,
        'evaluate_denoising': True,
        'evaluate_recall': True,
        'generate_plots': True
    }
    
    print("Testing MU-SH-RBM evaluation...")
    print("Evaluation test setup completed!")
