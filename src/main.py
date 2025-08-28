"""
MU-SH-RBM Main Experimental Script
==================================

This script implements the complete Multi-Scale Unitary Spectral Hopfield RBM experiment
as described in the research proposal. It orchestrates training, evaluation, and visualization
of the MU-SH-RBM model on MNIST dataset.

Key Features:
1. Wavelet-circulant Hopfield core with undecimated 2D Haar wavelets
2. Strict complex-unitary weights using Householder reflections
3. Differentiable criticality tracker for adaptive k scheduling
4. Hamiltonian-tied single-layer attention
5. Comprehensive evaluation including generation, denoising, and associative recall

The experiment is designed to run within Tesla T4 16GB VRAM constraints.
"""

import os
import sys
import json
import time
import datetime
import traceback
from pathlib import Path
import torch
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_mushrbm, MUSHRBMTrainer
from evaluate import evaluate_mushrbm, MUSHRBMEvaluator
from preprocess import setup_data_loaders
from mu_sh_rbm import MUSHRBM

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_info():
    """Get device information for logging."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
            'gpu_memory_allocated': torch.cuda.memory_allocated(0),
            'gpu_memory_reserved': torch.cuda.memory_reserved(0),
        })
    
    return info

def create_experiment_config():
    """Create experiment configuration optimized for Tesla T4 16GB."""
    config = {
        'dataset': 'mnist',
        'data_dir': './data',
        
        'input_shape': (1, 28, 28),
        'hidden_per_scale': 48,  # Reduced for memory efficiency
        'scales': 3,
        'attention_dim': 32,
        'wavelet': 'haar',
        
        'batch_size': 64,  # Conservative for memory
        'num_epochs': 15,  # Reduced for quick testing
        'learning_rate': 3e-3,
        'beta1': 0.5,
        'beta2': 0.99,
        
        'num_train': 5000,
        'num_test': 1000,
        
        'num_gen_samples': 500,
        'num_denoise_images': 200,
        'num_recall_patterns': 100,
        
        'save_dir': './models',
        'plot_dir': './.research/iteration1/images',
        'log_dir': './.research/iteration1',
        
        'seed': 42,
        'max_k': 8,  # Reduced for faster training
        'evaluate_generation': True,
        'evaluate_denoising': True,
        'evaluate_recall': True,
        'generate_plots': True,
    }
    
    return config

def log_experiment_info(config, device_info, log_dir):
    """Log experiment configuration and device information."""
    os.makedirs(log_dir, exist_ok=True)
    
    experiment_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': config,
        'device_info': device_info,
        'git_commit': os.popen('git rev-parse HEAD').read().strip(),
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
    }
    
    with open(os.path.join(log_dir, 'experiment_info.json'), 'w') as f:
        json.dump(experiment_info, f, indent=2)
    
    print("=== EXPERIMENT CONFIGURATION ===")
    print(f"Timestamp: {experiment_info['timestamp']}")
    print(f"Device: {device_info['device']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['gpu_name']}")
        print(f"GPU Memory: {device_info['gpu_memory_total'] / 1e9:.1f} GB")
    print(f"Dataset: {config['dataset']}")
    print(f"Model: MU-SH-RBM with {config['scales']} scales, {config['hidden_per_scale']} hidden/scale")
    print(f"Training: {config['num_epochs']} epochs, batch size {config['batch_size']}")
    print(f"Data: {config['num_train']} train, {config['num_test']} test samples")
    print("=" * 40)

def run_experiment():
    """Run the complete MU-SH-RBM experiment."""
    
    print("Starting MU-SH-RBM Experiment...")
    print("=" * 50)
    
    config = create_experiment_config()
    
    set_random_seeds(config['seed'])
    
    device_info = get_device_info()
    
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['plot_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    log_experiment_info(config, device_info, config['log_dir'])
    
    try:
        print("\n=== PHASE 1: TRAINING ===")
        start_time = time.time()
        
        model, training_history = train_mushrbm(config)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds")
        
        training_results = {
            'training_time': training_time,
            'final_loss': training_history['loss'][-1] if training_history['loss'] else None,
            'final_k': training_history['k_used'][-1] if training_history['k_used'] else None,
            'final_tau': training_history['tau'][-1] if training_history['tau'] else None,
            'history': training_history
        }
        
        with open(os.path.join(config['log_dir'], 'training_results.json'), 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print("\n=== PHASE 2: EVALUATION ===")
        start_time = time.time()
        
        best_model_path = os.path.join(config['save_dir'], 'best_model.pth')
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(config['save_dir'], 'final_model.pth')
        
        evaluation_results = evaluate_mushrbm(best_model_path, config)
        
        evaluation_time = time.time() - start_time
        print(f"Evaluation completed in {evaluation_time:.1f} seconds")
        
        evaluation_results['evaluation_time'] = evaluation_time
        
        with open(os.path.join(config['log_dir'], 'evaluation_results.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print("\n=== PHASE 3: EXPERIMENT SUMMARY ===")
        
        total_time = training_time + evaluation_time
        
        print(f"Total experiment time: {total_time:.1f} seconds")
        print(f"Final training loss: {training_results['final_loss']:.4f}")
        print(f"Average k used: {training_results['final_k']:.1f}")
        print(f"Final criticality τ: {training_results['final_tau']:.4f}")
        
        if 'generation' in evaluation_results:
            gen_results = evaluation_results['generation']
            print(f"Generation FID: {gen_results['fid_score']:.2f}")
            print(f"Sample diversity: {gen_results['sample_diversity']:.4f}")
        
        if 'denoising' in evaluation_results:
            denoise_results = evaluation_results['denoising']
            for corruption, metrics in denoise_results.items():
                print(f"Denoising {corruption}: PSNR={metrics['psnr_mean']:.2f}±{metrics['psnr_std']:.2f}")
        
        if 'associative_recall' in evaluation_results:
            recall_results = evaluation_results['associative_recall']
            print(f"Mean recall accuracy: {recall_results['mean_recall_accuracy']:.3f}")
        
        summary = {
            'experiment_completed': True,
            'total_time': total_time,
            'training_time': training_time,
            'evaluation_time': evaluation_time,
            'config': config,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'device_info': device_info,
            'status_enum': 'stopped'  # Set status to stopped as required
        }
        
        with open(os.path.join(config['log_dir'], 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
        print(f"Results saved to: {config['log_dir']}")
        print(f"Plots saved to: {config['plot_dir']}")
        print(f"Models saved to: {config['save_dir']}")
        print(f"Status: {summary['status_enum']}")
        
        return summary
        
    except Exception as e:
        print(f"\n=== EXPERIMENT FAILED ===")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
        error_info = {
            'experiment_completed': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'config': config,
            'device_info': device_info,
            'status_enum': 'stopped'  # Still set to stopped even on failure
        }
        
        with open(os.path.join(config['log_dir'], 'experiment_error.json'), 'w') as f:
            json.dump(error_info, f, indent=2, default=str)
        
        raise e

def main():
    """Main entry point for the experiment."""
    try:
        summary = run_experiment()
        
        print(f"\nExperiment status: {summary['status_enum']}")
        
        return 0
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
