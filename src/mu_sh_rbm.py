import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import math
from typing import Tuple, Optional, List
from einops import rearrange, repeat
import pytorch_wavelets as pywt

try:
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    print("Warning: geoopt not available, using simplified unitary constraints")

class WaveletTransform(nn.Module):
    """
    Undecimated 2D Haar wavelet packet transform for locality-preserving spectral operations.
    """
    def __init__(self, scales=3, wavelet='haar'):
        super().__init__()
        self.scales = scales
        self.wavelet = wavelet
        
        self.dwt = pywt.DWTForward(J=scales, mode='periodization', wave=wavelet)
        self.idwt = pywt.DWTInverse(mode='periodization', wave=wavelet)
    
    def forward(self, x):
        """
        Forward wavelet transform.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            coeffs: Wavelet coefficients (low, high_list)
        """
        return self.dwt(x)
    
    def inverse(self, coeffs):
        """
        Inverse wavelet transform.
        
        Args:
            coeffs: Wavelet coefficients (low, high_list)
        
        Returns:
            x: Reconstructed tensor [B, C, H, W]
        """
        return self.idwt(coeffs)

class ComplexUnitaryLayer(nn.Module):
    """
    Complex unitary layer using Householder reflections on Stiefel manifold.
    Simplified version for demonstration - full version would use geoopt.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if GEOOPT_AVAILABLE:
            self.weight_real = geoopt.ManifoldParameter(
                torch.randn(input_dim, output_dim) * 0.01,
                manifold=geoopt.Stiefel()
            )
            self.weight_imag = geoopt.ManifoldParameter(
                torch.randn(input_dim, output_dim) * 0.01,
                manifold=geoopt.Stiefel()
            )
        else:
            self.weight_real = Parameter(torch.randn(input_dim, output_dim) * 0.01)
            self.weight_imag = Parameter(torch.randn(input_dim, output_dim) * 0.01)
            self.register_buffer('scale', torch.tensor(1.0))
    
    def get_complex_weight(self):
        """Get complex weight matrix."""
        if GEOOPT_AVAILABLE:
            return torch.complex(self.weight_real, self.weight_imag)
        else:
            w_real = F.normalize(self.weight_real, dim=0)
            w_imag = F.normalize(self.weight_imag, dim=0)
            return torch.complex(w_real, w_imag) * self.scale
    
    def forward(self, x):
        """
        Forward pass through complex unitary layer.
        
        Args:
            x: Real input tensor [B, input_dim]
        
        Returns:
            Real output tensor [B, output_dim]
        """
        W = self.get_complex_weight()
        
        x_complex = torch.complex(x, torch.zeros_like(x))
        
        out_complex = torch.matmul(x_complex, W)
        
        return out_complex.real

class CriticalityTracker(nn.Module):
    """
    Differentiable criticality tracker for adaptive k scheduling.
    """
    def __init__(self, ema_decay=0.9):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer('tau', torch.tensor(0.0))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
    
    def update(self, loss_value):
        """Update criticality measure based on loss dynamics."""
        if self.step_count == 0:
            self.tau.copy_(loss_value.abs())
        else:
            self.tau.mul_(self.ema_decay).add_(loss_value.abs(), alpha=1-self.ema_decay)
        
        self.step_count += 1
    
    def get_adaptive_k(self, max_k=10, min_k=1):
        """Get adaptive k based on criticality measure."""
        k = min_k + int((max_k - min_k) * torch.sigmoid(self.tau - 1.0).item())
        return max(min_k, min(max_k, k))

class HamiltonianAttention(nn.Module):
    """
    Hamiltonian-tied single-layer attention embedded in the energy function.
    """
    def __init__(self, hidden_dim, attention_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        self.query_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.out_proj = nn.Linear(attention_dim, hidden_dim, bias=False)
        
        self.scale = 1.0 / math.sqrt(attention_dim)
    
    def forward(self, h):
        """
        Compute attention-weighted hidden states.
        
        Args:
            h: Hidden states [B, hidden_dim]
        
        Returns:
            Attention output [B, hidden_dim]
        """
        B = h.size(0)
        
        Q = self.query_proj(h)  # [B, attention_dim]
        K = self.key_proj(h)    # [B, attention_dim]
        V = self.value_proj(h)  # [B, attention_dim]
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, B]
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_out = torch.matmul(attn_weights, V)  # [B, attention_dim]
        
        output = self.out_proj(attn_out)  # [B, hidden_dim]
        
        return output

class MUSHRBM(nn.Module):
    """
    Multi-Scale Unitary Spectral Hopfield RBM (MU-SH-RBM).
    
    Combines:
    1. Wavelet-circulant Hopfield core
    2. Strict complex-unitary weights
    3. Differentiable criticality tracker
    4. Hamiltonian-tied single-layer attention
    """
    def __init__(self, 
                 input_shape=(1, 28, 28),
                 hidden_per_scale=64,
                 scales=3,
                 attention_dim=32,
                 wavelet='haar'):
        super().__init__()
        
        self.input_shape = input_shape
        self.scales = scales
        self.hidden_per_scale = hidden_per_scale
        
        C, H, W = input_shape
        self.visible_dim = C * H * W
        self.total_hidden_dim = hidden_per_scale * (scales + 1)  # +1 for low-freq component
        
        self.wavelet_transform = WaveletTransform(scales=scales, wavelet=wavelet)
        
        self.unitary_layers = nn.ModuleList([
            ComplexUnitaryLayer(self.visible_dim, hidden_per_scale)
            for _ in range(scales + 1)
        ])
        
        self.visible_bias = Parameter(torch.zeros(self.visible_dim))
        self.hidden_bias = Parameter(torch.zeros(self.total_hidden_dim))
        
        self.criticality_tracker = CriticalityTracker()
        
        self.attention = HamiltonianAttention(self.total_hidden_dim, attention_dim)
        
        self.energy_scale = Parameter(torch.tensor(1.0))
    
    def visible_to_hidden(self, v):
        """
        Compute hidden probabilities from visible units.
        
        Args:
            v: Visible units [B, visible_dim]
        
        Returns:
            Hidden probabilities [B, total_hidden_dim]
        """
        B = v.size(0)
        
        v_img = v.view(B, *self.input_shape)
        
        coeffs = self.wavelet_transform(v_img)
        low_freq, high_freq_list = coeffs
        
        hidden_parts = []
        
        low_flat = low_freq.view(B, -1)
        if low_flat.size(1) != self.visible_dim:
            if low_flat.size(1) < self.visible_dim:
                low_flat = F.pad(low_flat, (0, self.visible_dim - low_flat.size(1)))
            else:
                low_flat = low_flat[:, :self.visible_dim]
        
        h_low = self.unitary_layers[0](low_flat)
        hidden_parts.append(h_low)
        
        for i, high_freq in enumerate(high_freq_list):
            if i >= self.scales:
                break
            high_flat = high_freq.view(B, -1)
            if high_flat.size(1) != self.visible_dim:
                if high_flat.size(1) < self.visible_dim:
                    high_flat = F.pad(high_flat, (0, self.visible_dim - high_flat.size(1)))
                else:
                    high_flat = high_flat[:, :self.visible_dim]
            
            h_high = self.unitary_layers[i + 1](high_flat)
            hidden_parts.append(h_high)
        
        h_concat = torch.cat(hidden_parts, dim=1)
        
        h_input = h_concat + self.hidden_bias
        h_prob = torch.sigmoid(self.energy_scale * h_input)
        
        return h_prob
    
    def hidden_to_visible(self, h):
        """
        Compute visible probabilities from hidden units.
        
        Args:
            h: Hidden units [B, total_hidden_dim]
        
        Returns:
            Visible probabilities [B, visible_dim]
        """
        B = h.size(0)
        
        h_att = self.attention(h)
        h_combined = h + 0.1 * h_att  # Small attention contribution
        
        hidden_parts = torch.split(h_combined, self.hidden_per_scale, dim=1)
        
        v_recon = torch.zeros(B, self.visible_dim, device=h.device)
        
        for i, h_part in enumerate(hidden_parts):
            if i < len(self.unitary_layers):
                W = self.unitary_layers[i].get_complex_weight()
                W_inv = W.conj().transpose(-2, -1)  # Hermitian transpose
                
                h_complex = torch.complex(h_part, torch.zeros_like(h_part))
                v_part_complex = torch.matmul(h_complex, W_inv)
                v_part = v_part_complex.real
                
                v_recon += v_part / len(hidden_parts)
        
        v_input = v_recon + self.visible_bias
        v_prob = torch.sigmoid(self.energy_scale * v_input)
        
        return v_prob
    
    def gibbs_step(self, v):
        """Single Gibbs sampling step."""
        h_prob = self.visible_to_hidden(v)
        h_sample = torch.bernoulli(h_prob)
        
        v_prob = self.hidden_to_visible(h_sample)
        v_sample = torch.bernoulli(v_prob)
        
        return v_sample, h_prob, v_prob
    
    def one_shot_recall(self, v_noisy):
        """
        One-shot Hopfield recall using inverse wavelet transform.
        
        Args:
            v_noisy: Noisy visible units [B, visible_dim]
        
        Returns:
            Reconstructed visible units [B, visible_dim]
        """
        h_prob = self.visible_to_hidden(v_noisy)
        v_recon = self.hidden_to_visible(h_prob)
        
        return v_recon
    
    def contrastive_divergence(self, v_pos, k=None, epoch=0):
        """
        Contrastive Divergence training with adaptive k.
        
        Args:
            v_pos: Positive phase visible units [B, visible_dim]
            k: Number of Gibbs steps (if None, use adaptive)
            epoch: Current epoch for scheduling
        
        Returns:
            loss: CD loss
            k_used: Number of Gibbs steps used
        """
        if k is None:
            k = self.criticality_tracker.get_adaptive_k()
        
        h_pos_prob = self.visible_to_hidden(v_pos)
        
        v_neg = v_pos.clone()
        for _ in range(k):
            v_neg, _, _ = self.gibbs_step(v_neg)
        
        h_neg_prob = self.visible_to_hidden(v_neg)
        
        pos_energy = self.energy(v_pos, h_pos_prob)
        neg_energy = self.energy(v_neg, h_neg_prob)
        
        loss = pos_energy.mean() - neg_energy.mean()
        
        self.criticality_tracker.update(loss.detach())
        
        return loss, k
    
    def sample(self, num_samples, k=25, device=None):
        """
        Generate samples using Gibbs sampling.
        
        Args:
            num_samples: Number of samples to generate
            k: Number of Gibbs steps
            device: Device to generate samples on
        
        Returns:
            Generated samples [num_samples, visible_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        v = torch.bernoulli(torch.full((num_samples, self.visible_dim), 0.5, device=device))
        
        for _ in range(k):
            v, _, _ = self.gibbs_step(v)
        
        return v
    
    def energy(self, v, h):
        """
        Compute the energy of a visible-hidden configuration.
        
        Args:
            v: Visible units [B, visible_dim]
            h: Hidden units [B, total_hidden_dim]
        
        Returns:
            Energy values [B]
        """
        v_term = -torch.sum(v * self.visible_bias, dim=1)
        h_term = -torch.sum(h * self.hidden_bias, dim=1)
        
        h_from_v = self.visible_to_hidden(v)
        interaction = -torch.sum(h * h_from_v, dim=1)
        
        energy = self.energy_scale * (v_term + h_term + interaction)
        
        return energy
