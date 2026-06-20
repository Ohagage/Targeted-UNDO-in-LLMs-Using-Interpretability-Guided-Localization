#!/usr/bin/env python3
"""
Visualize mask comparisons between original and unlearned models.

Creates heatmaps showing:
1. Original model mask
2. Unlearned model mask  
3. Difference between them

Usage:
    python -m src.targeted_undo.visualize_mask_comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Configuration
MASK_PAIRS = {
    'supervised': ('masks/div_mult_mask.pt', 'masks/div_mult_mask_unlearned.pt'),
    'intersection': ('masks/div_mult_mask_intersection.pt', 'masks/div_mult_mask_unlearned_intersection.pt'),
    'unsupervised': ('masks/div_mult_mask_unsupervised.pt', 'masks/div_mult_mask_unlearned_unsupervised.pt'),
    'unsupervised_strict': ('masks/div_mult_mask_unsupervised_strict.pt', 'masks/div_mult_mask_unlearned_unsupervised_strict.pt'),
}

NUM_LAYERS = 14


def extract_mask_stats(mask_data):
    """
    Extract mask statistics per layer.
    
    Mask format: 1 = targeted (apply intervention), 0 = keep unchanged
    
    Returns:
        dict with per-layer statistics
    """
    stats = {}
    masks = mask_data.get('masks', {})
    
    for name, mask_tensor in masks.items():
        parts = name.split('.')
        layer_idx = int(parts[2])
        
        total = mask_tensor.numel()
        ones = (mask_tensor == 1).sum().item()
        
        stats[layer_idx] = {
            'total': total,
            'targeted': ones,
            'kept': total - ones,
            'pct_targeted': 100 * ones / total
        }
    
    return stats


def extract_mask_density_matrix(mask_data, granularity='row'):
    """
    Extract a matrix showing targeting density per row/column per layer.
    
    Mask format: 1 = targeted (apply intervention), 0 = keep unchanged
    
    Args:
        mask_data: loaded mask data
        granularity: 'row' (320 rows) or 'col' (1280 cols)
    
    Returns:
        numpy array (num_layers, size) with percentage of ones (targeted) in each row/col
    """
    masks = mask_data.get('masks', {})
    
    # Determine dimensions from first mask
    first_mask = next(iter(masks.values()))
    if granularity == 'row':
        size = first_mask.shape[0]  # 320
    else:
        size = first_mask.shape[1]  # 1280
    
    matrix = np.zeros((NUM_LAYERS, size))
    
    for name, mask_tensor in masks.items():
        parts = name.split('.')
        layer_idx = int(parts[2])
        
        if granularity == 'row':
            # Percentage of ones (targeted) in each row
            ones_per_row = (mask_tensor == 1).sum(dim=1).float() / mask_tensor.shape[1]
            matrix[layer_idx, :] = ones_per_row.numpy()
        else:
            # Percentage of ones (targeted) in each column (intermediate neuron)
            ones_per_col = (mask_tensor == 1).sum(dim=0).float() / mask_tensor.shape[0]
            matrix[layer_idx, :] = ones_per_col.numpy()
    
    return matrix


def plot_comparison(method_name, orig_path, unl_path, save_path):
    """Create a 3-panel comparison figure showing targeting density per row."""
    orig_data = torch.load(orig_path, weights_only=False)
    unl_data = torch.load(unl_path, weights_only=False)
    
    # Get statistics
    orig_stats = extract_mask_stats(orig_data)
    unl_stats = extract_mask_stats(unl_data)
    
    # Get density matrices (percentage of ones/targeted per row)
    orig_matrix = extract_mask_density_matrix(orig_data, granularity='row')
    unl_matrix = extract_mask_density_matrix(unl_data, granularity='row')
    
    # Total targeted params
    orig_total_targeted = sum(s['targeted'] for s in orig_stats.values())
    unl_total_targeted = sum(s['targeted'] for s in unl_stats.values())
    
    # Difference matrix
    diff_matrix = unl_matrix - orig_matrix
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common colormap for density (0% to max%)
    max_density = max(orig_matrix.max(), unl_matrix.max())
    if max_density == 0:
        max_density = 0.01  # Avoid division by zero
    
    # Plot 1: Original model density
    im1 = axes[0].imshow(orig_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=max_density)
    axes[0].set_title(f'Original Model\n({orig_total_targeted:,} params targeted)', fontsize=12)
    axes[0].set_xlabel('Row Index (output features)')
    axes[0].set_ylabel('Layer')
    axes[0].set_yticks(range(NUM_LAYERS))
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('% of row targeted')
    
    # Plot 2: Unlearned model density
    im2 = axes[1].imshow(unl_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=max_density)
    axes[1].set_title(f'Unlearned Model\n({unl_total_targeted:,} params targeted)', fontsize=12)
    axes[1].set_xlabel('Row Index (output features)')
    axes[1].set_ylabel('Layer')
    axes[1].set_yticks(range(NUM_LAYERS))
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('% of row targeted')
    
    # Plot 3: Difference
    max_diff = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    if max_diff == 0:
        max_diff = 0.01
    
    im3 = axes[2].imshow(diff_matrix, aspect='auto', cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f'Difference (Unlearned - Original)\nRed: more targeted in unlearned\nBlue: less targeted in unlearned', fontsize=11)
    axes[2].set_xlabel('Row Index (output features)')
    axes[2].set_ylabel('Layer')
    axes[2].set_yticks(range(NUM_LAYERS))
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Δ % targeted')
    
    plt.suptitle(f'Mask Comparison: {method_name.upper()}\n(Showing targeting density per row of down_proj.weight)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    for method, (orig_path, unl_path) in MASK_PAIRS.items():
        save_path = output_dir / f'mask_comparison_{method}.png'
        plot_comparison(method, orig_path, unl_path, str(save_path))
    
    print("\nAll heatmaps generated in outputs/")


if __name__ == "__main__":
    main()
