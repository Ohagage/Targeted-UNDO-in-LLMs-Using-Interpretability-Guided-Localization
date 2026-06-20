#!/usr/bin/env python3
"""
SNMF Neuron-Concept Visualization

Computes and visualizes neuron-concept importance using two methods:
1. Per-Neuron Scalar Method: importance(n, c) = sum_f(|F[n,f]| * S[f,c]) / sum_f(S[f,c])
2. Direction-Based Method: d_c = sum_f(S[f,c] * F[:,f]) / sum_f(S[f,c])

Both methods share the S matrix (feature-concept scores) computed from the full G matrix.

Usage:
    python -m src.targeted_undo.snmf_analysis.visualize_snmf_neurons \
        --snmf-dir outputs/snmf_full_v2 \
        --data-path src/snmf-mlp-decomposition/data/arithmetic.json \
        --output-dir src/targeted_undo/snmf_analysis/outputs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def log(msg: str) -> None:
    """Print timestamped message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==============================================================================
# Concept Grouping (Forget vs Retain)
# ==============================================================================

FORGET_SET = [
    'division_symbolic', 'division_riddle',
    'multiplication_symbolic', 'multiplication_riddle'
]

RETAIN_SET = [
    'addition_symbolic', 'addition_riddle',
    'subtraction_symbolic', 'subtraction_riddle',
    'english'
]


def get_concept_group(concept: str) -> str:
    """Return 'forget', 'retain', or 'unknown' for a concept."""
    if concept in FORGET_SET:
        return 'forget'
    elif concept in RETAIN_SET:
        return 'retain'
    else:
        return 'unknown'


def aggregate_by_group(
    importance: np.ndarray,
    concepts: List[str]
) -> Dict[str, np.ndarray]:
    """
    Aggregate importance scores by Forget/Retain groups.
    
    Args:
        importance: Shape (n_neurons, n_concepts)
        concepts: List of concept names
        
    Returns:
        Dict with 'forget' and 'retain' arrays of shape (n_neurons,)
    """
    forget_indices = [i for i, c in enumerate(concepts) if c in FORGET_SET]
    retain_indices = [i for i, c in enumerate(concepts) if c in RETAIN_SET]
    
    result = {}
    
    if forget_indices:
        result['forget'] = importance[:, forget_indices].mean(axis=1)
    else:
        result['forget'] = np.zeros(importance.shape[0])
        
    if retain_indices:
        result['retain'] = importance[:, retain_indices].mean(axis=1)
    else:
        result['retain'] = np.zeros(importance.shape[0])
    
    return result


# ==============================================================================
# Data Loading
# ==============================================================================

def load_concept_dataset(data_path: str) -> Tuple[List[str], List[str]]:
    """
    Load concept dataset and return (prompts, labels).
    
    Labels are ordered by sample index as they were during SNMF training.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = []
    labels = []
    
    for concept, texts in data.items():
        for text in texts:
            prompts.append(text)
            labels.append(concept)
    
    return prompts, labels


def load_snmf_data(snmf_dir: str, layer: int) -> Dict:
    """
    Load SNMF factors for a specific layer.
    
    Returns dict with keys: F, G, token_ids, sample_ids
    """
    layer_dir = Path(snmf_dir) / f"layer_{layer}"
    factors_path = layer_dir / "snmf_factors.pt"
    
    if not factors_path.exists():
        raise FileNotFoundError(f"SNMF factors not found at {factors_path}")
    
    data = torch.load(factors_path, weights_only=False)
    return data


def get_token_labels(sample_ids: List[int], labels: List[str]) -> List[str]:
    """Map each token to its concept label using sample_ids."""
    return [labels[sid] for sid in sample_ids]


def get_unique_concepts(labels: List[str]) -> List[str]:
    """Get unique concepts in order of first appearance."""
    seen = set()
    unique = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


# ==============================================================================
# Core Computation (SHARED)
# ==============================================================================

def compute_feature_concept_scores(
    G: torch.Tensor,
    token_labels: List[str],
    concepts: List[str]
) -> np.ndarray:
    """
    Compute S matrix: feature-concept scores using the FULL G matrix.
    
    S[f, c] = sum of G[t, f] for all tokens t belonging to concept c
    
    Args:
        G: Activation weights matrix (n_tokens, n_features)
        token_labels: Concept label for each token
        concepts: List of unique concepts
        
    Returns:
        S matrix of shape (n_features, n_concepts)
    """
    n_tokens, n_features = G.shape
    n_concepts = len(concepts)
    
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    
    S = np.zeros((n_features, n_concepts), dtype=np.float32)
    
    G_np = G.numpy() if isinstance(G, torch.Tensor) else G
    
    for t in range(n_tokens):
        concept = token_labels[t]
        c_idx = concept_to_idx[concept]
        S[:, c_idx] += G_np[t, :]
    
    return S


# ==============================================================================
# Method 1: Per-Neuron Scalar Importance
# ==============================================================================

def compute_neuron_importance(
    F: torch.Tensor,
    S: np.ndarray
) -> np.ndarray:
    """
    Compute per-neuron scalar importance for each concept.
    
    importance(n, c) = sum_f(|F[n,f]| * S[f,c]) / sum_f(S[f,c])
    
    Takes absolute value BEFORE aggregation (no cancellation).
    
    Args:
        F: Feature directions matrix (n_neurons, n_features)
        S: Feature-concept scores (n_features, n_concepts)
        
    Returns:
        Importance matrix of shape (n_neurons, n_concepts)
    """
    F_np = F.numpy() if isinstance(F, torch.Tensor) else F
    F_abs = np.abs(F_np)  # Take absolute value FIRST
    
    n_neurons, n_features = F_abs.shape
    n_concepts = S.shape[1]
    
    I = np.zeros((n_neurons, n_concepts), dtype=np.float32)
    
    for c in range(n_concepts):
        weights = S[:, c]  # Shape: (n_features,)
        weight_sum = weights.sum()
        
        if weight_sum > 0:
            # Weighted average of |F| values
            I[:, c] = (F_abs @ weights) / weight_sum
        
    return I


# ==============================================================================
# Method 2: Direction-Based
# ==============================================================================

def compute_direction_vectors(
    F: torch.Tensor,
    S: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute direction vectors for each concept.
    
    d_c = sum_f(S[f,c] * F[:,f]) / sum_f(S[f,c])
    
    Preserves sign during aggregation (cancellation possible).
    
    Args:
        F: Feature directions matrix (n_neurons, n_features)
        S: Feature-concept scores (n_features, n_concepts)
        
    Returns:
        Tuple of:
        - d_raw: Raw direction vectors (n_neurons, n_concepts)
        - d_unit: L2-normalized unit vectors (n_neurons, n_concepts)
    """
    F_np = F.numpy() if isinstance(F, torch.Tensor) else F
    
    n_neurons, n_features = F_np.shape
    n_concepts = S.shape[1]
    
    d_raw = np.zeros((n_neurons, n_concepts), dtype=np.float32)
    d_unit = np.zeros((n_neurons, n_concepts), dtype=np.float32)
    
    for c in range(n_concepts):
        weights = S[:, c]  # Shape: (n_features,)
        weight_sum = weights.sum()
        
        if weight_sum > 0:
            # Weighted average of F values (with sign)
            d_raw[:, c] = (F_np @ weights) / weight_sum
            
            # L2 normalize to get unit direction
            norm = np.linalg.norm(d_raw[:, c])
            if norm > 1e-8:
                d_unit[:, c] = d_raw[:, c] / norm
        
    return d_raw, d_unit


# ==============================================================================
# Normalization
# ==============================================================================

def global_minmax_normalize(data_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Apply global min-max normalization across all layers.
    
    Args:
        data_dict: Dict mapping layer -> importance matrix
        
    Returns:
        Dict with normalized matrices (values in [0, 1])
    """
    all_values = np.concatenate([d.flatten() for d in data_dict.values()])
    global_min = all_values.min()
    global_max = all_values.max()
    
    range_val = global_max - global_min
    if range_val < 1e-8:
        range_val = 1.0
    
    normalized = {}
    for layer, data in data_dict.items():
        normalized[layer] = (data - global_min) / range_val
    
    return normalized


# ==============================================================================
# Visualization
# ==============================================================================

def plot_neuron_concept_heatmap(
    data: np.ndarray,
    concepts: List[str],
    layer: int,
    title: str,
    save_path: str,
    cmap: str = 'YlOrRd',
    show_values: bool = False,
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Create heatmap showing neuron-concept importance.
    
    Args:
        data: Importance matrix (n_neurons, n_concepts)
        concepts: List of concept names
        layer: Layer index
        title: Plot title
        save_path: Where to save the figure
        cmap: Colormap name
        show_values: Whether to annotate cells with values
        figsize: Figure size
    """
    n_neurons, n_concepts = data.shape
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transpose for display: rows=concepts, cols=neurons
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_xlabel('Neuron Index', fontsize=12)
    ax.set_ylabel('Concept', fontsize=12)
    ax.set_title(f'{title}\nLayer {layer}', fontsize=14, fontweight='bold')
    
    # Y-axis: concepts
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concepts, fontsize=10)
    
    # X-axis: sample neuron indices
    n_ticks = min(20, n_neurons)
    tick_positions = np.linspace(0, n_neurons - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_positions, fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance Score (normalized)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_summary_heatmap(
    layer_data: Dict[int, np.ndarray],
    concepts: List[str],
    title: str,
    save_path: str,
    cmap: str = 'YlOrRd',
    figsize: Tuple[int, int] = (14, 8)
) -> None:
    """
    Create summary heatmap: rows=concepts, cols=layers.
    
    Shows mean importance per concept per layer.
    """
    layers = sorted(layer_data.keys())
    n_layers = len(layers)
    n_concepts = len(concepts)
    
    # Aggregate: mean importance per concept per layer
    summary = np.zeros((n_concepts, n_layers), dtype=np.float32)
    
    for i, layer in enumerate(layers):
        data = layer_data[layer]  # (n_neurons, n_concepts)
        summary[:, i] = data.mean(axis=0)  # Mean across neurons
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(summary, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Concept', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # X-axis: layers
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers, fontsize=10)
    
    # Y-axis: concepts
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concepts, fontsize=10)
    
    # Annotate with values
    for i in range(n_concepts):
        for j in range(n_layers):
            val = summary[i, j]
            text_color = 'white' if val > summary.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=text_color)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Importance', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_comparison_heatmap(
    scalar_data: Dict[int, np.ndarray],
    direction_data: Dict[int, np.ndarray],
    concepts: List[str],
    save_path: str,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    Create side-by-side comparison of both methods (summary view).
    """
    layers = sorted(scalar_data.keys())
    n_layers = len(layers)
    n_concepts = len(concepts)
    
    # Aggregate summaries
    scalar_summary = np.zeros((n_concepts, n_layers), dtype=np.float32)
    direction_summary = np.zeros((n_concepts, n_layers), dtype=np.float32)
    
    for i, layer in enumerate(layers):
        scalar_summary[:, i] = scalar_data[layer].mean(axis=0)
        direction_summary[:, i] = direction_data[layer].mean(axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Per-Neuron Scalar
    im1 = axes[0].imshow(scalar_summary, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Layer Index', fontsize=11)
    axes[0].set_ylabel('Concept', fontsize=11)
    axes[0].set_title('Per-Neuron Scalar Method\nsum(|F| * S) / sum(S)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(n_layers))
    axes[0].set_xticklabels(layers, fontsize=9)
    axes[0].set_yticks(range(n_concepts))
    axes[0].set_yticklabels(concepts, fontsize=9)
    
    for i in range(n_concepts):
        for j in range(n_layers):
            val = scalar_summary[i, j]
            text_color = 'white' if val > scalar_summary.max() * 0.6 else 'black'
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=text_color)
    
    plt.colorbar(im1, ax=axes[0], label='Mean Importance')
    
    # Plot 2: Direction-Based
    im2 = axes[1].imshow(direction_summary, aspect='auto', cmap='YlOrRd')
    axes[1].set_xlabel('Layer Index', fontsize=11)
    axes[1].set_ylabel('Concept', fontsize=11)
    axes[1].set_title('Direction-Based Method\n|sum(F * S) / sum(S)|', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(n_layers))
    axes[1].set_xticklabels(layers, fontsize=9)
    axes[1].set_yticks(range(n_concepts))
    axes[1].set_yticklabels(concepts, fontsize=9)
    
    for i in range(n_concepts):
        for j in range(n_layers):
            val = direction_summary[i, j]
            text_color = 'white' if val > direction_summary.max() * 0.6 else 'black'
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=text_color)
    
    plt.colorbar(im2, ax=axes[1], label='Mean |d_c|')
    
    plt.suptitle('Neuron-Concept Importance: Method Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_forget_retain_heatmap(
    forget_data: Dict[int, np.ndarray],
    retain_data: Dict[int, np.ndarray],
    save_path: str,
    figsize: Tuple[int, int] = (18, 6)
) -> None:
    """
    Create Forget vs Retain vs Selectivity heatmaps.
    
    Shows neurons (rows) × layers (cols) with normalized 0-1 values.
    """
    layers = sorted(forget_data.keys())
    n_layers = len(layers)
    n_neurons = forget_data[layers[0]].shape[0]
    
    # Build matrices: (n_neurons, n_layers)
    forget_matrix = np.zeros((n_neurons, n_layers), dtype=np.float32)
    retain_matrix = np.zeros((n_neurons, n_layers), dtype=np.float32)
    
    for i, layer in enumerate(layers):
        forget_matrix[:, i] = forget_data[layer]
        retain_matrix[:, i] = retain_data[layer]
    
    # Selectivity: Forget - Retain (positive = more important for forget)
    selectivity_matrix = forget_matrix - retain_matrix
    
    # Global min-max normalize each to [0, 1]
    def minmax_norm(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-8:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    forget_norm = minmax_norm(forget_matrix)
    retain_norm = minmax_norm(retain_matrix)
    # Selectivity: normalize to [-1, 1] style but display raw for interpretation
    sel_abs_max = max(abs(selectivity_matrix.min()), abs(selectivity_matrix.max()))
    if sel_abs_max < 1e-8:
        sel_abs_max = 1.0
    selectivity_norm = selectivity_matrix / sel_abs_max  # Now in [-1, 1]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Forget Set
    im1 = axes[0].imshow(forget_norm, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    axes[0].set_xlabel('Layer Index', fontsize=11)
    axes[0].set_ylabel('Neuron Index', fontsize=11)
    axes[0].set_title('Forget Set Importance\n(division + multiplication)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(n_layers))
    axes[0].set_xticklabels(layers, fontsize=9)
    n_yticks = min(10, n_neurons)
    ytick_pos = np.linspace(0, n_neurons - 1, n_yticks, dtype=int)
    axes[0].set_yticks(ytick_pos)
    axes[0].set_yticklabels(ytick_pos, fontsize=8)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Importance (0-1)', fontsize=10)
    
    # Plot 2: Retain Set
    im2 = axes[1].imshow(retain_norm, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    axes[1].set_xlabel('Layer Index', fontsize=11)
    axes[1].set_ylabel('Neuron Index', fontsize=11)
    axes[1].set_title('Retain Set Importance\n(addition + subtraction + english)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(n_layers))
    axes[1].set_xticklabels(layers, fontsize=9)
    axes[1].set_yticks(ytick_pos)
    axes[1].set_yticklabels(ytick_pos, fontsize=8)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Importance (0-1)', fontsize=10)
    
    # Plot 3: Selectivity (Forget - Retain)
    im3 = axes[2].imshow(selectivity_norm, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_xlabel('Layer Index', fontsize=11)
    axes[2].set_ylabel('Neuron Index', fontsize=11)
    axes[2].set_title('Selectivity (Forget - Retain)\nRed=Forget, Blue=Retain', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(n_layers))
    axes[2].set_xticklabels(layers, fontsize=9)
    axes[2].set_yticks(ytick_pos)
    axes[2].set_yticklabels(ytick_pos, fontsize=8)
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Selectivity (-1 to 1)', fontsize=10)
    
    plt.suptitle('Neuron Importance: Forget vs Retain Sets', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


def plot_forget_retain_summary(
    forget_data: Dict[int, np.ndarray],
    retain_data: Dict[int, np.ndarray],
    save_path: str,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Create summary bar chart showing mean Forget vs Retain importance per layer.
    """
    layers = sorted(forget_data.keys())
    
    forget_means = [forget_data[l].mean() for l in layers]
    retain_means = [retain_data[l].mean() for l in layers]
    selectivity_means = [f - r for f, r in zip(forget_means, retain_means)]
    
    # Normalize to 0-1
    all_vals = forget_means + retain_means
    val_min, val_max = min(all_vals), max(all_vals)
    if val_max - val_min > 1e-8:
        forget_norm = [(v - val_min) / (val_max - val_min) for v in forget_means]
        retain_norm = [(v - val_min) / (val_max - val_min) for v in retain_means]
    else:
        forget_norm = [0] * len(layers)
        retain_norm = [0] * len(layers)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    x = np.arange(len(layers))
    width = 0.35
    
    # Plot 1: Forget vs Retain bars
    bars1 = axes[0].bar(x - width/2, forget_norm, width, label='Forget Set', color='#d62728')
    bars2 = axes[0].bar(x + width/2, retain_norm, width, label='Retain Set', color='#1f77b4')
    axes[0].set_xlabel('Layer Index', fontsize=11)
    axes[0].set_ylabel('Mean Importance (0-1)', fontsize=11)
    axes[0].set_title('Forget vs Retain Importance by Layer', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, fontsize=9)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    
    # Add value labels
    for bar, val in zip(bars1, forget_norm):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, retain_norm):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Plot 2: Selectivity (Forget - Retain)
    colors = ['#d62728' if s > 0 else '#1f77b4' for s in selectivity_means]
    axes[1].bar(x, selectivity_means, color=colors)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Layer Index', fontsize=11)
    axes[1].set_ylabel('Selectivity (Forget - Retain)', fontsize=11)
    axes[1].set_title('Selectivity by Layer\n(Positive = Forget-biased)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SNMF neuron-concept importance using both methods."
    )
    
    parser.add_argument("--snmf-dir", type=str, default="outputs/snmf_full_v2",
                        help="Path to SNMF results directory")
    parser.add_argument("--data-path", type=str, 
                        default="src/snmf-mlp-decomposition/data/arithmetic.json",
                        help="Path to concept dataset JSON")
    parser.add_argument("--output-dir", type=str,
                        default="src/targeted_undo/snmf_analysis/outputs",
                        help="Output directory for visualizations")
    parser.add_argument("--layers", type=str, default=None,
                        help="Specific layers to process (e.g., '0,1,2' or '0-5'). Default: all")
    parser.add_argument("--concepts", type=str, default=None,
                        help="Specific concepts to visualize (comma-separated). Default: all")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("SNMF Neuron-Concept Visualization")
    log("=" * 60)
    
    # Load concept dataset to get labels
    log(f"Loading dataset from {args.data_path}...")
    prompts, labels = load_concept_dataset(args.data_path)
    all_concepts = get_unique_concepts(labels)
    log(f"  Found {len(all_concepts)} concepts: {all_concepts}")
    
    # Filter concepts if specified
    if args.concepts:
        target_concepts = [c.strip() for c in args.concepts.split(',')]
        concepts = [c for c in all_concepts if c in target_concepts]
    else:
        concepts = all_concepts
    
    log(f"  Analyzing concepts: {concepts}")
    
    # Determine layers to process
    snmf_dir = Path(args.snmf_dir)
    available_layers = sorted([
        int(p.name.split('_')[1]) 
        for p in snmf_dir.glob('layer_*') 
        if p.is_dir()
    ])
    
    if args.layers:
        layers = []
        for chunk in args.layers.split(','):
            if '-' in chunk:
                a, b = chunk.split('-')
                layers.extend(range(int(a), int(b) + 1))
            else:
                layers.append(int(chunk))
        layers = sorted(set(layers) & set(available_layers))
    else:
        layers = available_layers
    
    log(f"  Processing layers: {layers}")
    
    # Storage for all layers
    importance_per_layer = {}  # Per-neuron scalar method
    direction_raw_per_layer = {}  # Direction vectors (raw)
    direction_unit_per_layer = {}  # Direction vectors (L2 normalized)
    direction_mag_per_layer = {}  # |d_c| for visualization
    S_matrices = {}  # Feature-concept score matrices
    
    # Process each layer
    log("\nComputing importance scores...")
    
    for layer in layers:
        log(f"\n--- Layer {layer} ---")
        
        # Load SNMF data
        data = load_snmf_data(args.snmf_dir, layer)
        F = data['F']
        G = data['G']
        sample_ids = data['sample_ids']
        
        log(f"  F shape: {F.shape}, G shape: {G.shape}")
        
        # Get token labels
        token_labels = get_token_labels(sample_ids, labels)
        
        # SHARED: Compute S matrix (feature-concept scores)
        S = compute_feature_concept_scores(G, token_labels, concepts)
        S_matrices[layer] = S
        log(f"  S matrix shape: {S.shape}")
        
        # METHOD 1: Per-neuron scalar importance
        I = compute_neuron_importance(F, S)
        importance_per_layer[layer] = I
        log(f"  Per-neuron importance shape: {I.shape}")
        
        # METHOD 2: Direction vectors
        d_raw, d_unit = compute_direction_vectors(F, S)
        direction_raw_per_layer[layer] = d_raw
        direction_unit_per_layer[layer] = d_unit
        # Use |d_raw| for visualization: shows actual magnitude per neuron
        # This is comparable to scalar method (both use same global min-max norm)
        # d_unit is kept separately for masking purposes
        direction_mag_per_layer[layer] = np.abs(d_raw)
        log(f"  Direction vectors shape: {d_raw.shape}")
    
    # Global normalization for visualization
    log("\nApplying global min-max normalization...")
    importance_norm = global_minmax_normalize(importance_per_layer)
    direction_mag_norm = global_minmax_normalize(direction_mag_per_layer)
    
    # ==== Generate Visualizations ====
    log("\nGenerating visualizations...")
    
    # 1. Per-layer heatmaps (per-neuron scalar method)
    for layer in layers:
        plot_neuron_concept_heatmap(
            importance_norm[layer],
            concepts,
            layer,
            title="Per-Neuron Scalar Importance",
            save_path=str(output_dir / f"neuron_importance_layer_{layer}.png")
        )
    
    # 2. Per-layer heatmaps (direction magnitude)
    for layer in layers:
        plot_neuron_concept_heatmap(
            direction_mag_norm[layer],
            concepts,
            layer,
            title="Direction Magnitude |d_c|",
            save_path=str(output_dir / f"direction_magnitude_layer_{layer}.png")
        )
    
    # 3. Summary heatmap: per-neuron scalar
    plot_summary_heatmap(
        importance_norm,
        concepts,
        title="Per-Neuron Scalar Importance (Mean per Layer)",
        save_path=str(output_dir / "summary_scalar_importance.png")
    )
    
    # 4. Summary heatmap: direction magnitude
    plot_summary_heatmap(
        direction_mag_norm,
        concepts,
        title="Direction Magnitude |d_c| (Mean per Layer)",
        save_path=str(output_dir / "summary_direction_magnitude.png")
    )
    
    # 5. Comparison heatmap
    plot_comparison_heatmap(
        importance_norm,
        direction_mag_norm,
        concepts,
        save_path=str(output_dir / "method_comparison.png")
    )
    
    # ==== Forget vs Retain Visualizations ====
    log("\nComputing Forget vs Retain aggregations...")
    
    # Aggregate importance by Forget/Retain groups (per-neuron scalar method)
    forget_per_layer = {}
    retain_per_layer = {}
    
    for layer in layers:
        grouped = aggregate_by_group(importance_norm[layer], concepts)
        forget_per_layer[layer] = grouped['forget']
        retain_per_layer[layer] = grouped['retain']
    
    # Aggregate direction magnitude by Forget/Retain groups (direction-based method)
    forget_direction_per_layer = {}
    retain_direction_per_layer = {}
    
    for layer in layers:
        grouped = aggregate_by_group(direction_mag_norm[layer], concepts)
        forget_direction_per_layer[layer] = grouped['forget']
        retain_direction_per_layer[layer] = grouped['retain']
    
    log(f"  Forget Set: {FORGET_SET}")
    log(f"  Retain Set: {RETAIN_SET}")
    
    # 6. Forget vs Retain heatmap (neurons × layers)
    plot_forget_retain_heatmap(
        forget_per_layer,
        retain_per_layer,
        save_path=str(output_dir / "forget_retain_heatmap.png")
    )
    
    # 7. Forget vs Retain summary (bar chart per layer) - scalar method
    plot_forget_retain_summary(
        forget_per_layer,
        retain_per_layer,
        save_path=str(output_dir / "forget_retain_summary.png")
    )
    
    # 8. Direction-based Forget vs Retain heatmap
    plot_forget_retain_heatmap(
        forget_direction_per_layer,
        retain_direction_per_layer,
        save_path=str(output_dir / "forget_retain_direction_heatmap.png")
    )
    
    # 9. Direction-based Forget vs Retain summary
    plot_forget_retain_summary(
        forget_direction_per_layer,
        retain_direction_per_layer,
        save_path=str(output_dir / "forget_retain_direction_summary.png")
    )
    
    # ==== Save Data ====
    log("\nSaving computed data...")
    
    # Save importance matrices
    torch.save({
        'importance': {layer: torch.from_numpy(data) for layer, data in importance_per_layer.items()},
        'importance_normalized': {layer: torch.from_numpy(data) for layer, data in importance_norm.items()},
        'concepts': concepts,
        'layers': layers,
    }, output_dir / "importance_data.pt")
    log(f"  Saved: {output_dir / 'importance_data.pt'}")
    
    # Save direction vectors
    torch.save({
        'd_raw': {layer: torch.from_numpy(data) for layer, data in direction_raw_per_layer.items()},
        'd_unit': {layer: torch.from_numpy(data) for layer, data in direction_unit_per_layer.items()},
        'd_magnitude': {layer: torch.from_numpy(data) for layer, data in direction_mag_per_layer.items()},
        'd_magnitude_normalized': {layer: torch.from_numpy(data) for layer, data in direction_mag_norm.items()},
        'concepts': concepts,
        'layers': layers,
    }, output_dir / "direction_vectors.pt")
    log(f"  Saved: {output_dir / 'direction_vectors.pt'}")
    
    # Save S matrices (feature-concept scores)
    torch.save({
        'S': {layer: torch.from_numpy(data) for layer, data in S_matrices.items()},
        'concepts': concepts,
        'layers': layers,
    }, output_dir / "feature_concept_scores.pt")
    log(f"  Saved: {output_dir / 'feature_concept_scores.pt'}")
    
    # Save Forget/Retain aggregated importance (for masking) - both methods
    torch.save({
        # Per-neuron scalar method
        'scalar_forget': {layer: torch.from_numpy(data) for layer, data in forget_per_layer.items()},
        'scalar_retain': {layer: torch.from_numpy(data) for layer, data in retain_per_layer.items()},
        # Direction-based method
        'direction_forget': {layer: torch.from_numpy(data) for layer, data in forget_direction_per_layer.items()},
        'direction_retain': {layer: torch.from_numpy(data) for layer, data in retain_direction_per_layer.items()},
        # Metadata
        'forget_concepts': FORGET_SET,
        'retain_concepts': RETAIN_SET,
        'layers': layers,
    }, output_dir / "forget_retain_importance.pt")
    log(f"  Saved: {output_dir / 'forget_retain_importance.pt'}")
    
    log("\n" + "=" * 60)
    log("Visualization Complete!")
    log(f"Results saved to: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
