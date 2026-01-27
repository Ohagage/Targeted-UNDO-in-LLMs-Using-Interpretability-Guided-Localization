#!/usr/bin/env python3
"""
Create parameter masks based on SNMF feature analysis.

This script identifies features associated with specific concepts (e.g., division, multiplication)
and creates masks to zero out the corresponding model parameters.

Usage:
    python -m src.targeted_undo.create_snmf_mask \
        --snmf-dir outputs/snmf_full_v2 \
        --concepts division multiplication \
        --output-path masks/div_mult_mask.pt \
        --threshold 0.1
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
import torch


def load_snmf_results(snmf_dir: str, layer: int, load_unsupervised: bool = False) -> Tuple[torch.Tensor, Dict, Dict]:
    """Load F matrix and analysis results for a layer."""
    layer_dir = Path(snmf_dir) / f"layer_{layer}"
    
    # Load F matrix
    factors_path = layer_dir / "snmf_factors.pt"
    factors = torch.load(factors_path, weights_only=False)
    F = factors['F']  # (d_activation, rank)
    
    # Load supervised analysis
    supervised_path = layer_dir / "feature_analysis_supervised.json"
    with open(supervised_path) as f:
        supervised = json.load(f)
    
    # Load unsupervised analysis if requested
    unsupervised = {}
    if load_unsupervised:
        unsupervised_path = layer_dir / "feature_analysis_unsupervised.json"
        if unsupervised_path.exists():
            with open(unsupervised_path) as f:
                unsupervised = json.load(f)
    
    return F, supervised, unsupervised


def find_concept_features_supervised(supervised: Dict, target_concepts: List[str]) -> List[int]:
    """Find feature indices that match target concepts (supervised method)."""
    matching_features = []
    
    for feat_idx, feat_data in supervised.items():
        dominant_concept = feat_data.get('dominant_concept', '')
        
        # Check if any target concept is in the dominant concept name
        for target in target_concepts:
            if target.lower() in dominant_concept.lower():
                matching_features.append(int(feat_idx))
                break
    
    return matching_features


def find_concept_features_unsupervised(unsupervised: Dict, target_tokens: List[str], 
                                        min_matches: int = 2, top_k_check: int = 10) -> List[int]:
    """
    Find feature indices based on vocabulary projection tokens (unsupervised method).
    
    Args:
        unsupervised: Unsupervised analysis results
        target_tokens: Tokens to look for (e.g., ['/', 'divide', 'split'] for division)
        min_matches: Minimum number of target tokens that must appear in top-k
        top_k_check: How many top tokens to check for matches
    
    Returns:
        List of matching feature indices
    """
    matching_features = []
    
    for feat_idx, feat_data in unsupervised.items():
        positive_tokens = feat_data.get('positive_tokens', [])[:top_k_check]
        
        # Count how many target tokens appear in the feature's top tokens
        matches = 0
        matched_tokens = []
        for token in positive_tokens:
            token_clean = token.strip().lower()
            for target in target_tokens:
                if target.lower() in token_clean or token_clean in target.lower():
                    matches += 1
                    matched_tokens.append(token)
                    break
        
        if matches >= min_matches:
            matching_features.append(int(feat_idx))
    
    return matching_features


# Default token patterns for common arithmetic operations
DIVISION_TOKENS = [
    '/', 'divide', 'divided', 'split', 'share', 'each', 'per', 
    'equally', 'distribute', 'quotient', 'half', 'quarter',
    'among', 'between', 'ratio'
]

MULTIPLICATION_TOKENS = [
    '*', '×', 'times', 'multiply', 'multiplied', 'product', 
    'total', 'groups', 'rows', 'columns', 'double', 'triple',
    'twice', 'thrice'
]

ADDITION_TOKENS = [
    '+', 'add', 'plus', 'sum', 'total', 'more', 'increase',
    'together', 'combine', 'join', 'additional'
]

SUBTRACTION_TOKENS = [
    '-', 'minus', 'subtract', 'less', 'fewer', 'remain', 'left',
    'difference', 'remove', 'take away', 'decrease'
]

CONCEPT_TOKEN_MAP = {
    'division': DIVISION_TOKENS,
    'multiplication': MULTIPLICATION_TOKENS,
    'addition': ADDITION_TOKENS,
    'subtraction': SUBTRACTION_TOKENS,
}


def get_active_neurons(F: torch.Tensor, feature_indices: List[int], 
                       threshold: float = 0.1, method: str = "relative") -> Set[int]:
    """
    Identify neurons that are active for the given features.
    
    Args:
        F: Feature directions matrix (d_activation, rank)
        feature_indices: Which features to consider
        threshold: Threshold for considering a neuron "active"
        method: "relative" (top % of values) or "absolute" (above threshold)
    
    Returns:
        Set of neuron indices that are active
    """
    active_neurons = set()
    
    for feat_idx in feature_indices:
        feature_vec = F[:, feat_idx].abs()
        
        if method == "relative":
            # Top threshold% of neurons
            k = max(1, int(threshold * len(feature_vec)))
            _, top_indices = torch.topk(feature_vec, k)
            active_neurons.update(top_indices.tolist())
        else:
            # Neurons above absolute threshold
            max_val = feature_vec.max()
            thresh = threshold * max_val
            active_indices = (feature_vec >= thresh).nonzero(as_tuple=True)[0]
            active_neurons.update(active_indices.tolist())
    
    return active_neurons


def create_mlp_mask(
    model_config: Dict,
    active_neurons_per_layer: Dict[int, Set[int]],
    mode: str = "mlp"
) -> Dict[str, torch.Tensor]:
    """
    Create masks for MLP parameters.
    
    For mode="mlp" (320d output):
        - Mask applies to down_proj output dimensions
        - down_proj.weight: (hidden_size, intermediate_size)
        - Active neurons correspond to hidden_size dimension (rows)
    
    Returns:
        Dictionary mapping parameter names to mask tensors (1=keep, 0=remove)
    """
    hidden_size = model_config['hidden_size']
    intermediate_size = model_config['intermediate_size']
    num_layers = model_config['num_hidden_layers']
    
    masks = {}
    
    for layer in range(num_layers):
        active_neurons = active_neurons_per_layer.get(layer, set())
        
        if not active_neurons:
            # No active neurons for this layer - keep everything
            continue
        
        # Create mask for down_proj (hidden_size, intermediate_size)
        # Active neurons are in the output dimension (rows)
        down_proj_mask = torch.ones(hidden_size, intermediate_size)
        for neuron in active_neurons:
            if neuron < hidden_size:
                down_proj_mask[neuron, :] = 0  # Zero out this row
        
        masks[f"model.layers.{layer}.mlp.down_proj.weight"] = down_proj_mask
        
        # Also mask the corresponding rows in gate_proj and up_proj outputs
        # that feed into these neurons (transposed perspective)
        # gate_proj.weight: (intermediate_size, hidden_size)
        # up_proj.weight: (intermediate_size, hidden_size)
        # These project FROM hidden_size TO intermediate_size
        # The intermediate activations then go through down_proj
        
        # For now, we focus on down_proj since that's where the 
        # MLP output features are expressed
    
    return masks


def main():
    parser = argparse.ArgumentParser(description="Create SNMF-based parameter masks")
    parser.add_argument("--snmf-dir", type=str, required=True,
                        help="Directory containing SNMF results")
    parser.add_argument("--model-path", type=str, 
                        default="gemma-2-0.1B_all_arithmetic+eng/final_model",
                        help="Path to model (for config)")
    parser.add_argument("--concepts", nargs="+", required=True,
                        help="Concepts to mask (e.g., division multiplication)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output path for mask file")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Threshold for neuron activation (default: 0.1 = top 10%%)")
    parser.add_argument("--method", type=str, default="relative",
                        choices=["relative", "absolute"],
                        help="Thresholding method")
    parser.add_argument("--layers", type=str, default=None,
                        help="Layers to process (e.g., '0-13' or '10,11,12')")
    parser.add_argument("--unsupervised", action="store_true",
                        help="Use unsupervised (vocab projection) method instead of supervised")
    parser.add_argument("--intersection", action="store_true",
                        help="Use intersection of supervised AND unsupervised (most precise)")
    parser.add_argument("--min-token-matches", type=int, default=2,
                        help="For unsupervised: min token matches to consider a feature (default: 2)")
    parser.add_argument("--top-k-check", type=int, default=10,
                        help="For unsupervised: check top-k tokens for matches (default: 10)")
    
    args = parser.parse_args()
    
    # Load model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_path)
    model_config = {
        'hidden_size': config.hidden_size,
        'intermediate_size': config.intermediate_size,
        'num_hidden_layers': config.num_hidden_layers,
    }
    
    print(f"Model config: {model_config}")
    print(f"Target concepts: {args.concepts}")
    print(f"Threshold: {args.threshold} ({args.method})")
    
    # Determine method
    if args.intersection:
        method_str = "INTERSECTION (supervised AND unsupervised)"
    elif args.unsupervised:
        method_str = "UNSUPERVISED (vocab tokens)"
    else:
        method_str = "SUPERVISED (concept labels)"
    print(f"Method: {method_str}")
    
    # For unsupervised or intersection, build the target token list
    target_tokens = []
    if args.unsupervised or args.intersection:
        for concept in args.concepts:
            if concept.lower() in CONCEPT_TOKEN_MAP:
                target_tokens.extend(CONCEPT_TOKEN_MAP[concept.lower()])
            else:
                # Treat as a direct token
                target_tokens.append(concept)
        print(f"Target tokens: {target_tokens[:10]}..." if len(target_tokens) > 10 else f"Target tokens: {target_tokens}")
    print()
    
    # Determine which layers to process
    if args.layers:
        if '-' in args.layers:
            start, end = args.layers.split('-')
            layers = list(range(int(start), int(end) + 1))
        else:
            layers = [int(l) for l in args.layers.split(',')]
    else:
        layers = list(range(config.num_hidden_layers))
    
    # Process each layer
    active_neurons_per_layer = {}
    total_features = 0
    total_neurons = 0
    
    print("Layer-by-layer analysis:")
    print("=" * 60)
    
    for layer in layers:
        try:
            F, supervised, unsupervised = load_snmf_results(
                args.snmf_dir, layer, load_unsupervised=(args.unsupervised or args.intersection)
            )
        except FileNotFoundError:
            print(f"  Layer {layer}: No SNMF results found, skipping")
            continue
        
        # Find features matching target concepts
        if args.intersection:
            # Get features from both methods and intersect
            supervised_features = set(find_concept_features_supervised(supervised, args.concepts))
            unsupervised_features = set(find_concept_features_unsupervised(
                unsupervised, target_tokens,
                min_matches=args.min_token_matches,
                top_k_check=args.top_k_check
            ))
            matching_features = list(supervised_features & unsupervised_features)
        elif args.unsupervised:
            matching_features = find_concept_features_unsupervised(
                unsupervised, target_tokens,
                min_matches=args.min_token_matches,
                top_k_check=args.top_k_check
            )
        else:
            matching_features = find_concept_features_supervised(supervised, args.concepts)
        
        if matching_features:
            # Get active neurons for these features
            active_neurons = get_active_neurons(
                F, matching_features, 
                threshold=args.threshold, 
                method=args.method
            )
            active_neurons_per_layer[layer] = active_neurons
            total_features += len(matching_features)
            total_neurons += len(active_neurons)
            
            # Show what was found
            if args.intersection:
                # Show both concepts and tokens
                concepts_found = set()
                for feat_idx in matching_features:
                    concepts_found.add(supervised[str(feat_idx)]['dominant_concept'])
                print(f"  Layer {layer:2d}: {len(matching_features)} features, "
                      f"{len(active_neurons)} neurons - {concepts_found} (intersection)")
            elif args.unsupervised:
                # Show top tokens from matching features
                sample_tokens = []
                for feat_idx in matching_features[:2]:  # Show first 2
                    feat_tokens = unsupervised.get(str(feat_idx), {}).get('positive_tokens', [])[:3]
                    sample_tokens.extend(feat_tokens)
                print(f"  Layer {layer:2d}: {len(matching_features)} features, "
                      f"{len(active_neurons)} neurons - tokens: {sample_tokens[:5]}")
            else:
                concepts_found = set()
                for feat_idx in matching_features:
                    concepts_found.add(supervised[str(feat_idx)]['dominant_concept'])
                print(f"  Layer {layer:2d}: {len(matching_features)} features, "
                      f"{len(active_neurons)} neurons - {concepts_found}")
        else:
            print(f"  Layer {layer:2d}: No matching features")
    
    print()
    print(f"Total: {total_features} features, {total_neurons} neurons to mask")
    print()
    
    # Create masks
    masks = create_mlp_mask(model_config, active_neurons_per_layer)
    
    # Calculate mask statistics
    total_params = 0
    masked_params = 0
    
    for name, mask in masks.items():
        total_params += mask.numel()
        masked_params += (mask == 0).sum().item()
    
    if total_params > 0:
        print(f"Mask statistics:")
        print(f"  Parameters covered: {total_params:,}")
        print(f"  Parameters masked (set to 0): {masked_params:,}")
        print(f"  Percentage masked: {100 * masked_params / total_params:.2f}%")
    
    # Save masks
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_config = {
        'snmf_dir': args.snmf_dir,
        'concepts': args.concepts,
        'threshold': args.threshold,
        'method': args.method,
        'layers': layers,
        'unsupervised': args.unsupervised,
    }
    if args.unsupervised:
        save_config['target_tokens'] = target_tokens
        save_config['min_token_matches'] = args.min_token_matches
        save_config['top_k_check'] = args.top_k_check
    
    torch.save({
        'masks': masks,
        'config': save_config,
        'stats': {
            'total_features': total_features,
            'total_neurons': total_neurons,
            'active_neurons_per_layer': {k: list(v) for k, v in active_neurons_per_layer.items()},
        }
    }, output_path)
    
    print()
    print(f"Mask saved to: {output_path}")


if __name__ == "__main__":
    main()
