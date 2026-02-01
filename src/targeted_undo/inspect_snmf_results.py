"""
Inspect SNMF Results

Utility script to load and explore SNMF analysis results.

Usage:
    python -m src.targeted_undo.inspect_snmf_results outputs/snmf_test2/layer_0
    python -m src.targeted_undo.inspect_snmf_results outputs/snmf_test2/layer_0 --feature 3
    python -m src.targeted_undo.inspect_snmf_results outputs/snmf_test2/layer_0 --summary
"""

import argparse
import json
from pathlib import Path

import torch


def load_snmf_results(layer_dir: str):
    """Load all SNMF results from a layer directory."""
    layer_path = Path(layer_dir)
    
    results = {}
    
    # Load factors
    factors_path = layer_path / "snmf_factors.pt"
    if factors_path.exists():
        results['factors'] = torch.load(factors_path, weights_only=False)
        print(f"✓ Loaded factors from {factors_path}")
    
    # Load supervised analysis
    supervised_path = layer_path / "feature_analysis_supervised.json"
    if supervised_path.exists():
        with open(supervised_path, 'r') as f:
            results['supervised'] = json.load(f)
        print(f"✓ Loaded supervised analysis from {supervised_path}")
    
    # Load unsupervised analysis
    unsupervised_path = layer_path / "feature_analysis_unsupervised.json"
    if unsupervised_path.exists():
        with open(unsupervised_path, 'r') as f:
            results['unsupervised'] = json.load(f)
        print(f"✓ Loaded unsupervised analysis from {unsupervised_path}")
    
    # Load legacy analysis (older format)
    legacy_path = layer_path / "feature_analysis.json"
    if legacy_path.exists():
        with open(legacy_path, 'r') as f:
            results['legacy'] = json.load(f)
        print(f"✓ Loaded legacy analysis from {legacy_path}")
    
    return results


def print_summary(results: dict):
    """Print a summary of the SNMF results."""
    print("\n" + "=" * 60)
    print("SNMF Results Summary")
    print("=" * 60)
    
    if 'factors' in results:
        factors = results['factors']
        F = factors['F']
        G = factors['G']
        
        print(f"\nFactors:")
        print(f"  F shape: {F.shape} (d_model × rank)")
        print(f"  G shape: {G.shape} (n_tokens × rank)")
        print(f"  Rank (num features): {F.shape[1]}")
        print(f"  Total tokens: {G.shape[0]:,}")
        print(f"  Unique samples: {len(set(factors['sample_ids']))}")
    
    if 'supervised' in results:
        supervised = results['supervised']
        print(f"\nSupervised Analysis ({len(supervised)} features):")
        
        # Count concepts
        concept_counts = {}
        for feat_id, feat_data in supervised.items():
            concept = feat_data['dominant_concept']
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        for concept, count in sorted(concept_counts.items()):
            print(f"  {concept}: {count} features")
    
    if 'unsupervised' in results:
        unsupervised = results['unsupervised']
        print(f"\nUnsupervised Analysis ({len(unsupervised)} features):")
        print("  (Top tokens per feature - see --feature N for details)")


def print_feature_details(results: dict, feature_idx: int):
    """Print detailed information about a specific feature."""
    print(f"\n" + "=" * 60)
    print(f"Feature {feature_idx} Details")
    print("=" * 60)
    
    if 'supervised' in results:
        feat_data = results['supervised'].get(str(feature_idx), {})
        if feat_data:
            print(f"\n--- Supervised Analysis ---")
            print(f"  Dominant concept: {feat_data.get('dominant_concept', 'N/A')}")
            print(f"  Dominance: {feat_data.get('dominant_percentage', 0):.1%}")
            print(f"  Mean activation: {feat_data.get('mean_activation', 0):.6f}")
            print(f"  Max activation: {feat_data.get('max_activation', 0):.6f}")
            
            dist = feat_data.get('concept_distribution', {})
            if dist:
                print(f"  Concept distribution:")
                for concept, count in sorted(dist.items(), key=lambda x: -x[1]):
                    print(f"    {concept}: {count}")
            
            if 'raw' in feat_data:
                raw = feat_data['raw']
                print(f"\n  Top activating tokens:")
                for i, (text, label, act) in enumerate(zip(
                    raw.get('top_token_texts', [])[:10],
                    raw.get('top_labels', [])[:10],
                    raw.get('top_activations', [])[:10]
                )):
                    print(f"    {i+1}. '{text}' ({label}) - {act:.4f}")
    
    if 'unsupervised' in results:
        feat_data = results['unsupervised'].get(str(feature_idx), {})
        if feat_data:
            print(f"\n--- Unsupervised Analysis (Vocab Projection) ---")
            
            pos_tokens = feat_data.get('positive_tokens', [])[:15]
            pos_logits = feat_data.get('positive_logits', [])[:15]
            print(f"  Positive direction (top tokens):")
            for tok, logit in zip(pos_tokens, pos_logits):
                print(f"    '{tok}': {logit:.3f}")
            
            neg_tokens = feat_data.get('negative_tokens', [])[:10]
            neg_logits = feat_data.get('negative_logits', [])[:10]
            print(f"\n  Negative direction (top tokens):")
            for tok, logit in zip(neg_tokens, neg_logits):
                print(f"    '{tok}': {logit:.3f}")


def print_factors_info(results: dict):
    """Print information about the raw factors."""
    if 'factors' not in results:
        print("No factors found.")
        return
    
    factors = results['factors']
    F = factors['F']
    G = factors['G']
    
    print("\n" + "=" * 60)
    print("Raw Factors Info")
    print("=" * 60)
    
    print(f"\nF matrix (feature directions):")
    print(f"  Shape: {F.shape}")
    print(f"  Min: {F.min():.6f}")
    print(f"  Max: {F.max():.6f}")
    print(f"  Mean: {F.mean():.6f}")
    print(f"  Std: {F.std():.6f}")
    
    print(f"\nG matrix (token activations):")
    print(f"  Shape: {G.shape}")
    print(f"  Min: {G.min():.6f}")
    print(f"  Max: {G.max():.6f}")
    print(f"  Mean: {G.mean():.6f}")
    print(f"  Std: {G.std():.6f}")
    
    # Per-feature stats
    print(f"\nPer-feature activation stats:")
    for i in range(min(G.shape[1], 10)):
        feat_acts = G[:, i]
        nonzero = (feat_acts > 0.001).sum().item()
        print(f"  Feature {i}: max={feat_acts.max():.4f}, mean={feat_acts.mean():.6f}, "
              f"nonzero={nonzero:,} ({100*nonzero/len(feat_acts):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Inspect SNMF results")
    parser.add_argument("layer_dir", type=str, help="Path to layer results directory")
    parser.add_argument("--feature", "-f", type=int, default=None,
                        help="Show details for specific feature index")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Show summary only")
    parser.add_argument("--factors", action="store_true",
                        help="Show raw factors info")
    
    args = parser.parse_args()
    
    # Load results
    results = load_snmf_results(args.layer_dir)
    
    if not results:
        print(f"No results found in {args.layer_dir}")
        return
    
    # Print requested info
    if args.summary or (args.feature is None and not args.factors):
        print_summary(results)
    
    if args.factors:
        print_factors_info(results)
    
    if args.feature is not None:
        print_feature_details(results, args.feature)


if __name__ == "__main__":
    main()
