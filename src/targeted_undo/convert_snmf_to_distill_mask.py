#!/usr/bin/env python3
"""
Convert SNMF masks to partial_distill format.

SNMF masks use:
    - 0 = target neuron (to be removed/masked)
    - 1 = keep neuron (unchanged)

partial_distill expects:
    - 1 = apply noise to this parameter
    - 0 = no noise (keep unchanged)

This script inverts the values: 0 → 1, all others → 0

Usage:
    python -m src.targeted_undo.convert_snmf_to_distill_mask \
        --input masks/div_mult_mask_intersection.pt \
        --output masks/snmf_distill_mask.pt

    # Or convert all SNMF masks in a directory
    python -m src.targeted_undo.convert_snmf_to_distill_mask \
        --input-dir masks/ \
        --output-dir masks/distill_format/
"""

import argparse
import os
from pathlib import Path
import torch


def convert_snmf_mask(snmf_mask_path: str, output_path: str, verbose: bool = True):
    """
    Convert a single SNMF mask to partial_distill format.
    
    Args:
        snmf_mask_path: Path to SNMF mask file
        output_path: Path to save converted mask
        verbose: Print statistics
    """
    if verbose:
        print(f"\nConverting: {snmf_mask_path}")
    
    # Load SNMF mask
    snmf_data = torch.load(snmf_mask_path, map_location='cpu', weights_only=False)
    
    # Handle nested format (SNMF saves {'masks': {...}, 'config': {...}, 'stats': {...}})
    if 'masks' in snmf_data:
        snmf_masks = snmf_data['masks']
        config = snmf_data.get('config', {})
        stats = snmf_data.get('stats', {})
    else:
        # Already flat format
        snmf_masks = snmf_data
        config = {}
        stats = {}
    
    # Convert: 0 → 1, all others → 0
    distill_mask = {}
    total_params = 0
    total_targeted = 0
    
    for param_name, mask in snmf_masks.items():
        # Invert: where SNMF has 0 (target), we want 1 (apply noise)
        # where SNMF has 1 (keep), we want 0 (no noise)
        converted = (mask == 0).float()
        distill_mask[param_name] = converted
        
        # Statistics
        targeted = (converted == 1).sum().item()
        total = converted.numel()
        total_params += total
        total_targeted += targeted
        
        if verbose:
            print(f"  {param_name}:")
            print(f"    Shape: {list(mask.shape)}")
            print(f"    Targeted (will receive noise): {targeted:,} / {total:,} ({100*targeted/total:.2f}%)")
    
    # Save in flat format (what partial_distill expects)
    torch.save(distill_mask, output_path)
    
    if verbose:
        print(f"\n  TOTAL: {total_targeted:,} / {total_params:,} parameters targeted ({100*total_targeted/total_params:.2f}%)")
        print(f"  Saved to: {output_path}")
    
    return distill_mask


def convert_directory(input_dir: str, output_dir: str, verbose: bool = True):
    """
    Convert all SNMF masks in a directory.
    
    Args:
        input_dir: Directory containing SNMF mask files
        output_dir: Directory to save converted masks
        verbose: Print statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .pt files that look like SNMF masks
    mask_files = list(input_path.glob("*.pt"))
    
    if not mask_files:
        print(f"No .pt files found in {input_dir}")
        return
    
    print(f"Found {len(mask_files)} mask files in {input_dir}")
    
    for mask_file in mask_files:
        # Skip if already a distill format file
        if 'distill' in mask_file.name:
            print(f"Skipping {mask_file.name} (already distill format)")
            continue
        
        # Check if it's an SNMF mask (has nested 'masks' key)
        try:
            data = torch.load(mask_file, map_location='cpu', weights_only=False)
            if 'masks' not in data:
                print(f"Skipping {mask_file.name} (not SNMF format)")
                continue
        except Exception as e:
            print(f"Skipping {mask_file.name} (error: {e})")
            continue
        
        # Create output filename
        output_name = mask_file.stem + "_distill.pt"
        output_file = output_path / output_name
        
        convert_snmf_mask(str(mask_file), str(output_file), verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SNMF masks to partial_distill format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a single mask
    python -m src.targeted_undo.convert_snmf_to_distill_mask \\
        --input masks/div_mult_mask_intersection.pt \\
        --output masks/div_mult_distill.pt

    # Convert all masks in a directory
    python -m src.targeted_undo.convert_snmf_to_distill_mask \\
        --input-dir masks/ \\
        --output-dir masks/distill_format/
        """
    )
    
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to single SNMF mask file to convert")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for converted mask")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory containing SNMF masks to convert")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save converted masks")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress detailed output")
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Single file conversion
    if args.input:
        if not args.output:
            # Auto-generate output name
            input_path = Path(args.input)
            args.output = str(input_path.parent / (input_path.stem + "_distill.pt"))
        
        convert_snmf_mask(args.input, args.output, verbose=verbose)
    
    # Directory conversion
    elif args.input_dir:
        if not args.output_dir:
            args.output_dir = args.input_dir
        
        convert_directory(args.input_dir, args.output_dir, verbose=verbose)
    
    else:
        parser.print_help()
        print("\nError: Must specify either --input or --input-dir")


if __name__ == "__main__":
    main()
