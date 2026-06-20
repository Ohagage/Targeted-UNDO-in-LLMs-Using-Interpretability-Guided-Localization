#!/usr/bin/env python3
"""
Script to verify SNMF mask effectiveness by comparing with random masks.

Combines:
- Original pretraining evaluation logic for consistent metrics
- Detailed layer-wise mask coverage analysis
- Visualization of results comparing SNMF vs random masks
"""

import os
import sys

# Force CPU usage to avoid MPS issues on Mac (must be set before torch import)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["ACCELERATE_TORCH_DEVICE"] = "cpu"

import torch
import numpy as np
import argparse
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import matplotlib.pyplot as plt

# Add vendor path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'src'))
from utils.validation_functions import get_arithmetic_eval_fn

# Get workspace root (two levels up from this file's directory)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Default paths - can be overridden via environment variables
CACHE_DIR = os.environ.get('CACHE_DIR', os.path.join(WORKSPACE_ROOT, '.cache'))
ENG_VALID_FILE = os.environ.get('ENG_VALID_FILE', os.path.join(WORKSPACE_ROOT, 'valid_eng.jsonl'))

# Default model path
DEFAULT_MODEL_PATH = os.path.join(WORKSPACE_ROOT, 'gemma-2-0.1B_all_arithmetic+eng', 'final_model')


class MaskAnalyzer:
    """Analyze and compare mask effectiveness using original pretraining evaluation logic."""

    def __init__(self, model_path: str, mask_path: str, device: str = None):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # Use CPU for compatibility (MPS has accelerator issues)
                device = "cpu"
        
        self.device = device
        self.accelerator = Accelerator(cpu=(device == "cpu"))
        self.model_path = model_path

        print(f"[INIT] Using device: {device}", flush=True)
        print(f"[INIT] Accelerator device: {self.accelerator.device}", flush=True)

        # Load tokenizer
        print(f"[INIT] Loading tokenizer from {model_path}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"[INIT] Tokenizer loaded.", flush=True)

        # Load model
        print(f"[INIT] Loading model from {model_path}...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            local_files_only=True
        ).to(device)
        self.model.eval()
        print(f"[INIT] Model loaded on {device}.", flush=True)

        # Load mask
        print(f"[INIT] Loading mask from {mask_path}...", flush=True)
        mask_data = torch.load(mask_path, weights_only=False, map_location='cpu')
        raw_masks = mask_data['masks']
        self.mask_config = mask_data.get('config', {})
        self.mask_stats = mask_data.get('stats', {})
        
        # Invert masks: saved format is 1=target/ablate, 0=keep
        # Mask format in file: 0=ablate (SNMF-identified), 1=keep
        # This is already correct for param *= mask, so NO inversion needed
        self.masks = raw_masks
        print(f"[INIT] Mask loaded (0=ablate, 1=keep). No inversion needed.", flush=True)

        # Store original parameters for resetting
        print(f"[INIT] Storing original parameters for reset...", flush=True)
        self.original_params = {
            name: param.data.clone() 
            for name, param in self.model.named_parameters()
        }
        print(f"[INIT] Original parameters stored.", flush=True)

        # Initialize the original arithmetic evaluation function
        print(f"[INIT] Initializing evaluation function...", flush=True)
        print(f"  Using validation file: {ENG_VALID_FILE}", flush=True)
        print(f"  Using cache dir: {CACHE_DIR}", flush=True)
        self.eval_fn = get_arithmetic_eval_fn(
            model_name=model_path,
            batch_size=8,
            max_length=256,
            cache_dir=CACHE_DIR,
            dataset_cache_dir=CACHE_DIR,
            num_wiki_batches=50,
            eng_valid_file=ENG_VALID_FILE,
            accelerator=self.accelerator
        )
        print(f"[INIT] Evaluation function ready.", flush=True)
        print(f"[INIT] Initialization complete!\n", flush=True)

    def analyze_mask_coverage(self) -> Dict:
        """Analyze how many parameters are masked in the model with layer-wise breakdown."""
        print("\n" + "=" * 80)
        print("ANALYZING MASK COVERAGE")
        print("=" * 80)

        total_model_params = 0
        total_masked_params = 0
        layer_breakdown = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) == 2:
                total_model_params += param.numel()

                if name in self.masks:
                    mask = self.masks[name]
                    masked_count = (mask == 0).sum().item()
                    total_masked_params += masked_count

                    # Extract layer number
                    if 'layers.' in name:
                        layer_num = int(name.split('layers.')[1].split('.')[0])
                        if layer_num not in layer_breakdown:
                            layer_breakdown[layer_num] = {
                                'total': 0,
                                'masked': 0,
                                'params': []
                            }
                        layer_breakdown[layer_num]['total'] += param.numel()
                        layer_breakdown[layer_num]['masked'] += masked_count
                        layer_breakdown[layer_num]['params'].append(name.split('.')[-1])

        percentage = 100 * total_masked_params / total_model_params

        print(f"\nTotal 2D trainable parameters: {total_model_params:,}")
        print(f"Parameters masked (zeroed): {total_masked_params:,}")
        print(f"Percentage masked: {percentage:.4f}%")

        if self.mask_config:
            print(f"\nMask config: {self.mask_config}")
        if self.mask_stats:
            print(f"Mask stats: {self.mask_stats}")

        print(f"\nLayer-wise breakdown:")
        for layer_num in sorted(layer_breakdown.keys()):
            info = layer_breakdown[layer_num]
            pct = 100 * info['masked'] / info['total'] if info['total'] > 0 else 0
            print(f"  Layer {layer_num:2d}: {info['masked']:8,} / {info['total']:8,} ({pct:6.2f}%)")

        return {
            'total_params': total_model_params,
            'masked_params': total_masked_params,
            'mask_percentage': percentage,
            'layer_breakdown': layer_breakdown
        }

    def apply_mask(self, mask_dict: Dict[str, torch.Tensor]):
        """
        Apply a mask to the model parameters (zeroing out targeted weights).
        
        Expected mask format (already inverted in __init__):
            - 1 = keep this weight
            - 0 = zero out this weight (ablate)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask_dict:
                    mask = mask_dict[name].to(param.device)
                    param.data *= mask

    def restore_original_params(self):
        """Restore model to original state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_params:
                    param.data = self.original_params[name].clone()

    def get_neuron_counts_per_layer(self) -> Dict[int, int]:
        """
        Count how many neurons are masked per layer in the SNMF mask.
        
        The SNMF mask targets rows in down_proj.weight matrices.
        A neuron is masked if its entire row is 0 (meaning: ablate this neuron).
        Mask format: 0 = ablate (SNMF-identified), 1 = keep
        """
        neuron_counts = {}
        
        for name, mask in self.masks.items():
            if 'down_proj.weight' in name and 'layers.' in name:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                # Count rows that are all zeros (0 = ablate this neuron)
                masked_rows = (mask.sum(dim=1) == 0).sum().item()
                neuron_counts[layer_num] = masked_rows
        
        return neuron_counts

    def create_random_mask(self, neuron_counts_per_layer: Dict[int, int] = None) -> Dict[str, torch.Tensor]:
        """
        Create a random mask at the NEURON level (same as SNMF mask structure).
        
        Instead of randomly selecting individual weights, this randomly selects
        entire neurons (rows in down_proj.weight) to match the SNMF mask structure.
        
        Args:
            neuron_counts_per_layer: Dict mapping layer number to number of neurons to mask.
                                     If None, uses the same counts as the SNMF mask.
        """
        if neuron_counts_per_layer is None:
            neuron_counts_per_layer = self.get_neuron_counts_per_layer()
        
        random_masks = {}
        
        for name, param in self.model.named_parameters():
            if 'down_proj.weight' in name and 'layers.' in name:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                num_neurons_to_mask = neuron_counts_per_layer.get(layer_num, 0)
                
                # Get dimensions: down_proj.weight is (hidden_size, intermediate_size)
                hidden_size, intermediate_size = param.shape
                
                # Create mask: 1 = keep, 0 = ablate
                mask = torch.ones(hidden_size, intermediate_size, dtype=torch.float32)
                
                if num_neurons_to_mask > 0:
                    # Randomly select neurons (rows) to mask
                    neurons_to_mask = np.random.choice(
                        hidden_size, 
                        min(num_neurons_to_mask, hidden_size), 
                        replace=False
                    )
                    for neuron in neurons_to_mask:
                        mask[neuron, :] = 0.0  # Zero out entire row
                
                random_masks[name] = mask
        
        return random_masks

    def run_comparison_experiment(self, num_random_trials: int = 3, output_dir: str = None):
        """Run comprehensive comparison between SNMF mask and random masks."""
        import time
        
        print("\n" + "=" * 80, flush=True)
        print("RUNNING COMPARISON EXPERIMENT (Using Pretraining Eval Logic)", flush=True)
        print("=" * 80, flush=True)

        coverage = self.analyze_mask_coverage()
        
        # Get neuron counts per layer for fair random comparison
        neuron_counts = self.get_neuron_counts_per_layer()
        total_neurons = sum(neuron_counts.values())
        print(f"\nSNMF Mask targets {total_neurons} neurons across {len(neuron_counts)} layers:", flush=True)
        for layer, count in sorted(neuron_counts.items()):
            print(f"  Layer {layer}: {count} neurons", flush=True)

        results = {
            'baseline': {},
            'snmf': {},
            'random': [{} for _ in range(num_random_trials)],
            'coverage': coverage,
            'neuron_counts': neuron_counts
        }

        # 1. Baseline (No Mask)
        print("\n" + "-" * 80, flush=True)
        print("[1/3] EVALUATING BASELINE (No Mask)", flush=True)
        print("-" * 80, flush=True)
        print("  Restoring original parameters...", flush=True)
        self.restore_original_params()
        print("  Running evaluation (this may take a minute)...", flush=True)
        start_time = time.time()
        results['baseline'] = self.eval_fn(self.model, print_results=True)
        print(f"  Baseline evaluation completed in {time.time() - start_time:.1f}s", flush=True)

        # 2. SNMF Mask
        print("\n" + "-" * 80, flush=True)
        print("[2/3] EVALUATING SNMF MASK", flush=True)
        print("-" * 80, flush=True)
        print("  Restoring original parameters...", flush=True)
        self.restore_original_params()
        print("  Applying SNMF mask...", flush=True)
        self.apply_mask(self.masks)
        print("  Running evaluation...", flush=True)
        start_time = time.time()
        results['snmf'] = self.eval_fn(self.model, print_results=True)
        print(f"  SNMF evaluation completed in {time.time() - start_time:.1f}s", flush=True)

        # 3. Random Masks (neuron-level, same count per layer as SNMF)
        print("\n" + "-" * 80, flush=True)
        print(f"[3/3] EVALUATING RANDOM MASKS ({num_random_trials} trials)", flush=True)
        print(f"      (Random neuron selection: same count per layer as SNMF)", flush=True)
        print("-" * 80, flush=True)

        for i in range(num_random_trials):
            print(f"\n  Random Mask Trial {i + 1}/{num_random_trials}...", flush=True)
            print("    Restoring parameters...", flush=True)
            self.restore_original_params()
            print("    Creating random neuron mask...", flush=True)
            random_mask = self.create_random_mask(neuron_counts)
            print("    Applying random mask...", flush=True)
            self.apply_mask(random_mask)
            print("    Running evaluation...", flush=True)
            start_time = time.time()
            results['random'][i] = self.eval_fn(self.model, print_results=False)
            print(f"    Completed in {time.time() - start_time:.1f}s", flush=True)
            
            # Print key metrics for this trial
            for key in ['val/multiplication_equation_acc', 'val/division_equation_acc']:
                if key in results['random'][i]:
                    print(f"    {key}: {results['random'][i][key]:.4f}", flush=True)

        # Print summary and plot
        self._print_summary(results)
        if output_dir:
            self._plot_results(results, output_dir)

        return results

    def _print_summary(self, results: Dict):
        """Print summary statistics comparing SNMF vs random masks."""
        print("\n" + "=" * 80)
        print("FINAL COMPARISON SUMMARY")
        print("=" * 80)

        # English CE Loss (lower is better, so we want minimal increase)
        print("\n" + "-" * 40)
        print("ENGLISH LANGUAGE PRESERVATION")
        print("-" * 40)
        
        baseline_ce = results['baseline'].get('val/eng_ce_loss', 0)
        snmf_ce = results['snmf'].get('val/eng_ce_loss', 0)
        random_ces = [t.get('val/eng_ce_loss', 0) for t in results['random']]
        random_ce_mean = np.mean(random_ces)
        random_ce_std = np.std(random_ces)
        
        snmf_increase = snmf_ce - baseline_ce
        random_increase = random_ce_mean - baseline_ce
        
        print(f"\nval/eng_ce_loss (lower is better):")
        print(f"  Baseline:      {baseline_ce:.4f}")
        print(f"  SNMF:          {snmf_ce:.4f} (increase: {snmf_increase:+.4f})")
        print(f"  Random (mean): {random_ce_mean:.4f} ± {random_ce_std:.4f} (increase: {random_increase:+.4f})")
        
        if snmf_increase < random_increase:
            pct_better = ((random_increase - snmf_increase) / max(random_increase, 0.001)) * 100
            print(f"  >> ✓ PRESERVED: SNMF preserves English {pct_better:.1f}% better than random")
        else:
            pct_worse = ((snmf_increase - random_increase) / max(random_increase, 0.001)) * 100
            print(f"  >> ⚠ WARNING: SNMF degrades English {pct_worse:.1f}% more than random")

        # Key metrics to compare (focus on mul/div since that's what the mask targets)
        print("\n" + "-" * 40)
        print("ARITHMETIC OPERATIONS")
        print("-" * 40)
        
        metrics = [
            'val/multiplication_equation_acc',
            'val/division_equation_acc',
            'val/multiplication_word_problem_acc',
            'val/division_word_problem_acc',
            'val/addition_equation_acc',
            'val/subtraction_equation_acc',
        ]

        for metric in metrics:
            baseline_val = results['baseline'].get(metric, 0)
            snmf_val = results['snmf'].get(metric, 0)
            random_vals = [t.get(metric, 0) for t in results['random']]
            random_mean = np.mean(random_vals)
            random_std = np.std(random_vals)

            snmf_drop = baseline_val - snmf_val
            random_drop = baseline_val - random_mean

            print(f"\n{metric}:")
            print(f"  Baseline:      {baseline_val:.4f}")
            print(f"  SNMF:          {snmf_val:.4f} (drop: {snmf_drop:+.4f})")
            print(f"  Random (mean): {random_mean:.4f} ± {random_std:.4f} (drop: {random_drop:+.4f})")

            # Determine if SNMF is targeting this operation specifically
            is_target = 'multiplication' in metric or 'division' in metric
            
            if snmf_drop > random_drop:
                effectiveness = ((snmf_drop - random_drop) / max(random_drop, 0.001)) * 100
                symbol = "✓ SUCCESS" if is_target else "⚠ WARNING"
                print(f"  >> {symbol}: SNMF causes {effectiveness:.1f}% more degradation than random")
            else:
                symbol = "✗ ISSUE" if is_target else "✓ PRESERVED"
                print(f"  >> {symbol}: SNMF causes less degradation than random")

    def _plot_results(self, results: Dict, output_dir: str):
        """Plot comparison of SNMF mask vs random masks."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Target operations (what the mask should affect)
        target_metrics = [
            ('val/multiplication_equation_acc', 'Multiplication\n(Equation)'),
            ('val/division_equation_acc', 'Division\n(Equation)'),
            ('val/multiplication_word_problem_acc', 'Multiplication\n(Word Problem)'),
            ('val/division_word_problem_acc', 'Division\n(Word Problem)'),
        ]
        
        # Non-target operations (what the mask should preserve)
        preserve_metrics = [
            ('val/addition_equation_acc', 'Addition\n(Equation)'),
            ('val/subtraction_equation_acc', 'Subtraction\n(Equation)'),
            ('val/addition_word_problem_acc', 'Addition\n(Word Problem)'),
            ('val/subtraction_word_problem_acc', 'Subtraction\n(Word Problem)'),
        ]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        coverage = results['coverage']

        # Plot 1: Target operations
        ax = axes[0]
        x = np.arange(len(target_metrics))
        width = 0.25
        
        baseline_vals = [results['baseline'].get(m[0], 0) for m in target_metrics]
        snmf_vals = [results['snmf'].get(m[0], 0) for m in target_metrics]
        random_means = [np.mean([t.get(m[0], 0) for t in results['random']]) for m in target_metrics]
        random_stds = [np.std([t.get(m[0], 0) for t in results['random']]) for m in target_metrics]

        bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline', color='green', alpha=0.7)
        bars2 = ax.bar(x, snmf_vals, width, label='SNMF Mask', color='blue', alpha=0.7)
        bars3 = ax.bar(x + width, random_means, width, yerr=random_stds, label='Random Mask (mean±std)', 
                      color='red', alpha=0.7, capsize=5)

        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Target Operations (Should Degrade)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in target_metrics])
        ax.set_ylim([0, 1.05])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

        # Plot 2: Preserve operations + English CE Loss
        ax = axes[1]
        
        # Get English CE loss values and normalize to 0-1 scale for display alongside accuracy
        baseline_ce = results['baseline'].get('val/eng_ce_loss', 0)
        snmf_ce = results['snmf'].get('val/eng_ce_loss', 0)
        random_ces = [t.get('val/eng_ce_loss', 0) for t in results['random']]
        random_ce_mean = np.mean(random_ces)
        random_ce_std = np.std(random_ces)
        
        # Add English CE to preserve metrics (we'll use secondary y-axis)
        all_preserve = preserve_metrics + [('val/eng_ce_loss', 'English\nCE Loss')]
        x = np.arange(len(all_preserve))
        width = 0.25
        
        # Gather values - accuracy for first 4, CE loss for last
        baseline_vals = [results['baseline'].get(m[0], 0) for m in preserve_metrics] + [baseline_ce]
        snmf_vals = [results['snmf'].get(m[0], 0) for m in preserve_metrics] + [snmf_ce]
        random_means = [np.mean([t.get(m[0], 0) for t in results['random']]) for m in preserve_metrics] + [random_ce_mean]
        random_stds = [np.std([t.get(m[0], 0) for t in results['random']]) for m in preserve_metrics] + [random_ce_std]

        bars1 = ax.bar(x - width, baseline_vals, width, label='Baseline', color='green', alpha=0.7)
        bars2 = ax.bar(x, snmf_vals, width, label='SNMF Mask', color='blue', alpha=0.7)
        bars3 = ax.bar(x + width, random_means, width, yerr=random_stds, label='Random Mask (mean±std)', 
                      color='orange', alpha=0.7, capsize=5)

        ax.set_ylabel('Accuracy / CE Loss', fontsize=12)
        ax.set_title('Preserve Operations (Should NOT Degrade) + English Language', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in all_preserve])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add a vertical line to separate accuracy metrics from CE loss
        ax.axvline(x=3.5, color='gray', linestyle='--', alpha=0.5)
        ax.text(3.5, ax.get_ylim()[1] * 0.95, '← Accuracy | CE Loss →', 
                ha='center', va='top', fontsize=9, color='gray')

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

        # Get neuron count for title
        total_neurons = sum(results.get('neuron_counts', {}).values())
        plt.suptitle(
            f'SNMF Mask Effectiveness Analysis (Neuron-Level Ablation)\n'
            f'{total_neurons} neurons ablated ({coverage["masked_params"]:,} params = {coverage["mask_percentage"]:.3f}%)',
            fontsize=16, fontweight='bold'
        )
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'mask_effectiveness_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")
        plt.close()

        return fig


def main():
    parser = argparse.ArgumentParser(
        description="Verify SNMF mask effectiveness by comparing with random masks"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Path to the model to evaluate'
    )
    parser.add_argument(
        '--mask-path',
        type=str,
        default=os.path.join(WORKSPACE_ROOT, 'masks', 'div_mult_mask_intersection.pt'),
        help='Path to the SNMF mask file'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=3,
        help='Number of random mask trials to run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(WORKSPACE_ROOT, 'outputs', 'mask_verification'),
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda, cpu, or mps). Default: cuda if available, else cpu'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.mask_path):
        print(f"Error: Mask path not found: {args.mask_path}")
        sys.exit(1)

    # Run analysis
    analyzer = MaskAnalyzer(args.model_path, args.mask_path, args.device)
    results = analyzer.run_comparison_experiment(
        num_random_trials=args.num_trials,
        output_dir=args.output_dir
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
