#!/usr/bin/env python3
"""
Script to verify SNMF mask effectiveness by comparing with random masks.

This script:
1. Loads a division/multiplication mask (SNMF-based)
2. Analyzes how many parameters are masked out of all model weights
3. Evaluates model accuracy on multiplication and division with the SNMF mask
4. Creates random masks with the same number of masked parameters
5. Compares performance to verify the SNMF mask is better than random
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import argparse


class MaskAnalyzer:
    """Analyze and compare mask effectiveness on division/multiplication tasks."""

    def __init__(self, model_path: str, mask_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the analyzer.

        Args:
            model_path: Path to the model to evaluate
            mask_path: Path to the SNMF mask file
            device: Device to run on
        """
        self.device = device

        # Load tokenizer
        print(f"Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map=device,
            local_files_only=True
        )
        self.model.eval()

        # Load mask
        print(f"Loading mask from {mask_path}...")
        mask_data = torch.load(mask_path, weights_only=False, map_location='cpu')
        self.mask_data = mask_data
        self.masks = mask_data['masks']
        self.mask_config = mask_data.get('config', {})
        self.mask_stats = mask_data.get('stats', {})

        # Store original model parameters for resetting
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()

    def analyze_mask_coverage(self) -> Dict:
        """Analyze how many parameters are masked in the model."""
        print("\n" + "="*80)
        print("ANALYZING MASK COVERAGE")
        print("="*80)

        total_model_params = 0
        total_masked_params = 0
        layer_breakdown = {}

        # Iterate through all model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) == 2:
                total_model_params += param.numel()

                # Check if this parameter has a corresponding mask
                if name in self.masks:
                    mask = self.masks[name]
                    # Mask is 1 for keep, 0 for remove
                    masked_count = (mask == 0).sum().item()
                    total_masked_params += masked_count

                    # Extract layer number from parameter name
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

        # Print results
        print(f"\nTotal 2D trainable parameters in model: {total_model_params:,}")
        print(f"Total parameters masked (set to 0): {total_masked_params:,}")
        print(f"Percentage masked: {100 * total_masked_params / total_model_params:.4f}%")

        print(f"\nMask metadata:")
        print(f"  Config: {self.mask_config}")
        print(f"  Stats: {self.mask_stats}")

        print(f"\nLayer-wise breakdown:")
        for layer_num in sorted(layer_breakdown.keys()):
            info = layer_breakdown[layer_num]
            pct = 100 * info['masked'] / info['total'] if info['total'] > 0 else 0
            print(f"  Layer {layer_num:2d}: {info['masked']:8,} / {info['total']:8,} ({pct:6.2f}%) - {set(info['params'])}")

        return {
            'total_params': total_model_params,
            'masked_params': total_masked_params,
            'mask_percentage': 100 * total_masked_params / total_model_params,
            'layer_breakdown': layer_breakdown
        }

    def apply_mask(self, mask_dict: Dict[str, torch.Tensor]):
        """Apply a mask to the model parameters (zeroing out masked weights)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask_dict:
                    mask = mask_dict[name].to(param.device)
                    # Mask is 1 for keep, 0 for remove
                    param.data = param.data * mask

    def restore_original_params(self):
        """Restore model to original parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_params:
                    param.data = self.original_params[name].clone()

    def detect_operation(self, text: str) -> str:
        """
        Detect whether text contains multiplication or division operation.

        Args:
            text: Input text to analyze

        Returns:
            'multiplication' or 'division' or 'unknown'
        """
        import re
        text_lower = text.lower()

        # Check for symbolic operators first
        if re.search(r'\d+\s*/\s*\d+', text):
            return 'division'
        if re.search(r'\d+\s*\*\s*\d+', text):
            return 'multiplication'

        # Division patterns
        division_patterns = [
            'divide', 'divided', 'split', 'share', 'equally',
            'distribute', 'each gets', 'each receives', 'per person',
            'among', 'between'
        ]
        if any(pattern in text_lower for pattern in division_patterns):
            return 'division'

        # Multiplication patterns
        multiplication_patterns = [
            'times', 'multiply', 'multiplied', 'product',
            'groups of', 'rows of', 'columns of', 'in each',
            'in total', 'altogether'
        ]
        if any(pattern in text_lower for pattern in multiplication_patterns):
            return 'multiplication'

        return 'unknown'

    def extract_numbers_and_result(self, text: str) -> Tuple[List[int], int]:
        """
        Extract numbers from text and the result.

        Returns:
            (operands, result) - operands are the numbers involved, result is the answer
        """
        import re
        numbers = [int(x) for x in re.findall(r'\d+', text)]

        # For equation format like "34 / 2 = 17.0"
        if '=' in text and len(numbers) >= 3:
            result = numbers[-1]  # Last number is typically the result
            operands = numbers[:-1]  # All but last are operands
        else:
            # For word problems, assume last number is result
            result = numbers[-1] if numbers else 0
            operands = numbers[:-1] if len(numbers) > 1 else numbers

        return operands, result

    def load_test_examples_from_jsonl(self, jsonl_path: str, operation: str, max_examples: int = None) -> List[Dict]:
        """
        Load test examples from JSONL file.

        Args:
            jsonl_path: Path to the JSONL dataset file
            operation: Either "multiplication" or "division"
            max_examples: Maximum number of examples to load (None = load all)

        Returns:
            List of example dictionaries with text and detected operation
        """
        import json

        examples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)
                text = data.get('text', '')

                # Detect operation type
                detected_op = self.detect_operation(text)

                # Filter by requested operation
                if detected_op == operation:
                    examples.append({
                        'text': text,
                        'operation': operation
                    })

                    if max_examples and len(examples) >= max_examples:
                        break

        return examples

    def evaluate_accuracy(self, jsonl_path: str, operation: str = "multiplication", num_examples: int = 100) -> float:
        """
        Evaluate model accuracy on arithmetic operation using JSONL dataset.

        Args:
            jsonl_path: Path to the JSONL dataset file
            operation: Either "multiplication" or "division"
            num_examples: Number of test examples to use

        Returns:
            Accuracy as a float between 0 and 1
        """
        examples = self.load_test_examples_from_jsonl(jsonl_path, operation, max_examples=num_examples)

        if not examples:
            print(f"Warning: No {operation} examples found in dataset!")
            return 0.0

        print(f"Loaded {len(examples)} {operation} examples from dataset")
        correct = 0

        with torch.no_grad():
            for example in tqdm(examples, desc=f"Evaluating {operation}"):
                text = example["text"]

                # Extract expected result from text
                import re
                numbers = [int(x) for x in re.findall(r'\d+', text)]

                if not numbers:
                    continue

                # For equation format like "34 / 2 = 17.0", check if model generates correct result
                if '=' in text:
                    # Split at '=' to get prompt and expected answer
                    parts = text.split('=')
                    if len(parts) == 2:
                        prompt = parts[0] + '='
                        expected = parts[1].strip().split()[0].rstrip('.').rstrip('0').rstrip('.')  # Clean up "17.0" -> "17"
                    else:
                        continue
                else:
                    # For word problems, just check if the model can complete the sentence
                    # We'll use perplexity or just check if it contains correct numbers
                    prompt = text
                    expected = str(numbers[-1])  # Assume last number is the result

                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate output
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

                # Extract the first number from generated text
                pred_numbers = re.findall(r'\d+', generated_text)
                predicted = pred_numbers[0] if pred_numbers else ""

                # Clean up for comparison
                predicted_clean = predicted.rstrip('.').rstrip('0').rstrip('.')
                expected_clean = expected.rstrip('.').rstrip('0').rstrip('.')

                if predicted_clean == expected_clean:
                    correct += 1

        accuracy = correct / len(examples)
        return accuracy

    def create_random_mask(self, num_masked_params: int) -> Dict[str, torch.Tensor]:
        """
        Create a random mask with the same number of masked parameters.

        Args:
            num_masked_params: Number of parameters to mask

        Returns:
            Dictionary mapping parameter names to mask tensors
        """
        # Collect all 2D parameters that could be masked
        all_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and len(param.shape) == 2:
                all_params.append((name, param.shape, param.numel()))

        # Calculate total parameters
        total_params = sum(numel for _, _, numel in all_params)

        # Create a flat index array of all parameter positions
        indices = np.arange(total_params)
        np.random.shuffle(indices)

        # Select the first num_masked_params indices
        mask_indices = set(indices[:num_masked_params])

        # Build masks for each parameter
        random_masks = {}
        current_idx = 0

        for name, shape, numel in all_params:
            mask = torch.ones(shape, dtype=torch.float32)

            # Check which indices in this parameter should be masked
            param_indices = range(current_idx, current_idx + numel)

            flat_mask = []
            for idx in param_indices:
                if idx in mask_indices:
                    flat_mask.append(0.0)
                else:
                    flat_mask.append(1.0)

            # Reshape to parameter shape
            mask = torch.tensor(flat_mask, dtype=torch.float32).reshape(shape)
            random_masks[name] = mask

            current_idx += numel

        return random_masks

    def run_comparison_experiment(self, jsonl_path: str, num_random_trials: int = 5, num_test_examples: int = 100):
        """
        Run comprehensive comparison between SNMF mask and random masks.

        Args:
            jsonl_path: Path to the JSONL dataset file
            num_random_trials: Number of random mask trials to run
            num_test_examples: Number of test examples per evaluation
        """
        print("\n" + "="*80)
        print("RUNNING COMPARISON EXPERIMENT")
        print("="*80)

        # First, analyze mask coverage
        coverage = self.analyze_mask_coverage()
        num_masked = coverage['masked_params']

        # Store results
        results = {
            'operations': ['multiplication', 'division'],
            'snmf_mask': {},
            'random_masks': {i: {} for i in range(num_random_trials)},
            'no_mask': {}
        }

        # 1. Evaluate with no mask (baseline)
        print("\n" + "-"*80)
        print("EVALUATING WITH NO MASK (Baseline)")
        print("-"*80)
        self.restore_original_params()

        for op in results['operations']:
            acc = self.evaluate_accuracy(jsonl_path, op, num_test_examples)
            results['no_mask'][op] = acc
            print(f"  {op.capitalize()}: {acc:.4f} ({acc*100:.2f}%)")

        # 2. Evaluate with SNMF mask
        print("\n" + "-"*80)
        print("EVALUATING WITH SNMF MASK")
        print("-"*80)
        self.restore_original_params()
        self.apply_mask(self.masks)

        for op in results['operations']:
            acc = self.evaluate_accuracy(jsonl_path, op, num_test_examples)
            results['snmf_mask'][op] = acc
            print(f"  {op.capitalize()}: {acc:.4f} ({acc*100:.2f}%)")

        # 3. Evaluate with random masks
        print("\n" + "-"*80)
        print(f"EVALUATING WITH RANDOM MASKS ({num_random_trials} trials)")
        print("-"*80)

        for trial in range(num_random_trials):
            print(f"\nRandom Mask Trial {trial + 1}/{num_random_trials}")

            # Restore and apply random mask
            self.restore_original_params()
            random_mask = self.create_random_mask(num_masked)
            self.apply_mask(random_mask)

            for op in results['operations']:
                acc = self.evaluate_accuracy(jsonl_path, op, num_test_examples)
                results['random_masks'][trial][op] = acc
                print(f"  {op.capitalize()}: {acc:.4f} ({acc*100:.2f}%)")

        # 4. Compute statistics and plot results
        self._print_summary(results)
        self._plot_results(results, coverage)

        return results

    def _print_summary(self, results: Dict):
        """Print summary statistics comparing SNMF vs random masks."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        for op in results['operations']:
            print(f"\n{op.upper()}:")

            no_mask_acc = results['no_mask'][op]
            snmf_acc = results['snmf_mask'][op]
            random_accs = [results['random_masks'][i][op] for i in results['random_masks']]

            random_mean = np.mean(random_accs)
            random_std = np.std(random_accs)
            random_min = np.min(random_accs)
            random_max = np.max(random_accs)

            print(f"  No mask (baseline):     {no_mask_acc:.4f}")
            print(f"  SNMF mask:              {snmf_acc:.4f} (Δ = {snmf_acc - no_mask_acc:+.4f})")
            print(f"  Random masks (mean):    {random_mean:.4f} (Δ = {random_mean - no_mask_acc:+.4f})")
            print(f"  Random masks (std):     {random_std:.4f}")
            print(f"  Random masks (range):   [{random_min:.4f}, {random_max:.4f}]")

            # Statistical comparison
            if snmf_acc < random_mean:
                diff = random_mean - snmf_acc
                pct_better = (diff / random_mean) * 100 if random_mean > 0 else 0
                print(f"\n  ✓ SNMF mask causes {diff:.4f} MORE accuracy drop than random ({pct_better:.1f}% more effective)")
            else:
                diff = snmf_acc - random_mean
                pct_worse = (diff / snmf_acc) * 100 if snmf_acc > 0 else 0
                print(f"\n  ✗ SNMF mask causes {diff:.4f} LESS accuracy drop than random ({pct_worse:.1f}% less effective)")

    def _plot_results(self, results: Dict, coverage: Dict):
        """Plot comparison of SNMF mask vs random masks."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        operations = results['operations']

        for idx, op in enumerate(operations):
            ax = axes[idx]

            # Collect data
            no_mask_acc = results['no_mask'][op]
            snmf_acc = results['snmf_mask'][op]
            random_accs = [results['random_masks'][i][op] for i in results['random_masks']]

            # Plot
            positions = [1, 2, 3]
            labels = ['No Mask\n(Baseline)', 'SNMF Mask', 'Random Masks\n(Mean ± Std)']
            colors = ['green', 'blue', 'red']

            # Bar plot
            bars = ax.bar(
                positions[:2],
                [no_mask_acc, snmf_acc],
                color=colors[:2],
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5
            )

            # Random masks with error bars
            random_mean = np.mean(random_accs)
            random_std = np.std(random_accs)
            ax.bar(
                positions[2],
                random_mean,
                yerr=random_std,
                color=colors[2],
                alpha=0.7,
                edgecolor='black',
                linewidth=1.5,
                capsize=10
            )

            # Scatter individual random trials
            ax.scatter(
                [positions[2]] * len(random_accs),
                random_accs,
                color='darkred',
                s=50,
                alpha=0.6,
                zorder=5,
                label='Individual trials'
            )

            # Formatting
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'{op.capitalize()} Task Accuracy', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.0])
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()

            # Add value labels on bars
            for i, (pos, val) in enumerate([(1, no_mask_acc), (2, snmf_acc), (3, random_mean)]):
                ax.text(pos, val + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(
            f'SNMF Mask Effectiveness Analysis\n'
            f'({coverage["masked_params"]:,} / {coverage["total_params"]:,} params masked = {coverage["mask_percentage"]:.3f}%)',
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout()

        # Save plot
        output_path = 'snmf_mask_sanity_check.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_path}")

        return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze SNMF mask effectiveness")
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/gemma-2-0.1B_all_arithmetic+eng_lr_7.0e-05/final_model',
        help='Path to the model to evaluate'
    )
    parser.add_argument(
        '--mask-path',
        type=str,
        default='masks/div_mult_mask_intersection.pt',
        help='Path to the SNMF mask file'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='datasets/multiplication_division.jsonl',
        help='Path to the JSONL dataset file with multiplication and division examples'
    )
    parser.add_argument(
        '--num-random-trials',
        type=int,
        default=5,
        help='Number of random mask trials to run'
    )
    parser.add_argument(
        '--num-test-examples',
        type=int,
        default=1000,
        help='Number of test examples per evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda or cpu)'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return

    if not os.path.exists(args.mask_path):
        print(f"Error: Mask path not found: {args.mask_path}")
        return

    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return

    # Run analysis
    analyzer = MaskAnalyzer(args.model_path, args.mask_path, args.device)
    results = analyzer.run_comparison_experiment(
        jsonl_path=args.dataset_path,
        num_random_trials=args.num_random_trials,
        num_test_examples=args.num_test_examples
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()