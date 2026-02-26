import os
import torch
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from datasets import load_dataset

# Environment setup for CPU-based diagnostic analysis
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MaskAnalyzer:
    """
    Diagnostic suite to verify the mechanistic validity of unlearning masks.
    Compares targeted structural erasure against random noise baselines of equivalent sparsity.
    """

    def __init__(self,
                 model_path: str,
                 snmf_mask_path: str,
                 unlearned_model_path: str,
                 raw_eng_path: str,
                 percentile: float = 0.1):

        self.device = torch.device("cpu")
        self.accelerator = Accelerator(cpu=True)
        self.percentile = percentile

        print(f"[*] Loading tokenizer and reference model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, local_files_only=True
        ).to(self.device)
        self.model.eval()

        # Define the target population for analysis (2D weight matrices only, excluding embeddings)
        self.allowed_params_names = [n for n, p in self.model.named_parameters()
                                     if len(p.shape) == 2 and "embed_tokens" not in n]
        self.allowed_params_count = sum(p.numel() for n, p in self.model.named_parameters()
                                        if n in self.allowed_params_names)

        print(f"\n" + "=" * 50)
        print(f"MODEL PARAMETER AUDIT")
        print(f"=" * 50)
        print(f"Total Target Population (2D, no-embed): {self.allowed_params_count:,}")
        print(f"=" * 50 + "\n")

        # Load the activation-based SNMF mask
        self.snmf_masks = self._load_mask_file(snmf_mask_path)

        # Dynamically generate the Delta Mask to ensure consistency with current sparsity constraints
        self.delta_masks = self.generate_delta_mask(model_path, unlearned_model_path)

        # Cache original weights to allow restoration between randomized trials
        self.original_params = {name: param.data.clone() for name, param in self.model.named_parameters()}

        # Setup English utility dataset to measure collateral damage
        self.clean_eng_path = "cleaned_valid_eng.jsonl"
        raw_ds = load_dataset("json", data_files=raw_eng_path, split="train")
        raw_ds.select_columns(["input_ids", "attention_mask", "text"]).to_json(self.clean_eng_path)

        # Initialize vendor evaluation logic
        base_vendor_path = "/Users/shirashko/PycharmProjects/Targeted-UNDO/src/vendor"
        cache_path = os.path.join(base_vendor_path, ".cache")

        from src.vendor.src.utils.validation_functions import get_arithmetic_eval_fn
        self._raw_eval_fn = get_arithmetic_eval_fn(
            model_name=model_path, batch_size=8, max_length=256, cache_dir=cache_path,
            dataset_cache_dir=cache_path, num_wiki_batches=50,
            eng_valid_file=self.clean_eng_path, accelerator=self.accelerator
        )

    def generate_delta_mask(self, original_path: str, unlearned_path: str) -> Dict[str, torch.Tensor]:
        """
        Calculates weight discrepancy between models and creates a binary mask.
        Parameters in the top-k percentile of absolute change are assigned 0 (erased/noised).
        """
        print(f"[*] Generating localized Delta Mask (Top {self.percentile * 100}% discrepancy)...")
        model_unl = AutoModelForCausalLM.from_pretrained(
            unlearned_path, torch_dtype=torch.float32, local_files_only=True
        ).to(self.device)

        fixed_masks = {}
        total_targeted = 0

        with torch.no_grad():
            for name, p_orig in self.model.named_parameters():
                if name not in self.allowed_params_names:
                    # Non-target layers remain unaffected (assigned 1)
                    fixed_masks[name] = torch.ones_like(p_orig.data)
                    continue

                p_unl = dict(model_unl.named_parameters())[name]
                diff = torch.abs(p_orig - p_unl)

                n_elements = diff.numel()
                k = int(n_elements * (1 - self.percentile))

                if k >= n_elements:
                    mask = torch.ones_like(diff)
                else:
                    threshold = torch.kthvalue(diff.flatten(), k).values
                    # 0 for high-discrepancy parameters, 1 for retained parameters
                    mask = (diff < threshold).float()

                fixed_masks[name] = mask
                total_targeted += (mask == 0).sum().item()

        print(f"[+] Delta Mask generated. Parameters Targeted: {total_targeted:,} "
              f"({(total_targeted / self.allowed_params_count) * 100:.2f}% of target space)")
        del model_unl
        return fixed_masks

    def _load_mask_file(self, path: str) -> Dict[str, torch.Tensor]:
        print(f"[*] Loading mask from {path}...")
        data = torch.load(path, weights_only=False, map_location='cpu')
        return data['masks'] if (isinstance(data, dict) and 'masks' in data) else data

    def _count_masked_params(self, mask_dict: Dict[str, torch.Tensor]) -> int:
        count = 0
        for name, mask in mask_dict.items():
            if name in self.allowed_params_names:
                count += (mask == 0).sum().item()
        return count

    def apply_mask(self, mask_dict: Dict[str, torch.Tensor]):
        """Zeroes out parameters indicated by the mask."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in mask_dict:
                    m = mask_dict[name].to(param.device)
                    param.data *= m.view(param.shape)

    def restore_original_params(self):
        """Resets model weights to their cached baseline state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = self.original_params[name].clone()

    def create_random_mask(self, num_masked_params: int) -> Dict[str, torch.Tensor]:
        """Generates a random binary mask with a matched number of zeroed parameters."""
        all_params = [(n, p.shape, p.numel()) for n, p in self.model.named_parameters()
                      if n in self.allowed_params_names]
        total_available = sum(p[2] for p in all_params)

        indices = np.random.choice(total_available, num_masked_params, replace=False)
        mask_indices = set(indices)

        random_masks = {}
        current_idx = 0
        for name, shape, numel in all_params:
            flat_mask = torch.ones(numel)
            local_hits = [i - current_idx for i in range(current_idx, current_idx + numel) if i in mask_indices]
            flat_mask[local_hits] = 0.0
            random_masks[name] = flat_mask.reshape(shape)
            current_idx += numel
        return random_masks

    def run_full_analysis(self, num_random_trials: int = 3):
        results = {}
        print("\n[*] Measuring Baseline Performance...")
        results["Baseline"] = self._raw_eval_fn(self.model, print_results=False)

        # Diagnostic suite for SNMF localization
        snmf_count = self._count_masked_params(self.snmf_masks)
        print(f"\n[Audit] SNMF Mask Sparsity: {snmf_count:,} parameters.")
        results["SNMF"] = self._run_mask_suite("SNMF", self.snmf_masks, snmf_count, num_random_trials)

        # Diagnostic suite for Delta localization
        delta_count = self._count_masked_params(self.delta_masks)
        print(f"\n[Audit] Delta Mask Sparsity: {delta_count:,} parameters.")
        results["Delta"] = self._run_mask_suite("Delta", self.delta_masks, delta_count, num_random_trials)

        self.plot_comparative_results(results, snmf_count, delta_count)
        with open("comparison_logs.json", "w") as f:
            json.dump(results, f, indent=4)

    def _run_mask_suite(self, label: str, mask: Dict, count: int, trials: int):
        print(f"\n--- Running Evaluation Suite: {label} ---")
        suite = {"targeted": None, "random_trials": []}

        # Evaluate performance drop using the structural mask
        self.restore_original_params()
        self.apply_mask(mask)
        suite["targeted"] = self._raw_eval_fn(self.model, print_results=False)

        # Evaluate performance drops using random erasure baselines
        for i in range(trials):
            print(f"Trial {i + 1}/{trials}: Computing random erasure baseline...")
            self.restore_original_params()
            self.apply_mask(self.create_random_mask(count))
            suite["random_trials"].append(self._raw_eval_fn(self.model, print_results=False))
        return suite

    def plot_comparative_results(self, results, snmf_n, delta_n):
        """Visualizes localization specificity vs random baseline."""

        # Define Metric Groups
        forget_metrics = [
            'val/multiplication_equation_acc', 'val/division_equation_acc',
            'val/multiplication_word_problem_acc', 'val/division_word_problem_acc'
        ]
        retain_metrics = [
            'val/addition_equation_acc', 'val/subtraction_equation_acc',
            'val/addition_word_problem_acc', 'val/subtraction_word_problem_acc'
        ]

        def get_agg_scores(suite):
            if not suite or not suite.get("targeted"):
                return (0, 0, 0), (0, 0, 0), (0, 0, 0)

            # Targeted scores
            f_target = np.mean([suite["targeted"].get(m, 0) for m in forget_metrics])
            r_target = np.mean([suite["targeted"].get(m, 0) for m in retain_metrics])
            l_target = suite["targeted"].get('val/eng_ce_loss', 0)

            # Random trials aggregation
            if not suite["random_trials"]:
                return (f_target, r_target, l_target), (0, 0, 0), (0, 0, 0)

            f_rand = [np.mean([t.get(m, 0) for m in forget_metrics]) for t in suite["random_trials"]]
            r_rand = [np.mean([t.get(m, 0) for m in retain_metrics]) for t in suite["random_trials"]]
            l_rand = [t.get('val/eng_ce_loss', 0) for t in suite["random_trials"]]

            return (f_target, r_target, l_target), \
                (np.mean(f_rand), np.mean(r_rand), np.mean(l_rand)), \
                (np.std(f_rand), np.std(r_rand), np.std(l_rand))

        # Extract all scores
        (snmf_f, snmf_r, snmf_l), (snmf_rf, snmf_rr, snmf_rl), (snmf_sf, snmf_sr, snmf_sl) = get_agg_scores(
            results["SNMF"])
        (delta_f, delta_r, delta_l), (delta_rf, delta_rr, delta_rl), (delta_sf, delta_sr, delta_sl) = get_agg_scores(
            results["Delta"])

        base_f = np.mean([results["Baseline"].get(m, 0) for m in forget_metrics])
        base_r = np.mean([results["Baseline"].get(m, 0) for m in retain_metrics])
        base_l = results["Baseline"].get('val/eng_ce_loss', 0)

        # Initialize Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Subplot 1: Accuracy (Targeted vs Random)
        labels = ['Forget Set\n(Goal: Low Acc)', 'Retain Set\n(Goal: High Acc)']
        x = np.arange(len(labels))
        width = 0.2

        # Plot bars
        ax1.bar(x - 1.5 * width, [base_f, base_r], width, label='Pretrained Baseline', color='gray', alpha=0.3)

        # SNMF Bars
        ax1.bar(x - 0.5 * width, [snmf_f, snmf_r], width, label=f'SNMF Targeted ({snmf_n:,} params)', color='orange',
                edgecolor='black')
        ax1.errorbar(x - 0.5 * width, [snmf_rf, snmf_rr], yerr=[snmf_sf, snmf_sr], fmt='o', color='darkorange',
                     label='SNMF Random Baseline', capsize=5)

        # Delta Bars
        ax1.bar(x + 0.5 * width, [delta_f, delta_r], width, label=f'Delta Targeted ({delta_n:,} params)', color='blue',
                edgecolor='black')
        ax1.errorbar(x + 0.5 * width, [delta_rf, delta_rr], yerr=[delta_sf, delta_sr], fmt='o', color='darkblue',
                     label='Delta Random Baseline', capsize=5)

        ax1.set_ylabel('Accuracy Score', fontweight='bold')
        ax1.set_title('Localization Precision: Targeted vs. Random Erasure', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontweight='bold')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)

        # Subplot 2: Collateral Damage (English CE Loss)
        methods = ['Baseline', 'SNMF Targeted', 'SNMF Random', 'Delta Targeted', 'Delta Random']
        losses = [base_l, snmf_l, snmf_rl, delta_l, delta_rl]
        stds = [0, 0, snmf_sl, 0, delta_sl]
        colors = ['gray', 'orange', 'wheat', 'blue', 'lightblue']

        bars = ax2.bar(methods, losses, yerr=stds, color=colors, capsize=7, edgecolor='black')
        ax2.set_ylabel('English CE Loss (Lower is Better)', fontweight='bold')
        ax2.set_title('Collateral Damage to General Language Utility', fontsize=14, fontweight='bold')
        ax2.set_ylim(min(losses) * 0.9, max(losses) * 1.1)

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'{height:.3f}', ha='center', va='bottom',
                     fontweight='bold')

        plt.xticks(rotation=15)
        plt.tight_layout()

        output_name = 'localization_diagnostic_results.png'
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"\n[+] Analysis complete. Visualization saved as '{output_name}'.")


def main():
    parser = argparse.ArgumentParser()
    # Maintaining your original default paths
    parser.add_argument('--model-path', type=str, default="models/gemma-2-0.1B_all_arithmetic+eng/final_model")
    parser.add_argument('--unlearned-path', type=str,
                        default="models/gemma-2-0.1B_all_arithmetic+eng_lr_7.0e-05/final_model")
    parser.add_argument('--snmf-path', type=str, default="div_mult_mask.pt")
    parser.add_argument('--eng-path', type=str, default="valid_eng.jsonl")
    parser.add_argument('--percentile', type=float, default=0.1)
    args = parser.parse_args()

    analyzer = MaskAnalyzer(args.model_path, args.snmf_path, args.unlearned_path, args.eng_path, args.percentile)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()