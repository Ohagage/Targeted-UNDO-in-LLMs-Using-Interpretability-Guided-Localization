import torch
from transformers import AutoModelForCausalLM
import os
import sys
import json


class MaskFactory:
    """
    Generates a range of localization masks based on weight discrepancy
    between an original and an unlearned model.
    """

    def __init__(self, orig_path, unl_path, device="cpu"):
        print(f"Loading models for discrepancy analysis on {device}...")
        self.device = device
        # Using bfloat16 to match typical training precision for Gemma-2
        self.m_orig = AutoModelForCausalLM.from_pretrained(
            orig_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(device)
        self.m_unl = AutoModelForCausalLM.from_pretrained(
            unl_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(device)

        self.diffs = self._calculate_weight_diffs()

    def _calculate_weight_diffs(self):
        """Pre-calculates absolute weight differences for all 2D layers."""
        diffs = {}
        with torch.no_grad():
            for (name, p_orig), (_, p_unl) in zip(self.m_orig.named_parameters(), self.m_unl.named_parameters()):
                # Only target weight matrices (Attention and MLP projections)
                # Biases and Norm layers are typically excluded to maintain stability
                if len(p_orig.shape) == 2:
                    diffs[name] = torch.abs(p_orig - p_unl).cpu()
        return diffs

    def generate_mask(self, percentile, top_alpha, bottom_alpha, label):
        """
        Creates a hybrid mask configuration.
        percentile: Fraction of weights (0.0 to 1.0) receiving top_alpha.
        top_alpha: Noise intensity for the most changed parameters.
        bottom_alpha: Noise intensity for the background/remaining parameters.
        """
        mask_dict = {}
        for name, diff in self.diffs.items():
            flattened_diff = diff.view(-1).float()
            # Calculate threshold for the top percentile
            k = int(flattened_diff.numel() * (1 - percentile))

            # Use kthvalue for efficient thresholding per layer
            threshold = torch.kthvalue(flattened_diff, max(1, k)).values

            # Create the hybrid mapping: top values get top_alpha, rest get bottom_alpha
            binary_mask = (diff >= threshold)
            hybrid_mask = torch.where(
                binary_mask,
                torch.tensor(top_alpha, dtype=torch.bfloat16),
                torch.tensor(bottom_alpha, dtype=torch.bfloat16)
            )
            mask_dict[name] = hybrid_mask

        # Package the mask with metadata for experimental tracking
        return {
            "name": f"mask_p{percentile}t{top_alpha}_b{bottom_alpha}{label}",
            "config": {
                "percentile": percentile,
                "top_alpha": top_alpha,
                "bottom_alpha": bottom_alpha,
                "label": label
            },
            "weights": mask_dict
        }


def save_mask_package(mask_package, base_dir):
    """Saves the mask dictionary and a config JSON for reproducibility."""
    folder = os.path.join(base_dir, mask_package["name"])
    os.makedirs(folder, exist_ok=True)

    # Save weights as a .pt file for the UNDO training script
    torch.save(mask_package["weights"], os.path.join(folder, "mask.pt"))

    # Save metadata for later analysis/paper writing
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump(mask_package["config"], f, indent=4)

    print(f"Successfully generated and saved: {mask_package['name']}")


if __name__ == "__main__":
    # Environment Paths
    BASE_PATH = "/home/ADV_2526a/rashkovits/distillation-robustify-unlearning/models/non-wmdp"
    ORIG = f"{BASE_PATH}/pretrained_models/gemma-2-0.1B_all_arithmetic+eng/final_model"
    UNL = f"{BASE_PATH}/unlearned_models/MaxEnt/gemma-2-0.1B_all_arithmetic+eng_lr_7.0e-05/final_model"

    OUT_DIR = "/home/ADV_2526a/rashkovits/distillation-robustify-unlearning/outputs/masks"

    factory = MaskFactory(ORIG, UNL)

    # Format: (percentile, top_alpha, bottom_alpha, informative_label)
    experiments = [
        # 1. Focuses purely on the "Unlearning Trace".
        # Goal: Test if minimal intervention on behavioral weights is sufficient.
        (0.05, 1.0, 0.0, "Trace_Minimalist"),

        # 2. Targeted Hybrid (Balanced): Hypothesized Sweet Spot.
        # Goal: Break the primary circuit while lightly disrupting the surrounding latent space.
        (0.10, 1.0, 0.1, "Trace_Hybrid_Surgical"),

        # 3. Targeted Hybrid (Aggressive):
        # Goal: Ensure robust erasure by perturbing a wider neighborhood of the unlearning delta.
        (0.30, 1.0, 0.2, "Trace_Hybrid_Aggressive"),

        # 4. Feature-Level Distribution:
        # Goal: Test the impact of widely distributed noise on knowledge recovery.
        (0.50, 0.8, 0.3, "Distributed_Stochastic_Erasure"),

        # 5. Near-Global Baseline (Control Group):
        # Goal: Represent the standard UNDO framework for collateral damage comparison.
        (0.90, 0.5, 0.5, "Global_Parity_Baseline")
    ]

    for p, ta, ba, label in experiments:
        package = factory.generate_mask(p, ta, ba, label)
        save_mask_package(package, OUT_DIR)
