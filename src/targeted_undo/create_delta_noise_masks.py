import torch
from transformers import AutoModelForCausalLM
import os


def create_masks(original_path, unlearned_path, save_dir, percentile=0.1):
    """
    Generates two types of masks based on weight differences:
    1. Binary Mask: High-change parameters get 1, others 0.
    2. Relative Mask: Scaled differences, where magnitude indicates unlearning intensity.
    """
    print(f"Loading models to CPU...")
    model_orig = AutoModelForCausalLM.from_pretrained(original_path, torch_dtype=torch.bfloat16, device_map="cpu")
    model_unl = AutoModelForCausalLM.from_pretrained(unlearned_path, torch_dtype=torch.bfloat16, device_map="cpu")

    binary_masks = {}
    relative_masks = {}

    print(f"Processing parameters...")
    with torch.no_grad():
        for (name, p_orig), (_, p_unl) in zip(model_orig.named_parameters(), model_unl.named_parameters()):
            # Calculate absolute difference (delta)
            diff = torch.abs(p_orig - p_unl)

            if diff.numel() == 0:
                continue

            # 1. Create Relative Mask (Scaled between 0 and 1)
            # This indicates how much each parameter changed relative to the max change in this layer
            max_diff = torch.max(diff)
            if max_diff > 0:
                relative_masks[name] = (diff / max_diff).to(torch.bfloat16)
            else:
                relative_masks[name] = torch.zeros_like(diff)

            # 2. Create Binary Mask (Top K percentile)
            # Identifying the threshold for the most significant changes
            threshold = torch.quantile(diff.float(), 1 - percentile)
            binary_masks[name] = (diff >= threshold).to(torch.bfloat16)

    # Save both masks to the outputs directory in the server
    os.makedirs(save_dir, exist_ok=True)
    torch.save(binary_masks, os.path.join(save_dir, "binary_delta_mask.pt"))
    torch.save(relative_masks, os.path.join(save_dir, "relative_delta_mask.pt"))

    print(f"Successfully saved masks to: {save_dir}")


if __name__ == "__main__":
    # Paths provided for your specific setup on the server
    BASE_DIR = "/home/ADV_2526a/hagage/distillation-robustify-unlearning-copy/models/non-wmdp"
    ORIG_PATH = f"{BASE_DIR}/pretrained_models/gemma-2-0.1B_all_arithmetic+eng/final_model"
    UNL_PATH = f"{BASE_DIR}/unlearned_models/MaxEnt/gemma-2-0.1B_all_arithmetic+eng_lr_4.0e-04/final_model"

    # Saving to the outputs/masks directory
    OUTPUT_SAVE_DIR = "/home/ADV_2526a/rashkovits/distillation-robustify-unlearning-copy/outputs/masks"

    create_masks(ORIG_PATH, UNL_PATH, OUTPUT_SAVE_DIR, percentile=0.1)