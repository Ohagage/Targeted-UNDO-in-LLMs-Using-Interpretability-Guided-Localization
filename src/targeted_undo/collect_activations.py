"""
Activation Collection Script

Collects activations from specified layers of an LLM and saves them as a dataset.
Supports different activation modes (mlp, residual, mlp_out) and organizes output
by layer and mode.

Usage:
    python -m targeted_undo.collect_activations \
        --model-name google/gemma-2-2b \
        --layers 0-31 \
        --mode mlp \
        --data-path data/wmdp_dataset.json \
        --save-path outputs/activations \
        --model-device cuda \
        --data-device cpu \
        --batch-size 4 \
        --seed 42
"""

import sys
import argparse
import random
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "snmf-mlp-decomposition"))

from transformer_lens import HookedTransformer, utils


# ------------------------------
# Logging Helper
# ------------------------------
def log(txt: str) -> None:
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {txt}", flush=True)


# ------------------------------
# Seed Setting
# ------------------------------
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log(f"Set random seed to {seed}")


# ------------------------------
# Argument Parsing Helpers
# ------------------------------
def parse_int_list(spec: str) -> List[int]:
    """
    Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints.
    """
    out = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if '-' in chunk:
            a, b = chunk.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return sorted(set(out))


def default_device() -> str:
    """Return best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------
# Dataset Loading
# ------------------------------
def load_dataset_from_json(data_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from JSON file.
    
    Expects JSON with a list of objects, each containing at least a 'prompt' or 'text' field.
    Optionally includes 'label' field.
    """
    log(f"Loading dataset from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # If it's a dict, look for common keys
        if 'data' in data:
            data = data['data']
        elif 'samples' in data:
            data = data['samples']
        else:
            # Convert dict to list if needed
            data = list(data.values()) if not isinstance(list(data.values())[0], dict) else list(data.values())
    
    log(f"Loaded {len(data)} samples")
    return data


def get_prompts_and_labels(data: List[Dict[str, Any]]) -> Tuple[List[str], List[Any]]:
    """Extract prompts and labels from dataset."""
    prompts = []
    labels = []
    
    for item in data:
        # Try different common field names for text
        if isinstance(item, str):
            prompts.append(item)
            labels.append(None)
        elif isinstance(item, dict):
            text = item.get('prompt') or item.get('text') or item.get('question') or item.get('input', '')
            label = item.get('label') or item.get('category') or item.get('concept')
            prompts.append(text)
            labels.append(label)
    
    return prompts, labels


# ------------------------------
# Activation Generator
# ------------------------------
class ActivationCollector:
    """
    Collects activations from specified layers of an LLM.
    """
    
    SUPPORTED_MODES = ['mlp', 'residual', 'mlp_out', 'attn', 'attn_out']
    
    def __init__(
        self,
        model_name: str,
        model_device: str = "cpu",
        data_device: str = "cpu",
        mode: str = "mlp"
    ):
        """
        Initialize the collector with a pretrained model.
        
        Args:
            model_name: Name of the pretrained model (HuggingFace format)
            model_device: Device to load the model onto
            data_device: Device to store collected activations
            mode: Which activation to collect ('mlp', 'residual', 'mlp_out', 'attn', 'attn_out')
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Mode '{mode}' not supported. Choose from: {self.SUPPORTED_MODES}")
        
        log(f"Loading model '{model_name}' on device '{model_device}'")
        self.model = HookedTransformer.from_pretrained(model_name, device=model_device)
        self.model_name = model_name
        self.model_device = model_device
        self.data_device = data_device
        self.mode = mode
        
        # Get model info
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        log(f"Model loaded: {self.n_layers} layers, d_model={self.d_model}")
    
    def _get_hook_string(self, layer: int) -> str:
        """Get the hook string for a given layer based on the mode."""
        if self.mode == 'mlp':
            return f"blocks.{layer}.mlp.hook_post"
        elif self.mode == 'mlp_out':
            return f"blocks.{layer}.hook_mlp_out"
        elif self.mode == 'residual':
            return utils.get_act_name("resid_post", layer)
        elif self.mode == 'attn':
            return f"blocks.{layer}.attn.hook_result"
        elif self.mode == 'attn_out':
            return f"blocks.{layer}.hook_attn_out"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _get_activation_dim(self, layer: int) -> int:
        """Get the dimension of activations for the given mode."""
        if self.mode in ['mlp', 'mlp_out']:
            # MLP activations typically have d_mlp dimension for post, d_model for out
            if self.mode == 'mlp':
                return self.model.cfg.d_mlp
            return self.d_model
        elif self.mode in ['residual', 'attn_out']:
            return self.d_model
        elif self.mode == 'attn':
            return self.model.cfg.d_head * self.model.cfg.n_heads
        return self.d_model
    
    def collect_activations(
        self,
        prompts: List[str],
        layers: List[int],
        batch_size: int = 4,
        max_length: Optional[int] = None,
        include_padding: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """
        Collect activations from the model for given prompts.
        
        Args:
            prompts: List of text prompts
            layers: List of layer indices to collect from
            batch_size: Batch size for processing
            max_length: Maximum sequence length (None = no truncation)
            include_padding: Whether to include padding token activations
        
        Returns:
            Dictionary mapping layer index to tensor of shape (num_tokens, d_activation)
        """
        # Validate layers
        for layer in layers:
            if layer < 0 or layer >= self.n_layers:
                raise ValueError(f"Layer {layer} out of range [0, {self.n_layers})")
        
        log(f"Collecting activations from layers {layers} using mode '{self.mode}'")
        log(f"Processing {len(prompts)} prompts with batch_size={batch_size}")
        
        # Initialize storage
        layer_activations = {layer: [] for layer in layers}
        
        # Get special token IDs
        pad_token_id = self.model.tokenizer.pad_token_id
        bos_token_id = self.model.tokenizer.bos_token_id
        
        # Process in batches
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Collecting activations"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompts))
                batch_prompts = prompts[start_idx:end_idx]
                
                # Tokenize
                tokens = self.model.to_tokens(batch_prompts, padding_side="left")
                if max_length is not None:
                    tokens = tokens[:, :max_length]
                
                tokens = tokens.to(self.model_device)
                
                # Run model with cache
                _, cache = self.model.run_with_cache(tokens)
                
                # Create mask for valid tokens
                if include_padding:
                    mask = torch.ones_like(tokens, dtype=torch.bool)
                else:
                    mask = (tokens != pad_token_id)
                    if bos_token_id is not None:
                        mask = mask & (tokens != bos_token_id)
                
                # Extract activations for each layer
                for layer in layers:
                    hook_str = self._get_hook_string(layer)
                    acts = cache[hook_str].detach()  # (batch, seq_len, d_activation)
                    
                    # Extract non-padding activations
                    nonpad_acts = acts[mask].view(-1, acts.size(-1))
                    layer_activations[layer].append(nonpad_acts.to(self.data_device))
                
                # Cleanup
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all activations
        log("Concatenating activations...")
        final_activations = {}
        for layer in layers:
            final_activations[layer] = torch.cat(layer_activations[layer], dim=0)
            log(f"  Layer {layer}: {final_activations[layer].shape}")
        
        self.model.reset_hooks()
        
        return final_activations


# ------------------------------
# Saving Functions
# ------------------------------
def save_activations(
    activations: Dict[int, torch.Tensor],
    save_path: Path,
    mode: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save activations organized by mode and layer.
    
    Output structure:
        save_path/
        ├── {mode}/
        │   ├── layer_{i}/
        │   │   └── activations.pt
        │   └── ...
        └── config.json
    """
    log(f"Saving activations to {save_path}")
    
    mode_dir = save_path / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activations per layer
    for layer, acts in activations.items():
        layer_dir = mode_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(acts, layer_dir / "activations.pt")
        log(f"  Saved layer {layer}: {acts.shape}")
    
    # Save config
    if config is not None:
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        log(f"  Saved config to {config_path}")


# ------------------------------
# Main Entry Point
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect and save activations from LLM layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect MLP activations from layers 0-31
  python -m targeted_undo.collect_activations --model-name google/gemma-2-2b --layers 0-31 --mode mlp

  # Collect residual stream from specific layers
  python -m targeted_undo.collect_activations --model-name meta-llama/Llama-3.1-8B --layers 10,15,20 --mode residual
        """
    )
    
    # Model configuration
    parser.add_argument("--model-name", type=str, required=True,
                        help="HuggingFace model name (e.g., 'google/gemma-2-2b')")
    parser.add_argument("--layers", type=str, required=True,
                        help="Layers to collect from. Format: '0-31' or '0,4,10-12'")
    parser.add_argument("--mode", type=str, default="mlp",
                        choices=['mlp', 'residual', 'mlp_out', 'attn', 'attn_out'],
                        help="Activation type to collect (default: mlp)")
    
    # Data configuration
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to dataset JSON file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--max-length", type=int, default=None,
                        help="Maximum sequence length (default: no truncation)")
    
    # Device configuration
    parser.add_argument("--model-device", type=str, default=default_device(),
                        help=f"Device for the model (default: {default_device()})")
    parser.add_argument("--data-device", type=str, default="cpu",
                        help="Device for storing activations (default: cpu)")
    
    # Output configuration
    parser.add_argument("--save-path", type=str, required=True,
                        help="Directory to save activations")
    
    # Processing configuration
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for processing (default: 4)")
    parser.add_argument("--include-padding", action="store_true",
                        help="Include padding token activations")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Parse layers
    layers = parse_int_list(args.layers)
    
    # Set seed
    set_seed(args.seed)
    
    # Resolve paths
    data_path = Path(args.data_path).resolve()
    save_path = Path(args.save_path).resolve()
    
    # Log configuration
    config = {
        'model_name': args.model_name,
        'layers': layers,
        'mode': args.mode,
        'data_path': str(data_path),
        'max_samples': args.max_samples,
        'max_length': args.max_length,
        'model_device': args.model_device,
        'data_device': args.data_device,
        'save_path': str(save_path),
        'batch_size': args.batch_size,
        'include_padding': args.include_padding,
        'seed': args.seed,
    }
    
    log("=" * 50)
    log("Activation Collection Configuration")
    log("=" * 50)
    for key, value in config.items():
        log(f"  {key}: {value}")
    log("=" * 50)
    
    # Load dataset
    data = load_dataset_from_json(str(data_path))
    prompts, _ = get_prompts_and_labels(data)
    
    # Limit samples if specified
    if args.max_samples is not None:
        prompts = prompts[:args.max_samples]
        log(f"Limited to {len(prompts)} samples")
    
    # Initialize collector
    collector = ActivationCollector(
        model_name=args.model_name,
        model_device=args.model_device,
        data_device=args.data_device,
        mode=args.mode,
    )
    
    # Collect activations
    activations = collector.collect_activations(
        prompts=prompts,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_padding=args.include_padding,
    )
    
    # Save results
    save_activations(
        activations=activations,
        save_path=save_path,
        mode=args.mode,
        config=config,
    )
    
    # Count total tokens
    total_tokens = sum(acts.shape[0] for acts in activations.values()) // len(layers)
    
    log("=" * 50)
    log("Activation collection complete!")
    log(f"Total tokens collected: {total_tokens}")
    log(f"Output saved to: {save_path}")
    log("=" * 50)


if __name__ == "__main__":
    main()

