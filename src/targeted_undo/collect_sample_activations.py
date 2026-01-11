"""
Sample-Level Activation Collection Script

Collects SAMPLE-LEVEL (mean-pooled) activations from LLM layers with labels.
This is designed for training linear probes.

Output format per layer:
    - activations: Tensor of shape (num_samples, d_activation)
    - labels: List of labels for each sample

Usage:
    python -m targeted_undo.collect_sample_activations \
        --model-name google/gemma-2-2b \
        --layers 0-31 \
        --mode mlp \
        --data-path data/wmdp_dataset.json \
        --save-path outputs/sample_activations \
        --aggregation mean
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


# ------------------------------
# Argument Parsing Helpers
# ------------------------------
def parse_int_list(spec: str) -> List[int]:
    """Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints."""
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
    """Load dataset from JSON file."""
    log(f"Loading dataset from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'data' in data:
            data = data['data']
        elif 'samples' in data:
            data = data['samples']
        else:
            data = list(data.values())
    
    log(f"Loaded {len(data)} samples")
    return data


def get_prompts_and_labels(data: List[Dict[str, Any]]) -> Tuple[List[str], List[Any]]:
    """Extract prompts and labels from dataset."""
    prompts = []
    labels = []
    
    for item in data:
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
# Sample Activation Collector
# ------------------------------
class SampleActivationCollector:
    """
    Collects SAMPLE-LEVEL activations from LLM layers.
    Aggregates token activations per sample using mean/last/first pooling.
    """
    
    SUPPORTED_MODES = ['mlp', 'residual', 'mlp_out', 'attn', 'attn_out']
    SUPPORTED_AGGREGATIONS = ['mean', 'last', 'first', 'max']
    
    def __init__(
        self,
        model_name: str,
        model_device: str = "cpu",
        data_device: str = "cpu",
        mode: str = "mlp"
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Mode '{mode}' not supported. Choose from: {self.SUPPORTED_MODES}")
        
        log(f"Loading model '{model_name}' on device '{model_device}'")
        self.model = HookedTransformer.from_pretrained(model_name, device=model_device)
        self.model_name = model_name
        self.model_device = model_device
        self.data_device = data_device
        self.mode = mode
        
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
    
    def _aggregate_activations(
        self, 
        acts: torch.Tensor, 
        mask: torch.Tensor, 
        aggregation: str
    ) -> torch.Tensor:
        """
        Aggregate token activations to sample-level.
        
        Args:
            acts: (batch, seq_len, d_activation)
            mask: (batch, seq_len) - True for valid tokens
            aggregation: 'mean', 'last', 'first', 'max'
        
        Returns:
            (batch, d_activation) - aggregated activations
        """
        batch_size = acts.shape[0]
        d_act = acts.shape[-1]
        result = torch.zeros(batch_size, d_act, device=acts.device, dtype=acts.dtype)
        
        for i in range(batch_size):
            valid_acts = acts[i][mask[i]]  # (num_valid_tokens, d_activation)
            
            if valid_acts.shape[0] == 0:
                continue
                
            if aggregation == 'mean':
                result[i] = valid_acts.mean(dim=0)
            elif aggregation == 'last':
                result[i] = valid_acts[-1]
            elif aggregation == 'first':
                result[i] = valid_acts[0]
            elif aggregation == 'max':
                result[i] = valid_acts.max(dim=0)[0]
        
        return result
    
    def collect_activations(
        self,
        prompts: List[str],
        layers: List[int],
        batch_size: int = 4,
        max_length: Optional[int] = None,
        aggregation: str = 'mean',
    ) -> Dict[int, torch.Tensor]:
        """
        Collect sample-level activations from the model.
        
        Args:
            prompts: List of text prompts
            layers: List of layer indices to collect from
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            aggregation: How to aggregate tokens ('mean', 'last', 'first', 'max')
        
        Returns:
            Dictionary mapping layer index to tensor of shape (num_samples, d_activation)
        """
        if aggregation not in self.SUPPORTED_AGGREGATIONS:
            raise ValueError(f"Aggregation '{aggregation}' not supported. Choose from: {self.SUPPORTED_AGGREGATIONS}")
        
        for layer in layers:
            if layer < 0 or layer >= self.n_layers:
                raise ValueError(f"Layer {layer} out of range [0, {self.n_layers})")
        
        log(f"Collecting sample-level activations from layers {layers}")
        log(f"Aggregation: {aggregation}, Processing {len(prompts)} samples")
        
        # Initialize storage
        layer_activations = {layer: [] for layer in layers}
        
        pad_token_id = self.model.tokenizer.pad_token_id
        bos_token_id = self.model.tokenizer.bos_token_id
        
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
                
                # Run model
                _, cache = self.model.run_with_cache(tokens)
                
                # Create mask for valid tokens
                mask = (tokens != pad_token_id)
                if bos_token_id is not None:
                    mask = mask & (tokens != bos_token_id)
                
                # Extract and aggregate activations for each layer
                for layer in layers:
                    hook_str = self._get_hook_string(layer)
                    acts = cache[hook_str].detach()  # (batch, seq_len, d_activation)
                    
                    # Aggregate to sample level
                    sample_acts = self._aggregate_activations(acts, mask, aggregation)
                    layer_activations[layer].append(sample_acts.to(self.data_device))
                
                del cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all samples
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
def save_sample_activations(
    activations: Dict[int, torch.Tensor],
    labels: List[Any],
    save_path: Path,
    mode: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save sample-level activations with labels.
    
    Output structure:
        save_path/
        ├── {mode}/
        │   ├── layer_{i}.pt  (contains {'activations': tensor, 'labels': list})
        │   └── ...
        └── config.json
    """
    log(f"Saving sample activations to {save_path}")
    
    mode_dir = save_path / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    # Save activations + labels per layer
    for layer, acts in activations.items():
        layer_data = {
            'activations': acts,
            'labels': labels,
        }
        layer_path = mode_dir / f"layer_{layer}.pt"
        torch.save(layer_data, layer_path)
        log(f"  Saved layer {layer}: {acts.shape}")
    
    # Save config
    if config is not None:
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)


# ------------------------------
# Main Entry Point
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect sample-level activations for linear probe training.",
    )
    
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--layers", type=str, required=True)
    parser.add_argument("--mode", type=str, default="mlp",
                        choices=['mlp', 'residual', 'mlp_out', 'attn', 'attn_out'])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=['mean', 'last', 'first', 'max'],
                        help="How to aggregate token activations (default: mean)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model-device", type=str, default=default_device())
    parser.add_argument("--data-device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    layers = parse_int_list(args.layers)
    set_seed(args.seed)
    
    data_path = Path(args.data_path).resolve()
    save_path = Path(args.save_path).resolve()
    
    config = {
        'model_name': args.model_name,
        'layers': layers,
        'mode': args.mode,
        'aggregation': args.aggregation,
        'data_path': str(data_path),
        'max_samples': args.max_samples,
        'max_length': args.max_length,
        'seed': args.seed,
    }
    
    log("=" * 50)
    log("Sample Activation Collection")
    log("=" * 50)
    for key, value in config.items():
        log(f"  {key}: {value}")
    log("=" * 50)
    
    # Load dataset
    data = load_dataset_from_json(str(data_path))
    prompts, labels = get_prompts_and_labels(data)
    
    if args.max_samples is not None:
        prompts = prompts[:args.max_samples]
        labels = labels[:args.max_samples]
        log(f"Limited to {len(prompts)} samples")
    
    # Collect activations
    collector = SampleActivationCollector(
        model_name=args.model_name,
        model_device=args.model_device,
        data_device=args.data_device,
        mode=args.mode,
    )
    
    activations = collector.collect_activations(
        prompts=prompts,
        layers=layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        aggregation=args.aggregation,
    )
    
    # Save
    save_sample_activations(
        activations=activations,
        labels=labels,
        save_path=save_path,
        mode=args.mode,
        config=config,
    )
    
    log("=" * 50)
    log(f"Done! Saved {len(prompts)} samples to {save_path}")
    log("=" * 50)


if __name__ == "__main__":
    main()

