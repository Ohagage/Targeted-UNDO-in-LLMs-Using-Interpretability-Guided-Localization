"""
Linear Probe Training Script

Trains linear probes on sample-level activations to evaluate how well
concepts/labels are linearly separable in the activation space.

Usage:
    python -m targeted_undo.train_linear_probe \
        --activations-path outputs/sample_activations \
        --mode mlp \
        --layers 0-31 \
        --output-path outputs/probe_results.json
"""

import argparse
import copy
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


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


# ------------------------------
# Argument Parsing Helpers
# ------------------------------
def parse_int_list(spec: str) -> List[int]:
    """Parse '0,1,2' or '0-3' or '0,2,5-7' into a list of ints."""
    out: List[int] = []
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
# Linear Probe Model
# ------------------------------
class LinearProbe(nn.Module):
    """
    Linear classifier probe for classification tasks.
    
    Encapsulates the model, optimizer, scheduler, and training logic
    similar to the course solution structure.
    
    Args:
        input_dim: Dimension of input features (activation size)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization weight
        max_epochs: Maximum training epochs
        early_stopping_patience: Epochs to wait before early stopping
        device: Device to run on ('cuda', 'mps', 'cpu')
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        learning_rate: float = 0.01,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        
        # Model layer
        self.linear = nn.Linear(input_dim, num_classes)
        self._initialize_weights()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (cosine annealing like course solution)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max_epochs
        )
        
        # Best model tracking
        self.best_weights = copy.deepcopy(self.linear.state_dict())
        self.best_val_acc = 0.0
        self.final_val_df = None  # Store final validation results
        
        # Move to device
        self.to(device)
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.linear(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for input.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
    
    def validate(
        self, 
        X_val: torch.Tensor, 
        y_val: torch.Tensor
    ) -> Tuple[pd.DataFrame, float, float, float]:
        """
        Validate the model on a validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            result_df: DataFrame with predictions and targets
            loss: Validation loss
            accuracy: Validation accuracy
            f1: Validation F1 score (macro)
        """
        self.eval()
        with torch.no_grad():
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            
            logits = self.forward(X_val)
            loss = self.criterion(logits, y_val).item()
            
            preds = logits.argmax(dim=1)
            accuracy = (preds == y_val).float().mean().item()
            
            # Compute F1
            preds_np = preds.cpu().numpy()
            targets_np = y_val.cpu().numpy()
            f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
            
            # Create result DataFrame
            result_df = pd.DataFrame({
                "preds": preds_np,
                "target": targets_np
            })
            
        return result_df, loss, accuracy, f1
    
    def set_to_best_weights(self) -> None:
        """Restore model to best weights found during training."""
        self.linear.load_state_dict(self.best_weights)
    
    def fit(
        self,
        train_loader: DataLoader,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the probe on the given data.
        
        Args:
            train_loader: DataLoader for training data
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print progress
            
        Returns:
            training_history: Dict containing loss and metric history
        """
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
        }
        
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.forward(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Update learning rate
            self.scheduler.step()
            
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            result_df, val_loss, val_acc, val_f1 = self.validate(X_val, y_val)
            
            # Record history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Log epoch progress
            if verbose:
                improved = ""
                if val_acc > self.best_val_acc:
                    improved = " *"  # Mark improvement
                log(f"    Epoch {epoch + 1:3d}/{self.max_epochs}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_acc={val_acc:.4f}, "
                    f"val_f1={val_f1:.4f}{improved}")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_weights = copy.deepcopy(self.linear.state_dict())
                self.final_val_df = result_df
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                if verbose:
                    log(f"    Early stopping at epoch {epoch + 1} (no improvement for {self.early_stopping_patience} epochs)")
                break
        
        # Restore best weights
        self.set_to_best_weights()
        
        return history
    
    def get_metrics(
        self, 
        X_test: torch.Tensor, 
        y_test: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute final metrics on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        result_df, loss, accuracy, f1_macro = self.validate(X_test, y_test)
        
        # Additional metrics
        y_pred = result_df['preds'].values
        y_true = result_df['target'].values
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'loss': loss,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
        }


# ------------------------------
# Data Loading
# ------------------------------
def load_layer_data(
    activations_path: Path, 
    mode: str, 
    layer: int
) -> Tuple[torch.Tensor, List[Any]]:
    """Load activations and labels for a specific layer."""
    layer_path = activations_path / mode / f"layer_{layer}.pt"
    
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_path}")
    
    data = torch.load(layer_path, weights_only=False)
    activations = data['activations']
    labels = data['labels']
    
    return activations, labels


def prepare_data(
    activations: torch.Tensor,
    labels: List[Any],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LabelEncoder]:
    """
    Prepare data for training.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder
    """
    # Filter out None labels
    valid_indices = [i for i, label in enumerate(labels) if label is not None]
    
    if len(valid_indices) < len(labels):
        log(f"  Filtered {len(labels) - len(valid_indices)} samples with None labels")
    
    activations = activations[valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    y = torch.tensor(y, dtype=torch.long)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        activations.numpy(), y.numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=y.numpy()
    )
    
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
        label_encoder
    )


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes on activation datasets.",
    )
    
    parser.add_argument("--activations-path", type=str, required=True,
                        help="Path to sample activations directory")
    parser.add_argument("--mode", type=str, default="mlp",
                        choices=['mlp', 'residual', 'mlp_out', 'attn', 'attn_out'])
    parser.add_argument("--layers", type=str, required=True,
                        help="Layers to train probes on (e.g., '0-31' or '10,15,20')")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to save results JSON")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true",
                        help="Save trained probe models")
    
    args = parser.parse_args()
    
    layers = parse_int_list(args.layers)
    set_seed(args.seed)
    
    activations_path = Path(args.activations_path).resolve()
    output_path = Path(args.output_path).resolve()
    
    log("=" * 60)
    log("Linear Probe Training")
    log("=" * 60)
    log(f"  Activations path: {activations_path}")
    log(f"  Mode: {args.mode}")
    log(f"  Layers: {layers}")
    log(f"  Device: {args.device}")
    log("=" * 60)
    
    # Results storage
    results = {
        'config': {
            'activations_path': str(activations_path),
            'mode': args.mode,
            'layers': layers,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'test_size': args.test_size,
            'seed': args.seed,
        },
        'layer_results': {}
    }
    
    # Train probe for each layer
    for layer in tqdm(layers, desc="Training probes"):
        log(f"\n--- Layer {layer} ---")
        
        try:
            # Load data
            activations, labels = load_layer_data(activations_path, args.mode, layer)
            log(f"  Loaded {len(labels)} samples, activation dim: {activations.shape[1]}")
            
            # Prepare data
            X_train, X_test, y_train, y_test, label_encoder = prepare_data(
                activations, labels,
                test_size=args.test_size,
                seed=args.seed,
            )
            
            num_classes = len(label_encoder.classes_)
            log(f"  Classes: {list(label_encoder.classes_)}")
            log(f"  Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Create DataLoader
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True
            )
            
            # Create and train probe
            probe = LinearProbe(
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_epochs=args.epochs,
                early_stopping_patience=args.early_stopping_patience,
                device=args.device,
            )
            
            # Train
            probe.fit(train_loader, X_test, y_test, verbose=True)
            
            # Get final metrics
            metrics = probe.get_metrics(X_test, y_test)
            metrics['num_train_samples'] = len(X_train)
            metrics['num_test_samples'] = len(X_test)
            
            log(f"  Accuracy: {metrics['accuracy']:.4f}")
            log(f"  F1 (macro): {metrics['f1_macro']:.4f}")
            
            # Store results
            results['layer_results'][layer] = {
                'metrics': metrics,
                'classes': list(label_encoder.classes_),
            }
            
            # Save model if requested
            if args.save_models:
                model_dir = output_path.parent / "probe_models" / args.mode
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(probe.state_dict(), model_dir / f"probe_layer_{layer}.pt")
                
        except Exception as e:
            log(f"  ERROR: {e}")
            results['layer_results'][layer] = {'error': str(e)}
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {output_path}")
    
    # Print summary
    log("\n" + "=" * 60)
    log("Summary (Accuracy per layer)")
    log("=" * 60)
    for layer in layers:
        if layer in results['layer_results'] and 'metrics' in results['layer_results'][layer]:
            acc = results['layer_results'][layer]['metrics']['accuracy']
            log(f"  Layer {layer:2d}: {acc:.4f}")
        else:
            log(f"  Layer {layer:2d}: ERROR")


if __name__ == "__main__":
    main()
