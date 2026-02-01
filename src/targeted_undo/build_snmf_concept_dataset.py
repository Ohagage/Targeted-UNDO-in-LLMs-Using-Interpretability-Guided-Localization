"""
SNMF Concept Dataset Builder

Creates a labeled concept dataset from arithmetic and English training data
for use with Semi-Nonnegative Matrix Factorization (SNMF) analysis.

The output format is compatible with the snmf-mlp-decomposition project's
SupervisedConceptDataset class.

Usage:
    python -m targeted_undo.build_snmf_concept_dataset \
        --data-dir /path/to/jsonl/files \
        --output-path data/snmf_concepts.json \
        --samples-per-concept 300

Output Format:
    {
        "addition_symbolic": ["18 + 13 = 31", ...],
        "addition_riddle": ["There are 5 apples...", ...],
        "subtraction_symbolic": ["47 - 12 = 35", ...],
        ...
        "english": ["The quick brown fox...", ...]
    }
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


# ------------------------------
# Logging Helper
# ------------------------------
def log(txt: str) -> None:
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {txt}", flush=True)


# ------------------------------
# Classification Functions
# ------------------------------
def detect_operation(text: str) -> Optional[str]:
    """
    Detect the arithmetic operation in a text.
    Returns: 'addition', 'subtraction', 'multiplication', 'division', or None
    """
    # Check for symbolic operators first
    if re.search(r'\d+\s*\+\s*\d+', text):
        return 'addition'
    if re.search(r'\d+\s*-\s*\d+', text):
        return 'subtraction'
    if re.search(r'\d+\s*\*\s*\d+', text):
        return 'multiplication'
    if re.search(r'\d+\s*/\s*\d+', text):
        return 'division'
    
    # Arithmetic riddles are SHORT (1-3 sentences). Long text is likely English prose.
    # Count sentences roughly by counting periods
    if len(text) > 500 or text.count('.') > 5:
        return None  # Too long to be an arithmetic riddle
    
    # Must contain at least one number to be an arithmetic problem
    if not re.search(r'\d+', text):
        return None
    
    # Check for word-based indicators (for riddles)
    text_lower = text.lower()
    
    # DIVISION indicators (check FIRST - most specific patterns)
    # Division: splitting X items among Y groups/people, result is "each gets Z"
    division_patterns = [
        'divide', 'divides', 'divided',
        'split', 'splits', 
        'share equally', 'shared equally', 'shares equally',
        'distribute', 'distributes', 'distributed',
        'packed equally', 'pack equally', 'to be packed',
        'equally into', 'equally among', 'equally between',
        'evenly between', 'evenly among', 'evenly into',
        'equal piles', 'equal parts',
        'into equal', 'among equal', 'sort into',
        'each box gets', 'each shelf gets', 'each shelf holds',
        'each friend receives', 'each player gets', 'each gets',
        'each tray has', 'each school gets', 'each group gets',
        'each pile has', 'each classmate gets', 'each sibling receives',
        'each pack receives', 'each group has',
        'to distribute among', 'shares', 'each student gets',
        'each row has', 'each section has',
    ]
    if any(pattern in text_lower for pattern in division_patterns):
        return 'division'
    
    # MULTIPLICATION indicators (check BEFORE subtraction/addition)
    # Pattern: "X items in each Y" or "each Y has X" computing total
    multiplication_patterns = [
        'times', 'multiply', 'multiplied',
        'groups of', 'sets of', 'rows of', 'columns of',
        'repeats a pattern', 'repeat a pattern',
        'equal groups of',
        'in each row', 'in each pile', 'in each box', 'in each bag',
        'in each container', 'in each jar', 'in each bundle',
        'on each plate', 'per plate', 'per shelf',
        'if there are', 'in total there are',
        'altogether', 'in total',
    ]
    # Key: multiplication talks about "each X has Y" computing TOTAL
    # Division talks about "each X gets Y" from splitting
    has_mult_phrase = any(p in text_lower for p in multiplication_patterns)
    # Distinguish: multiplication says "there are X in total" (computing up)
    # NOT "each X gets" (that's division splitting down)
    if has_mult_phrase and 'each' in text_lower and 'gets' not in text_lower and 'receives' not in text_lower:
        return 'multiplication'
    if has_mult_phrase and ('in total' in text_lower or 'altogether' in text_lower):
        return 'multiplication'
    
    # ADDITION indicators - check for "adds more", "more X to it"
    # These are STRONG addition signals - check before subtraction
    addition_strong = [
        'adds', 'add more', 'more to it',
        'gives them', 'gives her', 'gives him',
        'brings', 'contributes', 'puts in',
        'places more', 'find more', 'finds more',
        'receive more', 'receives more', 'earn more', 'earns more',
        'collect more', 'collects more',
    ]
    if any(word in text_lower for word in addition_strong):
        return 'addition'
    
    # SUBTRACTION indicators - removal/decrease
    subtraction_words = [
        'fall off', 'falls off',
        'lose', 'loses', 'lost',
        'gave away', 'give away', 'gives away',
        'taken out', 'takes out', 'take out',
        'removed', 'removes', 'remove',
        'are sold', 'is sold', 'get broken', 'gets broken',
        'are eaten', 'is eaten',
        'drop', 'drops', 'dropped',
        'use', 'uses', 'used',
        'spend', 'spends', 'spent',
        'break', 'breaks', 'broken',
        'are left', 'is left', 'left over',
        'remain', 'remains', 'remaining',
    ]
    if any(word in text_lower for word in subtraction_words):
        return 'subtraction'
    
    # Weak addition patterns (only if nothing else matched)
    addition_weak = ['more', 'together', 'now has', 'now have', 'now they have']
    if any(word in text_lower for word in addition_weak):
        return 'addition'
    
    return None


def is_symbolic(text: str) -> bool:
    """
    Check if the text is primarily a symbolic equation (e.g., "18 + 13 = 31")
    vs a word problem/riddle.
    """
    # Count words vs numbers/operators
    words = re.findall(r'[a-zA-Z]+', text)
    
    # If very few words (just the equation), it's symbolic
    if len(words) <= 2:
        return True
    
    # If it starts with a number and has operators, likely symbolic
    if re.match(r'^\d+\s*[\+\-\*\/]', text.strip()):
        return True
    
    return False


def classify_sample(text: str, source_file: str = "") -> Tuple[str, str]:
    """
    Classify a sample into (operation, format) tuple.
    
    Returns:
        (operation, format) where:
        - operation: 'addition', 'subtraction', 'multiplication', 'division', 'english'
        - format: 'symbolic', 'riddle', 'prose'
    """
    # Try to detect arithmetic operation
    operation = detect_operation(text)
    
    if operation is None:
        # No arithmetic detected - it's English prose
        return ('english', 'prose')
    
    # Determine if symbolic or riddle
    fmt = 'symbolic' if is_symbolic(text) else 'riddle'
    
    return (operation, fmt)


# ------------------------------
# Data Loading
# ------------------------------
def load_jsonl_samples(filepath: Path, max_samples: int = 10000) -> List[str]:
    """Load text samples from a JSONL file."""
    samples = []
    
    log(f"Loading from {filepath.name}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                record = json.loads(line)
                text = record.get('text', '')
                if text and len(text.strip()) > 5:  # Skip empty/tiny samples
                    samples.append(text.strip())
            except json.JSONDecodeError:
                continue
    
    log(f"  Loaded {len(samples)} samples")
    return samples


# ------------------------------
# Dataset Building
# ------------------------------
def build_concept_dataset(
    data_dir: Path,
    samples_per_concept: int = 300,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Build the concept dataset from available JSONL files.
    
    Args:
        data_dir: Directory containing JSONL files
        samples_per_concept: Target number of samples per concept
        seed: Random seed for sampling
        
    Returns:
        Dictionary mapping concept labels to lists of text samples
    """
    random.seed(seed)
    
    # Initialize concept buckets
    concepts: Dict[str, List[str]] = defaultdict(list)
    
    # Define source files and their expected content
    source_files = {
        'addition_subtraction.jsonl': ['addition', 'subtraction'],
        'multiplication_division.jsonl': ['multiplication', 'division'],
        'all_arithmetic.jsonl': ['addition', 'subtraction', 'multiplication', 'division'],
        'train_eng.jsonl': ['english'],
    }
    
    # Process each source file
    for filename, expected_ops in source_files.items():
        filepath = data_dir / filename
        
        if not filepath.exists():
            log(f"  Skipping {filename} (not found)")
            continue
        
        # Load more samples than needed to allow for filtering
        # Use higher multiplier for arithmetic files to ensure enough riddles
        multiplier = 50 if 'arithmetic' in filename or 'division' in filename else 20
        samples = load_jsonl_samples(filepath, max_samples=samples_per_concept * multiplier)
        
        # Handle English file specially - ALL samples are English
        if 'eng' in filename:
            for text in samples:
                concepts['english'].append(text)
            continue
        
        # For arithmetic files, use text-based classification
        # but ONLY accept operations that match the source file
        for text in samples:
            operation, fmt = classify_sample(text, filename)
            
            if operation is None or operation == 'english':
                # Skip samples that don't classify as arithmetic
                continue
            
            # Only accept if operation matches expected ops from source file
            if operation not in expected_ops:
                continue
                
            # Create concept key like "addition_symbolic" or "multiplication_riddle"
            concept_key = f"{operation}_{fmt}"
            concepts[concept_key].append(text)
    
    # Balance and sample from each concept (remove duplicates first)
    balanced_concepts: Dict[str, List[str]] = {}
    
    for concept, texts in concepts.items():
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)
        texts = unique_texts
        
        if len(texts) == 0:
            log(f"  WARNING: No samples for concept '{concept}'")
            continue
        
        # Shuffle and take up to samples_per_concept
        random.shuffle(texts)
        sampled = texts[:samples_per_concept]
        balanced_concepts[concept] = sampled
        log(f"  {concept}: {len(sampled)} samples")
    
    return balanced_concepts


# ------------------------------
# Main Entry Point
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build SNMF concept dataset from training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory containing JSONL data files"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Output path for the concept dataset JSON"
    )
    parser.add_argument(
        "--samples-per-concept", type=int, default=300,
        help="Target number of samples per concept (default: 300)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    output_path = Path(args.output_path).resolve()
    
    log("=" * 60)
    log("SNMF Concept Dataset Builder")
    log("=" * 60)
    log(f"  Data directory: {data_dir}")
    log(f"  Output path: {output_path}")
    log(f"  Samples per concept: {args.samples_per_concept}")
    log(f"  Seed: {args.seed}")
    log("=" * 60)
    
    # Check data directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Build the dataset
    concepts = build_concept_dataset(
        data_dir=data_dir,
        samples_per_concept=args.samples_per_concept,
        seed=args.seed,
    )
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)
    
    # Summary
    log("=" * 60)
    log("Dataset Summary")
    log("=" * 60)
    total_samples = 0
    for concept, samples in sorted(concepts.items()):
        log(f"  {concept}: {len(samples)} samples")
        total_samples += len(samples)
    log(f"  TOTAL: {total_samples} samples across {len(concepts)} concepts")
    log(f"\nSaved to: {output_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
