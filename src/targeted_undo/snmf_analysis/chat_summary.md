# SNMF Neuron-Concept Importance Analysis: Complete Summary

This document summarizes the development and implementation of the SNMF (Semi-Nonnegative Matrix Factorization) neuron-concept importance visualization tool, including all technical decisions, mathematical foundations, and bug fixes.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Background: SNMF Pipeline](#background-snmf-pipeline)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Two Methods for Computing Importance](#two-methods-for-computing-importance)
5. [Forget Set vs Retain Set Division](#forget-set-vs-retain-set-division)
6. [Normalization Strategies](#normalization-strategies)
7. [Implementation Details](#implementation-details)
8. [Important Limitations and Scope](#important-limitations-and-scope)
9. [Bug Fixes and Corrections](#bug-fixes-and-corrections)
10. [Output Files](#output-files)
11. [Key Takeaways](#key-takeaways)
12. [Technical Discussions](#technical-discussions)

---

## Project Goal

### Original User Statement

> "My final goal is to present some data about the information found in the SNMF for each neuron, how much it effects each concept/feature. To find metrics to accumulate the importance of each neuron to each input category, and make a mask that is not binary but contains this numeric number showing how much this neuron is connected to the forget set. But first I want to understand better the information collected by the SNMF and how it can be presented (how many samples from each concept activate it, or how much it was activated)."

### Summary

The goal is to quantify and visualize how important each MLP neuron is to each concept in the dataset, using SNMF decomposition. This analysis supports **targeted unlearning** - the ability to selectively remove specific knowledge (e.g., division and multiplication) from a language model while preserving other knowledge (e.g., addition, subtraction, English).

The analysis enables creation of **non-binary masks** for targeted unlearning by identifying which neurons are most important for "forget" concepts vs "retain" concepts.

### Visualization Inspiration

The visualization style was inspired by `src/targeted_undo/masks_plots/mask_heatmap_functional.png`, which shows a heatmap of neurons × layers with color-coded importance values.

---

## Background: SNMF Pipeline

### Data Flow Overview

```
1. Load prompts from arithmetic.json (with concept labels)
2. Tokenize prompts and run through model
3. Extract MLP activations (down_proj output) at each layer
4. Apply SNMF decomposition: A ≈ F @ G.T
5. Save F, G, token_ids, sample_ids per layer
6. [This tool] Compute importance from F, G, and concept labels
```

### run_snmf.py Process

The `run_snmf.py` script in `src/targeted_undo/` collects activations:

1. **Batch Processing**: For each batch of prompts:
   - Tokenize with left padding
   - Run forward pass through model
   - Hook into MLP layers to capture activations
   - Extract activations only for real tokens (non-padding)
   - Track `token_ids` and `sample_ids` for each token

2. **SNMF Factorization**: Apply Semi-NMF to get F and G matrices

3. **Output**: Save `snmf_factors.pt` per layer containing:
   - `F`: Feature directions `(d_activation, rank)`
   - `G`: Activation coefficients `(n_tokens, rank)`
   - `token_ids`: Which token each row of G corresponds to
   - `sample_ids`: Which prompt each token came from

### Comparison with snmf-mlp-decomposition Repo

| Aspect | Original Repo | Our Implementation |
|--------|---------------|-------------------|
| Model loading | Remote API | Local HuggingFace |
| Activation hook | Similar | Adapted for local models |
| SNMF algorithm | Identical | Identical |
| Output format | Same | Same |
| Analysis tools | Separate scripts | Integrated visualization |

---

## Mathematical Foundations

### SNMF Decomposition

SNMF decomposes activations as:

```
A ≈ F @ G.T
```

Where:
- **A**: Original MLP activations `(n_tokens, d_activation)`
- **F**: Feature directions matrix `(d_activation, rank)` = `(320, 25)` per layer
- **G**: Activation coefficients matrix `(n_tokens, rank)` = `(166151, 25)` per layer

### Key Matrices

| Matrix | Shape | Description |
|--------|-------|-------------|
| **F** | `(320, 25)` per layer | How each neuron contributes to each of the 25 SNMF features |
| **G** | `(166151, 25)` per layer | How strongly each feature activates for each token |
| **S** | `(25, 9)` per layer | Feature-concept score matrix (computed from G and labels) |

### The S Matrix (Feature-Concept Scores)

The S matrix is the bridge between abstract SNMF features and interpretable concepts:

```python
S[f, c] = sum of G[t, f] for all tokens t belonging to concept c
```

This quantifies how much each SNMF feature `f` is associated with each concept `c`.

### Concepts in the Dataset

From `arithmetic.json`, 9 concepts were identified:

**Forget Set** (to unlearn):
- `division_symbolic`
- `division_riddle`
- `multiplication_symbolic`
- `multiplication_riddle`

**Retain Set** (to preserve):
- `addition_symbolic`
- `addition_riddle`
- `subtraction_symbolic`
- `subtraction_riddle`
- `english`

---

## Two Methods for Computing Importance

### Method 1: Per-Neuron Scalar Importance

**Formula:**
```
importance[n, c] = sum_f( |F[n, f]| * S[f, c] ) / sum_f( S[f, c] )
```

**Key Properties:**
- Takes absolute value of F **before** aggregation
- All contributions are additive (no cancellation)
- Produces a single scalar importance value per neuron-concept pair
- Represents **total magnitude of involvement** regardless of direction

### Method 2: Direction-Based Importance

**Formula (raw direction vector):**
```
d_raw[:, c] = sum_f( S[f, c] * F[:, f] ) / sum_f( S[f, c] )
```

**Formula (L2-normalized for masking):**
```
d_unit[:, c] = d_raw[:, c] / ||d_raw[:, c]||_2
```

**Key Properties:**
- Preserves sign during aggregation (allows cancellation)
- Opposite-signed contributions can cancel out
- Produces a direction vector in neuron space for each concept
- Represents **net directional effect**

### Visual Comparison

For visualization, both methods use `|value|` and global min-max normalization:

| Method | Visualization Value | What It Shows |
|--------|---------------------|---------------|
| Scalar | `\|importance\|` normalized | Total magnitude (no cancellation) |
| Direction | `\|d_raw\|` normalized | Net effect magnitude (with cancellation) |

### Why They Should Be Similar

Both methods:
1. Use the same F and S matrices
2. Apply the same weighted aggregation formula
3. Use the same global min-max normalization

The only mathematical difference:
- **Scalar**: `sum(|F| * S)` → takes absolute value first
- **Direction**: `|sum(F * S)|` → takes absolute value after sum

Since `|sum(...)| ≤ sum(|...|)`, the direction method tends to show slightly lower values due to cancellation effects. But the overall patterns should be similar.

---

## Forget Set vs Retain Set Division

### Purpose

For targeted unlearning, we need to identify neurons that are:
1. **High importance for Forget Set** (division + multiplication)
2. **Low importance for Retain Set** (addition + subtraction + English)

### Aggregation

```python
forget_importance[n] = mean(importance[n, c]) for c in FORGET_SET
retain_importance[n] = mean(importance[n, c]) for c in RETAIN_SET
selectivity[n] = forget_importance[n] - retain_importance[n]
```

### Selectivity Interpretation

| Selectivity Value | Meaning |
|-------------------|---------|
| Positive (red) | Neuron is biased toward Forget concepts |
| Negative (blue) | Neuron is biased toward Retain concepts |
| Near zero | Neuron is equally important for both |

### Visualizations Generated

1. **3-Panel Heatmap**: Forget | Retain | Selectivity (neurons × layers)
2. **Bar Chart Summary**: Mean Forget vs Retain importance per layer
3. **Both for scalar and direction methods**

---

## Normalization Strategies

### Global Min-Max Normalization

Applied to importance matrices for cross-layer comparison:

```python
global_min = min(all values across all layers and neurons)
global_max = max(all values across all layers and neurons)
normalized = (value - global_min) / (global_max - global_min)
```

**Result**: Values scaled to [0, 1] globally, preserving relative differences across layers.

### Per-Matrix Min-Max Normalization

Applied to Forget/Retain heatmaps:

```python
# Forget heatmap
forget_normalized = (forget - forget.min()) / (forget.max() - forget.min())

# Retain heatmap
retain_normalized = (retain - retain.min()) / (retain.max() - retain.min())
```

**Result**: Each heatmap uses full [0, 1] range independently.

### Symmetric Normalization (Selectivity)

For selectivity (which can be positive or negative):

```python
max_abs = max(|selectivity.min()|, |selectivity.max()|)
selectivity_normalized = selectivity / max_abs  # Range: [-1, 1]
```

**Result**: Zero-centered, symmetric range [-1, 1].

---

## Implementation Details

### File Structure

```
src/targeted_undo/snmf_analysis/
├── __init__.py
├── visualize_snmf_neurons.py
├── chat_summary.md
└── outputs/
    ├── neuron_importance_layer_*.png
    ├── direction_magnitude_layer_*.png
    ├── summary_scalar_importance.png
    ├── summary_direction_magnitude.png
    ├── method_comparison.png
    ├── forget_retain_heatmap.png
    ├── forget_retain_summary.png
    ├── forget_retain_direction_heatmap.png
    ├── forget_retain_direction_summary.png
    ├── importance_data.pt
    ├── direction_vectors.pt
    ├── feature_concept_scores.pt
    └── forget_retain_importance.pt
```

### Core Functions

```python
# Data Loading
load_concept_dataset(data_path)     # Load labels from arithmetic.json
load_snmf_data(snmf_dir, layer)     # Load F, G, token_ids, sample_ids

# Core Computation
get_token_labels(sample_ids, labels)                    # Map tokens → concepts
compute_feature_concept_scores(G, token_labels, concepts)  # Compute S matrix
compute_neuron_importance(F, S)                         # Per-neuron scalar method
compute_direction_vectors(F, S)                         # Direction-based method

# Normalization
global_minmax_normalize(data_dict)   # Global normalization across layers

# Aggregation
aggregate_by_group(importance, concepts)  # Forget/Retain grouping

# Visualization
plot_neuron_concept_heatmap(...)     # Per-layer heatmaps
plot_summary_heatmap(...)            # Cross-layer summary
plot_comparison_heatmap(...)         # Side-by-side method comparison
plot_forget_retain_heatmap(...)      # 3-panel Forget/Retain/Selectivity
plot_forget_retain_summary(...)      # Bar chart per layer
```

### CLI Usage

```bash
.venv/bin/python src/targeted_undo/snmf_analysis/visualize_snmf_neurons.py \
    --snmf-dir src/snmf-mlp-decomposition/outputs/snmf_results \
    --data-path src/snmf-mlp-decomposition/data/arithmetic.json \
    --output-dir src/targeted_undo/snmf_analysis/outputs \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13
```

---

## Important Limitations and Scope

### Only down_proj (MLP Output) Is Analyzed

**Current Limitation**: The SNMF analysis only covers the MLP `down_proj` layer (the output projection of the feed-forward network). This means:

| Component | Analyzed? | Notes |
|-----------|-----------|-------|
| MLP down_proj | ✓ Yes | 320 neurons per layer |
| MLP gate | ✗ No | Not included in SNMF |
| MLP up_proj | ✗ No | Not included in SNMF |
| Attention heads | ✗ No | Separate mechanism |

**Implication for Masking**: When creating masks for targeted unlearning:
- The mask will have values for `down_proj` neurons
- Other components (gate, up, attention) could be set to 0 (no masking) or require separate analysis

### The top_k Parameter

**Question raised**: "The decision of top_k = 20, where is it effects? Do we try *all* the tokens and store only the top 20?"

**Answer**: The `top_k` parameter is used in the **summary/interpretation** phase, not in the core SNMF computation:
- **G matrix**: Contains activations for ALL tokens (166,151 tokens)
- **top_k**: Only used when displaying "top activating examples" for interpretability
- **Our analysis**: Uses the FULL G matrix, not just top-k

The SNMF decomposition itself processes all tokens. The top_k is only for human-readable summaries.

### Importance Score Range

**Raw importance values**: Depend on the scale of F and S matrices. In our data:
- Raw scalar importance: ~0.03 to ~0.06 (mean per layer-concept)
- Raw direction magnitude: ~0.01 to ~0.045 (mean per layer-concept)

**After normalization**: [0, 1] for all visualizations

---

## Bug Fixes and Corrections

### Bug 1: ModuleNotFoundError for torch/numpy

**Problem**: Running script with system Python instead of virtual environment.

**Fix**: Use `.venv/bin/python` explicitly.

### Bug 2: Matplotlib Permission Denied

**Problem**: Sandbox restrictions preventing matplotlib from accessing cache directory.

**Fix**: Run with `required_permissions: ["all"]`.

### Bug 3: Results Not Normalized to [0, 1]

**Problem**: Initial summary heatmaps showed raw values (e.g., 0.05) instead of [0, 1].

**Fix**: 
- Implemented global min-max normalization
- Added per-matrix normalization for Forget/Retain heatmaps
- Ensured all visualizations use proper 0-1 scaling

### Bug 4: Missing Forget/Retain Visualization

**Problem**: Initial implementation didn't include the requested Forget Set vs Retain Set grouping.

**Fix**: 
- Added `FORGET_SET` and `RETAIN_SET` constants
- Implemented `aggregate_by_group()` function
- Created `plot_forget_retain_heatmap()` and `plot_forget_retain_summary()`

### Bug 5: Direction Vector Magnitude Confusion

**Problem**: User requested "magnitude of the direction vector" for visualization.

**Initial (incorrect) interpretation**: Used `|d_unit|` (L2-normalized vector components).

**Issue**: L2 normalization makes each concept's direction vector unit length, destroying magnitude information and making it incomparable to the scalar method.

**Final Fix**: Use `|d_raw|` (raw direction vector, before L2 normalization) with the same global min-max normalization as the scalar method.

**Key Insight**: 
- For **visualization**: Use `|d_raw|` (comparable to scalar method)
- For **masking**: Use `d_unit` (L2-normalized direction for geometric operations)

### Bug 6: Forget/Retain Not Applied to Direction Method

**Problem**: Forget/Retain aggregation was only computed for scalar method, not direction method.

**Fix**: Added parallel computation and visualization for direction-based Forget/Retain:
- `forget_retain_direction_heatmap.png`
- `forget_retain_direction_summary.png`
- Updated `forget_retain_importance.pt` to include both methods

---

## Output Files

### Visualization Files

| File | Description |
|------|-------------|
| `neuron_importance_layer_*.png` | Per-layer heatmaps (scalar method) |
| `direction_magnitude_layer_*.png` | Per-layer heatmaps (direction method) |
| `summary_scalar_importance.png` | Cross-layer summary (scalar) |
| `summary_direction_magnitude.png` | Cross-layer summary (direction) |
| `method_comparison.png` | Side-by-side comparison of both methods |
| `forget_retain_heatmap.png` | Forget/Retain/Selectivity (scalar) |
| `forget_retain_direction_heatmap.png` | Forget/Retain/Selectivity (direction) |
| `forget_retain_summary.png` | Bar chart per layer (scalar) |
| `forget_retain_direction_summary.png` | Bar chart per layer (direction) |

### Data Files

| File | Contents |
|------|----------|
| `importance_data.pt` | Raw and normalized importance matrices (scalar method) |
| `direction_vectors.pt` | `d_raw`, `d_unit`, and `d_magnitude` for all layers |
| `feature_concept_scores.pt` | S matrices for all layers |
| `forget_retain_importance.pt` | Aggregated Forget/Retain importance (both methods) |

### Data File Structure

**`forget_retain_importance.pt`:**
```python
{
    'scalar_forget': {layer: tensor},      # Per-neuron scalar, Forget Set
    'scalar_retain': {layer: tensor},      # Per-neuron scalar, Retain Set
    'direction_forget': {layer: tensor},   # Direction-based, Forget Set
    'direction_retain': {layer: tensor},   # Direction-based, Retain Set
    'forget_concepts': ['division_symbolic', ...],
    'retain_concepts': ['addition_symbolic', ...],
    'layers': [0, 1, ..., 13],
}
```

**`direction_vectors.pt`:**
```python
{
    'd_raw': {layer: tensor},      # Raw direction vectors (320, 9)
    'd_unit': {layer: tensor},     # L2-normalized direction vectors (320, 9)
    'd_magnitude': {layer: tensor}, # |d_raw| for visualization (320, 9)
    'concepts': [...],
    'layers': [...],
}
```

---

## Key Takeaways

### 1. Two Methods, Different Interpretations

| Aspect | Per-Neuron Scalar | Direction-Based |
|--------|-------------------|-----------------|
| Absolute value | Before aggregation | After aggregation |
| Sign preservation | No | Yes (allows cancellation) |
| Result | Single importance score | Direction vector in neuron space |
| Use case | Simple importance ranking | Geometric masking operations |

### 2. Normalization Matters

- **Global min-max** ensures cross-layer comparability
- **Per-matrix normalization** for Forget/Retain uses full color range
- **L2 normalization** destroys magnitude - use only for direction, not visualization

### 3. Scalar ≥ Direction (in magnitude)

Because `|sum(x)| ≤ sum(|x|)`, the direction method always shows equal or lower values than the scalar method for any given neuron-concept pair (when cancellation occurs).

### 4. Both Methods Show Similar Patterns

After fixing the L2 normalization issue, both methods show:
- High importance in early layers (0-5)
- Declining importance in later layers
- Similar concept-wise patterns
- Consistent Forget vs Retain selectivity

### 5. Data Saved for Masking

The `d_unit` vectors (L2-normalized) are saved separately for actual masking operations, while `|d_raw|` is used for visualization. This separates the concerns of:
- **Visualization**: Comparable magnitude across methods
- **Masking**: Geometric direction for targeted intervention

### 6. Observed Results Pattern

From the generated visualizations:
- **High importance in early layers (0-5)**: Stronger neuron-concept associations
- **Declining importance in later layers (6-13)**: More distributed representations
- **Similar patterns across concepts**: All arithmetic concepts show similar layer-wise distribution
- **English concept**: Slightly different pattern (more spread across layers)

---

## Technical Discussions

### Absolute Value vs L2 Normalization Confusion

**User Question**: "For direction, don't we use abs? Why do we need L2?"

**Clarification**:

| Operation | Purpose | When Used |
|-----------|---------|-----------|
| `\|F\|` (element-wise abs) | Remove sign, treat all as positive | Scalar importance: before aggregation |
| `\|d_raw\|` (element-wise abs) | Get magnitude per neuron for visualization | Direction: after aggregation, for display |
| `\|\|d\|\|_2` (L2 norm) | Normalize to unit length | Direction masking: geometric operations |

**Key Distinction**:
- **Scalar method**: Uses absolute value **before** weighted sum → all contributions add
- **Direction method**: Uses absolute value **after** weighted sum → allows cancellation

**L2 Normalization Purpose**:
- Makes direction vectors unit length for geometric operations
- Useful for masking where you want to project/subtract a direction
- **NOT for visualization** (destroys magnitude comparability)

### Weighted Mean vs Other Aggregation Options

**Discussion**: How to aggregate feature contributions to get neuron importance?

**Options considered**:
1. **Simple sum**: `sum(F * S)` - Larger S values dominate
2. **Weighted mean**: `sum(F * S) / sum(S)` - Normalizes by total concept activation ✓
3. **Max**: `max(F * S)` - Only considers strongest feature

**Choice**: Weighted mean was selected because:
- Accounts for concepts having different amounts of data
- Provides normalized importance independent of concept frequency
- More stable across different concept sizes

### Direction-Based Masking: Technical Approach

**User Question**: "Isn't direction-based masking like noising just a part of a neuron? How would it work technically?"

**Explanation**:

**Current Approach (Per-Neuron Scalar)**:
```python
# Simple scaling of neuron activations
masked_output = original_output * (1 - mask_strength * importance[n])
```
- Scales entire neuron output
- Binary or soft scaling per neuron

**Direction-Based Approach (Advanced)**:
```python
# Project out the concept direction from activation space
d_unit = direction_vector / ||direction_vector||  # Unit direction
projection = dot(activation, d_unit) * d_unit     # Component along direction
masked_activation = activation - mask_strength * projection
```
- Removes only the **component** of activation aligned with concept direction
- Preserves orthogonal information
- More surgical intervention

**Key Difference**:
- Scalar: "Turn down this neuron"
- Direction: "Remove this specific pattern from activation space"

**Trade-offs**:
| Aspect | Scalar Masking | Direction Masking |
|--------|----------------|-------------------|
| Simplicity | ✓ Simple | More complex |
| Precision | Affects all neuron functions | Targets specific pattern |
| Collateral damage | Higher | Lower |
| Implementation | Easy | Requires matrix operations |

### Aggregation Options for Multiple Concepts

**For Forget/Retain grouping**, we use **mean** across concepts:

```python
forget_importance = mean(importance[:, c] for c in FORGET_SET)
```

**Alternatives considered**:
- **Max**: Most important concept dominates
- **Sum**: Larger sets get higher values
- **Weighted mean**: Could weight by concept frequency

**Mean was chosen** for balanced representation of all concepts in each set.

---

## Future Work

1. **Create non-binary masks** using the Forget/Retain selectivity scores
2. **Threshold selection** for which neurons to mask
3. **Validation** of masks on actual unlearning tasks
4. **Comparison** of scalar vs direction-based masks in practice
5. **Extend analysis** to other MLP components (gate, up_proj)
6. **Attention head analysis** for complete model coverage

---

## Appendix: Full Conversation Topics Covered

1. Analytics code location in snmf-mlp-decomposition
2. F matrix interpretation to concepts
3. Existing targeted_undo code explanation
4. run_snmf.py in-depth explanation
5. Batch processing details (tokenization, extraction)
6. Comparison with original snmf-mlp-decomposition repo
7. Supervised method analysis details
8. Visual presentation planning
9. FFN components (gate, up, down) separation
10. Importance calculation math process
11. top_k = 20 effects and usage
12. Mask size considerations (attention, gate, intermediate)
13. Feature direction explanation
14. Weighted mean vs alternatives
15. Importance score range
16. Direction-based masking approach
17. Min-max normalization details
18. abs vs L2 normalization
19. Plan merger (simultaneous approaches)
20. Matrix S explanation
21. Implementation and execution
22. Normalization issues and fixes
23. Forget/Retain visualization
24. Direction magnitude visualization
25. Methods comparison and equivalence
