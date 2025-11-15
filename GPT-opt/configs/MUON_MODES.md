# Muon Optimizer Modes

This document explains the three different modes for applying the Muon optimizer to transformer model parameters.

## Overview

The Muon optimizer can be configured to apply its orthogonalization-based updates to different subsets of parameters, allowing you to explore different optimization strategies for attention and feed-forward layers.

## Three Modes

### Mode 1: `stacked_qkv` (Default)

**What it does:**
- Applies Muon to the full stacked QKV weight matrix `c_attn.weight` (shape: `[3*n_embd, n_embd]`)
- Applies Muon to all other 2D weight matrices (W_O, FFN, etc.)
- Uses AdamW only for embeddings, layer norms, and biases

**When to use:**
- This is the standard/default behavior
- Good baseline for comparison
- Works well when the QKV projections should be optimized together

**Config example:**
```yaml
optimizer_params:
  - name: muon-polarexpress
    lr: [0.01]
    muon_mode: stacked_qkv  # or omit this line (default)
```

---

### Mode 2: `split_qkv`

**What it does:**
- Splits the QKV matrix into separate Q, K, and V matrices
- Applies Muon independently to each projection (Q, K, V)
- Also splits by attention heads for finer-grained updates
- Applies Muon to W_O and FFN matrices (split by heads for W_O)
- Uses AdamW for embeddings, layer norms, and biases

**When to use:**
- When you want Q, K, and V to be optimized independently
- May provide more flexibility in how each projection learns
- Useful for experiments exploring per-head or per-projection optimization

**Config example:**
```yaml
optimizer_params:
  - name: muon-polarexpress
    lr: [0.01]
    muon_mode: split_qkv
```

---

### Mode 3: `voh_only` (Value, Output, Hidden/FFN only)

**What it does:**
- Applies Muon only to:
  - **V** (Value projection) - extracted from QKV matrix, split by heads
  - **W_O** (Output projection) - `c_proj.weight`
  - **FFN** (Feed-Forward Network) - `mlp.c_fc.weight` and `mlp.c_proj.weight`
- Uses AdamW for:
  - **Q** and **K** (Query and Key projections) - first 2/3 of QKV matrix
  - Embeddings, layer norms, biases, and other parameters

**When to use:**
- Hypothesis: Q and K projections may benefit from adaptive learning rates (AdamW)
- V, W_O, and FFN may benefit more from orthogonalization (Muon)
- Useful for ablation studies to understand which components benefit from Muon

**Config example:**
```yaml
optimizer_params:
  - name: muon-polarexpress
    lr: [0.01]
    muon_mode: voh_only
```

---

## Implementation Details

### Parameter Grouping

**GPT-2 Model Structure:**
```
transformer.h[i].attn.c_attn.weight   # QKV stacked: [3*n_embd, n_embd]
transformer.h[i].attn.c_proj.weight   # W_O: [n_embd, n_embd]  
transformer.h[i].mlp.c_fc.weight      # FFN up: [4*n_embd, n_embd]
transformer.h[i].mlp.c_proj.weight    # FFN down: [n_embd, 4*n_embd]
```

**QKV Matrix Layout:**
```
[0:n_embd, :]        # Query (Q)
[n_embd:2*n_embd, :] # Key (K)
[2*n_embd:3*n_embd, :]# Value (V)
```

### Mode Comparison Table

| Component | `stacked_qkv` | `split_qkv` | `voh_only` |
|-----------|---------------|-------------|------------|
| Q projection | Muon (stacked) | Muon (split) | **AdamW** |
| K projection | Muon (stacked) | Muon (split) | **AdamW** |
| V projection | Muon (stacked) | Muon (split) | Muon (split by heads) |
| W_O (output) | Muon | Muon (split by heads) | Muon (split by heads) |
| FFN matrices | Muon | Muon | Muon |
| Embeddings | AdamW | AdamW | AdamW |
| LayerNorms | AdamW | AdamW | AdamW |

---

## Usage

### In YAML Config

Add the `muon_mode` parameter to your optimizer configuration:

```yaml
optimizer_params:
  - name: muon-polarexpress
    lr: [0.01]
    weight_decay: 0.1
    momentum: 0.95
    ns_steps: 5
    rms_scaling: true
    polar_num_iters: 5
    polar_safety: 1.01
    polar_cushion: 0.024
    muon_mode: voh_only  # Choose: stacked_qkv, split_qkv, or voh_only
```

### In Python Code

```python
from gptopt.optim.muon import Muon

optimizer = Muon(
    model.named_parameters(),
    lr=0.01,
    weight_decay=0.1,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
    rms_scaling=True,
    polar_method='polarexpress',
    polar_num_iters=5,
    polar_safety=1.01,
    polar_cushion=0.024,
    muon_mode='voh_only',  # Choose mode here
    nheads=12  # Required for split_qkv and voh_only modes
)
```

---

## Experimental Recommendations

1. **Baseline**: Start with `stacked_qkv` to establish baseline performance
2. **Exploration**: Try `split_qkv` to see if independent Q/K/V optimization helps
3. **Hypothesis Testing**: Use `voh_only` to test if Q/K benefit from adaptive learning rates

### Metrics to Track

- Training loss and validation perplexity
- Gradient norms per component (Q, K, V, W_O, FFN)
- Orthogonality error (for Muon-optimized components)
- Convergence speed

---

## Notes

- All modes require `nheads` parameter for `split_qkv` and `voh_only` modes
- The `split_heads` parameter is separate and can be combined with any mode
- Weight decay is applied to all parameters in their respective optimizers
- AdamW uses the same learning rate as Muon but with separate momentum terms

---

## References

- Muon paper: [MomentUm Orthogonalized by Newton-schulz](https://kellerjordan.github.io/posts/muon/)
- PolarExpress: Optimized polar decomposition for Muon
