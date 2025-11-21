# Comprehensive Metrics Logging

This document describes all metrics logged during training, organized by category following best practices for LLM training monitoring.

## Logging Configuration

**Configurable via sweep YAML** (`logging_params`):
- `log_step`: Frequency for end-to-end metrics (in micro-steps, default: 160 = every 10 optimizer steps)
- `diag_log_step`: Frequency for diagnostics: PE, attention health, weight scales (in micro-steps, default: 160 = every 10 optimizer steps)
- `svd_log_step`: Frequency for expensive SVD/orthogonality metrics (in micro-steps, default: 800 = every 50 optimizer steps)
- `val_step`: Frequency for validation (in micro-steps, default: 160 = every 10 optimizer steps)

**Current Sweep Frequencies:**
- **Phase 0** (0.2 epochs, ~336 optimizer steps = 5,376 micro-steps): 
  - log_step=160 → ~33 end-to-end log points
  - diag_log_step=160 → ~33 diagnostic events
  - svd_log_step=800 → ~6 SVD events
  - val_step=8000 → 0 validation points (too infrequent for 0.2 epochs)
  
- **Phase 1** (1.0 epoch, ~1,680 optimizer steps = 26,880 micro-steps):
  - log_step=160 → ~168 end-to-end log points
  - diag_log_step=160 → ~168 diagnostic events
  - svd_log_step=800 → ~33 SVD events
  - val_step=160 → ~168 validation points

## Quick Summary

**Total Metrics: ~150** across 6 categories, automatically logged to W&B

| Category | Purpose | Count | Frequency | Phase 0 | Phase 1 | Overhead |
|----------|---------|-------|-----------|---------|---------|----------|
| **End-to-end** | Training quality & efficiency | 8 | Every log_step (10 steps) | 38 pts | 191 pts | 0.0008% |
| **PolarExpress** | Optimizer health (if using PE) | 3 | Every diag_log_step (10 steps) | 38 evt | 191 evt | <0.001% |
| **Attention Health** | Catch head collapse across depth | 21 | Every diag_log_step (10 steps) | 38 evt | 191 evt | 0.004% |
| **Weight & Gradient Scales** | Localize gradient/scale issues | 22 | Every diag_log_step (10 steps) | 38 evt | 191 evt | 0.002% |
| **SVD + Orthogonality** | Per-projection conditioning analysis | ~91 | Every svd_log_step (50 steps) | 0 evt | 38 evt | **0.029%** |
| **Stability Alarms** | Critical failure detection | 4 | Every 1-10 steps | 381 | 1,907 | <0.001% |
| **PHASE TOTAL** | — | **149** | — | **0.003%** | **0.035%** | — |

**Three Independent Logging Frequencies:**
- **log_step (160 microsteps)**: End-to-end metrics only (cheap)
- **diag_log_step (160 microsteps)**: PE + attention + scales diagnostics (moderate cost)
- **svd_log_step (800 microsteps)**: Expensive SVD computation (expensive but worth it)
- Combined overhead Phase 1: **0.035%** (~6.3 seconds out of 18,115 seconds)

**Key Features:**
- ✅ **Depth-aware**: Tracks layers 0, 5, 11 (first/middle/last) to diagnose where issues occur
- ✅ **Efficient**: ~0.14% average overhead across all steps (1.2ms per regular step)
- ✅ **Comprehensive**: Covers convergence, attention health, gradient flow, weight conditioning
- ✅ **Actionable**: Each metric has clear "good range" and alerts
- ✅ **Auto-logged to W&B**: Pass `wandb_run` and all metrics are automatically tracked
- ✅ **Production-ready**: Based on LLM training best practices with heavy optimizations

## A) End-to-end Metrics (Every `log_step`)

**Purpose:** Primary quality signal and generalization check

**Logged to W&B at frequency:** `log_step` (configurable, currently 160 microsteps = 10 optimizer steps)

**Note on Loss Logging:** 
- `train/loss` is computed every optimizer step for accumulation/averaging
- Logged to W&B only at `log_step` intervals to balance metric granularity vs W&B overhead
- At Phase 0/1 default (every 10 steps), provides sufficient granularity (~38-190 points) for analysis
- To log every step: Set `log_step=16` (1 optimizer step), but adds ~100x W&B calls (not recommended)

| Metric | Description | Good Range | Alerts |
|--------|-------------|------------|---------|
| `train/loss` | Cross-entropy training loss (averaged over grad_accum_steps) | Decreasing | Spikes, NaN |
| `val/loss` | Cross-entropy validation loss | Decreasing | Increases (overfitting) |
| `val/ppl` | Validation perplexity (exp(loss)) | Decreasing | > 1000 (poor model) |
| `train/lr` | Current learning rate | Config-dependent | - |
| `train/tokens_per_sec` | Training throughput | Maximize | Drops (bottleneck) |
| `train/step_time_ms` | Wall clock time per optimizer step (ms) | Minimize | Spikes indicate slowdown |

**Convergence & Speed/Cost:**
- `train/loss` should decrease smoothly
- `val/loss` tracks generalization (gap indicates overfitting)
- `tokens_per_sec` measures training efficiency
- `train/step_time_ms` tracks optimizer overhead (Muon > AdamW)

## B) PolarExpress-Internal Metrics (Every `diag_log_step` = 10 steps)

**Purpose:** Verify PE did its job and what it cost

| Metric | Description | Good Range | Alerts |
|--------|-------------|------------|---------|
| `pe/ortho_err_before` | ‖G^T G - I‖_F on input gradient (before PE, averaged) | 1-100 | N/A (input varies) |
| `pe/ortho_err_after` | ‖U^T U - I‖_F on output (after PE, averaged) | < 0.1 | > 0.5 (not orthogonalizing) |
| `pe/time_ms` | Direct cost of PE substep (last 10 steps) | < 10ms | > 50ms (bottleneck) |
| `pe/avg_time_ms` | Average PE time across entire training | < 10ms | > 50ms (bottleneck) |
| `ortho_err_before/layer{0,5,11}_stacked_qkv` | Per-layer ortho error before PE (stacked QKV) | 1-100 | N/A |
| `ortho_err_after/layer{0,5,11}_stacked_qkv` | Per-layer ortho error after PE (stacked QKV) | < 0.1 | > 0.5 |

**PE Quality Check:**
- `pe/ortho_err_before` measures input gradient orthogonality (averaged across all layers)
  - Computed on normalized gradient before any PE iterations
- `pe/ortho_err_after` measures how orthogonal the output U is (averaged across all layers)
  - Small values (< 0.1) = good orthogonalization
  - Large values (> 0.5) = PE failing, may need different coefficients
  - Recomputed after final iteration for accuracy (~100ms overhead every 100 steps)
- **Per-layer tracking**: `ortho_err_{before,after}/layer{0,5,11}_stacked_qkv`
  - Tracks ortho errors for sentinel layers (first, middle, last)
  - Enables depth-dependent PE quality analysis
  - Only logged for stacked QKV mode (most common)
- **Gap analysis**: `ortho_err_before - ortho_err_after` shows PE improvement
  - Large gap = PE doing useful work
  - Small gap = gradient already orthogonal (PE unnecessary?)
- `pe/time_ms` tracks computational cost per update (last 10 steps)
- `pe/avg_time_ms` tracks average PE time across entire training run

**Note:** Only logged when using Muon with `polar_method=polarexpress`. Per-layer metrics only for `muon_mode=stacked_qkv`.
**Overhead:** ~200ms every 100 steps for both before/after matmuls (0.02% of training time).

## C) Multi-Layer Attention Health (Every 10 Steps via `diag_log_step`)

**Purpose:** Catch head collapse early and track depth-dependent behavior

**Tracked Layers:** 0 (first), 5 (middle), 11 (last) - representative sampling across network depth

### Logits (Pre-Softmax, Unmasked) - Per Layer
| Metric | Description | Good Range | Alerts |
|--------|-------------|------------|---------|
| `logits/layer{0,5,11}/mean` | Mean attention logit per layer | -5 to 5 | Outside range |
| `logits/layer{0,5,11}/std` | Std of attention logits per layer | 0.5 - 5 | < 0.1 or > 10 |
| `logits/layer{0,5,11}/max_p95` | 95th percentile of logits per layer | < 20 | > 50 (over-scaled) |

**Purpose:** Detect over-scaled Q/K/d before softmax across network depth

### Attention (Post-Softmax) - Per Layer
| Metric | Description | Good Range | Alerts |
|--------|-------------|------------|---------|
| `attn/layer{0,5,11}/entropy/mean` | Mean attention entropy per layer | 2-6 | < 1 (collapse) |
| `attn/layer{0,5,11}/entropy/p05` | 5th percentile entropy per layer | > 0.5 | Near 0 (some heads collapsed) |
| `attn/layer{0,5,11}/entropy/p95` | 95th percentile entropy per layer | < 8 | > 10 (too diffuse) |
| `attn/layer{0,5,11}/maxA/frac>0.95` | Fraction with max attn > 0.95 per layer | < 0.1 | > 0.3 (peaky rows) |

**Attention Collapse Detection:**
- Low entropy (< 1) = attention focusing on single token (head collapse)
- High `maxA/frac>0.95` = many positions attending to one token (collapse proxy)
- **Depth patterns**: Early layers often have lower entropy (more specific), later layers higher (more diffuse)

**Total:** 21 attention metrics (7 metrics × 3 layers)

## D) Weight Scales & Gradient Flow (Every 10 Steps via `diag_log_step`)

**Purpose:** Localize problems via scale/gradient checks across network depth

### QKV Representation Scale Drift - Per Layer
| Metric | Description | Alerts |
|--------|-------------|---------|
| `qkv/layer{0,5,11}/q_norm/mean` | Mean norm of Q projection per layer | Drift over time |
| `qkv/layer{0,5,11}/k_norm/mean` | Mean norm of K projection per layer | Drift over time |
| `qkv/layer{0,5,11}/v_norm/mean` | Mean norm of V projection per layer | Drift over time |

**Purpose:** Track if Q/K/V representations are growing/shrinking across depth (often precedes issues)
**Total:** 9 QKV metrics (3 projections × 3 layers)

### Weight Norms - Per Layer
| Metric | Description | Alerts |
|--------|-------------|---------|
| `weights/layer{0,5,11}_attn_norm` | Attention weight matrix norm per layer | Sudden changes |
| `weights/layer{0,5,11}_mlp_norm` | MLP weight matrix norm per layer | Sudden changes |

**Purpose:** Monitor weight magnitude evolution across depth
**Total:** 6 weight metrics (2 matrices × 3 layers)

### Per-Layer Gradient Flow
| Metric | Description | Alerts |
|--------|-------------|---------|
| `grads/layer{0,5,11}_norm` | Mean gradient norm for all params in layer | Vanishing (< 1e-5) or exploding (> 100) |

**Purpose:** Detect vanishing/exploding gradients across depth
- Early layers (0) should have smaller but non-zero gradients
- Later layers (11) naturally have larger gradients (closer to loss)
- Sharp drops indicate vanishing gradient problem
**Total:** 3 per-layer metrics

### Gradient Norms by Subpath (Early Layers Only)
| Metric | Description | Alerts |
|--------|-------------|---------|
| `grads/W0_norm` | MLP up-projection gradients (layer 0) | Which path destabilizes first |
| `grads/WQ_norm` | Query projection gradients (layers 0-2) | " |
| `grads/WK_norm` | Key projection gradients (layers 0-2) | " |
| `grads/WV_norm` | Value projection gradients (layers 0-2) | " |

**Purpose:** Identify which subpath (MLP vs attention Q/K/V) has gradient issues first
**Why early layers only?** Gradients must flow through entire network to reach early layers, so problems manifest here most clearly. Later layers naturally have larger gradients.
**Total:** 4 subpath metrics

## E) SVD-Based Weight Analysis (Every 50 Steps via `svd_log_step`)

**Purpose:** Deep analysis of weight matrix conditioning and rank

**Tracked Layers:** 0 (first), 5 (middle), 11 (last) - sampled for efficiency

### Singular Value Histograms (Weights, Updates, Post-PE)

At each `svd_log_step`, we compute singular values for a small set of **sentinel matrices** and log
them as **W&B histograms**, not as separate scalar metrics:

| Metric prefix | Description |
|---------------|-------------|
| `svd/weights/layer{0,5,11}/{q,k,v,stacked}` | Histogram of singular values of attention Q/K/V and stacked QKV weight matrices. |
| `svd/pre_update/layer{0,5,11}/{q,k,v,stacked}` | Histogram of singular values of Muon momentum/update buffers before PE (if `use_muon=True`). |
| `svd/post_update/layer{0,5,11}/{q,k,v,stacked}` | Histogram of singular values of post-PE updates cached from the optimizer (if available). |
| `svd_log/weights/layer{0,5,11}/{q,k,v,stacked}` | Same as `svd/weights/...` but with **log10(σ)** applied elementwise. |
| `svd_log/pre_update/layer{0,5,11}/{q,k,v,stacked}` | Log10-scale histograms of pre-PE updates. |
| `svd_log/post_update/layer{0,5,11}/{q,k,v,stacked}` | Log10-scale histograms of post-PE updates. |
| `svd/total_time_ms` | Wall-clock time (ms) spent computing all SVD metrics at the current event. |

**Conventions:**
- `layer{0,5,11}` corresponds to the sentinel transformer blocks (first/middle/last).
- `{q,k,v,stacked}` refer to splitting the standard GPT-2 `c_attn` weight of shape `[2304, 768]`
  into three 768×768 blocks (Q, K, V) and the full stacked matrix (stacked QKV).
- Using `svd_log/*` in W&B **Distributions** views gives a clear picture of how the
  singular value distribution evolves over training on a log scale.

**How to use in W&B:**
- Plot `svd_log/weights/layer0/q` (or any other prefix) in the Distributions panel.
- Enable "Show as 2D" / "distribution over time" to see **step × log σ** density.
- Compare weights vs pre/post-update distributions to see how PE affects conditioning.

### Weight Orthogonality - Per Layer, Per Projection
| Metric | Description | Good Range | Alerts |
|--------|-------------|------------|---------|
| `ortho/layer{0,5,11}_{q,k,v}/err` | ‖W^T W - I‖_F per projection | < 10 | > 100 (far from orthogonal) |
| `ortho/layer{0,5,11}_{q,k,v}/err_normalized` | Orthogonality error / matrix size | < 0.1 | > 1.0 (severe deviation) |

**Weight Orthogonality Indicators:**
- **Per-projection tracking**: Separate Q, K, V orthogonality (each 768×768)
  - Identifies which projection loses orthogonality first
  - More granular than combined QKV analysis
- **Orthogonality error**: Measures how far weights are from being orthogonal
  - Complements `pe/ortho_err_after` which measures per-step orthogonalization quality
  - Shows if PE benefits accumulate: do weights stay orthogonal over training?
  - Zero = perfectly orthogonal, higher = more deviation
- **Normalized error**: Scaled by matrix dimension for comparison across layers
  - Makes it easier to compare shallow vs deep layers
  - Good baseline: < 0.1 means weights are reasonably orthogonal

**Key Insight:** Compare `pe/ortho_err_after` (per-step PE quality) vs `ortho/layer*/err` (accumulated weight orthogonality)
- Low `pe/ortho_err_after` + high `ortho/layer*/err` = PE works per-step but benefits don't persist
- Compare `pe/ortho_err_before` vs `pe/ortho_err_after` to see PE improvement per step
- Low both = PE successfully maintains long-term orthogonality
- High both = PE not working well (check hyperparameters)

**Total volume:** Dozens of SVD histogram series (weights, pre-update, post-update, linear + log10)
plus 18 orthogonality scalars. All are computed every `svd_log_step` (default: 800 micro-steps ≈ 50
optimizer steps), with total SVD + orthogonality overhead of ~140ms per event.

## F) Numeric Stability Alarms (Every Step)

| Metric | Description | Alerts |
|--------|-------------|---------|
| `train/naninf_flag` | NaN/Inf detected in loss | = 1.0 (critical) |
| `train/amp_scaler` | Max gradient magnitude | > 1e4 (overflow risk) |
| `train/amp_overflows` | Gradient overflow flag | = 1.0 (reduce LR) |
| `train/grad_norm` | Global gradient norm | Spikes (instability) |

**Critical Alerts:**
- `naninf_flag = 1` → Stop training, investigate
- `amp_overflows = 1` → Reduce learning rate or use gradient clipping

## Summary Dashboard Layout

### Primary Panel (Always visible)
```
train/loss  val/loss  val/ppl  tokens_per_sec
```

### Health Panel (Check every 100 steps)
```
pe/ortho_err_before  pe/ortho_err_after  attn/entropy/mean  logits/std
naninf_flag   amp_overflows
```

### Debug Panel (When investigating issues)
```
qkv/*_norm/mean
grads/*_norm
attn/maxA/frac>0.95
```

### Optional Panel: Weight Conditioning Snapshots (from SVD histograms)
```
svd_log/weights/layer0/stacked
svd_log/weights/layer5/stacked
svd_log/weights/layer11/stacked
ortho/layer0_q/err  ortho/layer5_q/err  ortho/layer11_q/err
```

## Frequency Summary

| Frequency | Metrics | Phase 0 Events | Phase 1 Events | Cost/Event | Phase 0 Total | Phase 1 Total | Phase 0 % | Phase 1 % |
|-----------|---------|-------|-----------|---------|-------|-----------|----------|----------|
| **log_step=160** (10 steps) | train/loss, train/*, tokens_per_sec, naninf_flag | 38 | 191 | 0.1ms | ~4ms | ~15ms | 0.0004% | 0.0008% |
| **diag_log_step=160** (10 steps) | pe/*, logits/*, attn/*, qkv/*, grads/*, weights/* | 38 | 191 | 0.8ms | ~30ms | ~150ms | 0.003% | **0.0083%** |
| **svd_log_step=800** (50 steps) | svd/*, ortho/* | 0 | 38 | 140ms | 0ms | **~5,320ms** | 0% | **0.029%** |
| **Every step** | train/grad_norm, train/amp_* | 381 | 1,907 | 0.1ms | ~38ms | ~200ms | 0.003% | 0.001% |
| **TOTAL OVERHEAD** | **~150 metrics** | — | — | — | **~72ms** | **~5,685ms** | **0.006%** | **0.031%** |

**Key Change from Previous Config:**
- Old: All diagnostics (PE, attention, scales) at svd_log_step (50 steps) → 38 events
- New: Diagnostics at diag_log_step (10 steps) → 191 events
- Old SVD: 38 events at 50 steps
- New SVD: 38 events at 50 steps (unchanged)
- **Result: 5x more diagnostic snapshots (38 → 191), only +0.0083% added overhead**

**Computational Cost Breakdown:**
- Per log_step event: 0.1ms (end-to-end only)
- Per diag_log_step event: ~0.8ms (attention/scales without SVD)
- Per svd_log_step event: ~140ms (expensive SVD computation)
- **Total per full cycle: ~0.15ms + ~0.8ms + 140ms ≈ ~141ms every ~50-100 steps**

**Trade-off Analysis (diag_log_step = 10 vs 50):**
- Diagnostic resolution: 38 → 191 snapshots (5x improvement)
- Additional overhead: +0.0083% Phase 1 (negligible)
- **Cost: +0.0083% | Benefit: 5x diagnostic resolution** ✅ Excellent trade-off

**Optimizations Applied:**
- ✅ Sequential layer forwarding (saves ~37% on attention metrics)
- ✅ Causal mask reuse across layers
- ✅ Cached XTX from PolarExpress (50-80× faster ortho check)
- ✅ Max gradient check only every 10 steps
- ✅ Sampled layers (0, 5, 11) instead of all 12
- ✅ Early layer gradient subpaths only

## Weights & Biases (W&B) Integration

**All metrics are automatically logged to W&B** when you provide a `wandb_run` object to the training function. 

### How It Works:
1. Initialize W&B in your training script:
   ```python
   import wandb
   wandb_run = wandb.init(project="your-project", name="your-run")
   ```

2. Pass `wandb_run` to the training function:
   ```python
   train(train_dataloader, val_dataloader, model, optimizer, 
         training_params, logging_params, wandb_run=wandb_run)
   ```

3. **All metrics described above are automatically logged** with no additional configuration:
   - Every step: loss, grad_norm, lr, TPS, stability alarms
   - Every 10 steps: AMP scaler/overflows
   - Every 100 steps: PE metrics, attention health, QKV norms, weight scales, gradients
   - Every 500 steps: SVD analysis

### Viewing in W&B Dashboard:
- Metrics are organized by prefix: `train/*`, `pe/*`, `attn/layer*/*`, `grads/*`, etc.
- Create custom panels using the examples in the "Summary Dashboard Layout" section
- Filter by layer using wildcards: `attn/layer*/entropy/mean` shows all 3 layers

### Disabling Logging:
If you don't want W&B logging, simply set `wandb_run=None`:
```python
train(..., wandb_run=None)  # No W&B logging
```

### Example: Check Training Health

```python
# In W&B dashboard, create panels:

# Panel 1: Convergence
plot: train/loss, val/loss, val/ppl

# Panel 2: PE Health (if using PolarExpress)
plot: pe/ortho_err_before, pe/ortho_err_after, pe/time_ms

# Panel 3: Attention Health Across Depth
plot: attn/layer0/entropy/mean, attn/layer5/entropy/mean, attn/layer11/entropy/mean
plot: attn/layer0/maxA/frac>0.95, attn/layer5/maxA/frac>0.95, attn/layer11/maxA/frac>0.95

# Panel 4: Gradient Flow Across Depth
plot: grads/layer0_norm, grads/layer5_norm, grads/layer11_norm

# Panel 5: Weight Conditioning (from SVD)
plot: svd/layer0/condition_number, svd/layer5/condition_number, svd/layer11/condition_number

# Panel 6: Stability Alarms
plot: train/naninf_flag, train/amp_overflows, train/grad_norm
```

## Metrics Design Rationale

### Why Sample Layers 0, 5, 11?
- **Layer 0 (First)**: Receives gradients through entire network (weakest signal), most prone to vanishing gradients
- **Layer 5 (Middle)**: Representative of mid-network behavior, balances early and late dynamics
- **Layer 11 (Last)**: Closest to loss, strongest gradients, most direct learning signal

This 3-layer sampling provides:
- **Coverage** across network depth without redundancy
- **Efficiency**: 3× cost instead of 12× for all layers
- **Actionable insights**: Can diagnose WHERE issues occur (early/middle/late)

### Why Different Frequencies?
- **Every step (6 metrics)**: Critical signals that must be caught immediately (NaN, loss spikes)
- **Every 10 steps (2 metrics)**: Moderate overhead checks (AMP scaling)
- **Every 100 steps (~45 metrics)**: Comprehensive monitoring with acceptable overhead (1.5ms)
- **Every 500 steps (21 metrics)**: Expensive analysis (SVD) that doesn't change rapidly

### Why These Specific Metrics?
Following LLM training best practices:
- **Attention health**: Catches head collapse (common failure mode)
- **Gradient flow**: Detects vanishing/exploding gradients early
- **Weight conditioning**: SVD reveals numerical instability before it causes NaNs
- **QKV scales**: Early warning sign of attention issues
- **PE metrics**: Verify optimizer is working correctly
- **Gradient subpaths**: Diagnose MLP vs Attention gradient flow separately

## Troubleshooting Guide

### Issue: Training diverges (loss → NaN)
1. Check `train/naninf_flag` - when did it first appear?
2. Check `train/amp_overflows` - gradient explosion?
3. Check `logits/layer*/std` - attention logits too large in any layer?
4. Check `grads/layer*_norm` - which layer's gradients exploded first?
5. Inspect `svd_log/weights/layer*/stacked` histograms - mass drifting to extreme log σ values (very large or very small) indicates ill-conditioned weights.
6. **Solution:** Reduce LR, increase grad clipping, adjust PE safety factor, or lower `polar_num_iters`

### Issue: Model not learning (loss plateau)
1. Check `val/loss` vs `train/loss` - overfitting?
2. Check `attn/layer*/entropy/mean` - attention collapsed in any layer?
3. Check `pe/ortho_err_after` - PE working correctly? Compare to `pe/ortho_err_before` for improvement.
4. Check `grads/layer0_norm` - gradients reaching early layers?
5. Inspect `svd_log/weights/layer*/stacked` histograms - collapsing spectrum (most mass concentrated in a few bins) can indicate loss of rank.
6. **Solution:** Adjust LR schedule, check data, verify PE coefficients, increase model capacity

### Issue: Slow training
1. Check `tokens_per_sec` - throughput bottleneck?
2. Check `pe/time_ms` - PE taking too long?
3. **Solution:** Reduce `polar_num_iters`, optimize batch size, use torch.compile, profile code

### Issue: Attention heads collapse
1. Check `attn/layer*/entropy/mean` - which layers dropping below 1?
2. Check `attn/layer*/maxA/frac>0.95` - which layers increasing?
3. Check `qkv/layer*/q_norm/mean` and `k_norm/mean` - Q/K scales drifting?
4. Check `logits/layer*/std` - logits becoming too peaky?
5. **Solution:** Adjust attention temperature, check initialization, verify PE orthogonalization, add attention dropout

### Issue: Gradients vanishing in early layers
1. Check `grads/layer0_norm` vs `grads/layer11_norm` - ratio should be ~0.01-0.1
2. Check `grads/W0_norm`, `grads/WQ_norm` - which subpath affected?
3. Inspect `svd_log/weights/layer0/stacked` - highly skewed or degenerate singular value distributions can correlate with vanishing gradients.
4. **Solution:** Increase LR for early layers (layer-wise LR), use skip connections, verify PE is helping

### Issue: Weights becoming ill-conditioned
1. Inspect `svd_log/weights/layer*/stacked` histograms - which layers have very wide log σ spread or most mass near extreme bins?
2. Check `pe/ortho_err_after` and `ortho/layer*/err` - PE failing to orthogonalize? Is gap from `pe/ortho_err_before` small while `ortho/layer*/err` remains high?
4. **Solution:** Adjust PE parameters (safety, cushion), add weight decay, reduce LR

## Configuration

Control logging frequency in your config file:

```yaml
logging_params:
  log_step: 50        # Print to console every N steps
  val_step: 500       # Run validation every N steps
  save_ckpt_step: 500 # Save checkpoint every N steps
```

Advanced metrics (categories C & D) are always computed every 100 steps for efficiency.
