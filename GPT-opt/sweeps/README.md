# W&B Sweep Configurations for CS182 PolarExpress Project

This directory contains W&B sweep configurations for running hyperparameter studies on PolarExpress.

## Setup

1. **Install W&B and login:**
```bash
pip install wandb
wandb login
```

2. **Make sure you're in the GPT-opt directory:**
```bash
cd GPT-opt
```

## Phase 0: Baseline Validation (Start Here!)

**Purpose**: Validate setup and establish baseline performance

**Config**: `phase0-baseline.yaml`
- 2 optimizers (AdamW, Muon+PE)
- 3 learning rates (0.001, 0.003, 0.005)
- Short training (3 epochs, ~100M tokens)
- **Total: 6 runs**

**Launch:**
```bash
# Initialize sweep
wandb sweep sweeps/phase0-baseline.yaml

# Start agent (run this on each GPU you want to use)
wandb agent <your-username>/cs182-polar-express/<sweep-id>
```

**Expected Time**: ~1-2 hours per run = 6-12 GPU-hours total

---

## Phase 1: PolarExpress Sensitivity Study

### Option A: Full Grid Search (12 runs)

**Config**: `phase1-pe-sensitivity.yaml`
- 3 num_iters (3, 5, 7)
- 2 safety factors (1.00, 1.01)
- 2 cushion values (0.024, 0.05)
- **Total: 3 × 2 × 2 = 12 runs**

**Launch:**
```bash
wandb sweep sweeps/phase1-pe-sensitivity.yaml
wandb agent <your-username>/cs182-polar-express/<sweep-id>
```

**Expected Time**: ~8-10 hours per run = 96-120 GPU-hours total

### Option B: Focused Search (Recommended, 3 runs)

**Config**: `phase1-pe-focused.yaml`
- Only varies num_iters (3, 5, 7)
- Fixes safety=1.01, cushion=0.024
- **Total: 3 runs**

**Strategy**: Test computational cost first, then create follow-up sweeps for other dimensions if needed.

**Launch:**
```bash
wandb sweep sweeps/phase1-pe-focused.yaml
wandb agent <your-username>/cs182-polar-express/<sweep-id>
```

**Expected Time**: ~8-10 hours per run = 24-30 GPU-hours total

---

## Running Agents on Multiple GPUs

### Parallel Execution:

**On your local machines:**
```bash
# Terminal 1 (A6000 #1)
CUDA_VISIBLE_DEVICES=0 wandb agent <sweep-id>

# Terminal 2 (A6000 #2)
CUDA_VISIBLE_DEVICES=1 wandb agent <sweep-id>

# Terminal 3 (RTX 4070)
CUDA_VISIBLE_DEVICES=2 wandb agent <sweep-id>

# Terminal 4 (RTX 5070)
CUDA_VISIBLE_DEVICES=3 wandb agent <sweep-id>
```

Each agent will automatically pull the next config from the sweep queue.

### Using tmux for Background Execution:

```bash
# Start tmux session
tmux new -s wandb-sweep

# Split into 4 panes (Ctrl+B, then %)
# In each pane, run:
CUDA_VISIBLE_DEVICES=X wandb agent <sweep-id>

# Detach with Ctrl+B, then D
# Reattach later with: tmux attach -t wandb-sweep
```

---

## Monitoring Your Sweeps

1. **W&B Dashboard**: 
   - Go to https://wandb.ai/your-username/cs182-polar-express
   - Click on "Sweeps" tab
   - View parallel coordinates plot, comparison tables

2. **Key Metrics to Watch**:
   - `val/loss` - Primary objective
   - `pe/ortho_err` - PE correctness
   - `pe/time_ms` - Computational cost
   - `svd/layer*/condition_number` - Weight conditioning
   - `attn/layer*/entropy/mean` - Attention health

3. **Command Line**:
```bash
wandb sweep <sweep-id> --status
```

---

## Stopping and Resuming

**Pause sweep** (stops assigning new runs):
```bash
wandb sweep --pause <sweep-id>
```

**Resume sweep**:
```bash
wandb sweep --resume <sweep-id>
```

**Cancel sweep** (stops everything):
```bash
wandb sweep --cancel <sweep-id>
```

**Stop individual agent**: Just Ctrl+C in the terminal

---

## Customizing Sweeps

### Adjust for Longer Training:

Edit the sweep YAML:
```yaml
training_data.training_params.num_epochs:
  value: 10  # Increase from 7

# Or adjust tokens directly
training_data.training_params.tokens_processed:
  value: 1048576  # ~1M tokens per step
```

### Change Learning Rate:

```yaml
optimizer_params.args.lr:
  value: 0.01  # Or use 'values: [0.003, 0.005, 0.01]' for sweep
```

### Add More Tags:

```yaml
logging_params.wandb.tags:
  value: ["phase1", "gpu-a6000", "my-experiment"]
```

---

## Troubleshooting

### Out of Memory (OOM):
- Reduce batch size in the config
- Use smaller model (already using gpt-small)
- Use gradient accumulation

### Run Diverged (NaN):
- Check `train/naninf_flag` metric
- Look at `train/grad_norm` - did it explode?
- Try lower learning rate
- Check `pe/ortho_err` - is PE working?

### Runs Too Slow:
- Reduce `tokens_processed` for faster iteration
- Reduce `num_epochs`
- Use smaller dataset (shakespeare instead of finewebmini)

### Agent Not Starting:
```bash
# Check if sweep exists
wandb sweep --status <sweep-id>

# Make sure you're in the right directory
pwd  # Should be in GPT-opt/

# Check python environment
which python
pip list | grep torch
```

---

## Tips for Efficient Experimentation

1. **Start small**: Run Phase 0 baseline first to validate everything works

2. **Monitor early**: Check first 1000 steps - if metrics look wrong, kill and fix

3. **Use focused sweep**: Phase1-focused (3 runs) before full grid (12 runs)

4. **Checkpoint often**: Set `save_ckpt_step: 1000` to save progress

5. **Compare live**: W&B parallel coordinates plot shows trends as runs complete

6. **Kill failures**: If run diverges early (NaN within 1000 steps), kill it and move on

7. **Parallelize smartly**: 
   - A6000s: Long Phase 1 runs
   - RTX 4070/5070: Short Phase 0 validation runs

---

## Expected Timeline

| Phase | Sweep | Runs | Hours/Run | Total GPU-Hours | Wall-Clock (4 GPUs) |
|-------|-------|------|-----------|-----------------|---------------------|
| Phase 0 | Baseline | 6 | 2 | 12 | 3 hours |
| Phase 1 | Focused | 3 | 10 | 30 | 7.5 hours |
| Phase 1 | Full (optional) | 12 | 10 | 120 | 30 hours |

**Recommendation**: Run Phase 0 + Phase 1 Focused = **~10 hours wall-clock time**

---

## Next Steps After Sweeps Complete

1. **Analyze in W&B**: Create comparison tables and plots
2. **Download data**: `wandb export` or use W&B API
3. **Identify best configs**: Pareto frontier of PE time vs val loss
4. **Move to Phase 2**: Use best config for attention stability experiments

Good luck with your experiments! 🚀
