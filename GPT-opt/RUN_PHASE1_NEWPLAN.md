# Phase 1 Execution Guide (Safety-First Strategy)

This guide walks through the new two-stage Phase 1 sweep strategy.

## Strategy Overview

**Stage 1 (Safety Sweep):**
- Fix standard PE params: `num_iters=[5]`, `cushion=0.024`
- Sweep `safety ∈ {1.00, 1.01}`
- Full evaluation: 3 seeds × 1 epoch
- **Total: 6 runs (~8 hours on 2 GPUs)**

**Stage 2 (Full Hyperparameter Sweep):**
- Fix best safety from Stage 1
- Sweep `cushion ∈ {0.01, 0.024, 0.05}` × 9 `num_iters` patterns
- Full evaluation: 3 seeds × 1 epoch
- **Total: 81 runs (~4.5 days on 2 GPUs)**

**Combined total: 87 runs, ~5 days on 2 GPUs**

## Prerequisites

1. Make sure you've pushed the latest changes from your Mac:
   ```bash
   # On Mac
   cd ~/Developer/2025/cs182-project
   git add -A
   git commit -m "Add Phase 1 safety-first sweep strategy"
   git push
   ```

2. Pull changes on the server:
   ```bash
   # On server
   cd ~/cs182-project
   git pull
   ```

3. Verify scripts are executable:
   ```bash
   cd GPT-opt
   chmod +x run_sweep.sh run_sweep_single_gpu.sh
   ```

---

## Stage 1: Safety Factor Sweep

### Step 1: Create the Sweep

```bash
cd ~/cs182-project/GPT-opt
wandb sweep sweeps/phase1-safety-sweep.yaml
```

Note the returned `SWEEP_ID`, e.g.:
```
wandb: Created sweep with ID: abc123
wandb: View sweep at: https://wandb.ai/entity/cs182-polar-express/sweeps/abc123
```

### Step 2: Launch Agents on Both GPUs

The sweep has 6 runs, so each GPU will process ~3 runs:

```bash
cd ~/cs182-project/GPT-opt
CUDA_VISIBLE_DEVICES=0,1 ./run_sweep.sh <SWEEP_ID>
```

Replace `<SWEEP_ID>` with the full path, e.g.:
```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_sweep.sh jasontrinh-university-of-california-berkeley/cs182-polar-express/abc123
```

### Step 3: Monitor Progress

The script will tail both GPU logs. In another terminal, you can also check:

```bash
cd ~/cs182-project/GPT-opt
tail -f logs/gpu_0.log logs/gpu_1.log
```

Or view real-time progress in W&B:
```
https://wandb.ai/<ENTITY>/cs182-polar-express/sweeps/<SWEEP_ID>
```

### Expected Timeline for Stage 1

- 6 runs total
- ~2.7 hours per run
- 2 GPUs processing in parallel
- **Total: ~8 hours**

---

## Stage 1 Analysis

After all 6 runs complete:

### 1. Compare Safety Factors in W&B

Go to your sweep page and group runs by `config.optimizer_params.args.polar_safety`:

- Look at `val/loss` (lower is better)
- Check if one safety factor is consistently better across all 3 seeds
- Note any stability issues (crashes, NaN losses)

### 2. Select Best Safety

Based on the analysis, choose the safety value (1.00 or 1.01) with:
- Lower average `val/loss` across seeds
- Better stability (fewer crashes/NaNs)
- Tighter variance across seeds

Example W&B analysis query:
```python
# In W&B workspace
runs = api.runs(f"{entity}/{project}")
safety_results = {}
for run in runs:
    if run.state == "finished":
        safety = run.config["optimizer_params"]["args"]["polar_safety"]
        val_loss = run.summary.get("val/loss")
        if safety not in safety_results:
            safety_results[safety] = []
        safety_results[safety].append(val_loss)

# Compare averages
for safety, losses in safety_results.items():
    print(f"Safety {safety}: mean={np.mean(losses):.4f}, std={np.std(losses):.4f}")
```

---

## Stage 2: Full Hyperparameter Sweep

### Step 1: Update the Template

Edit `sweeps/phase1-pe-stage2-template.yaml`:

```bash
nano sweeps/phase1-pe-stage2-template.yaml
```

Find line 60 and update with the best safety from Stage 1:

```yaml
# Line 60 - UPDATE THIS
optimizer_params.args.polar_safety:
  value: 1.01    # Change to 1.00 or 1.01 based on Stage 1 results
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 2: Create the Sweep

```bash
cd ~/cs182-project/GPT-opt
wandb sweep sweeps/phase1-pe-stage2-template.yaml
```

Note the returned `SWEEP_ID`.

### Step 3: Launch Agents on Both GPUs

```bash
cd ~/cs182-project/GPT-opt
CUDA_VISIBLE_DEVICES=0,1 ./run_sweep.sh <SWEEP_ID>
```

### Expected Timeline for Stage 2

- 81 runs total (9 num_iters × 3 cushions × 3 seeds)
- ~2.7 hours per run
- 2 GPUs processing in parallel (~40-41 runs each)
- **Total: ~109 hours (~4.5 days)**

---

## Final Analysis (After Stage 2)

After all 81 runs complete:

### 1. Compare Configurations

Group runs by:
- `config.optimizer_params.args.polar_num_iters`
- `config.optimizer_params.args.polar_cushion`

### 2. Key Questions to Answer

**Quality:**
- Does higher `num_iters` improve convergence? (3 vs 5 vs 7)
- Which cushion value works best? (0.01 vs 0.024 vs 0.05)

**Efficiency:**
- Can cycling patterns reduce compute without hurting convergence?
  - `[5,0]` vs `[5]`: 50% cost reduction
  - `[5,3]` vs `[5]`: 20% cost reduction
  - `[7,0]` vs `[7]`: 50% cost reduction
  - `[7,5]` vs `[7]`: 15% cost reduction

**Optimal Configuration:**
- What's the best `{num_iters, cushion}` combo?
- Is there a clear winner or multiple good options?

### 3. Export Results

```python
import wandb
api = wandb.Api()

# Get all Stage 2 runs
runs = api.runs(
    f"{entity}/cs182-polar-express",
    filters={"tags": "phase1", "state": "finished"}
)

# Export to CSV for analysis
import pandas as pd
data = []
for run in runs:
    data.append({
        "safety": run.config["optimizer_params"]["args"]["polar_safety"],
        "num_iters": run.config["optimizer_params"]["args"]["polar_num_iters"],
        "cushion": run.config["optimizer_params"]["args"]["polar_cushion"],
        "seed": run.config["seed"],
        "val_loss": run.summary.get("val/loss"),
        "train_loss": run.summary.get("train/loss"),
    })
df = pd.DataFrame(data)
df.to_csv("phase1_results.csv", index=False)
```

---

## Troubleshooting

**If a run fails:**
- Check the log file: `logs/gpu_X.log`
- The agent will automatically pick up the next run
- Failed runs appear in W&B as "crashed"

**To stop agents:**
- Press `Ctrl+C` in the terminal running `run_sweep.sh`
- Both agents will finish their current runs and exit

**To restart agents:**
- Just run the same `./run_sweep.sh <SWEEP_ID>` command again
- Agents will resume from where they left off

**If you need to kill a run immediately:**
```bash
# Find the process
ps aux | grep wandb

# Kill it
kill -9 <PID>
```

---

## Summary of Compute

| Stage | Runs | Time per Run | Wall-Clock (2 GPUs) | Epoch-Equivalents |
|-------|------|--------------|---------------------|-------------------|
| Stage 1 | 6 | 2.7h | ~8h | 6 |
| Stage 2 | 81 | 2.7h | ~109h (~4.5 days) | 81 |
| **Total** | **87** | - | **~5 days** | **87** |

This is ~1.5× more compute than the previous 0.5-epoch staged plan, but gives you:
- Very strong statistical evidence for best safety (6 full-epoch runs)
- Complete coverage of {cushion, num_iters} space at that safety (81 full-epoch runs)
- All results at full 1 epoch with 3 seeds for direct comparison
