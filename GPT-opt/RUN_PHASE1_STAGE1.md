# Phase 1 Stage 1 Execution Guide

This guide walks through running the Phase 1 Stage 1 sweeps in parallel on two GPUs.

## Overview

Phase 1 Stage 1 consists of two separate sweeps:
- **Safety 1.00**: 27 runs (9 num_iters × 3 cushions)
- **Safety 1.01**: 27 runs (9 num_iters × 3 cushions)

Total: 54 runs at 0.5 epochs each with seed=42

## Prerequisites

1. Make sure you've pushed the latest changes from your Mac:
   ```bash
   # On Mac
   cd ~/Developer/2025/cs182-project
   git add -A
   git commit -m "Add Phase 1 Stage 1 sweeps and single-GPU runner"
   git push
   ```

2. Pull changes on the server:
   ```bash
   # On server
   cd ~/cs182-project
   git pull
   ```

3. Make the new script executable:
   ```bash
   cd GPT-opt
   chmod +x run_sweep_single_gpu.sh
   ```

## Step 1: Create Both Sweeps

```bash
cd ~/cs182-project/GPT-opt

# Create sweep for safety=1.00
wandb sweep sweeps/phase1-pe-stage1-safety100.yaml
# Note the returned SWEEP_ID, e.g.: entity/project/sweep_abc123

# Create sweep for safety=1.01
wandb sweep sweeps/phase1-pe-stage1-safety101.yaml
# Note the returned SWEEP_ID, e.g.: entity/project/sweep_def456
```

## Step 2: Launch Agents on Both GPUs

Open **two separate terminal sessions** to the server.

### Terminal 1 (GPU 0 - Safety 1.00):

```bash
cd ~/cs182-project/GPT-opt
./run_sweep_single_gpu.sh 0 <SWEEP_ID_SAFETY100>
```

Replace `<SWEEP_ID_SAFETY100>` with the actual sweep ID from Step 1.

Example:
```bash
./run_sweep_single_gpu.sh 0 jasontrinh-university-of-california-berkeley/cs182-polar-express/abc123
```

### Terminal 2 (GPU 1 - Safety 1.01):

```bash
cd ~/cs182-project/GPT-opt
./run_sweep_single_gpu.sh 1 <SWEEP_ID_SAFETY101>
```

Replace `<SWEEP_ID_SAFETY101>` with the actual sweep ID from Step 1.

Example:
```bash
./run_sweep_single_gpu.sh 1 jasontrinh-university-of-california-berkeley/cs182-polar-express/def456
```

## Monitoring

Both agents will:
- Run in the foreground (you'll see output)
- Log to `logs/gpu_0.log` and `logs/gpu_1.log`
- Process runs from their respective sweeps

To view logs in another terminal:
```bash
cd ~/cs182-project/GPT-opt
tail -f logs/gpu_0.log logs/gpu_1.log
```

## Expected Timeline

- Each run: ~1.36 hours (0.5 epochs)
- 27 runs per GPU
- Sequential time per GPU: ~36.7 hours
- **Total wall-clock time: ~36.7 hours (~1.5 days)**

## When Complete

After both sweeps finish (all 54 runs):

1. **Analyze results in W&B:**
   - Compare runs by `val/loss`
   - Identify best `polar_safety` (1.00 or 1.01)
   - Identify best `polar_cushion` (0.01, 0.024, or 0.05)
   - Rank the 9 `polar_num_iters` patterns

2. **Select top ~10 configs** for Stage 2

3. **Update Stage 2 template:**
   ```bash
   # Edit sweeps/phase1-pe-stage2-template.yaml
   # - Set best safety value
   # - Set best cushion value
   # - Update num_iters list with top 10 patterns
   ```

4. **Launch Stage 2:**
   ```bash
   wandb sweep sweeps/phase1-pe-stage2-template.yaml
   # Run with both GPUs using regular run_sweep.sh
   CUDA_VISIBLE_DEVICES=0,1 ./run_sweep.sh <STAGE2_SWEEP_ID>
   ```

## Troubleshooting

**If a run fails:**
- Check the log file: `logs/gpu_X.log`
- The agent will automatically pick up the next run
- Failed runs appear in W&B as "crashed"

**To stop an agent:**
- Press `Ctrl+C` in the terminal
- The agent will finish its current run and exit

**To restart an agent:**
- Just run the same `./run_sweep_single_gpu.sh` command again
- It will resume from where it left off
