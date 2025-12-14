# Plotting Guide

Quick steps for reproducing sweeps with Polar Express parameters and generating the plots stored in this branch.

## Prerequisites
- Install the repo requirements (`pip install -r requirements.txt`) and log in to Weights & Biases (`wandb login`).
- The plotting notebook uses the `ENTITY` and `PROJECT` constants at the top of `plotting/main.ipynb`; update them if your W&B namespace differs.

## Run sweeps with Polar parameters
1. From `GPT-opt/`, create the sweep (includes `polar_method=polarexpress`, `polar_safety`, `polar_cushion`, and `polar_num_iters` overrides):
   ```bash
   cd GPT-opt
   wandb sweep sweeps/phase1-cushion-sweep.yaml
   ```
   Note the printed `SWEEP_ID`.
2. Launch agents (single node) that read the YAML overrides, including the Polar parameters:
   ```bash
   ./run_sweep.sh <SWEEP_ID>
   # or: ./run_sweep_single_gpu.sh <SWEEP_ID>
   ```
   Set `DATA_DIR` in the script if your dataset lives elsewhere.
3. For one-off runs instead of a sweep, you can directly override the Polar knobs:
   ```bash
   python run_hydra.py -cn gpt-small \\
     +optimizer_params.args.polar_method=polarexpress \\
     +optimizer_params.args.polar_safety=1.01 \\
     +optimizer_params.args.polar_cushion=0.24 \\
     +optimizer_params.args.polar_num_iters=[5]
   ```

## Generate plots
Work inside `plotting/main.ipynb`.
- (Optional) Run `download_all_plot_data()` to cache W&B data locally under `plotting/cache/` before plotting.
- **Attention entropy grid by optimizer family:** run the cell defining `lr_ranges` / `base_lrs` and call `plot_all_categories_combined(...)` to reproduce the combined figure.
- **Variant-filtered loss figures:** use the `_temporary_variants` context manager section to create the three paper figures (full vs. zoomed loss with different Muon/PE subsets).
- **Benji sweep plots:** the top cells (`plot_data(...)`) reproduce the condition number / effective rank / spectral gap plots from the original `benji-plots` branch.
- If you ran different sweeps, update `ATTN_CATEGORIES` and `VARIANT_SWEEP_IDS` in the notebook to point to your sweep IDs before plotting.

Plots saved via `save_plot=True` land under `plotting/plots/`; cached aggregates stay in `plotting/cache/`.
