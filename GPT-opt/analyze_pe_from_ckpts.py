# analyze_pe_from_ckpts.py
# Offline Polar Express SVD analysis with a single W&B table that shows
# how the SAME update evolves across PE iterations (iter curves on one plot).
#
# Usage:
#   python analyze_pe_from_ckpts.py \
#       --ckpt_glob "outputs/phase1/*/ckpt*.pth" \
#       --project your-wandb-project \
#       --run_name pe-offline-analysis \
#       --entity your-wandb-entity \
#       --device cuda
#
# In W&B:
#   - Add a Line Plot with source = pe_offline/iter_curves
#   - X = iter, Y = value, Group = ckpt_step
#   - Filter layer == 11 (or 0/5), part == stacked_qkv, metric == condition_number
#
# Notes:
#   • This script operates on the optimizer state (momentum/exp_avg) stored in checkpoints.
#   • It reconstructs the matrix update, runs PE iterations OFFLINE, and logs SVD metrics
#     per iteration as both (a) scalar summaries and (b) a single consolidated W&B Table.

import argparse
import glob
import os
from typing import Dict, Any, List

import torch
import wandb
import matplotlib.pyplot as plt

from gptopt.optim.polar_express import get_coeffs_for_config


# --------------------------- SVD helpers ---------------------------

@torch.no_grad()
def compute_svd_metrics(matrix: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Return basic SVD metrics for a 2D tensor, names prefixed for scalar logging."""
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        # Fall back to CPU if GPU SVD fails
        matrix_cpu = matrix.cpu()
        U, S, Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
        S = S.to(matrix.device)

    sigma_max = S[0].item()
    sigma_min = S[-1].item()
    cond = sigma_max / (sigma_min + 1e-10)
    eff_rank = (S.sum() / (sigma_max + 1e-10)).item()

    metrics = {
        f"{prefix}/sigma_max": sigma_max,
        f"{prefix}/sigma_min": sigma_min,
        f"{prefix}/condition_number": cond,
        f"{prefix}/effective_rank": eff_rank,
    }
    if len(S) > 1:
        metrics[f"{prefix}/spectral_gap"] = (S[0] / S[1]).item()
    return metrics


@torch.no_grad()
def svd_spectrum(matrix: torch.Tensor) -> torch.Tensor:
    """Return singular values (descending) as a 1D tensor."""
    try:
        S = torch.linalg.svdvals(matrix)
    except RuntimeError:
        S = torch.linalg.svdvals(matrix.cpu()).to(matrix.device)
    return S


@torch.no_grad()
def svd_vals_from_spectrum(S: torch.Tensor) -> Dict[str, float]:
    """Return SVD summary metrics given a singular value spectrum."""
    sigma_max = S[0].item()
    sigma_min = S[-1].item()
    cond = sigma_max / (sigma_min + 1e-10)
    eff_rank = (S.sum() / (sigma_max + 1e-10)).item()

    out = {
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "condition_number": cond,
        "effective_rank": eff_rank,
    }
    if S.numel() > 1:
        out["spectral_gap"] = (S[0] / S[1]).item()
    return out


@torch.no_grad()
def svd_vals(matrix: torch.Tensor) -> Dict[str, float]:
    """Return *unprefixed* SVD metrics (used for table rows)."""
    S = svd_spectrum(matrix)
    return svd_vals_from_spectrum(S)


# --------------------------- PE iteration (offline) ---------------------------

@torch.no_grad()
def run_pe_iterations(G: torch.Tensor, coeffs_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """Run Polar Express iterations offline and return intermediate states X_k.

    We mirror the left-orthogonalizing PE update:
        X_{k+1} = (a_k I + b_k (X_k X_k^T) + c_k (X_k X_k^T)^2) X_k
    We normalize X_0 similarly to the training code and emit [X_0, X_1, ..., X_T].

    Args:
        G: 2D tensor (the raw step/update you want to analyze). If >2D, flatten first.
        coeffs_list: list of (a, b, c) for each PE iteration.

    Returns:
        List[Tensor]: states after each iteration, all in float32 and original orientation.
    """
    assert G.ndim == 2, "run_pe_iterations expects a 2D matrix"

    # Follow training behavior: work in bf16 for matmuls, then convert back for SVD.
    X = G.to(dtype=torch.bfloat16)

    # Ensure we left-orthogonalize; if rows > cols, transpose to favor row Gram shape
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT  # now rows <= cols

    # Normalize Frobenius energy (include small safety like in training)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    states: List[torch.Tensor] = [X.clone()]

    eye_cache = None
    for (a, b, c) in coeffs_list:
        # Build Gram and polynomial
        A = X @ X.mT
        if eye_cache is None or eye_cache.size(0) != A.size(0):
            eye_cache = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        # (a I + b A + c A^2) X
        B = (b * A) + (c * (A @ A))
        X = (a * eye_cache + B) @ X
        states.append(X.clone())

    # Return to original orientation and dtype for SVD stability
    if transposed:
        states = [s.mT for s in states]
    states = [s.to(dtype=torch.float32) for s in states]
    return states


# --------------------------- W&B table utilities ---------------------------

def make_series_table() -> "wandb.sdk.data_types.table.Table":
    """Create a single table that collects all (ckpt, layer, part, iter) SVD points."""
    return wandb.Table(columns=["ckpt_step", "layer", "part", "metric", "iter", "value"])


def add_series_rows(
    table: "wandb.sdk.data_types.table.Table",
    ckpt_step: int,
    layer_id: int,
    part: str,
    iter_idx: int,
    metrics: Dict[str, float],
) -> None:
    """Append rows for each metric into the consolidated table."""
    if table is None:
        return
    for metric_name, val in metrics.items():
        table.add_data(ckpt_step, layer_id, part, metric_name, iter_idx, float(val))


def add_curve_points(
    curve_acc,
    ckpt_step: int,
    layer_id: int,
    part: str,
    iter_idx: int,
    metrics: Dict[str, float],
) -> None:
    """Accumulate per-(layer, part, metric) curves over iterations and checkpoints."""
    if curve_acc is None:
        return
    for metric_name, val in metrics.items():
        key = (layer_id, part, metric_name)
        by_ckpt = curve_acc.setdefault(key, {})
        series = by_ckpt.setdefault(ckpt_step, [])
        series.append(float(val))


# --------------------------- Main analysis per checkpoint ---------------------------

@torch.no_grad()
def analyze_checkpoint(
    path: str,
    device: str,
    series_table: "wandb.sdk.data_types.table.Table",
    curve_acc=None,
) -> Dict[str, Any]:
    """Load a checkpoint and compute offline PE SVD metrics for layers 0,5,11.

    Operates on optimizer_state_dict (no live optimizer reconstruction).
    Expects Muon to store:
      state[pid]['param_name'], state[pid]['is_W_QKV'], state[pid]['use_muon'],
      state[pid]['momentum_buffer'] (or equivalent update tensor).

    Logs:
      • Raw stacked update SVD (scalar metrics)
      • Per-iteration SVD for stacked QKV and split Q/K/V into a single W&B table
    """
    ckpt = torch.load(path, map_location=device)
    step = ckpt.get("step", None)
    ckpt_step = int(step) if step is not None else -1
    opt_state = ckpt["optimizer_state_dict"]

    state = opt_state["state"]
    param_groups = opt_state["param_groups"]

    # Map param id -> its optimizer param_group dict
    group_for_param: Dict[int, Dict[str, Any]] = {}
    for group in param_groups:
        for pid in group["params"]:
            group_for_param[pid] = group

    target_layers = [0, 5, 11]
    scalars: Dict[str, float] = {}

    for pid, s in state.items():
        if not s.get("use_muon", False):
            continue
        if not s.get("is_W_QKV", False):
            continue

        name = s.get("param_name", "")
        buf = s.get("momentum_buffer", None)
        if buf is None:
            continue

        # Identify attention layer index
        layer_id = None
        for L in target_layers:
            if f"h.{L}.attn.c_attn.weight" in name:
                layer_id = L
                break
        if layer_id is None:
            continue

        group = group_for_param.get(pid, {})
        polar_method   = group.get("polar_method", "polarexpress")
        polar_num_iters = group.get("polar_num_iters", None)
        polar_safety   = group.get("polar_safety", 1.01)
        polar_cushion  = group.get("polar_cushion", 0.024)

        if polar_method != "polarexpress":
            # analyze PE runs only
            continue

        # Prepare update matrix G on device (2D)
        G = buf.to(device)
        if G.ndim > 2:
            G = G.view(G.size(0), -1)

        # (1) Raw stacked update SVD (scalar logs)
        raw_prefix = f"pe_offline/layer{layer_id}_stacked_qkv/raw"
        scalars.update(compute_svd_metrics(G, raw_prefix))

        # (2) PE coefficients for this configuration
        coeffs_lists = get_coeffs_for_config(
            num_iters=polar_num_iters,
            safety=polar_safety,
            cushion=polar_cushion,
        )
        coeffs = coeffs_lists[0]

        # (3) Run PE iterations offline and append to the unified table and curve accumulator
        states = run_pe_iterations(G, coeffs)

        stacked_spectra = []
        q_spectra = []
        k_spectra = []
        v_spectra = []

        for iter_idx, X in enumerate(states):
            # Stacked QKV
            metrics_stacked = svd_vals(X)
            add_series_rows(series_table, ckpt_step, layer_id, "stacked_qkv", iter_idx, metrics_stacked)
            add_curve_points(curve_acc, ckpt_step, layer_id, "stacked_qkv", iter_idx, metrics_stacked)
            stacked_spectra.append(svd_spectrum(X))

            # Split Q/K/V (common GPT-2 shape: rows divisible by 3)
            if X.ndim == 2 and X.size(0) % 3 == 0:
                rows = X.size(0) // 3
                Q, K, V = X[0:rows, :], X[rows:2*rows, :], X[2*rows:3*rows, :]

                metrics_q = svd_vals(Q)
                add_series_rows(series_table, ckpt_step, layer_id, "q", iter_idx, metrics_q)
                add_curve_points(curve_acc, ckpt_step, layer_id, "q", iter_idx, metrics_q)
                q_spectra.append(svd_spectrum(Q))

                metrics_k = svd_vals(K)
                add_series_rows(series_table, ckpt_step, layer_id, "k", iter_idx, metrics_k)
                add_curve_points(curve_acc, ckpt_step, layer_id, "k", iter_idx, metrics_k)
                k_spectra.append(svd_spectrum(K))

                metrics_v = svd_vals(V)
                add_series_rows(series_table, ckpt_step, layer_id, "v", iter_idx, metrics_v)
                add_curve_points(curve_acc, ckpt_step, layer_id, "v", iter_idx, metrics_v)
                v_spectra.append(svd_spectrum(V))

        if stacked_spectra:
            spec_t = torch.stack([s.cpu() for s in stacked_spectra], dim=0)
            spec_log = torch.log10(spec_t + 1e-12)
            fig, ax = plt.subplots()
            im = ax.imshow(spec_log.numpy(), aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xlabel("singular_index")
            ax.set_ylabel("pe_iter")
            ax.set_title(f"log10 SV spectrum vs PE iter - ckpt{ckpt_step} layer{layer_id} stacked_qkv")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("log10(sigma)")
            wandb.log({f"pe_offline/heatmap/ckpt{ckpt_step}/layer{layer_id}_stacked_qkv": wandb.Image(fig)}, step=ckpt_step)
            plt.close(fig)

        if q_spectra:
            spec_t = torch.stack([s.cpu() for s in q_spectra], dim=0)
            spec_log = torch.log10(spec_t + 1e-12)
            fig, ax = plt.subplots()
            im = ax.imshow(spec_log.numpy(), aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xlabel("singular_index")
            ax.set_ylabel("pe_iter")
            ax.set_title(f"log10 SV spectrum vs PE iter - ckpt{ckpt_step} layer{layer_id} q")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("log10(sigma)")
            wandb.log({f"pe_offline/heatmap/ckpt{ckpt_step}/layer{layer_id}_q": wandb.Image(fig)}, step=ckpt_step)
            plt.close(fig)

        if k_spectra:
            spec_t = torch.stack([s.cpu() for s in k_spectra], dim=0)
            spec_log = torch.log10(spec_t + 1e-12)
            fig, ax = plt.subplots()
            im = ax.imshow(spec_log.numpy(), aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xlabel("singular_index")
            ax.set_ylabel("pe_iter")
            ax.set_title(f"log10 SV spectrum vs PE iter - ckpt{ckpt_step} layer{layer_id} k")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("log10(sigma)")
            wandb.log({f"pe_offline/heatmap/ckpt{ckpt_step}/layer{layer_id}_k": wandb.Image(fig)}, step=ckpt_step)
            plt.close(fig)

        if v_spectra:
            spec_t = torch.stack([s.cpu() for s in v_spectra], dim=0)
            spec_log = torch.log10(spec_t + 1e-12)
            fig, ax = plt.subplots()
            im = ax.imshow(spec_log.numpy(), aspect="auto", origin="lower", interpolation="nearest")
            ax.set_xlabel("singular_index")
            ax.set_ylabel("pe_iter")
            ax.set_title(f"log10 SV spectrum vs PE iter - ckpt{ckpt_step} layer{layer_id} v")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("log10(sigma)")
            wandb.log({f"pe_offline/heatmap/ckpt{ckpt_step}/layer{layer_id}_v": wandb.Image(fig)}, step=ckpt_step)
            plt.close(fig)

    if step is not None:
        scalars["pe_offline/ckpt_step"] = float(step)
    scalars["pe_offline/ckpt_path"] = os.path.basename(path)
    return scalars


# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline PolarExpress SVD analysis from checkpoints")
    parser.add_argument("--ckpt_glob", type=str, required=True,
                        help="Glob for checkpoint files, e.g. 'outputs/phase1/*/ckpt*.pth'")
    parser.add_argument("--project", type=str, required=True,
                        help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default="pe-offline-analysis",
                        help="W&B run name")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run analysis on (cuda or cpu)")
    parser.add_argument("--plot_dir", type=str, default="pe_offline_plots",
                        help="Directory to save Matplotlib PE iteration plots")
    args = parser.parse_args()

    ckpt_paths = sorted(glob.glob(args.ckpt_glob))
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints matched glob: {args.ckpt_glob}")

    wandb.init(project=args.project, entity=args.entity, name=args.run_name,
               config={"ckpt_glob": args.ckpt_glob})

    # We only log heatmaps from inside analyze_checkpoint; no scalars or tables here.
    series_table = None

    for idx, path in enumerate(ckpt_paths):
        scalars = analyze_checkpoint(path, device=args.device, series_table=series_table)
        step = int(scalars.get("pe_offline/ckpt_step", idx))
        print(f"[analyze_pe_from_ckpts] Analyzed {path} (step={step})")

    wandb.finish()


if __name__ == "__main__":
    main()
