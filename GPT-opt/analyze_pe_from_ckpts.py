import argparse
import glob
import os
from typing import Dict, Any, List

import torch
import wandb

from gptopt.optim.polar_express import get_coeffs_for_config


@torch.no_grad()
def compute_svd_metrics(matrix: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Return basic SVD metrics for a 2D tensor."""
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
def run_pe_iterations(G: torch.Tensor, coeffs_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """Re-implement PolarExpress iteration to expose per-iteration states.

    This mirrors gptopt.optim.polar_express.PolarExpress but returns a list of
    intermediate matrices after each iteration (including the normalized
    "iteration 0" state). This is for offline analysis only.
    """
    assert G.ndim >= 2

    # Cast and normalize as in PolarExpress
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    states: List[torch.Tensor] = [X.clone()]  # iter 0 (normalized input)

    for a, b, c in coeffs_list:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        states.append(X.clone())

    # Match output orientation with original G
    if transposed:
        states = [s.mT for s in states]
    # Convert back to float32 for SVD stability
    states = [s.to(dtype=torch.float32) for s in states]
    return states


def analyze_checkpoint(path: str, device: str = "cuda") -> List[Dict[str, Any]]:
    """Load a checkpoint and compute offline PE SVD metrics for layers 0,5,11.

    This operates **purely on the optimizer state_dict** using metadata stored by Muon:
    - state[pid]['param_name']
    - state[pid]['is_W_QKV']
    - state[pid]['use_muon']
    - state[pid]['momentum_buffer']

    It does **not** reconstruct a live optimizer; instead it pulls the momentum buffers
    and re-applies PolarExpress offline.

    Compared to the original version, this function now returns a list of metric
    dictionaries, one per PolarExpress iteration, so that W&B can plot a single
    time-series per metric with iteration index as the x-axis.
    """
    ckpt = torch.load(path, map_location=device)
    step = ckpt.get("step", None)
    opt_state = ckpt["optimizer_state_dict"]

    state = opt_state["state"]
    param_groups = opt_state["param_groups"]

    # Map param id -> param_group dict
    group_for_param: Dict[int, Dict[str, Any]] = {}
    for group in param_groups:
        for pid in group["params"]:
            group_for_param[pid] = group

    target_layers = [0, 5, 11]

    # One metrics dict per PE iteration (including iteration 0 normalized input)
    iter_metrics: List[Dict[str, Any]] = []

    for pid, s in state.items():
        if not s.get("use_muon", False):
            continue
        if not s.get("is_W_QKV", False):
            continue
        name = s.get("param_name", "")
        buf = s.get("momentum_buffer", None)
        if buf is None:
            continue

        # Identify which layer this is
        layer_id = None
        for L in target_layers:
            if f"h.{L}.attn.c_attn.weight" in name:
                layer_id = L
                break
        if layer_id is None:
            continue

        group = group_for_param.get(pid, {})
        ns_steps = group.get("ns_steps", 5)
        polar_method = group.get("polar_method", "polarexpress")
        polar_num_iters = group.get("polar_num_iters", None)
        polar_safety = group.get("polar_safety", 1.01)
        polar_cushion = group.get("polar_cushion", 0.024)

        if polar_method != "polarexpress":
            # Only analyze PE runs in this script.
            continue

        # Move buffer to device and flatten to 2D if needed
        G = buf.to(device)
        if G.ndim > 2:
            G = G.view(G.size(0), -1)

        # 1) SVD of raw stacked update (before normalization) -- stored in iter 0
        raw_metrics = compute_svd_metrics(G, f"pe_offline/layer{layer_id}_stacked_qkv/raw")

        # 2) Build PolarExpress coefficients for this configuration
        coeffs_lists = get_coeffs_for_config(
            num_iters=polar_num_iters,
            safety=polar_safety,
            cushion=polar_cushion,
        )
        coeffs = coeffs_lists[0]

        # 3) Run PE iterations and collect per-iteration SVD (stacked + split Q/K/V)
        states = run_pe_iterations(G, coeffs)

        for iter_idx, X in enumerate(states):
            # Ensure we have a metrics dict for this iteration
            while len(iter_metrics) <= iter_idx:
                iter_metrics.append({})

            base = f"pe_offline/layer{layer_id}"

            # Stacked QKV metrics (no iter suffix; x-axis will be iteration index)
            stacked_prefix = f"{base}/stacked_qkv"
            iter_metrics[iter_idx].update(compute_svd_metrics(X, stacked_prefix))

            # Split Q, K, V when in standard stacked shape (e.g., 2304x768)
            if X.ndim == 2 and X.size(0) % 3 == 0:
                rows = X.size(0) // 3
                Q = X[0:rows, :]
                K = X[rows:2*rows, :]
                V = X[2*rows:3*rows, :]
                iter_metrics[iter_idx].update(compute_svd_metrics(Q, f"{base}/q"))
                iter_metrics[iter_idx].update(compute_svd_metrics(K, f"{base}/k"))
                iter_metrics[iter_idx].update(compute_svd_metrics(V, f"{base}/v"))

            # Attach raw stacked update metrics to iteration 0 for convenience
            if iter_idx == 0:
                iter_metrics[0].update(raw_metrics)

    # Attach checkpoint metadata to every iteration's metrics
    for m in iter_metrics:
        if step is not None:
            m["pe_offline/ckpt_step"] = float(step)
        m["pe_offline/ckpt_path"] = os.path.basename(path)

    return iter_metrics


def main():
    parser = argparse.ArgumentParser(description="Offline PolarExpress SVD analysis from checkpoints")
    parser.add_argument("--ckpt_glob", type=str, required=True,
                        help="Glob for checkpoint files, e.g. 'outputs/phase1/*/ckpt*.pth'")
    parser.add_argument("--project", type=str, required=True,
                        help="wandb project name")
    parser.add_argument("--run_name", type=str, default="pe-offline-analysis",
                        help="wandb run name")
    parser.add_argument("--entity", type=str, default=None,
                        help="wandb entity (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run analysis on (cuda or cpu)")

    args = parser.parse_args()

    ckpt_paths = sorted(glob.glob(args.ckpt_glob))
    if not ckpt_paths:
        raise SystemExit(f"No checkpoints matched glob: {args.ckpt_glob}")

    wandb.init(project=args.project, entity=args.entity, name=args.run_name,
               config={"ckpt_glob": args.ckpt_glob})

    for idx, path in enumerate(ckpt_paths):
        iter_metrics_list = analyze_checkpoint(path, device=args.device)
        if not iter_metrics_list:
            print(f"No matching PE parameters found in checkpoint: {path}")
            continue

        # Log one W&B step per PE iteration so that each metric becomes
        # a time series over iterations. The x-axis is the PE iteration index.
        for iter_idx, metrics in enumerate(iter_metrics_list):
            metrics["pe_offline/iter"] = iter_idx
            wandb.log(metrics, step=iter_idx)

        print(f"Analyzed {path} ({len(iter_metrics_list)} PE iterations)")

    wandb.finish()


if __name__ == "__main__":
    main()
