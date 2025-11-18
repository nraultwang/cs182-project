import torch
import time 
import contextlib
import torch.distributed as dist
from gptopt.utils import get_worker_info, save_checkpoint, load_checkpoint
import json
import numpy as np

typedict = {"float16":torch.float16, "float32":torch.float32, "bfloat16":torch.bfloat16}

class Logging():

    def __init__(self):
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []



def compute_advanced_metrics(model, optimizer, batch, autocast_ctxt, compute_svd=False):
    """
    Compute comprehensive metrics for monitoring training health.
    
    Categories:
    A) End-to-end: val/ppl (computed separately in eval)
    B) PE-internal: pe/ortho_err, pe/time_ms (computed in optimizer)
    C) Attention health: logits/mean, logits/std, attn/entropy/mean
    D) Light scales: qkv norms, grad norms per subpath
    E) SVD metrics: singular values, condition number, effective rank (if compute_svd=True)
    """
    metrics = {}
    
    with torch.no_grad():
        # E) SVD metrics on sentinel matrices (expensive, only when requested)
        if compute_svd:
            svd_start_time = time.time()
            # Sample 3 representative matrices: first, middle, last
            svd_targets = [
                ('h.0.attn.c_attn.weight', 'layer0'),   # First layer attention
                ('h.5.attn.c_attn.weight', 'layer5'),   # Middle layer attention
                ('h.11.attn.c_attn.weight', 'layer11'), # Last layer attention
            ]
            
            # Helper function to compute SVD metrics
            def compute_svd_metrics(matrix, prefix):
                """Compute SVD metrics for a matrix and store with given prefix."""
                U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
                
                sigma_max = S[0].item()
                sigma_min = S[-1].item()
                condition_number = sigma_max / (sigma_min + 1e-10)
                effective_rank = (S.sum() / (sigma_max + 1e-10)).item()
                
                metrics[f'{prefix}/sigma_max'] = sigma_max
                metrics[f'{prefix}/sigma_min'] = sigma_min
                metrics[f'{prefix}/condition_number'] = condition_number
                metrics[f'{prefix}/effective_rank'] = effective_rank
                
                if len(S) > 1:
                    metrics[f'{prefix}/spectral_gap'] = (S[0] / S[1]).item()
            
            for target_name, layer_label in svd_targets:
                try:
                    for name, param in model.named_parameters():
                        if target_name in name:
                            W = param.data
                            
                            # Reshape if needed
                            if W.ndim > 2:
                                W = W.view(W.size(0), -1)
                            
                            # Split QKV weight matrix into Q, K, V (each 768 × 768)
                            # c_attn.weight is [2304, 768] = [Q; K; V] stacked vertically
                            if W.size(0) == 2304 and W.size(1) == 768:
                                Q = W[0:768, :]      # First 768 rows
                                K = W[768:1536, :]   # Middle 768 rows
                                V = W[1536:2304, :]  # Last 768 rows
                                
                                # Compute SVD for each projection
                                compute_svd_metrics(Q, f'svd/{layer_label}_q')
                                compute_svd_metrics(K, f'svd/{layer_label}_k')
                                compute_svd_metrics(V, f'svd/{layer_label}_v')
                            else:
                                # Fallback: compute on full matrix if not standard QKV shape
                                compute_svd_metrics(W, f'svd/{layer_label}_qkv')
                            
                            break  # Found the target, move to next
                except Exception as e:
                    print(f"Warning: Could not compute SVD for {layer_label}: {e}")
            
            # Also compute SVD on momentum buffers (updates) if using Muon
            # This directly measures what PolarExpress operates on
            if hasattr(optimizer, 'param_groups'):
                try:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if optimizer.state.get(p, {}).get('use_muon', False):
                                # Get the parameter name
                                param_name = None
                                for name, param in model.named_parameters():
                                    if param is p:
                                        param_name = name
                                        break
                                
                                if param_name is None:
                                    continue
                                
                                # Check if this is one of our target layers
                                for target_name, layer_label in svd_targets:
                                    if target_name in param_name:
                                        state = optimizer.state[p]
                                        if 'momentum_buffer' in state:
                                            buf = state['momentum_buffer']
                                            
                                            # Reshape if needed (same as Muon does)
                                            if buf.ndim > 2:
                                                buf = buf.view(buf.size(0), -1)
                                            
                                            # Split momentum buffer into Q, K, V updates
                                            # Note: Even though PE processes stacked (2304×768) when split_heads=False,
                                            # we still want to measure per-projection conditioning for analysis
                                            if buf.size(0) == 2304 and buf.size(1) == 768:
                                                Q_buf = buf[0:768, :]      # Q update
                                                K_buf = buf[768:1536, :]   # K update
                                                V_buf = buf[1536:2304, :]  # V update
                                                
                                                # Compute SVD for each update projection
                                                compute_svd_metrics(Q_buf, f'svd/update_{layer_label}_q')
                                                compute_svd_metrics(K_buf, f'svd/update_{layer_label}_k')
                                                compute_svd_metrics(V_buf, f'svd/update_{layer_label}_v')
                                                
                                                # Also compute on full stacked buffer (what PE actually sees)
                                                compute_svd_metrics(buf, f'svd/update_{layer_label}_stacked')
                                            else:
                                                # Fallback: compute on full buffer
                                                compute_svd_metrics(buf, f'svd/update_{layer_label}')
                                        
                                        break  # Found the target, move to next
                except Exception as e:
                    print(f"Warning: Could not compute SVD for momentum buffers: {e}")
            
            # Compute weight orthogonality: ||W^T W - I||_F
            # This measures if weights maintain orthogonality over training
            def compute_orthogonality(matrix, prefix):
                """Compute orthogonality error for a matrix."""
                # For non-square matrices, compute W^T W (smaller dimension)
                if matrix.size(0) > matrix.size(1):
                    WTW = matrix.T @ matrix  # cols × cols
                else:
                    WTW = matrix @ matrix.T  # rows × rows
                
                I = torch.eye(WTW.size(0), device=WTW.device, dtype=WTW.dtype)
                ortho_err = torch.norm(WTW - I, p='fro').item()
                ortho_err_normalized = ortho_err / WTW.size(0)
                
                metrics[f'{prefix}/err'] = ortho_err
                metrics[f'{prefix}/err_normalized'] = ortho_err_normalized
            
            for target_name, layer_label in svd_targets:
                try:
                    for name, param in model.named_parameters():
                        if target_name in name:
                            W = param.data
                            
                            # Reshape if needed
                            if W.ndim > 2:
                                W = W.view(W.size(0), -1)
                            
                            # Split QKV and compute orthogonality for each
                            if W.size(0) == 2304 and W.size(1) == 768:
                                Q = W[0:768, :]
                                K = W[768:1536, :]
                                V = W[1536:2304, :]
                                
                                compute_orthogonality(Q, f'ortho/{layer_label}_q')
                                compute_orthogonality(K, f'ortho/{layer_label}_k')
                                compute_orthogonality(V, f'ortho/{layer_label}_v')
                            else:
                                # Fallback for non-standard shapes
                                compute_orthogonality(W, f'ortho/{layer_label}')
                            
                            break  # Found the target, move to next
                except Exception as e:
                    print(f"Warning: Could not compute orthogonality for {layer_label}: {e}")
            
            # Log total SVD computation time
            svd_total_time = (time.time() - svd_start_time) * 1000  # Convert to ms
            metrics['svd/total_time_ms'] = svd_total_time
    
    with torch.no_grad():
        # C) Attention health - sample from first, middle, and last layers
        try:
            # Get model reference
            transformer = model.module.transformer if hasattr(model, 'module') else model.transformer
            
            # Adapt layer selection based on model depth
            n_layers = len(transformer.h)
            if n_layers >= 12:
                layers_to_check = [0, 5, 11]
                layer_labels = ['layer0', 'layer5', 'layer11']
            elif n_layers >= 6:
                layers_to_check = [0, n_layers // 2, n_layers - 1]
                layer_labels = ['layer0', f'layer{n_layers // 2}', f'layer{n_layers - 1}']
            else:
                layers_to_check = [0, n_layers - 1]
                layer_labels = ['layer0', f'layer{n_layers - 1}']
            
            # Compute embeddings once
            x = batch[0]
            tok_emb = transformer.wte(x)
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
            pos_emb = transformer.wpe(pos)
            x_current = tok_emb + pos_emb
            
            # Create causal mask once (reused for all layers)
            T = x.size(1)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            
            # Forward through layers sequentially, extracting metrics at checkpoints
            prev_checkpoint = -1
            for layer_idx, label in zip(layers_to_check, layer_labels):
                # Forward from previous checkpoint to current checkpoint
                for i in range(prev_checkpoint + 1, layer_idx):
                    x_current = transformer.h[i](x_current)
                
                # At checkpoint layer, extract attention metrics before forwarding
                block = transformer.h[layer_idx]
                x_ln = block.ln_1(x_current)
                
                # Get attention logits (pre-softmax)
                B, T, C = x_ln.size()
                qkv = block.attn.c_attn(x_ln)
                q, k, v = qkv.split(block.attn.n_embd, dim=2)
                k = k.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
                q = q.view(B, T, block.attn.n_head, C // block.attn.n_head).transpose(1, 2)
                
                att_logits = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
                
                # Mask and compute attention probabilities
                att_logits_masked = att_logits.masked_fill(causal_mask == 0, float('-inf'))
                att_probs = torch.softmax(att_logits_masked, dim=-1)
                
                # Logits statistics (pre-softmax, unmasked for health check)
                valid_logits = att_logits[causal_mask.expand_as(att_logits) == 1]
                metrics[f'logits/{label}/mean'] = valid_logits.mean().item()
                metrics[f'logits/{label}/std'] = valid_logits.std().item()
                # Subsample if tensor is too large for quantile computation
                if valid_logits.numel() > 1_000_000:
                    indices = torch.randperm(valid_logits.numel(), device=valid_logits.device)[:1_000_000]
                    sampled_logits = valid_logits[indices]
                else:
                    sampled_logits = valid_logits
                metrics[f'logits/{label}/max_p95'] = torch.quantile(sampled_logits, 0.95).item()
                
                # Attention entropy (post-softmax)
                valid_probs = att_probs[torch.isfinite(att_probs).all(dim=-1)]
                if len(valid_probs) > 0:
                    entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum(dim=-1)
                    metrics[f'attn/{label}/entropy/mean'] = entropy.mean().item()
                    # Subsample if tensor is too large for quantile computation
                    if entropy.numel() > 1_000_000:
                        indices = torch.randperm(entropy.numel(), device=entropy.device)[:1_000_000]
                        sampled_entropy = entropy[indices]
                    else:
                        sampled_entropy = entropy
                    metrics[f'attn/{label}/entropy/p05'] = torch.quantile(sampled_entropy, 0.05).item()
                    metrics[f'attn/{label}/entropy/p95'] = torch.quantile(sampled_entropy, 0.95).item()
                
                # Max attention fraction (attention collapse detector)
                max_attn = att_probs.max(dim=-1)[0]
                metrics[f'attn/{label}/maxA/frac>0.95'] = (max_attn > 0.95).float().mean().item()
                
                # Complete forward through this checkpoint layer
                x_current = block(x_current)
                prev_checkpoint = layer_idx
            
        except Exception as e:
            print(f"Warning: Could not compute attention metrics: {e}")
        
        # D) Light scales - QKV norms and weight scale tracking
        try:
            # QKV norms from layers 0, 5, 11
            for layer_idx in [0, 5, 11]:
                for name, param in model.named_parameters():
                    if f'h.{layer_idx}.attn.c_attn.weight' in name:
                        # Split into Q, K, V (assuming concatenated)
                        n_embd = param.shape[1]
                        q_w, k_w, v_w = param.chunk(3, dim=0)
                        metrics[f'qkv/layer{layer_idx}/q_norm/mean'] = q_w.norm(dim=1).mean().item()
                        metrics[f'qkv/layer{layer_idx}/k_norm/mean'] = k_w.norm(dim=1).mean().item()
                        metrics[f'qkv/layer{layer_idx}/v_norm/mean'] = v_w.norm(dim=1).mean().item()
                        break
            
            # Weight scale tracking (sample 3 layers: first, middle, last)
            for layer_idx in [0, 5, 11]:
                for name, param in model.named_parameters():
                    if f'h.{layer_idx}.attn.c_attn.weight' in name:
                        metrics[f'weights/layer{layer_idx}_attn_norm'] = param.data.norm().item()
                    elif f'h.{layer_idx}.mlp.c_fc.weight' in name:
                        metrics[f'weights/layer{layer_idx}_mlp_norm'] = param.data.norm().item()
        except Exception as e:
            print(f"Warning: Could not compute weight norms: {e}")
        
        # Gradient norms per subpath (W0=MLP up, WK/WV/WQ=attn)
        # Sample first 3 layers for efficiency
        try:
            mlp_up_grads, attn_grads = [], []
            layer_grads = {}  # Track per-layer gradient norms
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Extract layer number
                    for i in range(12):  # GPT-2 Small has 12 layers
                        if f'h.{i}.' in name:
                            # Track per-layer gradient flow (sample layers 0, 5, 11)
                            if i in [0, 5, 11]:
                                if i not in layer_grads:
                                    layer_grads[i] = []
                                layer_grads[i].append(param.grad.norm().item())
                            
                            # Subpath metrics (only first 3 layers for efficiency)
                            if i < 3:
                                if 'mlp.c_fc' in name:
                                    mlp_up_grads.append(param.grad.norm().item())
                                elif 'c_attn' in name:
                                    attn_grads.append(param.grad.norm().item())
                            break
            
            # Per-layer gradient flow (detects vanishing/exploding gradients)
            for layer_idx, grads in layer_grads.items():
                if grads:
                    metrics[f'grads/layer{layer_idx}_norm'] = np.mean(grads)
            
            # Subpath metrics
            if mlp_up_grads:
                metrics['grads/W0_norm'] = np.mean(mlp_up_grads)
            if attn_grads:
                # Use same value for all attention subpaths (QKV combined)
                avg_attn = np.mean(attn_grads)
                metrics['grads/WQ_norm'] = avg_attn
                metrics['grads/WK_norm'] = avg_attn
                metrics['grads/WV_norm'] = avg_attn
        except Exception as e:
            print(f"Warning: Could not compute gradient norms: {e}")
    
    return metrics


def eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt):

    world_size, rank, local_rank, device  = get_worker_info()
    model.eval()
    val_loss, counter = 0., 0
    with torch.no_grad():
        for batch in val_dataloader:
            with autocast_ctxt:
                val_loss += model(batch[0], batch[1], return_logits=False)[1]
            counter += 1
            if (val_accum_steps != 0) & (counter >= val_accum_steps): break
    val_loss = val_loss.detach().clone() / counter
    if world_size > 1: dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(f"Validation Loss: {val_loss.item()}")
    model.train()
    return val_loss


def train(train_dataloader, val_dataloader, model, optimizer, training_params, logging_params, scheduler=None, ckpt_dir="", wandb_run=None):
    
    world_size, rank, local_rank, device  = get_worker_info()
    master_process = (rank == 0)
    logger = Logging()
    optimizer_name = optimizer.__class__.__name__
    if 'Momo' in optimizer_name:
        pass_loss = True
    else:
        pass_loss = False
    if master_process: print(f"Set pass_loss to {pass_loss} for optimizer {optimizer_name}")

    # Track total training time for this run
    training_start_time = time.time()
    
    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    
    # Calculate total dataset size in tokens
    # len(train_dataloader) returns the total number of tokens in the dataset
    dataset_size_tokens = len(train_dataloader)
    if master_process: print(f"Dataset size: {dataset_size_tokens / 1e9:.2f}B tokens")
    
    # Calculate optimizer steps for the requested number of epochs
    num_epochs = training_params['num_epochs']
    tokens_to_process = int(num_epochs * dataset_size_tokens)
    total_iterations = tokens_to_process // training_params['tokens_processed']
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')
    
    # Support fractional epochs by calculating max micro-steps
    max_optimizer_steps = total_iterations
    max_microsteps = max_optimizer_steps * grad_accum_steps
    if master_process: print(f"Training for {num_epochs} epochs = {max_optimizer_steps} optimizer steps = {max_microsteps} micro-steps")

    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=None)
    if ckpt_dir == "":
        print("Will not save checkpoints as no directory is specified")
    
    # Training loop (supports fractional epochs)
    global_microstep = 0
    for epoch in range(int(num_epochs) + 1):  # +1 to handle fractional part
        if global_microstep >= max_microsteps:
            break
        if master_process:
            print(f"Epoch {epoch+1} (up to {num_epochs:.2f} total)")

        model.train()
        start_epoch = time.time()
        start_time = time.time() 
        loss_accum = 0.
        step = 1 if load_ckpt_step == 0 else int(load_ckpt_step)
        optimizer.zero_grad()
        if step != 1: print(train_dataloader.get_state())
        
        for batch in train_dataloader:            
            with autocast_ctxt:
                loss = model(batch[0], batch[1], return_logits=False)[1]
                loss /= grad_accum_steps
            loss_accum += loss.detach()
                
            # Check if accummulated enough gradients to take a step
            if step % grad_accum_steps != 0:
                with (model.no_sync() if world_size > 1 else contextlib.nullcontext()):
                    loss.backward()
            else:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if world_size > 1: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                if pass_loss:
                    optimizer.step(closure=None, loss=loss_accum)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                    
                #bookkeeping
                torch.cuda.synchronize()
                step_time = time.time() - start_time
                
                # Compute tokens per second
                tps = training_params["tokens_processed"] / step_time
                
                # Separate logging frequencies for different metric types
                log_step = logging_params.get('log_step', 160)
                diag_log_step = logging_params.get('diag_log_step', 160)  # PE, attention, scales (default: every 10 steps)
                svd_log_step = logging_params.get('svd_log_step', 800)    # SVD only (default: every 50 steps)
                
                # A) End-to-end metrics (at log_step frequency)
                if master_process and wandb_run is not None and (step % log_step == 0):
                    wandb_log_dict = {
                        "train/loss": loss_accum.item(), 
                        "train/grad_norm": norm.item(),
                        "train/step_time_ms": step_time * 1000,
                        "train/step": step,
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/tokens_per_sec": tps,
                    }
                    
                    # Check for NaN/Inf (numeric stability alarms)
                    if not torch.isfinite(loss_accum):
                        wandb_log_dict["train/naninf_flag"] = 1.0
                    else:
                        wandb_log_dict["train/naninf_flag"] = 0.0
                    
                    # Amplitude check (at log_step frequency or if NaN detected for immediate alarm)
                    if step % log_step == 0 or wandb_log_dict["train/naninf_flag"] == 1.0:
                        grad_list = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
                        if grad_list:  # Only compute if gradients exist
                            max_grad = max(grad_list)
                            wandb_log_dict["train/amp_scaler"] = max_grad
                            wandb_log_dict["train/amp_overflows"] = 1.0 if max_grad > 1e4 else 0.0
                    
                    if hasattr(optimizer, 'step_size_list'):
                        wandb_log_dict["train/step_size_list"] = optimizer.step_size_list
                    for param_group_ix, param_group in enumerate(optimizer.param_groups):
                        wandb_log_dict[f"train/lr_{param_group_ix}"] = param_group['lr']
                    
                    wandb_run.log(wandb_log_dict)
                
                # B) PE-internal metrics + C) Attention health + D) Weight/Scale metrics (at diag_log_step frequency)
                if master_process and wandb_run is not None and (step % diag_log_step == 0):
                    wandb_diag_dict = {}
                    
                    # PE metrics
                    if hasattr(optimizer, '_pe_ortho_errs_before') and len(optimizer._pe_ortho_errs_before) > 0:
                        wandb_diag_dict["pe/ortho_err_before"] = np.mean(optimizer._pe_ortho_errs_before[-10:])
                    if hasattr(optimizer, '_pe_ortho_errs_after') and len(optimizer._pe_ortho_errs_after) > 0:
                        wandb_diag_dict["pe/ortho_err_after"] = np.mean(optimizer._pe_ortho_errs_after[-10:])
                    if hasattr(optimizer, '_pe_times') and len(optimizer._pe_times) > 0:
                        # pe/time_ms reflects the average PE iteration time over the most recent window
                        wandb_diag_dict["pe/time_ms"] = np.mean(optimizer._pe_times[-10:])
                    
                    # Per-layer ortho errors for sentinel layers (0, 5, 11) in stacked mode
                    if hasattr(optimizer, '_pe_ortho_errs_before_per_layer'):
                        for layer_key, errs in optimizer._pe_ortho_errs_before_per_layer.items():
                            if len(errs) > 0:
                                wandb_diag_dict[f"ortho_err_before/{layer_key}_stacked_qkv"] = np.mean(errs[-10:])
                    if hasattr(optimizer, '_pe_ortho_errs_after_per_layer'):
                        for layer_key, errs in optimizer._pe_ortho_errs_after_per_layer.items():
                            if len(errs) > 0:
                                wandb_diag_dict[f"ortho_err_after/{layer_key}_stacked_qkv"] = np.mean(errs[-10:])
                    
                    # Attention health and weight scales (compute_svd=False to skip expensive SVD)
                    try:
                        advanced_metrics = compute_advanced_metrics(model, optimizer, batch, autocast_ctxt, compute_svd=False)
                        wandb_diag_dict.update(advanced_metrics)
                    except Exception as e:
                        print(f"Warning: Could not compute diagnostic metrics at step {step}: {e}")
                    
                    if wandb_diag_dict:
                        wandb_run.log(wandb_diag_dict)
                
                # E) SVD-based metrics (at svd_log_step frequency - expensive, keep at 50 steps)
                # NOTE: We explicitly ignore any PE timings that may occur inside SVD diagnostics,
                # so that pe/time_ms only reflects the cost of the PE call during the main optimizer step.
                if master_process and wandb_run is not None and (step % svd_log_step == 0):
                    # Cache current length of PE timing list (if present)
                    pe_times_len_before = None
                    if hasattr(optimizer, "_pe_times") and optimizer._pe_times is not None:
                        pe_times_len_before = len(optimizer._pe_times)

                    try:
                        # Expensive SVD computation
                        advanced_metrics = compute_advanced_metrics(model, optimizer, batch, autocast_ctxt, compute_svd=True)
                        wandb_run.log(advanced_metrics)
                    except Exception as e:
                        print(f"Warning: Could not compute SVD metrics at step {step}: {e}")
                    finally:
                        # Remove any PE timing entries that may have been added during SVD diagnostics
                        if pe_times_len_before is not None and hasattr(optimizer, "_pe_times") and optimizer._pe_times is not None:
                            optimizer._pe_times = optimizer._pe_times[:pe_times_len_before]
                logger.step_times.append(step_time)  # Are these different across ranks?
                logger.grad_norms.append(norm.item())
                for param_group in optimizer.param_groups:
                    logger.learning_rates.append(param_group['lr'])
                logger.losses.append(loss_accum.item())
                if hasattr(optimizer, 'step_size_list'):  
                    logger.step_size_list = optimizer.step_size_list  
                
                if (step % logging_params['log_step'] == 0) & master_process:
                    print(f"Step {step} of {total_iterations*grad_accum_steps}.")
                    print(f"Time taken : {step_time*1000:0.1f}ms | Tokens/s : {tps/1000:0.1f}k | Loss : {loss_accum.item():0.3f}")
                    
                if (step % logging_params['val_step'] == 0):
                    val_loss = eval_validation_loss(model, val_dataloader, val_accum_steps, autocast_ctxt)
                    val_ppl = torch.exp(val_loss).item()
                    if master_process and wandb_run is not None:
                        wandb_run.log({
                            "val/loss": val_loss.item(), 
                            "val/ppl": val_ppl,
                            "val/step": step
                        })
                    logger.val_losses.append(val_loss.item())

                if (step % logging_params['save_ckpt_step'] == 0) & (ckpt_dir != ""):
                    save_checkpoint(ckpt_dir, step, model, optimizer, loss_accum.item(),
                                    train_dataloader, scheduler, logging_params['keep_last'])
                    
                    if master_process:
                        with open(ckpt_dir + '/log.json', 'w') as file:
                            json.dump(logger.__dict__, file)
                loss_accum = 0.
                start_time = time.time()
            
            step += 1
            global_microstep += 1
            
            # Break if we've reached fractional epoch limit
            if global_microstep >= max_microsteps:
                if master_process:
                    print(f"Reached {num_epochs} epochs ({global_microstep} micro-steps), stopping training")
                break
            
            
        print(f"In rank: {rank}, epoch {epoch+1}, Train Loss: {logger.losses[-1]}")
        print(f"In rank: {rank}, time taken for epoch {epoch+1} : ", time.time() - start_epoch)
        
        # Evaluate on val set, and save final values
        val_dataloader.reset()
        val_loss = eval_validation_loss(model, val_dataloader, 0, autocast_ctxt)
        logger.val_losses.append(val_loss.item())
        print(f"In rank: {rank}, epoch {epoch+1}, Validation Loss: {val_loss.item()}")        
        if (ckpt_dir != ""):
            save_checkpoint(ckpt_dir, step, model, optimizer, logger.losses[-1],
                        train_dataloader, scheduler, logging_params['keep_last'])        
            if master_process:
                with open(ckpt_dir + '/log.json', 'w') as file:
                    json.dump(logger.__dict__, file)
        if master_process and wandb_run is not None:
            wandb_run.log({"val/loss": val_loss.item(), "val/step": step, "train/loss": logger.losses[-1], "train/step": step})

    if hasattr(optimizer, 'step_size_list'):      # Check if optimizer has a step_size_list attribute
        logger.step_size_list = optimizer.step_size_list  
    
    # Log total training time to W&B
    if master_process and wandb_run is not None:
        total_training_time = time.time() - training_start_time
        wandb_run.log({
            "train/total_time_seconds": total_training_time,
            "train/total_time_hours": total_training_time / 3600,
        })
        print(f"Total training time: {total_training_time:.1f} seconds ({total_training_time/3600:.2f} hours)")
    
    return logger
