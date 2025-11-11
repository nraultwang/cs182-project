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
            # Sample 3 representative matrices: first, middle, last
            svd_targets = [
                ('h.0.attn.c_attn.weight', 'layer0_qkv'),   # First layer attention
                ('h.5.attn.c_attn.weight', 'layer5_qkv'),   # Middle layer attention
                ('h.11.attn.c_attn.weight', 'layer11_qkv'), # Last layer attention
            ]
            
            for target_name, label in svd_targets:
                try:
                    for name, param in model.named_parameters():
                        if target_name in name:
                            # Compute SVD
                            U, S, Vh = torch.linalg.svd(param.data, full_matrices=False)
                            
                            # Key spectral metrics
                            sigma_max = S[0].item()
                            sigma_min = S[-1].item()
                            condition_number = sigma_max / (sigma_min + 1e-10)
                            effective_rank = (S.sum() / (sigma_max + 1e-10)).item()
                            
                            # Spectral decay: what fraction of energy in top k singular values
                            S_cumsum = torch.cumsum(S, dim=0)
                            S_total = S.sum()
                            energy_top10 = (S_cumsum[min(9, len(S)-1)] / S_total).item()
                            energy_top50 = (S_cumsum[min(49, len(S)-1)] / S_total).item()
                            
                            metrics[f'svd/{label}/sigma_max'] = sigma_max
                            metrics[f'svd/{label}/sigma_min'] = sigma_min
                            metrics[f'svd/{label}/condition_number'] = condition_number
                            metrics[f'svd/{label}/effective_rank'] = effective_rank
                            metrics[f'svd/{label}/energy_top10'] = energy_top10
                            metrics[f'svd/{label}/energy_top50'] = energy_top50
                            
                            # Spectral gap (ratio of 1st to 2nd singular value)
                            if len(S) > 1:
                                metrics[f'svd/{label}/spectral_gap'] = (S[0] / S[1]).item()
                            
                            break  # Found the target, move to next
                except Exception as e:
                    print(f"Warning: Could not compute SVD for {label}: {e}")
            
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
                                for target_name, label in svd_targets:
                                    if target_name in param_name:
                                        state = optimizer.state[p]
                                        if 'momentum_buffer' in state:
                                            buf = state['momentum_buffer']
                                            
                                            # Reshape if needed (same as Muon does)
                                            if buf.ndim > 2:
                                                buf = buf.view(buf.size(0), -1)
                                            
                                            # Compute SVD on the momentum buffer
                                            U_buf, S_buf, Vh_buf = torch.linalg.svd(buf, full_matrices=False)
                                            
                                            # Key spectral metrics for updates
                                            sigma_max_buf = S_buf[0].item()
                                            sigma_min_buf = S_buf[-1].item()
                                            condition_number_buf = sigma_max_buf / (sigma_min_buf + 1e-10)
                                            effective_rank_buf = (S_buf.sum() / (sigma_max_buf + 1e-10)).item()
                                            
                                            # Store with "update" prefix to distinguish from weight SVD
                                            metrics[f'svd/update_{label}/sigma_max'] = sigma_max_buf
                                            metrics[f'svd/update_{label}/sigma_min'] = sigma_min_buf
                                            metrics[f'svd/update_{label}/condition_number'] = condition_number_buf
                                            metrics[f'svd/update_{label}/effective_rank'] = effective_rank_buf
                                            
                                            if len(S_buf) > 1:
                                                metrics[f'svd/update_{label}/spectral_gap'] = (S_buf[0] / S_buf[1]).item()
                                        
                                        break  # Found the target, move to next
                except Exception as e:
                    print(f"Warning: Could not compute SVD for momentum buffers: {e}")
    
    with torch.no_grad():
        # C) Attention health - sample from first, middle, and last layers
        try:
            # Sample layers 0, 5, 11 for depth coverage
            layers_to_check = [0, 5, 11]
            layer_labels = ['layer0', 'layer5', 'layer11']
            
            # Get model reference
            transformer = model.module.transformer if hasattr(model, 'module') else model.transformer
            
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
                metrics[f'logits/{label}/max_p95'] = torch.quantile(valid_logits, 0.95).item()
                
                # Attention entropy (post-softmax)
                valid_probs = att_probs[torch.isfinite(att_probs).all(dim=-1)]
                if len(valid_probs) > 0:
                    entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum(dim=-1)
                    metrics[f'attn/{label}/entropy/mean'] = entropy.mean().item()
                    metrics[f'attn/{label}/entropy/p05'] = torch.quantile(entropy, 0.05).item()
                    metrics[f'attn/{label}/entropy/p95'] = torch.quantile(entropy, 0.95).item()
                
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
    val_loss = torch.tensor(val_loss.detach().clone(), device=device)/counter
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

    autocast_ctxt = contextlib.nullcontext()
    if training_params['autocast']:
        autocast_ctxt = torch.autocast(device_type=device, dtype=typedict[training_params['mixed_precision']])     
    B, T = training_params['batch_size'], training_params['context_length']
    grad_accum_steps = int(training_params['tokens_processed'] / (world_size*B*T))
    val_accum_steps = int(logging_params['val_tokens_processed'] / (world_size*B*T))
    if master_process: print(f"Accumulate gradient for {grad_accum_steps} steps")
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    max_grad_norm = training_params['gradnorm'] if training_params['gradnorm'] != 0. else float('inf')

    load_ckpt_step = logging_params['load_ckpt_step']
    if load_ckpt_step != 0:
        model, optimizer, train_dataloader, scheduler = load_checkpoint(ckpt_dir, load_ckpt_step, model, \
                                                        optimizer, train_dataloader, scheduler=None)
    if ckpt_dir == "":
        print("Will not save checkpoints as no directory is specified")
    
    # Training loop
    for epoch in range(training_params['num_epochs']):
        if master_process:
            print(f"Epoch {epoch+1} of {training_params['num_epochs']}")

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
                
                if master_process and wandb_run is not None:
                    # A) End-to-end metrics (every step)
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
                    
                    # Amplitude check (every 10 steps or if NaN detected)
                    if step % 10 == 0 or wandb_log_dict["train/naninf_flag"] == 1.0:
                        max_grad = max([p.grad.abs().max().item() for p in model.parameters() if p.grad is not None])
                        wandb_log_dict["train/amp_scaler"] = max_grad
                        wandb_log_dict["train/amp_overflows"] = 1.0 if max_grad > 1e4 else 0.0
                    
                    if hasattr(optimizer, 'step_size_list'):
                        wandb_log_dict["train/step_size_list"] = optimizer.step_size_list
                    for param_group_ix, param_group in enumerate(optimizer.param_groups):
                        wandb_log_dict[f"train/lr_{param_group_ix}"] = param_group['lr']
                    
                    # B) PE-internal metrics (every 100 steps)
                    if step % 100 == 0 and hasattr(optimizer, '_pe_ortho_errs'):
                        if len(optimizer._pe_ortho_errs) > 0:
                            wandb_log_dict["pe/ortho_err"] = np.mean(optimizer._pe_ortho_errs[-10:])  # Average last 10
                            wandb_log_dict["pe/time_ms"] = np.mean(optimizer._pe_times[-10:])
                    
                    # C) & D) Attention health and light scales (every 100 steps)
                    if step % 100 == 0:
                        try:
                            # Compute SVD every 500 steps (more expensive)
                            compute_svd = (step % 500 == 0)
                            advanced_metrics = compute_advanced_metrics(model, optimizer, batch, autocast_ctxt, compute_svd=compute_svd)
                            wandb_log_dict.update(advanced_metrics)
                        except Exception as e:
                            print(f"Warning: Could not compute advanced metrics at step {step}: {e}")
                    
                    wandb_run.log(wandb_log_dict)
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
    return logger
