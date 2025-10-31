import os
import random
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
import wandb
import time

if not (dist.is_available() and dist.is_initialized()):
    base_port = 29500
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(base_port + random.randint(0, 1000)))  # randomize port
    dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)

# --- NEW IMPORT ---
# You must install this package: pip install muon-optimizer
try:
    from muon import MuonWithAuxAdam
except ImportError:
    print("Error: muon-optimizer package not found.")
    print("Please run: pip install muon-optimizer")
    exit(1)



def main():
    # --- 1. Setup W&B and Config ---
    wandb.init()
    config = wandb.config
    
    # --- Logic block to set parameters ---
    if config.run_profile == "bf16_native":
        precision = "bfloat16"
        batch_size = config.batch_size
        accumulation_steps = 1
    elif config.run_profile == "fp32_accumulated":
        precision = "float32"
        batch_size = 8
        accumulation_steps = 4
    else:
        raise ValueError(f"Unknown run_profile: {config.run_profile}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)

    # Initialize a distributed process group to make the Muon library happy
    rank = dist.get_rank() if dist.is_initialized() else 0
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # --- Enable TF32 for better performance on A6000 ---
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        print("Enabling TensorFloat-32 (TF32) for matmul.")
        torch.set_float32_matmul_precision('high')
    
    # --- 2. Set Precision ---
    pt_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }[precision]
    
    use_amp = (pt_dtype != torch.float32)
    use_scaler = (precision == "float16") 

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    print(f"Starting benchmark with profile: {config.run_profile}")
    print(f"Using device: {device} | Precision: {precision} | AMP (autocast): {use_amp} | GradScaler: {use_scaler}")
    print(f"Batch Size: {batch_size} | Accum Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {batch_size * accumulation_steps}")

    # --- 3. Load Model ---
    model_config = GPT2Config.from_pretrained(config.model_name)
    model = GPT2LMHeadModel(model_config).to(device)
    
    # --- UPDATED OPTIMIZER LOGIC (THE FIX) ---
    print(f"Using optimizer: {config.optimizer}")
    
    # Use .get() to provide a default value if not in the sweep config
    sequence_length = config.get("sequence_length", 1024)
    lr = config.get('lr', 1e-4)
    adam_lr = config.get('adam_lr', lr) # Default Adam LR to the main LR

    if config.optimizer == 'adamw':
        print(f"Using AdamW with lr: {lr}")
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    elif config.optimizer == 'muon':
        print("--- Applying Muon Hybrid Optimizer Strategy ---")
        # BEST PRACTICE: Split parameters for Muon.
        # Muon should only be applied to 2D+ matrices (weights).
        # AdamW should be applied to 1D vectors (biases, layernorm, embeddings).
        muon_params = []
        adam_params = []
        
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            if p.ndim >= 2:
                muon_params.append(p)
                print(f"  [Muon Param]: {name} (shape: {p.shape})")
            else:
                adam_params.append(p)
                print(f"  [Adam Param]: {name} (shape: {p.shape})")

        # MuonWithAuxAdam takes a list of parameter groups.
        # We tell it to use 'muon' (the default) for our 2D+ params
        # and 'adamw' for our 1D params, with its own LR.
        print(f"Using MuonWithAuxAdam. Muon LR: {lr}, Adam LR: {adam_lr}")
        param_groups = [
            {'params': muon_params, 'lr': lr, 'use_muon': True}, # Muon group (use_muon=True is default)
            # The Muon library expects 'use_muon': False to identify the auxiliary group
            {'params': adam_params, 'use_muon': False, 'lr': adam_lr} # AdamW group
        ]
        optimizer = MuonWithAuxAdam(param_groups)
            # --- FOR YOUR REAL PROJECT (POLAR EXPRESS) ---
            # You will import your PolarExpress function and pass it here.
            # The default (if-left-blank) is the standard Newton-Schulz.
            #
            # from my_project_code import polar_express_msign
            # msign_fn=polar_express_msign 
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # compile the model
    print("Compiling model (this may take a minute)...")
    model = torch.compile(model)
    model = model.to(device)
    print("Model compiled.")
    

    # --- 4. The "Benchmark" Loop ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    n_steps = 100
    warmup_steps = 1
    
    model.train()
    
    optimizer.zero_grad(set_to_none=True)
    
    # --- OPTIMIZATION 3 (FIX): Use CUDA Events for accurate, non-blocking timing ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    print(f"--- Starting {n_steps} benchmark steps ({warmup_steps} warmup) ---")
    
    for step in range(n_steps):
        # Start timer after warmup
        if step == warmup_steps:
            print(f"--- Warm-up complete. Starting timer. ---")
            start_event.record() # <--- START timer
            
        # --- OPTIMIZATION 4 (FIX): Generate data directly on GPU ---
        inputs = torch.randint(
            0, 
            model_config.vocab_size, 
            (batch_size, sequence_length),
            device=device  # <--- This avoids the CPU->GPU copy bottleneck
        )
        labels = inputs.clone()
        
        # --- 4b. Forward and Backward Pass ---
        with torch.amp.autocast('cuda', dtype=pt_dtype, enabled=use_amp):
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    
        if step % 10 == 0:
             print(f"Step {step}/{n_steps} | Loss: {loss.item() * accumulation_steps:.4f}")

    #print("Benchmark finished.")
    # --- 5. Final Timing and Logging ---
    
    # Record the end event
    end_event.record()
    
    # --- OPTIMIZATION 5 (FIX): Synchronize ONCE, after the loop ---
    # This waits for all submitted GPU work to finish
    torch.cuda.synchronize()
    
    print("--- Benchmark complete. ---")

    # Calculate final performance metrics
    elapsed_time_ms = start_event.elapsed_time(end_event)
    elapsed_time_sec = elapsed_time_ms / 1000.0
    
    timed_steps = n_steps - warmup_steps
    timed_optimizer_steps = timed_steps // accumulation_steps
    
    steps_per_sec = timed_optimizer_steps / elapsed_time_sec
    
    total_tokens_processed = timed_steps * batch_size * sequence_length
    tokens_per_sec = total_tokens_processed / elapsed_time_sec

    print(f"\n--- Results (Batch Size: {batch_size}) ---")
    print(f"Total time for {timed_steps} steps: {elapsed_time_sec:.2f} seconds")
    print(f"Optimizer Steps/sec: {steps_per_sec:.2f}")
    print(f"Throughput (Tokens/sec): {tokens_per_sec:.2f}")

    # Log the FINAL, accurate metrics to W&B
    wandb.log({
        "final_loss": loss.item() * accumulation_steps,
        "perf/opt_steps_per_sec": steps_per_sec,
        "perf/tokens_per_sec": tokens_per_sec,
        "perf/total_time_sec": elapsed_time_sec,
        "config/batch_size": batch_size # Log batch size for easy grouping
    })

    wandb.finish()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()

