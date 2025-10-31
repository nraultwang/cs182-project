import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Config, GPT2LMHeadModel
import wandb
import time

def main():
    # --- 1. Setup W&B and Config ---
    wandb.init()
    config = wandb.config
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config.seed)
    
    # --- 2. Set Precision ---
    # Get the torch dtype from the config string
    pt_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }[config.precision]
    
    # Enable AMP (Automatic Mixed Precision) if not using float32
    # The A6000 (Ampere) natively supports bfloat16
    use_amp = (pt_dtype != torch.float32)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Starting benchmark with config:\n{config}")
    print(f"Using device: {device} | Precision: {config.precision} | AMP: {use_amp}")

    # --- 3. Load Model ---
    # This is a real, compute-heavy 355M parameter model
    model_config = GPT2Config.from_pretrained(config.model_name)
    model = GPT2LMHeadModel(model_config).to(device)
    
    # --- THIS IS THE MAGIC ---
    # Compile the model. This fuses kernels and DRAMATICALLY
    # reduces CPU overhead, letting the GPU run free.
    # The first few steps will be slow as it compiles.
    print("Compiling model (this may take a minute)...")
    model = torch.compile(model)
    print("Model compiled.")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # --- 4. The "Benchmark" Loop ---
    # We will run for 150 steps
    n_steps = 150
    warmup_steps = 50 # Ignore these steps for performance timing
    
    model.train()
    start_time = time.time()
    
    for step in range(n_steps):
        # --- 4a. Synthetic Data Generation ---
        # We create a random batch on the CPU.
        # This is very fast and simulates the dataloader.
        # We make the labels the same as inputs (standard for LM)
        inputs = torch.randint(
            0, 
            model_config.vocab_size, 
            (config.batch_size, config.sequence_length),
            device=device
        )
        labels = inputs.clone()
        
        # --- 4b. Forward and Backward Pass ---
        optimizer.zero_grad(set_to_none=True)
        
        # Use Autocast for bfloat16 / float16
        with torch.amp.autocast("cuda", dtype=pt_dtype, enabled=use_amp):
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # --- 4c. Performance Logging ---
        if step > warmup_steps:
            # We are now in the "steady state" after compilation
            # and warm-up. Start logging performance.
            torch.cuda.synchronize() # Wait for GPU to finish
            
            elapsed_time = time.time() - start_time
            steps_so_far = step - warmup_steps
            steps_per_sec = steps_so_far / elapsed_time

            print(f"Step {step}/{n_steps} | Loss: {loss.item():.4f} | Steps/sec: {steps_per_sec:.2f}")
            wandb.log({
                "train/loss": loss.item(),
                "perf/steps_per_sec": steps_per_sec
            })
        
        elif step == warmup_steps:
            print(f"--- Warm-up complete (ignored {warmup_steps} steps). Starting timer. ---")
            start_time = time.time() # Reset timer after warm-up

    print("Benchmark finished.")
    wandb.finish()

if __name__ == '__main__':
    main()

