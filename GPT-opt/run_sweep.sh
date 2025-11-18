#!/bin/bash

# Script to run wandb sweep agents on multiple GPUs
#
# USAGE:
# 1. Create the sweep:
#    $ wandb sweep sweeps/phase0-adamw.yaml
#
# 2. Run this script with the SWEEP_ID:
#    $ ./run_sweep.sh <SWEEP_ID>
#
# Example:
#    $ ./run_sweep.sh jasontrinh-university-of-california-berkeley/cs182-polar-express/abc123

if [ -z "$1" ]; then
  echo "Usage: $0 <SWEEP_ID>"
  echo "Example: $0 jasontrinh-university-of-california-berkeley/cs182-polar-express/abc123"
  exit 1
fi

SWEEP_ID=$1

echo "Starting agents for sweep: $SWEEP_ID"

# Set DATA_DIR environment variable
export DATA_DIR=~/data/huggingface/
echo "Using DATA_DIR: $DATA_DIR"

# Unset distributed training environment variables to ensure single-GPU mode
unset RANK
unset WORLD_SIZE
unset LOCAL_RANK
unset MASTER_ADDR
unset MASTER_PORT

# Create log directory if it doesn't exist
mkdir -p logs

# Check number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPU(s)"

if [ $NUM_GPUS -ge 2 ]; then
  echo "Launching 2 agents on GPUs 0 and 1"
  
  # Launch agent for GPU 0 in the background
  echo "Starting agent 1 on GPU 0. Log at logs/gpu_0.log"
  CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 wandb agent "$SWEEP_ID" > logs/gpu_0.log 2>&1 &
  
  # Launch agent for GPU 1 in the background
  echo "Starting agent 2 on GPU 1. Log at logs/gpu_1.log"
  CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 wandb agent "$SWEEP_ID" > logs/gpu_1.log 2>&1 &
  
  echo "Agents started in background. Tailing logs."
  echo "Press Ctrl+C to stop tailing (agents will continue to run)."
  tail -f logs/gpu_0.log logs/gpu_1.log
else
  echo "Only 1 GPU detected. Launching single agent."
  CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 wandb agent "$SWEEP_ID"
fi
