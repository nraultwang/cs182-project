#!/bin/bash

# Script to run a single wandb sweep agent on a specified GPU
#
# USAGE:
# 1. Create the sweep:
#    $ wandb sweep sweeps/phase1-pe-stage1-safety100.yaml
#
# 2. Run this script with GPU ID and SWEEP_ID:
#    $ ./run_sweep_single_gpu.sh <GPU_ID> <SWEEP_ID>
#
# Example:
#    $ ./run_sweep_single_gpu.sh 0 jasontrinh-university-of-california-berkeley/cs182-polar-express/abc123

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <GPU_ID> <SWEEP_ID>"
  echo "Example: $0 0 jasontrinh-university-of-california-berkeley/cs182-polar-express/abc123"
  exit 1
fi

GPU_ID=$1
SWEEP_ID=$2

echo "Starting agent for sweep: $SWEEP_ID on GPU $GPU_ID"

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

# Set unique MASTER_PORT based on GPU ID to avoid conflicts
MASTER_PORT=$((29500 + GPU_ID))

LOG_FILE="logs/gpu_${GPU_ID}.log"

echo "Starting agent on GPU $GPU_ID. Log at $LOG_FILE"
echo "Press Ctrl+C to stop the agent."

# Run the agent in foreground (or add '&' to run in background)
env CUDA_VISIBLE_DEVICES=$GPU_ID MASTER_PORT=$MASTER_PORT wandb agent $SWEEP_ID 2>&1 | tee $LOG_FILE
