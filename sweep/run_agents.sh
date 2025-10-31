#!/bin/bash

# This script launches two wandb agents in the background,
# one for each of your GPUs.
#
# USAGE:
# 1. Initialize the sweep:
#    $ wandb sweep sweep.yaml
#
# 2. Copy the SWEEP_ID from the output, which looks like:
#    your-username/your-project/abcdef12
#
# 3. Run this script with the SWEEP_ID:
#    $ ./run_agents.sh your-username/your-project/abcdef12

if [ -z "$1" ]; then
  echo "Usage: $0 <SWEEP_ID>"
  echo "You can get the SWEEP_ID by running 'wandb sweep sweep.yaml'"
  exit 1
fi

SWEEP_ID=$1

echo "Starting agents for sweep: $SWEEP_ID"

# Create log directory if it doesn't exist
mkdir -p logs

# Launch agent for GPU 0 in the background
echo "Starting agent 1 on GPU 0. Log at logs/gpu_0.log"
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29500 wandb agent $SWEEP_ID > logs/gpu_0.log 2>&1 &
#wandb agent $SWEEP_ID > logs/gpu_0.log 2>&1 &

# Launch agent for GPU 1 in the background
echo "Starting agent 2 on GPU 1. Log at logs/gpu_1.log"
CUDA_VISIBLE_DEVICES=1 MASTER_PORT=29501 wandb agent $SWEEP_ID > logs/gpu_1.log 2>&1 &

echo "Agents started in background. Tailing logs."
echo "Press Ctrl+C to stop tailing (agents will continue to run)."
tail -f logs/gpu_0.log logs/gpu_1.log
#tail -f logs/gpu_0.log
