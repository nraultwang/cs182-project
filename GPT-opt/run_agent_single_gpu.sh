#!/bin/bash
# Wrapper script to run wandb agent with single GPU
# Usage: ./run_agent_single_gpu.sh <GPU_ID> <SWEEP_ID>

GPU_ID=$1
SWEEP_ID=$2

export CUDA_VISIBLE_DEVICES=$GPU_ID
export MASTER_PORT=$((29500 + GPU_ID))

echo "Running agent on GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
wandb agent $SWEEP_ID
