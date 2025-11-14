# Data Setup Guide

## Overview

The training scripts expect preprocessed datasets in a specific binary format. The data is **NOT** in git (it's too large - ~3GB+).

## Setting Up Data

### 1. Choose a Data Directory

Pick a location with enough space (3GB+ for finewebmini):

```bash
# Option 1: Home directory
export DATA_DIR=~/data/huggingface/

# Option 2: Shared data location (check with your team)
export DATA_DIR=/data/shared/huggingface/

# Option 3: Project-specific location
export DATA_DIR=~/cs182-project/data/
```

**IMPORTANT**: Add this to your `~/.bashrc` so it persists:
```bash
echo 'export DATA_DIR=~/data/huggingface/' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create the Directory

```bash
mkdir -p $DATA_DIR
```

### 3. Download and Prepare a Dataset

#### For wikitext (quick test, ~100MB)
```bash
cd ~/cs182-project/GPT-opt
python process_data.py --name wikitext
```

This will download and tokenize the wikitext-103 dataset into `$DATA_DIR/wikitext-gpt2/`

**Note:** tiny_shakespeare has been deprecated by HuggingFace. Use wikitext for testing instead.

#### For finewebmini (~3GB)
```bash
cd ~/cs182-project/GPT-opt
python process_data.py --name finewebmini -s 100000000
```

This will:
- Download the FineWeb dataset from HuggingFace
- Tokenize it with GPT-2 tokenizer
- Save sharded binary files to `$DATA_DIR/finewebmini-gpt2/`
- Takes ~10-30 minutes depending on connection

#### For fineweb10B (full dataset, ~10GB+)
```bash
python process_data.py --name fineweb10B -s 100000000 -n 8
```

The `-n 8` flag uses 8 processes for faster processing.

## Verifying Your Data

Check that the data was created:

```bash
ls -lh $DATA_DIR/finewebmini-gpt2/
```

You should see files like:
```
train_000000.bin
train_000001.bin
...
val_000000.bin
```

## Running Training with Your Data

Once data is set up, you can run training:

```bash
# Make sure DATA_DIR is set
export DATA_DIR=~/data/huggingface/

# Run a test with shakespeare
python run_hydra.py training_data=shakespeare

# Run with finewebmini
python run_hydra.py training_data=finewebmini

# Run a wandb sweep
wandb agent <sweep-id>
```

## Dataset Info

| Dataset | Size | Processing Time | Use Case |
|---------|------|----------------|----------|
| tiny_shakespeare | ~1MB | <1 min | Quick testing |
| finewebmini | ~3GB | 10-30 min | Small experiments |
| fineweb10B | ~10GB+ | 1-2 hours | Full training runs |

## Troubleshooting

### Error: "FileNotFoundError: ... finewebmini-gpt2"

**Problem**: Data hasn't been downloaded yet.

**Fix**: Run `python process_data.py --name finewebmini`

### Error: "No such file or directory: '/mnt/ceph/...'"

**Problem**: `DATA_DIR` environment variable not set.

**Fix**: 
```bash
export DATA_DIR=~/data/huggingface/
# Or add to ~/.bashrc for persistence
```

### Check if DATA_DIR is set correctly

```bash
echo $DATA_DIR
ls -la $DATA_DIR
```

### Dataset won't download

**Problem**: Network issues or HuggingFace authentication required.

**Fix**: 
```bash
# Login to HuggingFace if needed
pip install huggingface_hub
huggingface-cli login

# Then try downloading again
python process_data.py --name finewebmini
```

## Quick Start (Recommended)

For first-time setup on a new server:

```bash
# 1. Set up data directory
export DATA_DIR=~/data/huggingface/
mkdir -p $DATA_DIR
echo 'export DATA_DIR=~/data/huggingface/' >> ~/.bashrc

# 2. Download small test dataset first
cd ~/cs182-project/GPT-opt
python process_data.py --name tiny_shakespeare

# 3. Test that training works
python run_hydra.py training_data=shakespeare optimizer_params.args.lr=0.001

# 4. If successful, download finewebmini for real experiments
python process_data.py --name finewebmini -s 100000000

# 5. Run your sweeps!
wandb sweep sweeps/phase0-adamw.yaml
```
