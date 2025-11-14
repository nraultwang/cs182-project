# Running Wandb Sweeps with Hydra

## The Problem

Wandb sweeps by default generate argparse-style arguments (`--key=value`), but Hydra expects override syntax without the `--` prefix (`key=value`).

## The Solution

All sweep YAML files now include this `command` section:

```yaml
command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}  # This tells wandb to NOT add -- prefix
```

## How to Run Sweeps

### 1. Create the sweep
```bash
wandb sweep sweeps/phase0-adamw.yaml
```

This will output:
```
wandb: Creating sweep with ID: ccwj8vud
wandb: Run sweep agent with: wandb agent <your-entity>/<project>/<sweep-id>
```

### 2. Run the sweep agent
```bash
wandb agent <your-entity>/<project>/<sweep-id>
```

Example:
```bash
wandb agent jasontrinh-university-of-california-berkeley/cs182-project-GPT-opt/ccwj8vud
```

### 3. Run on multiple machines (parallel)
On each machine/GPU:
```bash
wandb agent <same-sweep-id>
```

All agents will pull from the same sweep queue.

## Troubleshooting

### Error: "unrecognized arguments: --gpt_model=..."

**Problem**: The sweep YAML is missing the `command` section.

**Fix**: Add this to your sweep YAML:
```yaml
command:
  - ${env}
  - python
  - ${program}
  - ${args_no_hyphens}
```

### Error: "Could not override 'gpt_model'. No match in the defaults list"

**Problem**: The main `config.yaml` is missing or doesn't include `gpt_model` and `training_data` in its defaults.

**Fix**: Make sure `hydra_conf/config.yaml` exists with:
```yaml
defaults:
  - gpt_model: gpt-small
  - training_data: finewebmini
  - _self_
```

### Error: "Could not override 'training_data.training_params.num_epochs'"

**Problem**: Config groups use `# @package` directives to nest their contents under specific keys.

**Solution**: Our config groups use `# @package gpt_model` and `# @package training_data` directives, which means their contents are NESTED, not merged at the top level.

**Correct override syntax:**
- ✅ `training_data=finewebmini` (selects config group, nests under `training_data` key)
- ✅ `training_data.training_params.num_epochs=3` (overrides nested value)
- ✅ `gpt_model=gpt-large` (selects config group, nests under `gpt_model` key)
- ❌ `training_params.num_epochs=3` (WRONG - not at top level due to @package directive)

### Error: "Could not find config file"

**Problem**: Hydra can't find the config files in `hydra_conf/`.

**Fix**: Make sure you're running from the `GPT-opt/` directory:
```bash
cd GPT-opt
wandb agent <sweep-id>
```

### Check what command will run

Before creating the sweep, you can test the override syntax:
```bash
python run_hydra.py gpt_model=gpt-small optimizer_params.args.lr=0.0001
```

Notice: NO `--` prefix for Hydra!

## Available Sweeps

- `phase0-adamw.yaml` - Baseline AdamW with LR sweep (3 runs)
- `phase0-muon.yaml` - Baseline Muon with LR sweep (3 runs)
- `phase1-pe-focused.yaml` - PolarExpress num_iters sweep (3 runs)
- `phase1-pe-sensitivity.yaml` - Full PolarExpress hyperparameter grid (12 runs)

## Sweep Tips

1. **Start small**: Test with 1-2 runs before launching full sweeps
2. **Monitor early**: Check first few runs to catch config issues
3. **Use tags**: All sweeps include tags for easy filtering in wandb UI
4. **Parallel execution**: Run multiple agents for faster completion
5. **Stop bad runs**: Use wandb UI to stop runs that are clearly failing

## Example: Full Workflow

```bash
# 1. Create sweep
cd ~/cs182-project/GPT-opt
wandb sweep sweeps/phase0-adamw.yaml

# Output will show:
# wandb: Run sweep agent with: wandb agent <entity>/<project>/<sweep-id>

# 2. Copy that command and run
wandb agent <entity>/<project>/<sweep-id>

# 3. (Optional) On another GPU, run the same agent command for parallel execution
wandb agent <entity>/<project>/<sweep-id>
```
