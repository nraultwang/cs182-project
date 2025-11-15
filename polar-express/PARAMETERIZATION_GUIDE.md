# PolarExpress Parameterization Guide

## Overview
The `PolarExpress` optimizer now supports configurable coefficient settings through three hyperparameters:
- `polar_num_iters`: Number of iterations (3, 5, or 7)
- `polar_safety`: Safety factor (1.0 or 1.01)
- `polar_cushion`: Cushion parameter (0.1, 0.05, or 0.024)

This gives you **18 different coefficient configurations** to experiment with.

## Changes Made

### 1. `polar_express.py`
- Added `get_coeffs_for_config()` function to retrieve coefficients from the library
- Modified `PolarExpress()` to accept optional `coeffs_list` parameter
- Imports coefficient library from `polar-express/get-coeffs-2.py` at module load
- Falls back to default 8-iteration coefficients if library unavailable

### 2. `muon.py`
- Added three new parameters to `Muon.__init__()`:
  - `polar_num_iters=None` (default: 8-iteration config)
  - `polar_safety=1.01` (paper's default)
  - `polar_cushion=0.024` (paper's default)
- Modified `_initialize_polar_factorizer()` to pass coefficients to `PolarExpress`

### 3. `utils.py`
- Updated `get_optimizer_factory()` to pass coefficient parameters from config
- Supports: `polar_num_iters`, `polar_safety`, `polar_cushion` in YAML configs

## Usage

### In YAML Config Files
```yaml
optimizers:
  - name: muon-polarexpress
    lr: 0.01
    ns_steps: 5
    polar_num_iters: 5      # 3, 5, or 7 (omit for default 8)
    polar_safety: 1.01      # 1.0 or 1.01
    polar_cushion: 0.024    # 0.1, 0.05, or 0.024
```

### In Python Code
```python
from gptopt.optim.muon import Muon

optimizer = Muon(
    named_params=model.named_parameters(),
    lr=0.01,
    polar_method="polarexpress",
    polar_num_iters=5,      # Use 5-iteration coefficients
    polar_safety=1.01,       # With safety factor
    polar_cushion=0.024,     # Paper's cushion value
)
```

### Direct PolarExpress Usage
```python
from gptopt.optim.polar_express import PolarExpress, get_coeffs_for_config

# Get coefficients for specific config
coeffs = get_coeffs_for_config(num_iters=5, safety=1.01, cushion=0.024)

# Apply polar decomposition
U = PolarExpress(gradient, steps=5, coeffs_list=coeffs)
```

## Available Configurations

The library provides 18 pre-computed configurations:

### num_iters (3 options):
- `3`: Fast, fewer iterations
- `5`: **Paper's default**
- `7`: More accurate, slower

### safety (2 options):
- `1.0`: No safety factor (more aggressive)
- `1.01`: **Paper's default** (numerical stability)

### cushion (3 options):
- `0.1`: Large cushion (conservative)
- `0.05`: Medium cushion
- `0.024`: **Paper's default** (aggressive)

## Dictionary Keys
Coefficients are stored with keys: `n{iters}_s{safety}_c{cushion}`

Examples:
- `'n5_s1.01_c0.024'` - Paper's default (5 iters, safety=1.01, cushion=0.024)
- `'n3_s1.0_c0.1'` - Conservative (3 iters, no safety, large cushion)
- `'n7_s1.01_c0.05'` - Accurate (7 iters, with safety, medium cushion)

## Example Experiment Setup

See `configs/polarexpress-example.yaml` for a complete configuration testing multiple coefficient settings.

To run experiments:
```bash
python run.py --config configs/polarexpress-example.yaml
```

Or with Slurm:
```bash
./submit.sh configs/polarexpress-example.yaml
```

## Backward Compatibility

- **Default behavior unchanged**: If you don't specify the new parameters, it uses the original 8-iteration coefficients with safety factor
- Existing configs continue to work without modification
- Only affects `polar_method="polarexpress"` optimizer

## Testing Different Configurations

To systematically test all 18 configurations, create a sweep:

```yaml
optimizers:
  # Vary num_iters
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 3, polar_safety: 1.01, polar_cushion: 0.024}
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 5, polar_safety: 1.01, polar_cushion: 0.024}
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 7, polar_safety: 1.01, polar_cushion: 0.024}
  
  # Vary safety
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 5, polar_safety: 1.0, polar_cushion: 0.024}
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 5, polar_safety: 1.01, polar_cushion: 0.024}
  
  # Vary cushion
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 5, polar_safety: 1.01, polar_cushion: 0.1}
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 5, polar_safety: 1.01, polar_cushion: 0.05}
  - {name: muon-polarexpress, lr: 0.01, polar_num_iters: 5, polar_safety: 1.01, polar_cushion: 0.024}
```

## Research Questions

This parameterization enables you to investigate:
1. **Iteration count trade-off**: Does 3 vs 5 vs 7 iterations affect convergence speed?
2. **Safety factor impact**: Is the 1.01 safety factor necessary, or can 1.0 work?
3. **Cushion sensitivity**: How does the cushion parameter (spectral range) affect training?
4. **Interaction effects**: Are there optimal combinations of these parameters?

## Next Steps

1. **Generate baseline results**: Test paper's default (`n5_s1.01_c0.024`)
2. **Ablation studies**: Vary one parameter at a time
3. **Full grid search**: Test all 18 configurations
4. **Analysis**: Compare training curves, final loss, convergence speed
