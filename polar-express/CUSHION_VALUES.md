# Cushion Value Reference

## The Paper's Precise Cushion Value

The paper uses a **precise cushion value**: `0.02407327424182761`

However, for readability and simplicity, this is **keyed as `"0.024"`** in the coefficient library.

## How to Use in Your Config

All of these will work and map to the same pre-computed coefficients:

### Option 1: Simple (recommended)
```yaml
polar_cushion: 0.024
```

### Option 2: Precise (if you want to be explicit)
```yaml
polar_cushion: 0.02407327424182761
```

### Option 3: Any value close to 0.024
```yaml
polar_cushion: 0.0245  # Within tolerance of 0.001
```

All three map to the same key: `"n5_s1.01_c0.024"` ✅

## Complete Available Values

```yaml
# Large cushion (conservative)
polar_cushion: 0.1     # → key: "c0.10"

# Medium cushion
polar_cushion: 0.05    # → key: "c0.05"

# Small cushion (paper's default, aggressive)
polar_cushion: 0.024   # → key: "c0.024"
# OR
polar_cushion: 0.02407327424182761  # Same key: "c0.024"
```

## Technical Details

The tolerance check in the code:
```python
elif abs(cushion - 0.024) < 1e-3:  # Within 0.001
    c_str = "0.024"
```

This means any value in the range `[0.023, 0.025]` will match the `"0.024"` key.

## Why This Design?

1. **Precision**: The library pre-computes coefficients using the exact value `0.02407327424182761`
2. **Usability**: The key `"0.024"` is human-readable and easier to type
3. **Flexibility**: You can specify either the simple or precise value in configs
4. **Robustness**: Tolerant to small floating-point rounding differences

## Example Config

```yaml
optimizers:
  # These are equivalent:
  - name: muon-polarexpress
    polar_num_iters: 5
    polar_safety: 1.01
    polar_cushion: 0.024  # Simple

  - name: muon-polarexpress
    polar_num_iters: 5
    polar_safety: 1.01
    polar_cushion: 0.02407327424182761  # Precise
```

Both will use the exact same pre-computed coefficients from the library! 🎯
