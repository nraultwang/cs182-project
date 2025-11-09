"""
Example: Using the coefficient library in your training code
"""
# Import with proper module name (hyphen -> underscore in Python imports)
import importlib.util
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
coeffs_file = os.path.join(script_dir, "get-coeffs-2.py")

spec = importlib.util.spec_from_file_location("get_coeffs_2", coeffs_file)
get_coeffs_2 = importlib.util.module_from_spec(spec)
sys.modules["get_coeffs_2"] = get_coeffs_2
spec.loader.exec_module(get_coeffs_2)

generate_coeffs_library = get_coeffs_2.generate_coeffs_library
get_coeffs = get_coeffs_2.get_coeffs

# ============================================================
# METHOD 1: Pre-generate all combinations (recommended)
# ============================================================
print("Method 1: Pre-generate library")
print("="*60)

# One-time generation at start of training
coeffs_library = generate_coeffs_library(l0=1e-3, degree=5)
print(f"Generated {len(coeffs_library)} coefficient sets\n")

# Access by key name
paper_config = coeffs_library['n5_s1.01_c0.024']
print(f"Paper config: {len(paper_config)} iterations")
print(f"  First: {paper_config[0]}")

no_safety = coeffs_library['n7_s1.00_c0.10']
print(f"\nNo safety, 7 iters: {len(no_safety)} iterations")
print(f"  First: {no_safety[0]}")

# ============================================================
# METHOD 2: Use helper function (cleaner)
# ============================================================
print("\n" + "="*60)
print("Method 2: Use get_coeffs() helper")
print("="*60)

# Get specific configuration
coeffs = get_coeffs(
    num_iters=5, 
    safety=1.01, 
    cushion=0.02407327424182761,
    coeffs_library=coeffs_library  # Pass pre-generated library
)
print(f"Retrieved {len(coeffs)} coefficients")
print(f"  First: {coeffs[0]}")

# ============================================================
# METHOD 3: Generate on-demand (if not in library)
# ============================================================
print("\n" + "="*60)
print("Method 3: Generate on-demand")
print("="*60)

# Will generate if not in library
custom_coeffs = get_coeffs(
    num_iters=10,  # Not in standard library
    safety=1.05,   # Not in standard library
    cushion=0.01,  # Not in standard library
    coeffs_library=coeffs_library,
    l0=1e-3,
    degree=5
)
print(f"Generated custom config: {len(custom_coeffs)} iterations")
print(f"  First: {custom_coeffs[0]}")

# ============================================================
# RECOMMENDED USAGE IN TRAINING LOOP
# ============================================================
print("\n" + "="*60)
print("Recommended usage in training:")
print("="*60)
print("""
# At start of training script:
from get_coeffs_2 import generate_coeffs_library

coeffs_library = generate_coeffs_library()

# In your optimizer initialization:
polar_coeffs = coeffs_library['n5_s1.01_c0.024']  # Paper's default

# Or use config file:
config = {
    'num_iters': 5,
    'safety': 1.01,
    'cushion': 0.02407327424182761
}
polar_coeffs = get_coeffs(**config, coeffs_library=coeffs_library)

# Then pass to your optimizer
optimizer = PolarExpressOptimizer(
    params=model.parameters(),
    coeffs_list=polar_coeffs,
    ...
)
""")

# ============================================================
# ALL AVAILABLE CONFIGURATIONS
# ============================================================
print("\n" + "="*60)
print("All available pre-computed configurations:")
print("="*60)
print("\nnum_iters: [3, 5, 7]")
print("safety: [1.0, 1.01]")
print("cushion: [0.10, 0.05, 0.024 (paper)]")
print("\nTotal: 3 × 2 × 3 = 18 combinations\n")

print("Key naming convention:")
print("  n{iters}_s{safety}_c{cushion}")
print("\nExamples:")
print("  'n3_s1.00_c0.10'   - 3 iters, no safety, cushion=0.1")
print("  'n5_s1.01_c0.024'  - 5 iters, safety=1.01, paper's cushion")
print("  'n7_s1.00_c0.05'   - 7 iters, no safety, cushion=0.05")
