"""
Polar-Express-style polynomial coefficient generation
for Muon experiments.

Supports:
- degree = 3, 5, 7 (odd polynomials)
- configurable:
    - l0       (initial lower spectral bound)
    - num_iters
    - cushion
    - safety
Returns:
- coeffs_list suitable for use in a PE-style Muon step.
"""

from math import inf, sqrt
import numpy as np


# =========================
#  Degree-3 (cubic)
# =========================

def optimal_cubic(l: float, u: float):
    """
    Odd cubic:
        p(t) = a t + b t^3
    approximating sign(t) on [l, u].

    Uses:
    - limiting closed form when l/u ~ 1 (interval tiny),
    - Remez algorithm otherwise.
    """
    # Handle floating-point precision issues near convergence
    if abs(l - u) < 1e-14:
        l = u = (l + u) / 2.0
    assert 0.0 <= l <= u + 1e-14, f"Expected 0 <= l <= u, got l={l}, u={u}"
    l = min(l, u)  # Clamp l to be <= u

    if u > 0 and (1.0 - 5e-6) <= (l / u):
        a = 3.0 / (2.0 * u)
        b = -1.0 / (2.0 * u**3)
        return float(a), float(b)

    # Remez-like iteration with one interior alternation point q
    q = (l + u) / 2.0
    E, old_E = inf, None

    for _ in range(100):
        old_E = E

        # Alternating error at l, q, u: +E, -E, +E
        M = np.array([
            [l, l**3,  1.0],
            [q, q**3, -1.0],
            [u, u**3,  1.0],
        ])
        a, b, E = np.linalg.solve(M, np.ones(3))

        if old_E is not None and abs(old_E - E) < 1e-15:
            break

        # Update q as extremum of p'(x) = a + 3bx^2 = 0
        if abs(b) < 1e-15:
            break
        
        x_sq = -a / (3.0 * b)
        if x_sq <= 0:
            break
        
        q_new = sqrt(x_sq)
        q = max(l + 1e-12, min(u - 1e-12, q_new))

    return float(a), float(b)


# =========================
#  Degree-5 (quintic, Polar Express-style)
# =========================

def optimal_quintic(l: float, u: float):
    """
    Odd quintic:
        p(t) = a t + b t^3 + c t^5
    approximating sign(t) on [l, u].

    This matches the Polar Express paper exactly.
    """
    # Handle floating-point precision issues near convergence
    if abs(l - u) < 1e-14:
        l = u = (l + u) / 2.0
    assert 0.0 <= l <= u + 1e-14, f"Expected 0 <= l <= u, got l={l}, u={u}"
    l = min(l, u)  # Clamp l to be <= u

    # Limiting equioscillating polynomial as [l, u] collapses.
    # Paper: "Above this threshold, the equioscillating polynomial is numerically equal to..."
    if u > 0 and (1.0 - 5e-6) <= (l / u):
        a = (15.0 / 8.0) / u
        b = (-10.0 / 8.0) / (u**3)
        c = (3.0 / 8.0) / (u**5)
        return float(a), float(b), float(c)

    # Remez-like iteration with two interior alternation points q1, q2.
    # Paper: "We initialize q₁ and q₂ to be equally spaced in [l, u]"
    q1 = (2.0 * l + u) / 3.0
    q2 = (l + 2.0 * u) / 3.0
    E, old_E = inf, None

    for _ in range(100):
        old_E = E

        # Alternating error at l, q1, q2, u: +E, -E, +E, -E
        M = np.array([
            [l,  l**3,  l**5,   1.0],
            [q1, q1**3, q1**5, -1.0],
            [q2, q2**3, q2**5,  1.0],
            [u,  u**3,  u**5,  -1.0],
        ])
        a, b, c, E = np.linalg.solve(M, np.ones(4))

        if old_E is not None and abs(old_E - E) < 1e-15:
            break

        # Update interior points as extrema of p'(x) = a + 3bx² + 5cx⁴ = 0
        # Let y = x²: 5cy² + 3by + a = 0
        # Using quadratic formula: y = (-3b ± sqrt(9b² - 20ac)) / (10c)
        if abs(c) < 1e-15:
            break
        
        discriminant = 9.0 * b**2 - 20.0 * a * c
        if discriminant < 0:
            break
        
        sqrt_disc = sqrt(discriminant)
        y1 = (-3.0 * b - sqrt_disc) / (10.0 * c)
        y2 = (-3.0 * b + sqrt_disc) / (10.0 * c)
        
        # Take positive y values and compute x = sqrt(y)
        x_candidates = []
        for y in [y1, y2]:
            if y > 0:
                x = sqrt(y)
                if l < x < u:
                    x_candidates.append(x)
        
        if len(x_candidates) < 2:
            break
        
        x_candidates.sort()
        q1 = max(l + 1e-12, min(u - 1e-12, x_candidates[0]))
        q2 = max(l + 1e-12, min(u - 1e-12, x_candidates[1]))

    return float(a), float(b), float(c)


# =========================
#  Degree-7 (septic)
# =========================

def optimal_septic(l: float, u: float):
    """
    Odd septic:
        p(t) = a t + b t^3 + c t^5 + d t^7
    approximating sign(t) on [l, u].

    Uses:
    - limiting closed form when l/u ~ 1,
    - Remez algorithm otherwise.
    """
    # Handle floating-point precision issues near convergence
    if abs(l - u) < 1e-14:
        l = u = (l + u) / 2.0
    assert 0.0 <= l <= u + 1e-14, f"Expected 0 <= l <= u, got l={l}, u={u}"
    l = min(l, u)  # Clamp l to be <= u

    # Limiting form: enforce p(u)=1 and first 3 derivatives 0 at u.
    # Solving that system gives:
    #   a = 35/(16u)
    #   b = -35/(16u^3)
    #   c = 21/(16u^5)
    #   d = -5/(16u^7)
    if u > 0 and (1.0 - 5e-6) <= (l / u):
        a = 35.0 / (16.0 * u)
        b = -35.0 / (16.0 * u**3)
        c = 21.0 / (16.0 * u**5)
        d = -5.0 / (16.0 * u**7)
        return float(a), float(b), float(c), float(d)

    # Remez-like iteration with three interior alternation points
    q1 = (3.0 * l + u) / 4.0
    q2 = (2.0 * l + 2.0 * u) / 4.0
    q3 = (l + 3.0 * u) / 4.0
    E, old_E = inf, None

    for _ in range(100):
        old_E = E

        # Alternating error at l, q1, q2, q3, u: +E, -E, +E, -E, +E
        M = np.array([
            [l,  l**3,  l**5,  l**7,   1.0],
            [q1, q1**3, q1**5, q1**7, -1.0],
            [q2, q2**3, q2**5, q2**7,  1.0],
            [q3, q3**3, q3**5, q3**7, -1.0],
            [u,  u**3,  u**5,  u**7,   1.0],
        ])
        a, b, c, d, E = np.linalg.solve(M, np.ones(5))

        if old_E is not None and abs(old_E - E) < 1e-15:
            break

        # Update interior points as extrema of p'(x) = a + 3bx^2 + 5cx^4 + 7dx^6 = 0
        if abs(d) < 1e-15:
            break
        
        # Let y = x^2: 7dy^3 + 5cy^2 + 3by + a = 0
        cubic_coeffs = [7.0 * d, 5.0 * c, 3.0 * b, a]
        y_roots = np.roots(cubic_coeffs)
        
        # Filter real positive roots and take sqrt
        x_roots = []
        for y in y_roots:
            if np.isreal(y) and np.real(y) > 0:
                x_roots.append(sqrt(np.real(y)))
        
        if len(x_roots) < 3:
            break
        
        x_roots = sorted(x_roots)
        q1 = max(l + 1e-12, min(u - 1e-12, x_roots[0]))
        q2 = max(l + 1e-12, min(u - 1e-12, x_roots[1]))
        q3 = max(l + 1e-12, min(u - 1e-12, x_roots[2]))

    return float(a), float(b), float(c), float(d)


# =========================
#  Composition builder
# =========================

def optimal_composition(
    l0: float,
    num_iters: int,
    cushion: float = 0.02407327424182761,
    safety: float = 1.01,
    degree: int = 5,
):
    """
    Build coeffs_list for a Polar-Express-style composition.
    
    This matches the algorithm from the paper exactly:
    1. Apply cushioning to get effective lower bound
    2. Compute optimal polynomial on [eff_l, u]
    3. Recenter around 1 with respect to ORIGINAL (l, u) bounds
    4. Update interval using recentered polynomial
    5. After all iterations, apply safety factor to all coeffs except last
    
    Args:
      l0        : initial lower bound on singular values after scaling, in (0,1]
      num_iters : number of composition steps (T)
      cushion   : cushioning factor in (0,1]; effective lower bound per iter:
                    eff_l = max(l, cushion * u)
      safety    : safety factor >= 1.0; applied to all coefficients except last
      degree    : 3, 5, or 7

    Returns:
      coeffs_list: list of tuples per iteration
    """
    assert 0.0 < l0 <= 1.0, f"l0 must be in (0,1], got {l0}"
    assert num_iters >= 1, f"num_iters must be >= 1, got {num_iters}"
    assert degree in (3, 5, 7), f"degree must be 3, 5, or 7, got {degree}"
    assert 0.0 <= cushion <= 1.0, f"cushion must be in [0,1], got {cushion}"
    assert safety >= 1.0, f"safety must be >= 1.0, got {safety}"

    l = float(l0)
    u = 1.0
    coefficients = []

    for i in range(num_iters):
        # Apply cushioning to current interval
        eff_l = max(l, cushion * u)
        
        # Get base polynomial on [eff_l, u]
        if degree == 3:
            a, b = optimal_cubic(eff_l, u)
            
            # Recenter around 1 with respect to ORIGINAL (l, u):
            # Find rescalar such that 1 - rescalar*p(l) = rescalar*p(u) - 1
            # => rescalar = 2 / (p(l) + p(u))
            pl = a * l + b * l**3
            pu = a * u + b * u**3
            rescalar = 2.0 / (pl + pu)
            a *= rescalar
            b *= rescalar
            
            coefficients.append((float(a), float(b)))
            
            # Update interval: new l is p(l), new u = 2 - p(l)
            l = a * l + b * l**3
            u = 2.0 - l

        elif degree == 5:
            a, b, c = optimal_quintic(eff_l, u)
            
            # Recenter around 1 with respect to ORIGINAL (l, u)
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescalar = 2.0 / (pl + pu)
            a *= rescalar
            b *= rescalar
            c *= rescalar
            
            coefficients.append((float(a), float(b), float(c)))
            
            # Update interval
            l = a * l + b * l**3 + c * l**5
            u = 2.0 - l

        else:  # degree == 7
            a, b, c, d = optimal_septic(eff_l, u)
            
            # Recenter around 1 with respect to ORIGINAL (l, u)
            pl = a * l + b * l**3 + c * l**5 + d * l**7
            pu = a * u + b * u**3 + c * u**5 + d * u**7
            rescalar = 2.0 / (pl + pu)
            a *= rescalar
            b *= rescalar
            c *= rescalar
            d *= rescalar
            
            coefficients.append((float(a), float(b), float(c), float(d)))
            
            # Update interval
            l = a * l + b * l**3 + c * l**5 + d * l**7
            u = 2.0 - l

    # Apply safety factor to all coefficients except the last
    # Paper: "safety factor for numerical stability (but exclude last polynomial)"
    if safety != 1.0:
        coeffs_with_safety = []
        for i, coeff in enumerate(coefficients):
            if i < len(coefficients) - 1:  # Not the last coefficient
                if degree == 3:
                    a, b = coeff
                    coeffs_with_safety.append((a / safety, b / safety**3))
                elif degree == 5:
                    a, b, c = coeff
                    coeffs_with_safety.append((a / safety, b / safety**3, c / safety**5))
                else:  # degree == 7
                    a, b, c, d = coeff
                    coeffs_with_safety.append((a / safety, b / safety**3, c / safety**5, d / safety**7))
            else:  # Last coefficient - no safety factor
                coeffs_with_safety.append(coeff)
        coefficients = coeffs_with_safety

    return coefficients


# =========================
#  Helper: Verify convergence
# =========================

def analyze_convergence(coeffs_list, l0: float, degree: int):
    """
    Analyze the spectral interval evolution given coefficient list.
    
    Returns:
        intervals: list of (l, u) tuples at each iteration
        errors: list of maximum approximation errors
    """
    l, u = l0, 1.0
    intervals = [(l, u)]
    errors = []
    
    for coeffs in coeffs_list:
        if degree == 3:
            a, b = coeffs
            l_new = a * l + b * l**3
        elif degree == 5:
            a, b, c = coeffs
            l_new = a * l + b * l**3 + c * l**5
        else:  # degree == 7
            a, b, c, d = coeffs
            l_new = a * l + b * l**3 + c * l**5 + d * l**7
        
        u_new = 2.0 - l_new
        intervals.append((l_new, u_new))
        
        # Estimate max error as distance from ±1
        error = max(abs(1 - u_new), abs(1 - (2 - l_new)))
        errors.append(error)
        
        l, u = l_new, u_new
    
    return intervals, errors


# =========================
#  Pre-computed coefficient library
# =========================

def generate_coeffs_library(l0: float = 1e-3, degree: int = 5):
    """
    Generate a library of coefficient lists for all common hyperparameter combinations.
    
    Args:
        l0: Initial lower spectral bound
        degree: Polynomial degree (default 5 for quintic)
    
    Returns:
        Dictionary with keys like 'n3_s1.00_c0.10' mapping to coefficient lists
        
    Key format: 'n{num_iters}_s{safety}_c{cushion}'
        - n: number of iterations (3, 5, 7)
        - s: safety factor (1.00, 1.01)
        - c: cushion (0.10, 0.05, 0.024)
    """
    num_iters_options = [3, 5, 7]
    safety_options = [1.0, 1.01]
    cushion_options = [0.1, 0.05, 0.02407327424182761]
    
    coeffs_library = {}
    
    for num_iters in num_iters_options:
        for safety in safety_options:
            for cushion in cushion_options:
                # Create readable key
                # Format cushion to distinguish the three values
                if abs(cushion - 0.1) < 1e-10:
                    c_str = "0.10"
                elif abs(cushion - 0.05) < 1e-10:
                    c_str = "0.05"
                else:
                    c_str = "0.024"  # Paper's value
                
                key = f"n{num_iters}_s{safety:.2f}_c{c_str}"
                
                # Generate coefficients
                coeffs = optimal_composition(
                    l0=l0,
                    num_iters=num_iters,
                    cushion=cushion,
                    safety=safety,
                    degree=degree
                )
                
                coeffs_library[key] = coeffs
    
    return coeffs_library


def get_coeffs(num_iters: int, safety: float, cushion: float, 
               coeffs_library: dict = None, l0: float = 1e-3, degree: int = 5):
    """
    Get coefficient list from library or generate on-the-fly.
    
    Args:
        num_iters: Number of iterations (3, 5, or 7)
        safety: Safety factor (1.0 or 1.01)
        cushion: Cushion factor (0.1, 0.05, or 0.024...)
        coeffs_library: Pre-computed library (optional, will generate if None)
        l0: Initial lower bound
        degree: Polynomial degree
    
    Returns:
        List of coefficient tuples
    """
    # Format key
    if abs(cushion - 0.1) < 1e-10:
        c_str = "0.10"
    elif abs(cushion - 0.05) < 1e-10:
        c_str = "0.05"
    else:
        c_str = "0.024"
    
    key = f"n{num_iters}_s{safety:.2f}_c{c_str}"
    
    # Try to get from library
    if coeffs_library is not None and key in coeffs_library:
        return coeffs_library[key]
    
    # Generate on-the-fly if not in library
    return optimal_composition(l0=l0, num_iters=num_iters, cushion=cushion, 
                              safety=safety, degree=degree)


# =========================
#  Example usage
# =========================

if __name__ == "__main__":
    # Generate library of all combinations
    print("=" * 70)
    print("GENERATING COEFFICIENT LIBRARY")
    print("=" * 70)
    
    coeffs_library = generate_coeffs_library(l0=1e-3, degree=5)
    
    print(f"\nGenerated {len(coeffs_library)} coefficient sets:")
    for key in sorted(coeffs_library.keys()):
        num_coeffs = len(coeffs_library[key])
        print(f"  {key}: {num_coeffs} iterations")
    
    # Example: Access specific configuration
    print("\n" + "=" * 70)
    print("EXAMPLE: Accessing specific configurations")
    print("=" * 70)
    
    # Example 1: 5 iterations, safety=1.01, cushion=0.024 (paper's default)
    key1 = "n5_s1.01_c0.024"
    coeffs_1 = coeffs_library[key1]
    print(f"\n{key1} (Paper's config):")
    print(f"  First coeff: {coeffs_1[0]}")
    print(f"  Last coeff:  {coeffs_1[-1]}")
    
    # Example 2: 7 iterations, no safety, cushion=0.1
    key2 = "n7_s1.00_c0.10"
    coeffs_2 = coeffs_library[key2]
    print(f"\n{key2}:")
    print(f"  First coeff: {coeffs_2[0]}")
    print(f"  Last coeff:  {coeffs_2[-1]}")
    
    # Example 3: Using get_coeffs helper
    print("\n" + "=" * 70)
    print("EXAMPLE: Using get_coeffs() helper")
    print("=" * 70)
    
    coeffs_3 = get_coeffs(num_iters=3, safety=1.0, cushion=0.05, 
                          coeffs_library=coeffs_library)
    print(f"\nget_coeffs(num_iters=3, safety=1.0, cushion=0.05):")
    for i, c in enumerate(coeffs_3, 1):
        print(f"  Iter {i}: {c}")
    
    # Show all available keys
    print("\n" + "=" * 70)
    print("ALL AVAILABLE KEYS")
    print("=" * 70)
    print("\nFormat: n{iters}_s{safety}_c{cushion}")
    print("\nKeys:")
    for key in sorted(coeffs_library.keys()):
        print(f"  '{key}'")
    
    print("\n" + "=" * 70)
    print("USAGE IN YOUR CODE")
    print("=" * 70)
    print("""
# One-time setup: Generate the library
coeffs_library = generate_coeffs_library()

# Access by key:
my_coeffs = coeffs_library['n5_s1.01_c0.024']

# Or use helper function:
my_coeffs = get_coeffs(num_iters=5, safety=1.01, cushion=0.02407327424182761, 
                       coeffs_library=coeffs_library)
""")
    
