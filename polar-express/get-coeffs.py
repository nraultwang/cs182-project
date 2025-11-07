"""
Polar-Express-style polynomial coefficient generation.

Supports degrees 3, 5, and 7 with configurable:
- l0: initial lower spectral bound
- num_iters: number of composition iterations
- cushion: cushioning factor
- safety: safety factor (>= 1.0, where 1.0 means no safety)
"""

from math import inf, sqrt
import numpy as np


# =========================
#  Degree-3 (cubic)
# =========================

def optimal_cubic(l, u):
    """
    Optimal degree-3 odd polynomial: p(t) = a*t + b*t³
    """
    assert 0 <= l <= u, f"Invalid bounds: l={l}, u={u}"
    
    # Limiting form
    if u > 0 and (1 - 5e-6) <= (l / u):
        return (3/2)/u, (-1/2)/(u**3)
    
    # Closed-form solution
    alpha = np.sqrt(3 / (u**2 + l*u + l**2))
    beta = 4 / (2 + l*u*(l + u)*alpha**3)
    
    a = beta * (3/2) * alpha
    b = beta * (-1/2) * alpha**3
    
    return float(a), float(b)


# =========================
#  Degree-5 (quintic)
# =========================

def optimal_quintic(l, u):
    """
    Optimal degree-5 odd polynomial: p(t) = a*t + b*t³ + c*t⁵
    """
    assert 0 <= l <= u, f"Invalid bounds: l={l}, u={u}"
    
    if u > 0 and (1 - 5e-6) <= (l / u):
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    
    # Remez-style iteration
    q = (3*l + u) / 4
    r = (l + 3*u) / 4
    E, old_E = inf, None
    
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        
        # Find new equioscillation points
        discriminant = 9*b**2 - 20*a*c
        if discriminant < 0:
            break
        q, r = np.sqrt((-3*b + np.array([-1, 1]) * sqrt(discriminant)) / (10*c))
    
    return float(a), float(b), float(c)


# =========================
#  Degree-7 (septic)
# =========================

def optimal_septic(l, u):
    """
    Optimal degree-7 odd polynomial: p(t) = a*t + b*t³ + c*t⁵ + d*t⁷
    """
    assert 0 <= l <= u, f"Invalid bounds: l={l}, u={u}"
    
    # Limiting form
    if u > 0 and (l / u >= 0.99):
        return (35/16)/u, (-35/16)/(u**3), (21/16)/(u**5), (-5/16)/(u**7)
    
    # Remez-style iteration with 3 interior points
    q1 = l + 0.2*(u - l)
    q2 = l + 0.5*(u - l)
    q3 = l + 0.8*(u - l)
    
    E, old_E = inf, None
    
    for iteration in range(100):
        old_E = E
        
        LHS = np.array([
            [l, l**3, l**5, l**7, 1],
            [q1, q1**3, q1**5, q1**7, -1],
            [q2, q2**3, q2**5, q2**7, 1],
            [q3, q3**3, q3**5, q3**7, -1],
            [u, u**3, u**5, u**7, 1],
        ])
        
        try:
            a, b, c, d, E = np.linalg.solve(LHS, np.ones(5))
        except np.linalg.LinAlgError:
            return (35/16)/u, (-35/16)/(u**3), (21/16)/(u**5), (-5/16)/(u**7)
        
        if not (np.isfinite(a) and np.isfinite(b) and 
                np.isfinite(c) and np.isfinite(d)):
            return (35/16)/u, (-35/16)/(u**3), (21/16)/(u**5), (-5/16)/(u**7)
        
        # Find roots of derivative
        derivative_coeffs = [7*d, 0, 5*c, 0, 3*b, 0, a]
        roots = np.roots(derivative_coeffs)
        
        real_roots = roots[np.abs(np.imag(roots)) < 1e-10].real
        interior_roots = real_roots[(real_roots > l) & (real_roots < u)]
        interior_roots = np.sort(interior_roots)
        
        if len(interior_roots) >= 3:
            q1_new, q2_new, q3_new = interior_roots[:3]
            if l < q1_new < q2_new < q3_new < u:
                q1, q2, q3 = q1_new, q2_new, q3_new
        
        if old_E is not None and abs(old_E - E) < 1e-15:
            break
    
    return float(a), float(b), float(c), float(d)


# =========================
#  Main composition builder
# =========================

def optimal_composition(l0, num_iters, cushion=0.02407327424182761, 
                       safety=1.0, degree=5):
    """
    Build coefficient list for Polar-Express-style composition.
    
    Args:
        l0: Initial lower spectral bound (in range (0, 1])
        num_iters: Number of composition iterations
        cushion: Cushioning factor (default from paper: 0.024...)
        safety: Safety factor (default 1.0 = no safety, paper uses 1.01)
        degree: Polynomial degree (3, 5, or 7)
    
    Returns:
        List of coefficient tuples:
            degree=3: [(a, b), ...]
            degree=5: [(a, b, c), ...]
            degree=7: [(a, b, c, d), ...]
    """
    assert 0 < l0 <= 1, "l0 must be in (0, 1]"
    assert num_iters >= 1, "num_iters must be >= 1"
    assert degree in (3, 5, 7), "degree must be 3, 5, or 7"
    assert 0 <= cushion <= 1, "cushion must be in [0, 1]"
    assert safety >= 1.0, "safety must be >= 1.0"
    
    l = l0
    u = 1.0
    coefficients = []
    
    for i in range(num_iters):
        # Check if converged (l and u very close)
        if l >= u * 0.9999:
            # Use limiting form for remaining iterations
            if degree == 3:
                lim_coeffs = ((3/2)/u, (-1/2)/(u**3))
            elif degree == 5:
                lim_coeffs = ((15/8)/u, (-10/8)/(u**3), (3/8)/(u**5))
            else:  # degree == 7
                lim_coeffs = ((35/16)/u, (-35/16)/(u**3), (21/16)/(u**5), (-5/16)/(u**7))
            
            # Apply safety to limiting form
            if safety != 1.0:
                if degree == 3:
                    a, b = lim_coeffs
                    lim_coeffs = (a/safety, b/(safety**3))
                elif degree == 5:
                    a, b, c = lim_coeffs
                    lim_coeffs = (a/safety, b/(safety**3), c/(safety**5))
                else:
                    a, b, c, d = lim_coeffs
                    lim_coeffs = (a/safety, b/(safety**3), c/(safety**5), d/(safety**7))
            
            # Fill remaining iterations with limiting form
            for _ in range(num_iters - i):
                coefficients.append(lim_coeffs)
            break
        
        # Apply cushioning
        l_effective = max(l, cushion * u)
        
        # Get optimal polynomial on [l_effective, u]
        if degree == 3:
            a, b = optimal_cubic(l_effective, u)
            
            # Recenter around 1 with respect to CURRENT [l, u] (not l0!)
            pl = a * l + b * l**3
            pu = a * u + b * u**3
            rescalar = 2 / (pl + pu)
            a *= rescalar
            b *= rescalar
            
            # Save original coefficients for iteration
            a_iter, b_iter = a, b
            
            # Apply safety factor ONLY to output
            if safety != 1.0:
                a /= safety
                b /= safety**3
            
            coefficients.append((float(a), float(b)))
            
            # Update bounds using ORIGINAL (non-safety) coefficients
            l = a_iter * l + b_iter * l**3
            u = 2 - l
            
        elif degree == 5:
            a, b, c = optimal_quintic(l_effective, u)
            
            # Recenter around 1 with respect to CURRENT [l, u]
            pl = a * l + b * l**3 + c * l**5
            pu = a * u + b * u**3 + c * u**5
            rescalar = 2 / (pl + pu)
            a *= rescalar
            b *= rescalar
            c *= rescalar
            
            # Save original coefficients for iteration
            a_iter, b_iter, c_iter = a, b, c
            
            # Apply safety factor ONLY to output
            if safety != 1.0:
                a /= safety
                b /= safety**3
                c /= safety**5
            
            coefficients.append((float(a), float(b), float(c)))
            
            # Update bounds using ORIGINAL coefficients
            l = a_iter * l + b_iter * l**3 + c_iter * l**5
            u = 2 - l
            
        else:  # degree == 7
            a, b, c, d = optimal_septic(l_effective, u)
            
            # Recenter around 1 with respect to CURRENT [l, u]
            pl = a * l + b * l**3 + c * l**5 + d * l**7
            pu = a * u + b * u**3 + c * u**5 + d * u**7
            rescalar = 2 / (pl + pu)
            a *= rescalar
            b *= rescalar
            c *= rescalar
            d *= rescalar
            
            # Save original coefficients for iteration
            a_iter, b_iter, c_iter, d_iter = a, b, c, d
            
            # Apply safety factor ONLY to output
            if safety != 1.0:
                a /= safety
                b /= safety**3
                c /= safety**5
                d /= safety**7
            
            coefficients.append((float(a), float(b), float(c), float(d)))
            
            # Update bounds using ORIGINAL coefficients
            l = a_iter * l + b_iter * l**3 + c_iter * l**5 + d_iter * l**7
            u = 2 - l
    
    return coefficients


# =========================
#  Example usage
# =========================

if __name__ == "__main__":
    print("=" * 70)
    print("POLAR EXPRESS COEFFICIENT GENERATION")
    print("=" * 70)
    
    # Paper's default config (degree 5)
    print("\nDegree 5 (Paper's config: safety=1.01):")
    coeffs_5 = optimal_composition(1e-3, 10, cushion=0.02407327424182761, 
                                    safety=1.01, degree=5)
    print(f"Generated {len(coeffs_5)} coefficients:")
    for i, c in enumerate(coeffs_5):
        print(f"  {i}: {c}")
    
    # Compare first coefficient with paper
    paper_first = (8.28721201814563, -23.595886519098837, 17.300387312530933)
    our_first = coeffs_5[0]
    print(f"\nPaper's first: {paper_first}")
    print(f"Our first:     {our_first}")
    diffs = tuple(abs(a - b) for a, b in zip(paper_first, our_first))
    print(f"Differences:   {diffs}")
    
    # Degree 5 without safety
    print("\nDegree 5 (No safety: safety=1.0):")
    coeffs_5_no_safety = optimal_composition(1e-3, 10, safety=1.0, degree=5)
    print(f"Generated {len(coeffs_5_no_safety)} coefficients:")
    for i, c in enumerate(coeffs_5_no_safety):
        print(f"  {i}: {c}")
    
    # Degree 3
    print("\nDegree 3 (safety=1.0):")
    coeffs_3 = optimal_composition(1e-3, 10, safety=1.0, degree=3)
    print(f"Generated {len(coeffs_3)} coefficients:")
    for i, c in enumerate(coeffs_3):
        print(f"  {i}: {c}")
    
    # Degree 7
    print("\nDegree 7 (safety=1.01):")
    coeffs_7 = optimal_composition(1e-3, 10, cushion=0.02407327424182761, safety=1.00, degree=7)
    print(f"Generated {len(coeffs_7)} coefficients:")
    for i, c in enumerate(coeffs_7):
        print(f"  {i}: {c}")