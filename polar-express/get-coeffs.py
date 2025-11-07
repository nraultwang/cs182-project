import numpy as np

def optimal_cubic(l, u, tol=1e-15):
    """Optimal degree-3 odd polynomial (closed form)"""
    if 1 - 5e-6 <= l / u:
        return (3/2)/u, (-1/2)/(u**3)
    
    alpha = np.sqrt(3 / (u**2 + l*u + l**2))
    beta = 4 / (2 + l*u*(l + u)*alpha**3)
    
    a = beta * (3/2) * alpha
    b = beta * (-1/2) * alpha**3
    
    return float(a), float(b)


def optimal_quintic(l, u, max_iters=100, tol=1e-15):
    """Optimal degree-5 odd polynomial (Algorithm 3)"""
    if 1 - 5e-6 <= l / u:
        return (15/8)/u, (-10/8)/(u**3), (3/8)/(u**5)
    
    q = (3*l + u) / 4
    r = (l + 3*u) / 4
    E, old_E = np.inf, None
    
    for _ in range(max_iters):
        old_E = E
        LHS = np.array([
            [l, l**3, l**5, 1],
            [q, q**3, q**5, -1],
            [r, r**3, r**5, 1],
            [u, u**3, u**5, -1],
        ])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        
        roots = np.roots([5*c, 0, 3*b, 0, a])
        real_roots = roots[np.abs(np.imag(roots)) < 1e-10].real
        interior = np.sort(real_roots[(real_roots > l) & (real_roots < u)])
        
        if len(interior) >= 2:
            q, r = interior[:2]
        
        if abs(old_E - E) < tol:
            break
    
    return float(a), float(b), float(c)


def optimal_septic(l, u, max_iters=100, tol=1e-15):
    """Optimal degree-7 odd polynomial (numerical)"""
    # Expand the threshold for using limiting form
    if l / u >= 0.99:  # Changed from 1 - 5e-6
        return (35/8)/u, (-35/8)/(u**3), (21/8)/(u**5), (-5/8)/(u**7)
    
    # Better initialization
    q1 = l + 0.2*(u - l)
    q2 = l + 0.5*(u - l)
    q3 = l + 0.8*(u - l)
    
    E, old_E = np.inf, None
    
    for iteration in range(max_iters):
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
            print(f"  Singular matrix at iteration {iteration}, using limiting form")
            return (35/8)/u, (-35/8)/(u**3), (21/8)/(u**5), (-5/8)/(u**7)
        
        # Check for numerical issues
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and np.isfinite(d)):
            print(f"  Non-finite values at iteration {iteration}, using limiting form")
            return (35/8)/u, (-35/8)/(u**3), (21/8)/(u**5), (-5/8)/(u**7)
        
        derivative_coeffs = [7*d, 0, 5*c, 0, 3*b, 0, a]
        roots = np.roots(derivative_coeffs)
        
        real_roots = roots[np.abs(np.imag(roots)) < 1e-10].real
        interior_roots = real_roots[(real_roots > l) & (real_roots < u)]
        interior_roots = np.sort(interior_roots)
        
        if len(interior_roots) >= 3:
            q1_new, q2_new, q3_new = interior_roots[:3]
            
            # Safeguard: only update if reasonable
            if l < q1_new < q2_new < q3_new < u:
                q1, q2, q3 = q1_new, q2_new, q3_new
        
        if old_E is not None and abs(old_E - E) < tol:
            break
    
    return float(a), float(b), float(c), float(d)


def generate_coefficient_sequence(degree, l_init=1e-3, u=1.0, 
                                   cushion=0.02407327424182761, 
                                   num_iters=10):
    """Generate sequence of coefficients as in paper's Algorithm 1"""
    
    if degree == 3:
        optimal_fn = optimal_cubic
    elif degree == 5:
        optimal_fn = optimal_quintic
    elif degree == 7:
        optimal_fn = optimal_septic
    else:
        raise ValueError("degree must be 3, 5, or 7")
    
    l = l_init
    u = 1.0
    coeffs_list = []
    
    for i in range(num_iters):
        # Check for convergence
        if l / u >= 0.9999:
            print(f"  Converged at iteration {i+1}, using limiting form for remaining")
            # Use limiting form for remaining iterations
            if degree == 3:
                limiting_coeffs = ((3/2)/u, (-1/2)/(u**3))
            elif degree == 5:
                limiting_coeffs = ((15/8)/u, (-10/8)/(u**3), (3/8)/(u**5))
            elif degree == 7:
                limiting_coeffs = ((35/8)/u, (-35/8)/(u**3), (21/8)/(u**5), (-5/8)/(u**7))
            
            # Fill remaining with limiting form
            for _ in range(num_iters - i):
                coeffs_list.append(limiting_coeffs)
            break
        
        # Apply cushioning
        l_effective = max(l, cushion * u)
        
        # Get optimal polynomial
        coeffs = optimal_fn(l_effective, u)
        
        # Recenter (as paper does)
        if degree == 3:
            a, b = coeffs
            pl = a*l_init + b*l_init**3
            pu = a*u + b*u**3
            rescalar = 2 / (pl + pu)
            
            a_final = a * rescalar
            b_final = b * rescalar
            
            coeffs_list.append((a_final, b_final))
            
            # Update bounds
            l_new = a*l + b*l**3
            
        elif degree == 5:
            a, b, c = coeffs
            pl = a*l_init + b*l_init**3 + c*l_init**5
            pu = a*u + b*u**3 + c*u**5
            rescalar = 2 / (pl + pu)
            
            a_final = a * rescalar
            b_final = b * rescalar
            c_final = c * rescalar
            
            coeffs_list.append((a_final, b_final, c_final))
            
            # Update bounds
            l_new = a*l + b*l**3 + c*l**5
            
        elif degree == 7:
            a, b, c, d = coeffs
            pl = a*l_init + b*l_init**3 + c*l_init**5 + d*l_init**7
            pu = a*u + b*u**3 + c*u**5 + d*u**7
            
            # Check for numerical issues
            if not (np.isfinite(pl) and np.isfinite(pu)):
                print(f"  Non-finite values in recentering at iteration {i+1}")
                break
            
            rescalar = 2 / (pl + pu)
            
            a_final = a * rescalar
            b_final = b * rescalar
            c_final = c * rescalar
            d_final = d * rescalar
            
            coeffs_list.append((a_final, b_final, c_final, d_final))
            
            # Update bounds
            l_new = a*l + b*l**3 + c*l**5 + d*l**7
        
        # Safeguard: ensure l stays in valid range
        if l_new <= 0 or l_new >= u or not np.isfinite(l_new):
            print(f"  Invalid l_new = {l_new} at iteration {i+1}, stopping")
            break
        
        l = l_new
        u = 2 - l
        
        print(f"Iteration {i+1}: l = {l:.6f}, u = {u:.6f}")
    
    return coeffs_list


def print_coefficients_list(coeffs_list, degree):
    """Print coefficients in the format from Algorithm 1"""
    print(f"\ncoeffs_list_degree_{degree} = [")
    for coeffs in coeffs_list:
        print(f"    {coeffs},")
    print("]")


def generate_all_degrees():
    """Generate coefficient lists for degrees 3, 5, and 7"""
    print("=" * 70)
    print("GENERATING COEFFICIENT SEQUENCES")
    print("=" * 70)
    
    # Degree 3
    print("\n" + "=" * 70)
    print("DEGREE 3 (CUBIC)")
    print("=" * 70)
    coeffs_3 = generate_coefficient_sequence(degree=3, num_iters=10)
    print_coefficients_list(coeffs_3, 3)
    
    # Degree 5
    print("\n" + "=" * 70)
    print("DEGREE 5 (QUINTIC) - Should match paper")
    print("=" * 70)
    coeffs_5 = generate_coefficient_sequence(degree=5, num_iters=10)
    print_coefficients_list(coeffs_5, 5)
    
    paper_first = (8.28721201814563, -23.595886519098837, 17.300387312530933)
    our_first = coeffs_5[0]
    print(f"\nPaper's first: {paper_first}")
    print(f"Ours first:    {our_first}")
    diff = tuple(abs(a - b) for a, b in zip(paper_first, our_first))
    print(f"Differences:   {diff}")
    
    # Degree 7
    print("\n" + "=" * 70)
    print("DEGREE 7 (SEPTIC)")
    print("=" * 70)
    coeffs_7 = generate_coefficient_sequence(degree=7, num_iters=10)
    print_coefficients_list(coeffs_7, 7)
    
    return coeffs_3, coeffs_5, coeffs_7


if __name__ == "__main__":
    coeffs_3, coeffs_5, coeffs_7 = generate_all_degrees()
    
    print("\n" + "=" * 70)
    print("COPY-PASTE FORMAT")
    print("=" * 70)
    
    print("\n# Degree 3")
    print("coeffs_list_d3 = [")
    for c in coeffs_3:
        print(f"    {c},")
    print("]")
    
    print("\n# Degree 5")
    print("coeffs_list_d5 = [")
    for c in coeffs_5:
        print(f"    {c},")
    print("]")
    
    print("\n# Degree 7")
    print("coeffs_list_d7 = [")
    for c in coeffs_7:
        print(f"    {c},")
    print("]")