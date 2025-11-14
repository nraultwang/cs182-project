from itertools import chain, islice, repeat
import torch
import os
import warnings
import uuid
import sys
import importlib.util

# Import coefficient generation from polar-express
# Use importlib to handle the hyphenated filename
try:
    coeffs_file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'polar-express', 'get-coeffs-2.py')
    coeffs_file_path = os.path.abspath(coeffs_file_path)
    
    spec = importlib.util.spec_from_file_location("get_coeffs_2", coeffs_file_path)
    get_coeffs_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(get_coeffs_module)
    
    generate_coeffs_library = get_coeffs_module.generate_coeffs_library
    # Generate coefficient library once at module load
    COEFFS_LIBRARY = generate_coeffs_library(l0=1e-3, degree=5)
except (ImportError, FileNotFoundError, AttributeError) as e:
    warnings.warn(f"Could not import coefficient library from polar-express/get-coeffs-2.py: {e}")
    COEFFS_LIBRARY = None

# Default coefficients (paper's 8-iteration config with safety factor)
# These match: num_iters=8, safety=1.01, cushion=0.024
DEFAULT_COEFFS_LIST = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # limiting form
]
# safety factor applied (all but last)
DEFAULT_COEFFS_LIST = [(a / 1.01, b / 1.01**3, c / 1.01**5)
                       for (a, b, c) in DEFAULT_COEFFS_LIST[:-1]] + [DEFAULT_COEFFS_LIST[-1]]


def get_coeffs_for_config(num_iters=None, safety=1.01, cushion=0.024):
    """
    Get coefficient list for specified configuration.
    
    Args:
        num_iters: List of number of iterations (3, 5, or 7). If None, use default.
        safety: Safety factor (1.0 or 1.01)
        cushion: Cushion parameter. Library provides:
                 - 0.1 (or close)
                 - 0.05 (or close)
                 - 0.024 or 0.02407327424182761 (paper's precise value)
    
    Returns:
        list of list of (a, b, c) coefficient tuples for each num_iters
    
    Note:
        The library uses precise cushion value 0.02407327424182761 but keys it as "0.024".
        Values within 0.001 of 0.024 will match this key (e.g., 0.024, 0.02407..., 0.0245).
    """
    if num_iters is None:
        # Use default 8-iteration config
        return [DEFAULT_COEFFS_LIST]
    
    if COEFFS_LIBRARY is None:
        warnings.warn("Coefficient library not available, using default coefficients")
        return DEFAULT_COEFFS_LIST
    
    # Format cushion to match dictionary key format EXACTLY as in get-coeffs-2.py
    # The library uses the precise value 0.02407327424182761 but keys it as "0.024"
    if abs(cushion - 0.1) < 1e-10:
        c_str = "0.10"  # Note: with trailing zero
    elif abs(cushion - 0.05) < 1e-10:
        c_str = "0.05"
    elif abs(cushion - 0.024) < 1e-3:
        # Accept both 0.024 and the precise value 0.02407327424182761
        c_str = "0.024"
    else:
        # For arbitrary cushion values, format with full precision
        c_str = f"{cushion:.6f}".rstrip('0').rstrip('.')
    
    # Format safety with 2 decimals to match library format
    s_str = f"{safety:.2f}"
    
    keys = [f"n{num_iter}_s{s_str}_c{c_str}" for num_iter in num_iters]
    
    for key in keys:
        if key not in COEFFS_LIBRARY:
            # Try alternative formatting in case of rounding
            available_keys = [k for k in COEFFS_LIBRARY.keys() if k.startswith(f"n{num_iter}_s{s_str}_") for num_iter in num_iters]
            warnings.warn(
                f"Coefficient config '{key}' not found in library. "
                f"Available configs for n={num_iters}, s={s_str}: {available_keys}. "
                f"Using default coefficients."
            )
            return [DEFAULT_COEFFS_LIST]
    
    return [COEFFS_LIBRARY[key] for key in keys]


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int, coeffs_lists=None, return_ortho_info=False, iter_counter: int) -> torch.Tensor:
    """
    Polar decomposition using polynomial iteration.
    
    Args:
        G: Input gradient matrix
        steps: Number of iteration steps
        coeffs_list: Optional list of (a, b, c) coefficient tuples.
                    If None, uses DEFAULT_COEFFS_LIST (8 iterations, safety=1.01)
        return_ortho_info: If True, return (X, XTX) for computing orthogonality error
    """
    if coeffs_lists is None:
        coeffs_list = [DEFAULT_COEFFS_LIST]
        
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    transposed = G.size(-2) > G.size(-1)
    if transposed: X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 +1e-7)
    # hs = coeffs_list[:steps] + list( 
    #     repeat(coeffs_list[-1], steps - len(coeffs_list)))
    hs = coeffs_lists[iter_counter % len(coeffs_lists)]
    
    A = None  # Keep track of last A = X @ X.mT
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if transposed: X = X.mT
    
    if return_ortho_info and A is not None:
        # Return X and XTX (which is A if not transposed, or would be X.mT @ X if transposed)
        if transposed:
            # Need to compute XTX after transpose
            XTX = X.mT @ X
        else:
            XTX = A  # A is already X @ X.mT from last iteration
        return X, XTX
    return X


@torch.compile
def FastApplyPolarExpress(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0, coeffs_list=None) -> torch.Tensor:
    """
    Fast polar decomposition with restarts and numerical stability.
    
    Args:
        G: Input gradient matrix
        steps: Number of iteration steps
        restart_interval: How often to restart (apply Q and reset)
        shift_eps: Numerical stability shift for Y = XX^T
        coeffs_list: Optional coefficient list (uses DEFAULT_COEFFS_LIST if None)
    """
    if coeffs_list is None:
        coeffs_list = DEFAULT_COEFFS_LIST
        
    assert G.ndim >= 2
    X = G.double()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    hs = [(a * .99, b * .99, c * .99) for (a, b, c) in hs]  # safety factor
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    Y = X @ X.mT + shift_eps * I  # numerical stability
    Q = I.clone()
    for iter, (a, b, c) in enumerate(hs):
        if (iter % restart_interval == 0) and (iter > 0):
            X = Q @ X
            Y = X @ X.mT
            Q = I.clone()
        R = Q.mT @ Y @ Q
        Q = Q @ (a*I + R @ (b*I + c*R))  # Q <- Q(aI + bR + cR^2)
        # if verbose:
        #     print("-"*20)
        #     print(iter)
        #     print("R", torch.linalg.eigvalsh(R.double())[:10])
        #     print((R - R.T).norm().item())
        #     print("Q", torch.linalg.eigvalsh(Q.double())[:10])
        #     print((Q - Q.T).norm().item())
        #     print(torch.linalg.norm((Q @ X).double(), ord=2).item())
    X = Q @ X
    if (X.norm(dim=(-2, -1), keepdim=False) > 5 * I.shape[0]).any() or not (torch.isfinite(X).all()):
        warnings.warn("X.norm() is unusually large. Saving G to disk.")
        os.makedirs("bad_G", exist_ok=True)
        filename = f"bad_G_{uuid.uuid4().hex}.pt"
        torch.save(G, os.path.join("bad_G", filename))
    if G.size(-2) > G.size(-1): X = X.mT
    return X
