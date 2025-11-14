## Muon code from Moonlight
## https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py

# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
import torch
from functools import partial
import math
import warnings
from gptopt.optim.polar_express import PolarExpress, FastApplyPolarExpress

@torch.compile
def jiacheng(G, steps):
    """
    Jiacheng optimized polynomials
    """
    assert len(G.shape) >= 2
    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    if steps > len(abc_list):
        steps = len(abc_list)
    for a, b, c in abc_list[:steps]:
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315) 
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X



class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self,
                 named_params,
                 lr=1e-3,
                 weight_decay=0.1,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 rms_scaling=True,
                 nuclear_scaling=False,
                 polar_method="Keller",
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8,
                 split_heads=False,
                 nheads=None,
                 polar_num_iters=None,
                 polar_safety=1.01,
                 polar_cushion=0.024,
                 muon_mode="stacked_qkv",
                ):
        """
        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "Keller", "polarexpress")
            polar_num_iters: Number of iterations for PolarExpress coefficients (3, 5, or 7). None uses default 8-iter config.
            polar_safety: Safety factor for PolarExpress (1.0 or 1.01)
            polar_cushion: Cushion parameter for PolarExpress (0.1, 0.05, or 0.024)
            muon_mode: Mode for applying Muon to different parameter groups:
                - "stacked_qkv": Muon on stacked QKV matrices + rest of parameters (default)
                - "split_qkv": Muon on split Q/K/V matrices + rest of parameters
                - "voh_only": Muon on split heads for V, W_O, FFN only (AdamW on Q, K, and rest)
        """
        defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                rms_scaling=rms_scaling,
                nuclear_scaling=nuclear_scaling,
                adamw_betas=adamw_betas,
                adamw_eps=adamw_eps,
        )
        
        # Store mode and head configuration
        self.muon_mode = muon_mode
        self.split_heads = split_heads
        if self.split_heads or muon_mode in ["split_qkv", "voh_only"]:
            assert nheads is not None, "nheads must be specified for split_qkv or voh_only modes"
            self.nheads = nheads
        
        # Sort parameters based on muon_mode
        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        
        for name, p in named_params:
            # Embeddings and head always use AdamW
            if p.ndim < 2 or any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                adamw_params.append(p)
                adamw_params_names.append(name)
                continue
            
            # Mode-specific logic
            if muon_mode == "stacked_qkv":
                # Mode 1: All 2D params use Muon (current behavior)
                muon_params.append(p)
                muon_params_names.append(name)
            
            elif muon_mode == "split_qkv":
                # Mode 2: All 2D params use Muon (will split QKV in step())
                muon_params.append(p)
                muon_params_names.append(name)
            
            elif muon_mode == "voh_only":
                # Mode 3: Only V (from QKV), W_O, and FFN use Muon
                is_qkv = "attn.c_attn.weight" in name
                is_wo = "attn.c_proj.weight" in name
                is_ffn = "mlp." in name and ".weight" in name
                
                if is_wo or is_ffn:
                    # W_O and FFN always use Muon in this mode
                    muon_params.append(p)
                    muon_params_names.append(name)
                elif is_qkv:
                    # QKV will be split: V uses Muon, Q&K use AdamW
                    # We add to muon_params but mark it specially
                    muon_params.append(p)
                    muon_params_names.append(name)
                else:
                    # Everything else uses AdamW
                    adamw_params.append(p)
                    adamw_params_names.append(name)
            else:
                raise ValueError(f"Unknown muon_mode: {muon_mode}. Must be one of: stacked_qkv, split_qkv, voh_only")
        
        params = list(muon_params)
        params.extend(adamw_params)
        super().__init__(params, defaults)
        
        # Tag parameters with metadata for the step() function
        for p, p_name in zip(muon_params, muon_params_names):
            if muon_mode != "split_qkv" and not self.split_heads:
                assert p.ndim == 2, f"Expected 2D param, got {p.ndim}D for {p_name}"
            
            self.state[p]["use_muon"] = True
            self.state[p]["param_name"] = p_name
            
            # Tag special matrices
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True
            elif p_name.endswith("attn.c_proj.weight"):
                self.state[p]["is_W_O"] = True
            elif "mlp." in p_name and ".weight" in p_name:
                self.state[p]["is_FFN"] = True

        for p, p_name in zip(adamw_params, adamw_params_names):
            self.state[p]["use_muon"] = False
            self.state[p]["param_name"] = p_name

        # Store PolarExpress configuration
        self.polar_num_iters = polar_num_iters
        self.polar_safety = polar_safety
        self.polar_cushion = polar_cushion
        self.iter_counter = 0
        
        # Instantiate the polar factorization method
        self.polar_factorizer = self._initialize_polar_factorizer(polar_method)

    def _initialize_polar_factorizer(self, polar_method):
        """Initialize the polar factorization method(s) based on the provided name and parameters."""
        if polar_method == "Keller":
            return [zeropower_via_newtonschulz5]  # Wrap in list for consistency
        elif polar_method == "Jiacheng":
            return [jiacheng]  # Wrap in list for consistency
        elif polar_method == "polarexpress":
            # Get coefficients for the specified configuration
            from gptopt.optim.polar_express import get_coeffs_for_config
            coeffs_lists = get_coeffs_for_config(
                num_iters=self.polar_num_iters,
                safety=self.polar_safety,
                cushion=self.polar_cushion
            )
            # Create one partial function per coefficient list (for cycling)
            # This allows torch.compile to optimize each separately
            return [partial(PolarExpress, coeffs_list=coeffs) for coeffs in coeffs_lists]
        elif polar_method == "fast_polarexpress":
            return [partial(FastApplyPolarExpress, restart_interval=3, shift_eps=1e-3)]
        else:
            raise ValueError(f"Unknown polar method: {polar_method}")

    def adjust_lr_for_muon(self, lr, rms_scaling, nuclear_scaling, param_shape, grad, grad_sign):
        scale = 1.0
        if rms_scaling:
            fan_out, fan_in = param_shape[:2]
            scale *= math.sqrt(fan_out / fan_in)
        if nuclear_scaling:
            scale *= torch.trace(grad.T @ grad_sign)
        return lr * scale

    def step(self, closure=None):
        """Perform a single optimization step.
            Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
"""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                        
        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                g = p.grad
                if g is None:
                    continue
                if (g.ndim > 2) and not (self.split_heads):
                    g = g.view(g.size(0), -1)

                assert g is not None
                
                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Handle different modes for QKV splitting
                is_qkv = self.state[p].get("is_W_QKV", False)
                is_wo = self.state[p].get("is_W_O", False)
                old_shape = None
                qkv_split_indices = None
                
                # Mode-specific gradient preprocessing
                if self.muon_mode == "split_qkv" and is_qkv:
                    # Mode 2: Split QKV into Q, K, V and apply Muon to each separately
                    old_shape = g.shape
                    # Split into Q, K, V (each is [n_embd, n_embd])
                    n_embd = g.shape[1]
                    head_dim = n_embd // self.nheads
                    # Reshape to [3, nheads, head_dim, n_embd]
                    g = g.reshape(3, self.nheads, head_dim, n_embd)
                    qkv_split_indices = (0, 1, 2)  # Process Q, K, V separately
                    
                elif self.muon_mode == "voh_only" and is_qkv:
                    # Mode 3: Only apply Muon to V, handle Q and K with AdamW later
                    # Extract only V from the gradient (last third)
                    old_shape = g.shape
                    n_embd = g.shape[1]
                    head_dim = n_embd // self.nheads
                    # Get only V part: g[2*n_embd:, :]
                    g_v = g[2*n_embd:, :]
                    # Reshape V to [nheads, head_dim, n_embd] for per-head Muon
                    g = g_v.reshape(self.nheads, head_dim, n_embd)
                    qkv_split_indices = (2,)  # Only V
                    
                elif self.split_heads and is_qkv:
                    # Original split_heads mode
                    old_shape = g.shape
                    g = g.reshape(3 * self.nheads, g.shape[0] // (3 * self.nheads), g.shape[1])
                    
                elif (self.muon_mode == "split_qkv" or self.split_heads) and is_wo:
                    # Split W_O by heads
                    old_shape = g.shape
                    n_embd = g.shape[0]
                    head_dim = g.shape[1] // self.nheads
                    g = g.reshape(n_embd, self.nheads, head_dim).transpose(0, 1)

                # Use the selected polar factorization method
                import time as time_module
                pe_start = time_module.time()
                
                # Compute orthogonality error every 100 steps using cached XTX
                if not hasattr(self, '_pe_step_count'):
                    self._pe_step_count = 0
                self._pe_step_count += 1
                compute_ortho = (self._pe_step_count % 100 == 0)
                
                # Select the appropriate polar factorizer (cycles through list)
                current_factorizer = self.polar_factorizer[self.iter_counter % len(self.polar_factorizer)]
                
                # Check if using PolarExpress by seeing if it's a partial of PolarExpress
                use_polarexpress = (hasattr(current_factorizer, 'func') and 
                                   current_factorizer.func.__name__ == 'PolarExpress')
                
                if compute_ortho and use_polarexpress:
                    try:
                        # Request XTX to compute ortho error efficiently (before and after)
                        result = current_factorizer(g, group["ns_steps"], return_ortho_info=True)
                        if isinstance(result, tuple) and len(result) == 3:
                            u, XTX_before, XTX_after = result
                            # Compute ||XTX - I||_F using cached XTX values
                            with torch.no_grad():
                                u_sample = u if u.ndim == 2 else u[0]
                                XTX_before_sample = XTX_before if XTX_before.ndim == 2 else XTX_before[0]
                                XTX_after_sample = XTX_after if XTX_after.ndim == 2 else XTX_after[0]
                                I = torch.eye(XTX_after_sample.size(0), device=XTX_after_sample.device, dtype=XTX_after_sample.dtype)
                                ortho_err_before = torch.norm(XTX_before_sample - I, p='fro').item()
                                ortho_err_after = torch.norm(XTX_after_sample - I, p='fro').item()
                                
                                if not hasattr(self, '_pe_ortho_errs_before'):
                                    self._pe_ortho_errs_before = []
                                    self._pe_ortho_errs_after = []
                                    self._pe_times = []
                                self._pe_ortho_errs_before.append(ortho_err_before)
                                self._pe_ortho_errs_after.append(ortho_err_after)
                        else:
                            u = result if not isinstance(result, tuple) else result[0]
                    except Exception as e:
                        # Fallback if return_ortho_info not supported
                        u = current_factorizer(g, group["ns_steps"])
                else:
                    u = current_factorizer(g, group["ns_steps"])
                
                # Increment counter after call
                self.iter_counter += 1
                
                pe_time = (time_module.time() - pe_start) * 1000  # ms
                if hasattr(self, '_pe_times'):
                    self._pe_times.append(pe_time)
                
                # Reshape back to original shape after polar factorization
                if old_shape is not None:
                    if self.muon_mode == "split_qkv" and is_qkv:
                        # Reshape back from [3, nheads, head_dim, n_embd] to [3*n_embd, n_embd]
                        g = g.reshape(old_shape)
                        u = u.reshape(old_shape)
                    elif self.muon_mode == "voh_only" and is_qkv:
                        # u is only for V part: [nheads, head_dim, n_embd]
                        # Reshape back to [n_embd, n_embd] for V
                        u = u.reshape(old_shape[0] // 3, old_shape[1])
                        # We only update V part of the parameter
                        # The full gradient g is restored for momentum buffer update
                        g = state["momentum_buffer"]  # Use the full momentum buffer
                    elif (self.muon_mode == "split_qkv" or self.split_heads) and is_wo:
                        # Reshape back from [nheads, n_embd, head_dim] to [n_embd, nheads*head_dim]
                        g = g.transpose(0, 1).reshape(old_shape)
                        u = u.transpose(0, 1).reshape(old_shape)
                    elif self.split_heads and is_qkv:
                        # Original split_heads mode
                        g = g.reshape(old_shape)
                        u = u.reshape(old_shape)

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(
                    lr,
                    group["rms_scaling"],
                    group["nuclear_scaling"],
                    p.shape,
                    g.bfloat16(),  # convert to float16 to be compatible with u
                    u
                )
                
                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)
                
                # apply update
                if self.muon_mode == "voh_only" and is_qkv:
                    # Only update V part with Muon, Q and K will be updated with AdamW
                    n_embd = p.shape[1]
                    p.data[2*n_embd:, :].add_(u, alpha=-adjusted_lr)
                else:
                    p.data.add_(u, alpha=-adjusted_lr)
                
            ############################
            #       AdamW backup       #
            ############################

            # In voh_only mode, also handle Q and K from QKV matrices with AdamW
            adamw_params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            
            # Add Q and K parts of QKV matrices in voh_only mode
            if self.muon_mode == "voh_only":
                qkv_params_for_qk = [p for p in group["params"] if self.state[p].get("use_muon") and self.state[p].get("is_W_QKV")]
            else:
                qkv_params_for_qk = []
            
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in adamw_params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
            
            # Handle Q and K parts of QKV matrices with AdamW in voh_only mode
            for p in qkv_params_for_qk:
                g = p.grad
                if g is None:
                    continue
                
                # Extract Q and K gradients only (first 2/3 of the matrix)
                n_embd = p.shape[1]
                g_qk = g[:2*n_embd, :]
                
                # Initialize or get AdamW state for Q and K
                state = self.state[p]
                if "adamw_qk_step" not in state:
                    state["adamw_qk_step"] = 0
                    state["adamw_qk_moment1"] = torch.zeros_like(g_qk)
                    state["adamw_qk_moment2"] = torch.zeros_like(g_qk)
                
                state["adamw_qk_step"] += 1
                step = state["adamw_qk_step"]
                buf1 = state["adamw_qk_moment1"]
                buf2 = state["adamw_qk_moment2"]
                
                # Update moments
                buf1.lerp_(g_qk, 1 - beta1)
                buf2.lerp_(g_qk.square(), 1 - beta2)
                
                # Compute AdamW update
                g_qk_update = buf1 / (eps + buf2.sqrt())
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                
                # Apply weight decay and update to Q and K parts only
                p.data[:2*n_embd, :].mul_(1 - lr * weight_decay)
                p.data[:2*n_embd, :].add_(g_qk_update, alpha=-lr / scale)
                    
        return loss



