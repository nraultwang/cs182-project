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
                ):
        """
        Arguments:
            polar_method: The name of the polar factorization method to use (e.g., "Keller", "polarexpress")
            polar_num_iters: Number of iterations for PolarExpress coefficients (3, 5, or 7). None uses default 8-iter config.
            polar_safety: Safety factor for PolarExpress (1.0 or 1.01)
            polar_cushion: Cushion parameter for PolarExpress (0.1, 0.05, or 0.024)
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
        
        # print("EMBED TOKENS AND LM_HEAD ARE NOT HANDLED CORRECTLY FOR MUON, THEY SHOULD BE WITH ADAMW.")
        muon_params, muon_params_names = [], []
        adamw_params, adamw_params_names = [], []
        for name, p in named_params:
            if p.ndim >= 2 and not any(excluded in name for excluded in ["embeddings", "embed_tokens", "wte", "lm_head", "wpe"]):
                muon_params.append(p)
                muon_params_names.append(name)
            else:
                adamw_params.append(p)
                adamw_params_names.append(name)
        params = list(muon_params)
        params.extend(adamw_params)
        self.split_heads = split_heads
        if self.split_heads:
            assert nheads is not None, "nheads must be specified if split_heads is True"
            self.nheads = nheads
        super().__init__(params, defaults)
        
        # Sort parameters into those for which we will use Muon, and those for which we will not
# Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
        for p, p_name in zip(muon_params, muon_params_names):
            if not self.split_heads: assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
            if p_name.endswith("attn.c_attn.weight"):
                self.state[p]["is_W_QKV"] = True
            elif p_name.endswith("attn.c_proj.weight"):
                self.state[p]["is_W_O"] = True

        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

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

                if self.split_heads and self.state[p].get("is_W_QKV", False):
                    # For W_QKV, we split the gradients into 3 heads and process them separately
                    # print("before", g.shape, self.nheads)
                    old_shape = g.shape
                    g = g.reshape(3 * self.nheads, g.shape[0] // (3 * self.nheads), g.shape[1])
                    # print("after", g.shape)
                elif self.split_heads and self.state[p].get("is_W_O", False) and self.split_heads:
                    # print("before", g.shape, self.nheads)
                    old_shape = g.shape
                    g = g.reshape(g.shape[0], self.nheads, g.shape[1] // self.nheads).transpose(0, 1)
                    # print("after", g.shape)
                    # For W_O, we split the gradients into 3 heads and process them separately

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
                
                if self.split_heads and self.state[p].get("is_W_QKV", False):
                    g = g.reshape(old_shape)
                    u = u.reshape(old_shape)
                elif self.split_heads and self.state[p].get("is_W_O", False):
                    g = g.transpose(0, 1).reshape(old_shape)
                    u = u.transpose(0, 1).reshape(old_shape)

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
                p.data.add_(u, alpha=-adjusted_lr)
                
            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["weight_decay"]

            for p in params:
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
                    
        return loss



