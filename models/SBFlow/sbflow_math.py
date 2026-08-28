import warnings

import numpy as np
import torch


def lerp(a, b, t):
    return a + t * (b - a)


def sbflow_target(x0, x1, t, noise, sigma):
    t_view = t.view(-1, 1, 1, 1)
    mu_t = lerp(x0, x1, t_view)
    bridge_var = (t_view * (1.0 - t_view)).clamp_min(0.0)
    sigma_t = sigma * torch.sqrt(bridge_var)
    xt = mu_t + sigma_t * noise
    coeff = (1.0 - 2.0 * t_view) / (2.0 * t_view * (1.0 - t_view) + 1e-8)
    target = coeff * (xt - mu_t) + x1 - x0
    return xt, target


def _flatten_batch(x):
    if x.dim() > 2:
        return x.reshape(x.shape[0], -1)
    return x


class SBFlowOTPlanSampler:
    def __init__(self, method, reg=0.05, num_threads=1, warn=True):
        if method not in {"exact", "sinkhorn"}:
            raise ValueError(f"Unknown SBFlow OT method: {method}")
        try:
            pot = __import__("ot")
        except ImportError as exc:
            raise ImportError(
                "SBFlow OT coupling requires POT and its dependencies. "
                "Install a working POT stack before setting sbflow_use_ot=True."
            ) from exc

        self.method = method
        self.reg = float(reg)
        self.num_threads = num_threads
        self.warn = bool(warn)
        self.pot = pot

    def get_map(self, x0, x1):
        a = self.pot.unif(x0.shape[0])
        b = self.pot.unif(x1.shape[0])
        x0_flat = _flatten_batch(x0)
        x1_flat = _flatten_batch(x1)
        cost = torch.cdist(x0_flat, x1_flat) ** 2
        cost_np = cost.detach().cpu().numpy()

        if self.method == "exact":
            plan = self.pot.emd(a, b, cost_np, numThreads=self.num_threads)
        else:
            plan = self.pot.sinkhorn(a, b, cost_np, reg=self.reg)

        if not np.all(np.isfinite(plan)) or np.abs(plan.sum()) < 1e-8:
            if self.warn:
                warnings.warn(
                    "Numerical errors in SBFlow OT plan, reverting to uniform plan."
                )
            plan = np.ones_like(plan) / plan.size
        return plan

    def sample_map(self, plan, batch_size, replace=True):
        probs = plan.flatten()
        probs = probs / probs.sum()
        choices = np.random.choice(
            plan.shape[0] * plan.shape[1], p=probs, size=batch_size, replace=replace
        )
        return np.divmod(choices, plan.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        plan = self.get_map(x0, x1)
        i, j = self.sample_map(plan, x0.shape[0], replace=replace)
        return x0[i], x1[j]
