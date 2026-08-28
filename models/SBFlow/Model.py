import torch
import torch.nn.functional as F
import torchode as to
from torch import optim

from ..Model_system import System
from .sbflow_math import SBFlowOTPlanSampler
from .sbflow_math import sbflow_target


def _pad_size(size, multiple=64):
    return ((size + multiple - 1) // multiple) * multiple


def _pad_tensor(x, multiple=64, max_size=0):
    _, _, H, W = x.shape
    H_p = _pad_size(H, multiple)
    W_p = _pad_size(W, multiple)
    if max_size > 0 and max(H_p, W_p) > max_size:
        scale = max_size / max(H_p, W_p)
        H_r = _pad_size(round(H * scale), multiple)
        W_r = _pad_size(round(W * scale), multiple)
        return F.interpolate(x, (H_r, W_r), mode="bilinear"), (H, W)
    if H == H_p and W == W_p:
        return x, None
    return F.pad(x, (0, W_p - W, 0, H_p - H), mode="replicate"), (H, W)


def _unpad_tensor(x, original):
    if original is None:
        return x
    H, W = original
    if x.shape[2] == H and x.shape[3] == W:
        return x
    return F.interpolate(x, (H, W), mode="bilinear")


class Model(System):
    def __init__(self, configs):
        self._model_cfg_str = getattr(configs, "sbflow_model_config", "1_1_64")
        self._weights_path = getattr(configs, "sbflow_weights_path", None)
        self._max_size = int(getattr(configs, "sbflow_max_size", 0))
        self._sigma = float(getattr(configs, "sbflow_sigma", 1.0))
        if self._sigma <= 0:
            raise ValueError(f"sbflow_sigma must be strictly positive, got {self._sigma}.")
        self._t_eps = float(getattr(configs, "sbflow_t_eps", 1e-4))
        if not 0.0 <= self._t_eps < 0.5:
            raise ValueError(f"sbflow_t_eps must be in [0, 0.5), got {self._t_eps}.")
        self._use_ot = bool(getattr(configs, "sbflow_use_ot", False))
        self._ot_method = getattr(configs, "sbflow_ot_method", "sinkhorn")
        self._ot_replace = bool(getattr(configs, "sbflow_ot_replace", True))
        self._ot_num_threads = getattr(configs, "sbflow_ot_num_threads", 1)
        self._ot_sampler = None
        if self._use_ot:
            self._ot_sampler = SBFlowOTPlanSampler(
                method=self._ot_method,
                reg=2.0 * self._sigma ** 2,
                num_threads=self._ot_num_threads,
            )
        self._solver_atol = float(getattr(configs, "sbflow_solver_atol", 1e-2))
        self._solver_rtol = float(getattr(configs, "sbflow_solver_rtol", 1e-2))
        self._solver_dt0 = getattr(configs, "sbflow_solver_dt0", None)
        if self._solver_dt0 is not None:
            self._solver_dt0 = float(self._solver_dt0)
        super().__init__(configs)
        self.criterion = torch.nn.MSELoss()

    def get_model(self):
        from types import SimpleNamespace

        try:
            from .sbflow_network import get_model as _sbflow_get_model
        except ImportError as exc:
            raise ImportError(
                "SBFlow requires its local network runtime dependencies such as "
                "diffusers and safetensors to construct the model."
            ) from exc

        cfg = SimpleNamespace(
            config=self._model_cfg_str, weights_path=self._weights_path
        )
        return _sbflow_get_model(cfg)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=getattr(self.configs, "weight_decay", 0.0),
        )
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    def _sample_time(self, batch_size, device):
        if self._t_eps == 0.0:
            return torch.rand(batch_size, device=device)
        span = 1.0 - 2.0 * self._t_eps
        return self._t_eps + span * torch.rand(batch_size, device=device)

    def _sbflow_target(self, x0, x1, t, noise):
        return sbflow_target(x0, x1, t, noise, self._sigma)

    def _sbflow_loss(self, x0, x1, return_endpoint_velocity=False):
        batch_size, device = x0.shape[0], x0.device
        x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
        x1_p, _ = _pad_tensor(x1, max_size=self._max_size)
        if self._ot_sampler is not None:
            x0_p, x1_p = self._ot_sampler.sample_plan(
                x0_p, x1_p, replace=self._ot_replace
            )
        t = self._sample_time(batch_size, device)
        noise = torch.randn_like(x0_p)
        xt, target = self._sbflow_target(x0_p, x1_p, t, noise)
        v_p = self.model(xt, t)
        loss = torch.mean((v_p - target) ** 2)

        if not return_endpoint_velocity:
            return loss, None

        with torch.no_grad():
            v0_p = self.model(x0_p, torch.zeros(batch_size, device=device))
            v0 = _unpad_tensor(v0_p, orig)
        return loss, v0

    @torch.no_grad()
    def _predict_next_frame(self, x0):
        device, batch_size = x0.device, x0.shape[0]
        channels = x0.shape[1]
        x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
        _, _, padded_h, padded_w = x0_p.shape

        def _ode_f(t, y):
            y = y.reshape(-1, channels, padded_h, padded_w)
            return self.model(y, t).flatten(start_dim=1)

        term = to.ODETerm(_ode_f)
        step_method = to.Heun(term)

        if self._solver_dt0 is not None:
            controller = to.FixedStepController()
            dt0 = torch.full((batch_size,), self._solver_dt0, device=device)
        else:
            controller = to.IntegralController(
                atol=self._solver_atol, rtol=self._solver_rtol, term=term
            )
            dt0 = None

        adjoint = to.AutoDiffAdjoint(step_method, controller)
        t_eval = torch.linspace(0, 1, 2, device=device)[None, :].repeat(batch_size, 1)
        problem = to.InitialValueProblem(y0=x0_p.flatten(start_dim=1), t_eval=t_eval)
        sol = adjoint.solve(problem, dt0=dt0)

        result = sol.ys.reshape(batch_size, 2, channels, padded_h, padded_w)[:, -1]
        return _unpad_tensor(result, orig)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        x0 = batch_x[:, -1]
        x1 = batch_y[:, 0]
        loss, _ = self._sbflow_loss(x0, x1)

        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        x0 = batch_x[:, -1]
        x1 = batch_y[:, 0]
        loss, v = self._sbflow_loss(x0, x1, return_endpoint_velocity=True)
        if v is None:
            raise RuntimeError("SBFlow validation expected endpoint velocity output.")

        with torch.no_grad():
            nc, lo, hi = v.shape[1], self.label_idx[0], self.label_idx[1]
            out_slice = slice(lo, min(hi, nc))
            pred = (x0 + v).unsqueeze(1)[:, :, out_slice]
            label = x1.unsqueeze(1)[:, :, lo:hi]

        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred,
            "label": label,
        }

    def forward(self, batch_x, batch_y=None, **kwargs):
        x0 = batch_x[:, -1]
        n_pred = min(self.test_seq, self.aft_seq_length)

        frames = []
        current = x0
        for _ in range(n_pred):
            frame = self._predict_next_frame(current)
            frames.append(frame)
            current = frame

        pred_y = torch.stack(frames, dim=1)
        if self.test_seq < self.aft_seq_length:
            pred_y = pred_y[:, : self.test_seq]
        lo, hi = self.label_idx[0], self.label_idx[1]
        pred_y = pred_y[:, :, lo:hi, :, :]
        return pred_y
