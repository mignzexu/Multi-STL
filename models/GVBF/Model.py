import torch
import torch.nn.functional as F
import torchode as to
from torch import optim

from ..Model_system import System
from .gvbf_network import get_model as _gvbf_get_model


def _lerp(a, b, t):
    return a + t * (b - a)


def _joint_ode(model, start, end):
    t_0, a_0 = start
    t_1, a_1 = end

    def _func(x, k):
        t = _lerp(t_0, t_1, k)
        alpha = _lerp(a_0, a_1, k)
        v, d = torch.chunk(model(x, t, alpha), 2, dim=1)
        return v * (t_1 - t_0) + d * (a_1 - a_0)

    return _func


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
        return F.interpolate(x, (H_r, W_r), mode="bilinear"), ("resize", H, W)
    if H == H_p and W == W_p:
        return x, None
    return F.pad(x, (0, W_p - W, 0, H_p - H), mode="replicate"), ("crop", H, W)


def _unpad_tensor(x, original):
    if original is None:
        return x
    mode, H, W = original
    if mode == "crop":
        return x[:, :, :H, :W]
    if x.shape[2] == H and x.shape[3] == W:
        return x
    return F.interpolate(x, (H, W), mode="bilinear")


_MODE_DEFAULTS = {
    "flow": "1_1_64",
    "biflow": "1_2_64_cond",
    "condiff": "2_1_64",
}


class Model(System):
    def __init__(self, configs):
        self._mode = getattr(configs, "gvbf_mode", "biflow")
        self._noise_level = float(getattr(configs, "gvbf_noise_level", 0.1))
        self._model_cfg_str = getattr(
            configs, "gvbf_model_config", _MODE_DEFAULTS.get(self._mode, "1_2_64_cond")
        )
        self._weights_path = getattr(configs, "gvbf_weights_path", None)
        self._max_size = int(getattr(configs, "gvbf_max_size", 0))
        self._solver_atol = float(getattr(configs, "gvbf_solver_atol", 1e-2))
        self._solver_rtol = float(getattr(configs, "gvbf_solver_rtol", 1e-2))
        self._solver_dt0 = getattr(configs, "gvbf_solver_dt0", None)
        if self._solver_dt0 is not None:
            self._solver_dt0 = float(self._solver_dt0)
        super().__init__(configs)
        self.criterion = torch.nn.MSELoss()

    def get_model(self):
        from types import SimpleNamespace

        cfg = SimpleNamespace(
            config=self._model_cfg_str, weights_path=self._weights_path
        )
        return _gvbf_get_model(cfg)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=getattr(self.configs, "weight_decay", 0.0),
        )
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    # ── loss helpers ──────────────────────────────────────────────

    def _flow_loss(self, x0, x1):
        B, device = x0.shape[0], x0.device
        x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
        x1_p, _ = _pad_tensor(x1, max_size=self._max_size)
        t = torch.rand(B, device=device)
        xt = _lerp(x0_p, x1_p, t.view(-1, 1, 1, 1))
        v_p = self.model(xt, t)
        v = _unpad_tensor(v_p, orig)
        loss = torch.mean((v_p - (x1_p - x0_p)) ** 2)
        return loss, v

    def _biflow_loss(self, x0, x1):
        B, device = x0.shape[0], x0.device
        x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
        x1_p, _ = _pad_tensor(x1, max_size=self._max_size)
        t = torch.rand(B, device=device)
        alpha = torch.rand(B, device=device)
        noise = torch.randn_like(x0_p)
        xt = _lerp(x0_p, x1_p, t.view(-1, 1, 1, 1))
        xta = xt + alpha.view(-1, 1, 1, 1) * noise
        output = self.model(xta, t, alpha)
        v_p, d = torch.chunk(output, 2, dim=1)
        v = _unpad_tensor(v_p, orig)
        loss_v = torch.mean((v_p - (x1_p - x0_p)) ** 2)
        loss_d = torch.mean((d - noise) ** 2)
        return loss_v + loss_d, v

    def _condiff_loss(self, x0, x1):
        B, device = x0.shape[0], x0.device
        x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
        x1_p, _ = _pad_tensor(x1, max_size=self._max_size)
        alpha = torch.rand(B, device=device)
        noise = torch.randn_like(x1_p)
        xa = _lerp(noise, x1_p, alpha.view(-1, 1, 1, 1))
        d_p = self.model(torch.cat([xa, x0_p], dim=1), alpha)
        d = _unpad_tensor(d_p, orig)
        loss = torch.mean((d_p - (x1_p - noise)) ** 2)
        return loss, d

    # ── inference ─────────────────────────────────────────────────

    def _predict_next_frame(self, x0):
        device, B = x0.device, x0.shape[0]
        x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
        C, H, W = x0_p.shape[1:]

        if self._mode == "flow":
            def _ode_f(t, y):
                y = y.reshape(-1, C, H, W)
                return self.model(y, t).flatten(start_dim=1)

            y0 = x0_p.flatten(start_dim=1)

        elif self._mode == "biflow":
            ode_f = _joint_ode(self.model, (0.0, self._noise_level), (1.0, 0.0))

            def _ode_f(t, y):
                y = y.reshape(-1, C, H, W)
                return ode_f(y, t).flatten(start_dim=1)

            y0 = (
                x0_p + torch.randn_like(x0_p) * self._noise_level
            ).flatten(start_dim=1)

        elif self._mode == "condiff":
            def _ode_f(t, y):
                y = y.reshape(-1, C, H, W)
                return self.model(
                    torch.cat([y, x0_p], dim=1), t
                ).flatten(start_dim=1)

            y0 = torch.randn_like(x0_p).flatten(start_dim=1)

        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        term = to.ODETerm(_ode_f)
        step_method = to.Heun(term)

        if self._solver_dt0 is not None:
            controller = to.FixedStepController()
            dt0 = torch.full((B,), self._solver_dt0, device=device)
        else:
            controller = to.IntegralController(
                atol=self._solver_atol, rtol=self._solver_rtol, term=term
            )
            dt0 = None

        adjoint = to.AutoDiffAdjoint(step_method, controller)

        t_eval = torch.linspace(0, 1, 2, device=device)[None, :].repeat(B, 1)
        problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)

        with torch.no_grad():
            sol = adjoint.solve(problem, dt0=dt0)

        result = sol.ys.reshape(B, 2, C, H, W)[:, -1]
        return _unpad_tensor(result, orig)

    # ── lightning hooks ───────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        x0 = batch_x[:, -1]
        x1 = batch_y[:, 0]

        if self._mode == "flow":
            loss, _ = self._flow_loss(x0, x1)
        elif self._mode == "biflow":
            loss, _ = self._biflow_loss(x0, x1)
        elif self._mode == "condiff":
            loss, _ = self._condiff_loss(x0, x1)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        x0 = batch_x[:, -1]
        x1 = batch_y[:, 0]

        if self._mode == "flow":
            loss, v = self._flow_loss(x0, x1)
        elif self._mode == "biflow":
            loss, v = self._biflow_loss(x0, x1)
        elif self._mode == "condiff":
            loss, d = self._condiff_loss(x0, x1)
            v = d
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

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


if __name__ == "__main__":
    import json, os
    from pathlib import Path
    from types import SimpleNamespace

    torch.manual_seed(0)

    debug_dir = Path(__file__).resolve().parent / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    _BASE_CONFIG = SimpleNamespace(
        total_seq=[2, 2],
        test_seq=2,
        label_idx=[0, 1],
        in_category=["tp"],
        out_category=["tp"],
        learning_rate=1e-4,
        weight_decay=0.0,
        epoch=1,
        std_method="z_score",
        std_params={
            "dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]},
            "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]},
        },
        threshold=[[0.5]],
        metrics=["mae"],
        batch_size=2,
    )

    def summarize_gradients(module, tag):
        n = 0
        s = 0.0
        for _, p in module.named_parameters():
            if p.grad is None:
                continue
            n += 1
            s += p.grad.detach().abs().sum().item()
        if n == 0:
            raise RuntimeError(f"{tag}: no parameter gradients")
        if s == 0.0:
            raise RuntimeError(f"{tag}: all gradients are zero")
        return n, s

    def ensure_input_grad(tensor, tag):
        if tensor.grad is None:
            raise RuntimeError(f"{tag}: no input gradients")
        if not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"{tag}: non-finite input gradients")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    total = 0
    passed = 0

    for mode, model_cfg in [
        ("flow", "1_1_64"),
        ("biflow", "1_2_64_cond"),
        ("condiff", "2_1_64"),
    ]:
        for img_name, H, W, max_sz in [
            ("WB_64x128", 64, 128, 0),
            ("WB_32x64", 32, 64, 0),
            ("SD_500x900", 500, 900, 512),
        ]:
            total += 1
            case_dir = debug_dir / f"{mode}_{img_name}"
            case_dir.mkdir(parents=True, exist_ok=True)
            configs = SimpleNamespace(
                **vars(_BASE_CONFIG),
                obj_dir=str(case_dir),
                img_size=[H, W],
                gvbf_mode=mode,
                gvbf_model_config=model_cfg,
                gvbf_max_size=max_sz,
            )
            try:
                model = Model(configs).to(device)
                model.train()
                B, C = 2, 1
                bx = torch.randn(B, 2, C, H, W, device=device)
                by = torch.randn(B, 2, C, H, W, device=device)

                model.zero_grad(set_to_none=True)
                ts_out = model.training_step((bx, by), 0)
                ts_out["loss"].backward()
                n, s = summarize_gradients(model.model, "training_step")

                model.eval()
                with torch.no_grad():
                    fwd = model.forward(bx)

                model.train()
                vs_out = model.validation_step((bx, by), 0)

                print(
                    f"  [{mode:8s}] {img_name:12s} OK  "
                    f"loss={ts_out['loss'].item():.3f} "
                    f"grads={n} "
                    f"fwd={list(fwd.shape)} "
                    f"val={vs_out['val_loss'].item():.3f}"
                )
                passed += 1
            except Exception as e:
                print(f"  [{mode:8s}] {img_name:12s} FAIL  {e}")
            finally:
                del model
                torch.cuda.empty_cache()

    print(f"\n{passed}/{total} checks passed")
    if passed < total:
        raise SystemExit(1)
