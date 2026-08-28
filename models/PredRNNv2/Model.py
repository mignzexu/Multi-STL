import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn, optim

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_ROOT = PROJECT_ROOT / "models"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = [str(MODELS_ROOT)]
    sys.modules.setdefault("models", models_pkg)
    package_name = "models.PredRNNv2"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules.setdefault(package_name, package)
    __package__ = package_name

from ..Model_system import System
from .PredRNNv2_model import PredRNNv2_Model
from .utils import reshape_patch


def _as_config_dict(configs):
    return dict(vars(configs)) if not isinstance(configs, dict) else dict(configs)


class ScheduledSamplingWrapper(nn.Module):
    def __init__(self, configs, core_model_cls):
        super().__init__()
        self.configs = _as_config_dict(configs)
        self.label_idx = self.configs["label_idx"]
        self.pre_seq_length = self.configs["total_seq"][0]
        self.aft_seq_length = self.configs["total_seq"][1]
        self.img_channel = len(self.configs["in_category"])
        self.img_height, self.img_width = self.configs["img_size"]
        self.patch_size = self.configs["patch_size"]
        num_hidden = [int(x) for x in self.configs["num_hidden"].split(",")]
        self.core_model = core_model_cls(len(num_hidden), num_hidden, self.configs)

    @property
    def total_length(self):
        return self.pre_seq_length + self.aft_seq_length

    def _future_frames(self, batch_x, batch_y):
        if batch_y is None:
            return torch.zeros(batch_x.shape[0], self.aft_seq_length, self.img_channel, batch_x.shape[-2], batch_x.shape[-1], device=batch_x.device, dtype=batch_x.dtype)
        if batch_y.shape[2] == self.img_channel:
            return batch_y
        future = torch.zeros(batch_y.shape[0], batch_y.shape[1], self.img_channel, batch_y.shape[-2], batch_y.shape[-1], device=batch_y.device, dtype=batch_y.dtype)
        future[:, :, self.label_idx[0] : self.label_idx[1]] = batch_y
        return future

    def _zeros_flag(self, batch_size, length, device, dtype):
        return torch.zeros(batch_size, length, self.img_height // self.patch_size, self.img_width // self.patch_size, self.patch_size * self.patch_size * self.img_channel, device=device, dtype=dtype)

    def _inference_flag(self, batch_x):
        mask_input = 1 if self.configs.get("reverse_scheduled_sampling", 0) == 1 else self.pre_seq_length
        flag = self._zeros_flag(batch_x.shape[0], self.total_length - mask_input - 1, batch_x.device, batch_x.dtype)
        if self.configs.get("reverse_scheduled_sampling", 0) == 1:
            flag[:, : self.pre_seq_length - 1] = 1.0
        return flag

    def _scheduled_flag(self, batch_x, eta, step):
        if self.configs.get("reverse_scheduled_sampling", 0) == 1:
            return eta, self._reverse_scheduled_flag(batch_x, step)
        if not self.configs.get("scheduled_sampling", 1):
            return 0.0, self._zeros_flag(batch_x.shape[0], self.aft_seq_length - 1, batch_x.device, batch_x.dtype)
        eta = max(eta - self.configs["sampling_changing_rate"], 0.0) if step < self.configs["sampling_stop_iter"] else 0.0
        token = (torch.rand(batch_x.shape[0], self.aft_seq_length - 1, device=batch_x.device) < eta).to(batch_x.dtype)
        return eta, token[:, :, None, None, None] * torch.ones_like(self._zeros_flag(batch_x.shape[0], self.aft_seq_length - 1, batch_x.device, batch_x.dtype))

    def _reverse_scheduled_flag(self, batch_x, step):
        step_1, step_2 = self.configs["r_sampling_step_1"], self.configs["r_sampling_step_2"]
        if step < step_1:
            r_eta, eta = 0.5, 0.5
        elif step < step_2:
            r_eta = 1.0 - 0.5 * torch.exp(torch.tensor(-(step - step_1) / float(self.configs["r_exp_alpha"]), device=batch_x.device)).item()
            eta = 0.5 - (0.5 / (step_2 - step_1)) * (step - step_1)
        else:
            r_eta, eta = 1.0, 0.0
        flag = self._zeros_flag(batch_x.shape[0], self.total_length - 2, batch_x.device, batch_x.dtype)
        flag[:, : self.pre_seq_length - 1] = (torch.rand(batch_x.shape[0], self.pre_seq_length - 1, device=batch_x.device) < r_eta).to(batch_x.dtype)[:, :, None, None, None]
        flag[:, self.pre_seq_length - 1 :] = (torch.rand(batch_x.shape[0], self.aft_seq_length - 1, device=batch_x.device) < eta).to(batch_x.dtype)[:, :, None, None, None]
        return flag

    def forward(self, batch_x, batch_y=None, real_input_flag=None, return_loss=False):
        future = self._future_frames(batch_x, batch_y)
        frames = torch.cat([batch_x, future], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        frames = reshape_patch(frames, self.patch_size)
        if real_input_flag is None:
            real_input_flag = self._inference_flag(batch_x)
        return self.core_model(frames, real_input_flag, return_loss=return_loss)


class Model(System):
    def __init__(self, configs):
        super().__init__(configs)
        self.eta = float(getattr(configs, "sampling_start_value", 1.0))

    def get_model(self):
        return ScheduledSamplingWrapper(self.configs, PredRNNv2_Model)

    def _run_model(self, batch_x, batch_y, training):
        if training:
            self.eta, flag = self.model._scheduled_flag(batch_x, self.eta, int(getattr(self, "global_step", 0)))
        else:
            flag = self.model._inference_flag(batch_x)
        return self.model(batch_x, batch_y, real_input_flag=flag, return_loss=True)

    def _loss(self, result, label):
        loss = self.criterion(result["pred"], label)
        decouple_loss = result.get("decouple_loss")
        if isinstance(decouple_loss, torch.Tensor):
            loss = loss + float(getattr(self.configs, "decouple_beta", 0.0)) * decouple_loss
        return loss

    def forward(self, batch_x, batch_y=None, **kwargs):
        if self.test_seq <= self.aft_seq_length:
            return self.model(batch_x, batch_y, return_loss=False)["pred"][:, : self.test_seq]
        if len(self.configs.out_category) != len(self.configs.in_category):
            raise ValueError("Rolling prediction requires matching input and output channels.")
        pred_y, cur_seq, remaining = [], batch_x, self.test_seq
        while remaining > 0:
            out_seq = self.model(cur_seq, None, return_loss=False)["pred"]
            take = min(remaining, out_seq.shape[1])
            pred_y.append(out_seq[:, :take])
            remaining -= take
            cur_seq = torch.cat([cur_seq[:, out_seq.shape[1] :], out_seq], dim=1) if self.pre_seq_length > out_seq.shape[1] else out_seq[:, -self.pre_seq_length :]
        return torch.cat(pred_y, dim=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate, weight_decay=self.configs.weight_decay)
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        result = self._run_model(batch_x, batch_y, training=True)
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        loss = self._loss(result, label)
        return {"loss": loss, "train_loss": loss.detach(), "batch_size": batch_x.shape[0]}

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        result = self._run_model(batch_x, batch_y, training=False)
        pred = result["pred"]
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        loss = self._loss(result, label)
        return {"val_loss": loss.detach(), "batch_size": batch_x.shape[0], "output": pred.detach(), "label": label.detach()}


def _build_debug_configs():
    debug_dir = Path(__file__).resolve().parent / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(total_seq=[2, 2], test_seq=2, batch_size=2, label_idx=[0, 1], in_category=["tp"], out_category=["tp"], img_size=[8, 8], reverse_scheduled_sampling=0, r_sampling_step_1=25000, r_sampling_step_2=50000, r_exp_alpha=5000, scheduled_sampling=1, sampling_stop_iter=50000, sampling_start_value=1.0, sampling_changing_rate=0.00002, num_hidden="4,4", filter_size=3, stride=1, patch_size=2, layer_norm=0, decouple_beta=0.1, learning_rate=5e-4, weight_decay=0.0, epoch=1, std_method="z_score", std_params={"dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]}, "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]}}, threshold=[[0.5]], metrics=["mae"], obj_dir=str(debug_dir), save_interval=1)


def _summarize_gradients(module, tag):
    grad_param_count, grad_abs_sum = 0, 0.0
    for param in module.parameters():
        if param.grad is not None:
            grad_param_count += 1
            grad_abs_sum += param.grad.detach().abs().sum().item()
    if grad_param_count == 0 or grad_abs_sum == 0.0:
        raise RuntimeError(f"{tag} backward did not produce parameter gradients.")
    return grad_param_count, grad_abs_sum


if __name__ == "__main__":
    torch.manual_seed(0)
    configs = _build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(configs).to(device)
    batch_x = torch.randn(2, configs.total_seq[0], 1, 8, 8, device=device, requires_grad=True)
    batch_y = torch.randn(2, configs.total_seq[1], 1, 8, 8, device=device)
    debug_model.zero_grad(set_to_none=True)
    train_out = debug_model.training_step((batch_x, batch_y), 0)
    train_out["loss"].backward()
    grad_count, grad_sum = _summarize_gradients(debug_model.model, "training_step")
    print(f"[training_step] loss: {train_out['loss'].item():.6f}, grad params: {grad_count}, grad sum: {grad_sum:.4f}")
    debug_model.zero_grad(set_to_none=True)
    forward_input = batch_x.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_input)
    forward_output.square().mean().backward()
    grad_count, grad_sum = _summarize_gradients(debug_model.model, "forward")
    print(f"[forward] output shape: {tuple(forward_output.shape)}, grad params: {grad_count}, grad sum: {grad_sum:.4f}")
    print("All checks passed!")
