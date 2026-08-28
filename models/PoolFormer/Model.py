# pyright: reportMissingImports=false, reportIncompatibleMethodOverride=false
import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.Model_system import System
    gSTA_Model = importlib.import_module("Poolformer_model").gSTA_Model
else:
    from ..Model_system import System
    from .Poolformer_model import gSTA_Model


class Model(System):
    def __init__(self, configs):
        super().__init__(configs)

    def get_model(self):
        return gSTA_Model(vars(self.configs))

    def _predict(self, batch_x):
        return self.model(batch_x)["pred"]

    def forward(self, batch_x, batch_y=None, **kwargs):
        if self.test_seq > self.aft_seq_length:
            if len(self.configs.out_category) != len(self.configs.in_category):
                raise ValueError("输入输出通道一致才能滚动预测。")
            return self._rollout(batch_x)

        pred_y = self._predict(batch_x)
        if self.test_seq < self.aft_seq_length:
            pred_y = pred_y[:, :self.test_seq]

        return pred_y

    def _rollout(self, batch_x):
        if self.pre_seq_length < self.aft_seq_length:
            pred_y = []
            repeats = self.test_seq // self.pre_seq_length
            remainder = self.test_seq % self.pre_seq_length
            cur_seq = batch_x.clone()

            for _ in range(repeats):
                cur_seq = self._predict(cur_seq)[:, :self.pre_seq_length]
                pred_y.append(cur_seq)

            if remainder != 0:
                cur_seq = self._predict(cur_seq)
                pred_y.append(cur_seq[:, :remainder])

            return torch.cat(pred_y, dim=1)

        if self.pre_seq_length > self.aft_seq_length:
            pred_y = []
            repeats = self.test_seq // self.aft_seq_length
            remainder = self.test_seq % self.aft_seq_length
            differ = self.pre_seq_length - self.aft_seq_length
            in_seq = batch_x.clone()

            for _ in range(repeats):
                out_seq = self._predict(in_seq)
                pred_y.append(out_seq)
                in_seq = torch.cat((in_seq[:, differ:], out_seq), dim=1)

            if remainder != 0:
                out_seq = self._predict(in_seq)
                pred_y.append(out_seq[:, :remainder])

            return torch.cat(pred_y, dim=1)

        pred_y = []
        repeats = self.test_seq // self.pre_seq_length
        remainder = self.test_seq % self.pre_seq_length
        cur_seq = batch_x.clone()

        for _ in range(repeats):
            cur_seq = self._predict(cur_seq)
            pred_y.append(cur_seq)

        if remainder != 0:
            cur_seq = self._predict(cur_seq)
            pred_y.append(cur_seq[:, :remainder])

        return torch.cat(pred_y, dim=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self.model(batch_x)["pred"]
        label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.criterion(pred, label)
        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self.model(batch_x)["pred"]
        label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.criterion(pred, label)
        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred.detach(),
            "label": label.detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)

    def build_debug_configs():
        debug_dir = Path(__file__).resolve().parent / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            total_seq=[2, 2],
            test_seq=2,
            label_idx=[0, 1],
            in_category=["tp"],
            out_category=["tp"],
            img_size=[8, 8],
            spatio_kernel_enc=3,
            spatio_kernel_dec=3,
            hid_S=4,
            hid_T=8,
            N_T=2,
            N_S=2,
            mlp_ratio=2.0,
            drop=0.0,
            drop_path=0.0,
            learning_rate=5e-4,
            weight_decay=0.0,
            epoch=1,
            batch_size=2,
            std_method="z_score",
            std_params={
                "dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]},
                "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]},
            },
            threshold=[[0.5]],
            metrics=["mae"],
            obj_dir=str(debug_dir),
        )

    def summarize_gradients(module, tag):
        grad_param_count = 0
        grad_abs_sum = 0.0
        for _, param in module.named_parameters():
            if param.grad is None:
                continue
            grad_param_count += 1
            grad_abs_sum += param.grad.detach().abs().sum().item()
        if grad_param_count == 0:
            raise RuntimeError(f"{tag} backward did not produce parameter gradients.")
        if grad_abs_sum == 0.0:
            raise RuntimeError(f"{tag} backward gradients are all zeros.")
        return grad_param_count, grad_abs_sum

    def ensure_input_grad(tensor, tag):
        if tensor.grad is None:
            raise RuntimeError(f"{tag} backward did not produce input gradients.")
        if not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"{tag} input gradients contain non-finite values.")

    def ensure_shape(tensor, expected_shape, tag):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"{tag} expected shape {expected_shape}, but got {tuple(tensor.shape)}.")

    configs = build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(configs).to(device)
    debug_model.train()

    input_tensor = torch.randn(2, configs.total_seq[0], 1, 8, 8, device=device)
    label_tensor = torch.randn(2, configs.total_seq[1], 1, 8, 8, device=device)
    expected_shape = (2, configs.total_seq[1], 1, 8, 8)
    debug_summary: dict[str, object] = {"device": str(device), "input_shape": list(input_tensor.shape)}

    debug_model.zero_grad(set_to_none=True)
    model_input = input_tensor.detach().clone().requires_grad_(True)
    model_output = debug_model.model(model_input)["pred"]
    ensure_shape(model_output, expected_shape, "self.model")
    model_loss = model_output.square().mean()
    model_loss.backward()
    ensure_input_grad(model_input, "self.model")
    grad_count, grad_sum = summarize_gradients(debug_model.model, "self.model")
    debug_summary["self.model"] = {
        "output_shape": list(model_output.shape),
        "grad_param_count": grad_count,
        "grad_abs_sum": grad_sum,
    }

    debug_model.zero_grad(set_to_none=True)
    forward_input = input_tensor.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_input)
    ensure_shape(forward_output, expected_shape, "forward")
    forward_loss = forward_output.square().mean()
    forward_loss.backward()
    ensure_input_grad(forward_input, "forward")
    grad_count, grad_sum = summarize_gradients(debug_model.model, "forward")
    debug_summary["forward"] = {
        "output_shape": list(forward_output.shape),
        "grad_param_count": grad_count,
        "grad_abs_sum": grad_sum,
    }

    debug_model.zero_grad(set_to_none=True)
    train_output = debug_model.training_step((input_tensor, label_tensor), 0)
    train_output["loss"].backward()
    summarize_gradients(debug_model.model, "training_step")
    debug_summary["training_step_keys"] = sorted(train_output.keys())
    debug_summary["validation_step_keys"] = sorted(
        debug_model.validation_step((input_tensor, label_tensor), 0).keys()
    )

    print(json.dumps(debug_summary, indent=2, ensure_ascii=False))
    print("All checks passed!")
