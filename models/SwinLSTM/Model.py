import json
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
    from models.SwinLSTM.SwinLSTM_model import SwinLSTM_D_Model
else:
    from ..Model_system import System
    from .SwinLSTM_model import SwinLSTM_D_Model


class Model(System):
    def __init__(self, configs):
        super().__init__(configs)

    def _config_dict(self):
        return vars(self.configs).copy()

    def get_model(self):
        depths_downsample = [int(x) for x in self.configs.depths_downsample.split(",")]
        depths_upsample = [int(x) for x in self.configs.depths_upsample.split(",")]
        num_heads = [int(x) for x in self.configs.num_heads.split(",")]
        return SwinLSTM_D_Model(depths_downsample, depths_upsample, num_heads, self._config_dict())

    def _full_sequence(self, batch_x, batch_y=None):
        if batch_y is None:
            future_context = batch_x[:, -1:].expand(-1, self.aft_seq_length, -1, -1, -1)
        else:
            future_context = batch_y
        return torch.cat((batch_x, future_context), dim=1)

    def _future_prediction(self, batch_x, batch_y=None):
        frames_nthwc = self._full_sequence(batch_x, batch_y).permute(0, 1, 3, 4, 2).contiguous()
        output = self.model(frames_nthwc, return_loss=False)
        pred = output["pred"] if isinstance(output, dict) else output[0]
        pred = pred[:, -self.aft_seq_length :].permute(0, 1, 4, 2, 3).contiguous()
        return pred[:, :, self.label_idx[0] : self.label_idx[1], :, :]

    def _rolling_prediction(self, batch_x):
        if len(self.configs.in_category) != len(self.configs.out_category):
            raise ValueError("Rolling prediction requires matching input/output channels.")

        pred_parts = []
        cur_seq = batch_x
        generated = 0
        while generated < self.test_seq:
            pred = self._future_prediction(cur_seq)
            remaining = self.test_seq - generated
            pred_parts.append(pred[:, :remaining])
            generated += pred_parts[-1].shape[1]

            if self.pre_seq_length == self.aft_seq_length:
                cur_seq = pred
            elif self.pre_seq_length > self.aft_seq_length:
                cur_seq = torch.cat((cur_seq[:, self.aft_seq_length :], pred), dim=1)
            else:
                cur_seq = pred[:, : self.pre_seq_length]

        return torch.cat(pred_parts, dim=1)

    def forward(self, batch_x, batch_y=None, **kwargs):
        if self.test_seq > self.aft_seq_length and batch_y is None:
            return self._rolling_prediction(batch_x)

        pred_y = self._future_prediction(batch_x, batch_y)
        if self.test_seq < pred_y.shape[1]:
            pred_y = pred_y[:, : self.test_seq]
        return pred_y

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
        pred = self._future_prediction(batch_x, batch_y)
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        loss = self.criterion(pred, label)
        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self._future_prediction(batch_x, batch_y)
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
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
            img_size=[16, 16],
            depths_downsample="1,1",
            depths_upsample="1,1",
            num_heads="1,2",
            patch_size=2,
            window_size=2,
            embed_dim=8,
            learning_rate=1e-4,
            weight_decay=0.0,
            batch_size=2,
            epoch=1,
            std_method="z_score",
            std_params={
                "dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]},
                "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]},
            },
            threshold=[[0.5]],
            metrics=["mae"],
            obj_dir=str(debug_dir),
            save_interval=1,
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

    configs = build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(configs).to(device)
    debug_model.train()

    batch_size = configs.batch_size
    channels = len(configs.in_category)
    height, width = configs.img_size
    batch_x = torch.randn(batch_size, configs.total_seq[0], channels, height, width, device=device)
    batch_y = torch.randn(batch_size, configs.total_seq[1], channels, height, width, device=device)
    expected_shape = (batch_size, configs.total_seq[1], channels, height, width)

    debug_model.zero_grad(set_to_none=True)
    model_input = torch.cat((batch_x, batch_y), dim=1).detach().clone().requires_grad_(True)
    model_input_nthwc = model_input.permute(0, 1, 3, 4, 2).contiguous()
    model_output = debug_model.model(model_input_nthwc)["pred"][:, -configs.total_seq[1] :]
    model_output = model_output.permute(0, 1, 4, 2, 3).contiguous()
    if tuple(model_output.shape) != expected_shape:
        raise RuntimeError(f"self.model expected {expected_shape}, got {tuple(model_output.shape)}")
    model_loss = model_output.square().mean()
    model_loss.backward()
    ensure_input_grad(model_input, "self.model")
    model_grad_count, model_grad_sum = summarize_gradients(debug_model.model, "self.model")

    debug_model.zero_grad(set_to_none=True)
    forward_input = batch_x.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_input, batch_y)
    if tuple(forward_output.shape) != expected_shape:
        raise RuntimeError(f"forward expected {expected_shape}, got {tuple(forward_output.shape)}")
    forward_loss = forward_output.square().mean()
    forward_loss.backward()
    ensure_input_grad(forward_input, "forward")
    forward_grad_count, forward_grad_sum = summarize_gradients(debug_model.model, "forward")

    debug_model.test_seq = configs.total_seq[1] + 1
    rolling_output = debug_model(batch_x)
    rolling_expected_shape = (batch_size, debug_model.test_seq, channels, height, width)
    if tuple(rolling_output.shape) != rolling_expected_shape:
        raise RuntimeError(f"rolling forward expected {rolling_expected_shape}, got {tuple(rolling_output.shape)}")
    debug_model.test_seq = configs.test_seq

    print(json.dumps({
        "device": str(device),
        "self.model_shape": list(model_output.shape),
        "self.model_grad_params": model_grad_count,
        "self.model_grad_sum": model_grad_sum,
        "forward_shape": list(forward_output.shape),
        "forward_grad_params": forward_grad_count,
        "forward_grad_sum": forward_grad_sum,
    }, indent=2))
