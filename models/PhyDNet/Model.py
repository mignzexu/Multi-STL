import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.Model_system import System
    from models.PhyDNet.PhyDNet_model import PhyDNet_Model
else:
    from ..Model_system import System
    from .PhyDNet_model import PhyDNet_Model


class Model(System):
    def __init__(self, configs):
        super().__init__(configs)
        self.constraints = self._get_constraints()

    def _config_dict(self):
        return vars(self.configs).copy()

    def get_model(self):
        return PhyDNet_Model(self._config_dict(), torch.device("cpu"))

    def _get_constraints(self):
        constraints = torch.zeros((49, 7, 7))
        ind = 0
        for i in range(7):
            for j in range(7):
                constraints[ind, i, j] = 1
                ind += 1
        return constraints

    def _inference(self, batch_x, batch_y=None, return_loss=False):
        if batch_y is None:
            if return_loss:
                raise ValueError("batch_y is required when PhyDNet inference returns loss.")
            target_y = batch_x.new_empty(0)
        else:
            target_y = batch_y

        loss = batch_x.new_zeros(())
        for ei in range(self.pre_seq_length - 1):
            _, _, output_image, _, _ = self.model.encoder(batch_x[:, ei, :, :, :], ei == 0)
            if return_loss:
                loss = loss + self.model.criterion(output_image, batch_x[:, ei + 1, :, :, :])

        decoder_input = batch_x[:, -1, :, :, :]
        predictions = []
        for di in range(self.aft_seq_length):
            _, _, output_image, _, _ = self.model.encoder(decoder_input, False, False)
            decoder_input = output_image
            predictions.append(output_image)
            if return_loss:
                loss = loss + self.model.criterion(output_image, target_y[:, di, :, :, :])

        if return_loss:
            constraints = self.constraints.to(batch_x.device)
            for channel_idx in range(self.model.encoder.phycell.cell_list[0].input_dim):
                filters = self.model.encoder.phycell.cell_list[0].F.conv1.weight[:, channel_idx, :, :]
                moments = self.model.k2m(filters.double()).float()
                loss = loss + self.model.criterion(moments, constraints.to(moments.device))

        return torch.stack(predictions, dim=1), loss

    def _rolling_prediction(self, batch_x):
        if len(self.configs.in_category) != len(self.configs.out_category):
            raise ValueError("Rolling prediction requires matching input/output channels.")

        pred_parts = []
        cur_seq = batch_x
        generated = 0
        while generated < self.test_seq:
            pred_full, _ = self._inference(cur_seq, return_loss=False)
            pred = pred_full[:, :, self.label_idx[0] : self.label_idx[1], :, :]
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

        pred_y, _ = self._inference(batch_x, batch_y, return_loss=False)
        pred_y = pred_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
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
        teacher_forcing_ratio = np.maximum(0, 1 - self.current_epoch * 0.003)
        output = self.model(batch_x, batch_y, self.constraints.to(batch_x.device), teacher_forcing_ratio)
        loss = output["loss"]
        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_full, loss = self._inference(batch_x, batch_y, return_loss=True)
        pred = pred_full[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred.detach(),
            "label": label.detach(),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

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
            patch_size=2,
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
    batch_x = torch.rand(batch_size, configs.total_seq[0], channels, height, width, device=device)
    batch_y = torch.rand(batch_size, configs.total_seq[1], channels, height, width, device=device)
    expected_shape = (batch_size, configs.total_seq[1], channels, height, width)

    debug_model.zero_grad(set_to_none=True)
    model_input = batch_x.detach().clone().requires_grad_(True)
    model_output = debug_model.model(
        model_input,
        batch_y,
        debug_model.constraints.to(device),
        teacher_forcing_ratio=0.5,
    )
    if tuple(model_output["pred"].shape) != expected_shape:
        raise RuntimeError(f"self.model expected {expected_shape}, got {tuple(model_output['pred'].shape)}")
    model_output["loss"].backward()
    ensure_input_grad(model_input, "self.model")
    model_grad_count, model_grad_sum = summarize_gradients(debug_model.model, "self.model")

    debug_model.zero_grad(set_to_none=True)
    forward_input = batch_x.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_input)
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
        "self.model_shape": list(model_output["pred"].shape),
        "self.model_grad_params": model_grad_count,
        "self.model_grad_sum": model_grad_sum,
        "forward_shape": list(forward_output.shape),
        "forward_grad_params": forward_grad_count,
        "forward_grad_sum": forward_grad_sum,
    }, indent=2))
