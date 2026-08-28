import sys
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from models.Model_system import System
    from models.PreDiff.PreDiff_model import PreDiffModel
    from models.PreDiff.optimizer import build_optimizer
else:
    from ..Model_system import System
    from .PreDiff_model import PreDiffModel
    from .optimizer import build_optimizer


class Model(System):
    def __init__(self, configs):
        self.model_configs = self._to_config_dict(configs)
        self.opt_config = {"lr": float(self.model_configs.get("learning_rate", 1e-4))}
        self.metric_sampling_steps = int(self.model_configs.get("prediff_metric_sampling_steps", 8))
        super().__init__(configs)

    @staticmethod
    def _to_config_dict(configs):
        if isinstance(configs, dict):
            return dict(configs)
        return dict(vars(configs))

    def get_model(self):
        return PreDiffModel(self.model_configs)

    def _slice_label(self, batch_y):
        start, end = self.label_idx
        if batch_y.shape[2] == len(self.model_configs["out_category"]):
            return batch_y
        return batch_y[:, :, start:end, :, :]

    def _prepare_input_seq(self, batch_x):
        if batch_x.shape[1] >= self.pre_seq_length:
            return batch_x[:, -self.pre_seq_length :]
        padding_shape = (
            batch_x.shape[0],
            self.pre_seq_length - batch_x.shape[1],
            batch_x.shape[2],
            batch_x.shape[3],
            batch_x.shape[4],
        )
        padding = torch.zeros(padding_shape, device=batch_x.device, dtype=batch_x.dtype)
        return torch.cat([padding, batch_x], dim=1)

    def _sample_once(self, batch_x, sample_steps=None):
        return self.model.sample(
            self._prepare_input_seq(batch_x),
            steps=self.aft_seq_length,
            sample_steps=sample_steps,
        )

    def forward(self, batch_x, batch_y=None, **kwargs):
        sample_steps = kwargs.get("sample_steps", None)

        if self.test_seq <= self.aft_seq_length:
            pred_y = self._sample_once(batch_x, sample_steps=sample_steps)
            return pred_y[:, : self.test_seq]

        if len(self.configs.in_category) != len(self.configs.out_category):
            raise ValueError("PreDiff rolling inference requires matching input and output channels.")

        pred_y = []
        in_seq = self._prepare_input_seq(batch_x)
        remaining = self.test_seq

        while remaining > 0:
            out_seq = self._sample_once(in_seq, sample_steps=sample_steps)
            pred_y.append(out_seq[:, : min(remaining, out_seq.shape[1])])
            remaining -= out_seq.shape[1]

            if self.pre_seq_length > out_seq.shape[1]:
                in_seq = torch.cat([in_seq[:, out_seq.shape[1] :], out_seq], dim=1)
            else:
                in_seq = out_seq[:, -self.pre_seq_length :]

        return torch.cat(pred_y, dim=1)[:, : self.test_seq]

    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer(self.model, self.model_configs)
        self._last_configured_optimizer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        target = self._slice_label(batch_y)
        loss = self.model.training_loss(self._prepare_input_seq(batch_x), target)

        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        target = self._slice_label(batch_y)
        loss = self.model.training_loss(self._prepare_input_seq(batch_x), target)
        pred = self._sample_once(batch_x, sample_steps=self.metric_sampling_steps)

        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred.detach(),
            "label": target.detach(),
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
            batch_size=1,
            learning_rate=1e-4,
            weight_decay=0.0,
            optimizer="adamw",
            scheduler_T_max=1,
            epoch=1,
            prediff_use_vae=False,
            prediff_base_units=8,
            prediff_depth=[1, 1],
            prediff_downsample=1,
            prediff_num_heads=1,
            prediff_timesteps=4,
            prediff_inference_timesteps=2,
            prediff_metric_sampling_steps=2,
            prediff_use_relative_pos=False,
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

    configs = build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(configs).to(device)
    debug_model.train()

    batch_x = torch.randn(1, configs.total_seq[0], 1, 8, 8, device=device)
    batch_y = torch.randn(1, configs.total_seq[1], 1, 8, 8, device=device)

    debug_model.zero_grad(set_to_none=True)
    train_out = debug_model.training_step((batch_x, batch_y), 0)
    train_out["loss"].backward()
    grad_count, grad_sum = summarize_gradients(debug_model.model, "training_step")
    print(f"[training_step] loss: {train_out['loss'].item():.6f}, grad params: {grad_count}, grad sum: {grad_sum:.4f}")

    debug_model.eval()
    with torch.no_grad():
        forward_output = debug_model(batch_x)
    if forward_output.shape != (1, configs.test_seq, 1, 8, 8):
        raise RuntimeError(f"forward output shape mismatch: {forward_output.shape}")
    # PreDiffModel.sample and p_sample are decorated with torch.no_grad(), so the
    # diffusion generation path is an inference-only contract. Backward is checked
    # above through training_step, which uses the differentiable diffusion loss.
    print(f"[forward] output shape: {tuple(forward_output.shape)}")
    print("All checks passed!")
