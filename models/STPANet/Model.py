import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    CosineLRScheduler = None

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from models.Model_system import System
    from models.STPANet.loss import loss_fn
    from models.STPANet.STPANet_model import STPANet_Model
else:
    from .loss import loss_fn
    from .STPANet_model import STPANet_Model
    from ..Model_system import System


class OptConfig(TypedDict):
    lr: float
    weight_decay: float
    min_lr: float
    warmup_lr: float
    warmup_epoch: int
    lr_k_decay: float
    epoch: int
    scheduler: str | None


class Model(System):
    MODEL_CONFIG_KEYS = (
        "total_seq",
        "in_category",
        "out_category",
        "label_idx",
        "img_size",
        "N_S",
        "N_T",
        "hid_S",
        "hid_T",
        "mlp_ratio",
        "drop",
        "drop_path",
        "spatio_kernel_enc",
        "spatio_kernel_dec",
    )

    def __init__(self, configs):
        self.model_configs = self._build_model_configs(configs)
        super().__init__(configs)
        self.loss_bridge = loss_fn(self.model_configs)
        self.opt_config = self._build_opt_config()
        self._last_configured_optimizer = None

    @staticmethod
    def _read_config(configs, key):
        if hasattr(configs, key):
            return getattr(configs, key)
        if isinstance(configs, dict):
            return configs[key]
        raise AttributeError(f"Missing config field: {key}")

    @classmethod
    def _read_float_config(cls, configs, key, default=None):
        value = default if default is not None and not hasattr(configs, key) and not isinstance(configs, dict) else cls._read_config(configs, key) if (hasattr(configs, key) or (isinstance(configs, dict) and key in configs)) else default
        if value is None:
            raise AttributeError(f"Missing float config field: {key}")
        return float(value)

    @classmethod
    def _read_int_config(cls, configs, key, default=None):
        value = default if default is not None and not hasattr(configs, key) and not isinstance(configs, dict) else cls._read_config(configs, key) if (hasattr(configs, key) or (isinstance(configs, dict) and key in configs)) else default
        if value is None:
            raise AttributeError(f"Missing int config field: {key}")
        return int(value)

    @classmethod
    def _read_optional_str_config(cls, configs, key):
        has_attr = hasattr(configs, key)
        has_item = isinstance(configs, dict) and key in configs
        if not has_attr and not has_item:
            return None
        value = cls._read_config(configs, key)
        return None if value is None else str(value)

    @classmethod
    def _build_model_configs(cls, configs):
        return {key: cls._read_config(configs, key) for key in cls.MODEL_CONFIG_KEYS}

    def _build_opt_config(self) -> OptConfig:
        return {
            "lr": self._read_float_config(self.configs, "learning_rate"),
            "weight_decay": self._read_float_config(self.configs, "weight_decay", 0.0),
            "min_lr": self._read_float_config(self.configs, "lr_min", 1e-6),
            "warmup_lr": self._read_float_config(self.configs, "warmup_lr_init", 1e-5),
            "warmup_epoch": self._read_int_config(self.configs, "warmup_t", self._read_int_config(self.configs, "warmup_epoch", 0) if (hasattr(self.configs, "warmup_epoch") or (isinstance(self.configs, dict) and "warmup_epoch" in self.configs)) else 0),
            "lr_k_decay": self._read_float_config(self.configs, "k_decay", 1.0),
            "epoch": max(1, self._read_int_config(self.configs, "epoch", 1)),
            "scheduler": self._read_optional_str_config(self.configs, "scheduler"),
        }

    def get_model(self):
        return STPANet_Model(self.model_configs)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pred_y = None
        if self.test_seq == self.aft_seq_length:
            pred_y = self.model(batch_x)["pred"]
        elif self.test_seq < self.aft_seq_length:
            pred_y = self.model(batch_x)["pred"]
            pred_y = pred_y[:, : self.test_seq]
        elif self.test_seq > self.aft_seq_length:
            pred_y = []
            d = self.test_seq // self.pre_seq_length
            m = self.test_seq % self.pre_seq_length

            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)["pred"]
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)["pred"]
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)

        assert pred_y is not None
        return pred_y

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.opt_config["lr"],
            weight_decay=self.opt_config["weight_decay"],
        )
        self._last_configured_optimizer = optimizer

        if self.opt_config["scheduler"] != "cosine":
            return {"optimizer": optimizer}

        if CosineLRScheduler is not None:
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.opt_config["epoch"],
                lr_min=self.opt_config["min_lr"],
                warmup_t=self.opt_config["warmup_epoch"],
                t_in_epochs=True,
                k_decay=self.opt_config["lr_k_decay"],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.opt_config["epoch"],
                eta_min=self.opt_config["min_lr"],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any):
        if CosineLRScheduler is not None and isinstance(scheduler, CosineLRScheduler):
            trainer = self.__dict__.get("_trainer")
            current_epoch = getattr(trainer, "current_epoch", 0)
            scheduler.step(epoch=current_epoch)
            return

        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        loss, _ = self.loss_bridge(self, batch_x, batch_y)

        # 不在这里 self.log(sync_dist=True)。
        # loss 统计交给 System.on_train_batch_end + on_train_epoch_end。
        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        loss, output = self.loss_bridge(self, batch_x, batch_y)

        label = batch_y[:, :, self.label_idx[0]: self.label_idx[1], :, :]

        # 不在这里 self.log(sync_dist=True)。
        # val_loss 统计和 ModelCheckpoint 监控值由 System.on_validation_epoch_end 统一写入。
        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": output["pred"].detach(),
            "label": label.detach(),
        }


def _build_debug_configs():
    debug_dir = Path(__file__).resolve().parent / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        total_seq=[2, 2],
        test_seq=2,
        label_idx=[0, 1],
        in_category=["tp"],
        out_category=["tp"],
        img_size=[8, 8],
        batch_size=2,
        learning_rate=5e-4,
        weight_decay=0.0,
        lr_min=1e-6,
        warmup_lr_init=1e-5,
        warmup_t=0,
        warmup_epoch=0,
        k_decay=1.0,
        epoch=1,
        scheduler=None,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        hid_S=8,
        hid_T=16,
        N_T=2,
        N_S=2,
        mlp_ratio=2,
        drop_path=0.0,
        drop=0.0,
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


def _summarize_gradients(module, tag):
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


def _ensure_input_grad(tensor, tag):
    if tensor.grad is None:
        raise RuntimeError(f"{tag} backward did not produce input gradients.")
    if not torch.isfinite(tensor.grad).all():
        raise RuntimeError(f"{tag} input gradients contain non-finite values.")


if __name__ == "__main__":
    torch.manual_seed(0)
    configs = _build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(configs).to(device)
    debug_model.train()

    batch_size = 2
    channels = len(configs.in_category)
    height, width = configs.img_size
    batch_x = torch.randn(
        batch_size, configs.total_seq[0], channels, height, width, device=device
    )
    batch_y = torch.randn(
        batch_size, configs.total_seq[1], channels, height, width, device=device
    )

    debug_model.zero_grad(set_to_none=True)
    train_x = batch_x.detach().clone().requires_grad_(True)
    train_out = debug_model.training_step((train_x, batch_y), 0)
    train_out["loss"].backward()
    _ensure_input_grad(train_x, "training_step")
    grad_count, grad_sum = _summarize_gradients(debug_model.model, "training_step")
    print(
        f"[training_step] loss: {train_out['loss'].detach().item():.6f}, "
        f"grad params: {grad_count}, grad sum: {grad_sum:.4f}"
    )

    debug_model.zero_grad(set_to_none=True)
    forward_x = batch_x.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_x)
    forward_loss = forward_output.square().mean()
    forward_loss.backward()
    _ensure_input_grad(forward_x, "forward")
    grad_count, grad_sum = _summarize_gradients(debug_model.model, "forward")
    print(
        f"[forward] output shape: {tuple(forward_output.shape)}, "
        f"grad params: {grad_count}, grad sum: {grad_sum:.4f}"
    )

    val_out = debug_model.validation_step((batch_x, batch_y), 0)
    expected_val_keys = {"val_loss", "batch_size", "output", "label"}
    if set(val_out) != expected_val_keys:
        raise RuntimeError(f"validation_step returned unexpected keys: {sorted(val_out)}")

    print("All checks passed!")
