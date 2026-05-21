import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, TypedDict

try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    CosineLRScheduler = None

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
