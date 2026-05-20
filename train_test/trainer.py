import os

import lightning as l
import torch
import torch.utils.data as Data
from lightning.pytorch.callbacks import ModelCheckpoint


class Trainer(object):
    def __init__(self, configs, model, train_dataset=None, val_dataset=None):
        if train_dataset is None or val_dataset is None:
            raise ValueError("训练数据集或验证数据集不能为空")

        self.configs = configs
        self.model = model(self.configs)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.ckpt_dir = os.path.join(self.configs.obj_dir, "model")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.train_data = self._build_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

        self.val_data = self._build_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
            drop_last=True,
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_dir,
            filename="epoch{epoch:03d}-val_loss{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )

        self.trainer = self._build_trainer()

    def _build_dataloader(self, dataset, shuffle, drop_last):
        """
        mmap + /dev/shm 数据读取建议：
            num_workers 默认先设 0。
            稳定后再尝试 1、2。
        """

        num_workers = int(getattr(self.configs, "num_workers", 0))

        pin_memory = getattr(self.configs, "pin_memory", None)
        if pin_memory is None:
            pin_memory = self.configs.accelerator == "gpu"

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(self.configs.batch_size),
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "pin_memory": bool(pin_memory),
        }

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(
                getattr(self.configs, "persistent_workers", True)
            )
            loader_kwargs["prefetch_factor"] = int(
                getattr(self.configs, "prefetch_factor", 2)
            )

        return Data.DataLoader(**loader_kwargs)

    def _resolve_strategy(self):
        """
        自动适配：
            CPU / 单卡：auto
            多卡：默认 ddp
            用户显式指定：使用用户指定 strategy
        """

        user_strategy = getattr(self.configs, "strategy", "auto")

        if self.configs.accelerator != "gpu":
            return "auto"

        if int(self.configs.devices) <= 1:
            return "auto"

        if user_strategy == "auto":
            return "ddp"

        return user_strategy

    def _build_trainer(self):
        trainer_kwargs = {
            "default_root_dir": self.configs.obj_dir,
            "max_epochs": int(self.configs.epoch),

            # 关键：单卡 / 多卡 / CPU 自适应
            "accelerator": self.configs.accelerator,
            "devices": self.configs.devices,
            "strategy": self._resolve_strategy(),

            # 多卡时让 Lightning 自动给 DataLoader 加 DistributedSampler
            "use_distributed_sampler": True,

            # 可选：显存优化
            "precision": getattr(self.configs, "precision", "32-true"),
            "accumulate_grad_batches": int(
                getattr(self.configs, "accumulate_grad_batches", 1)
            ),

            "logger": False,
            "callbacks": [self.checkpoint_callback],
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 50,
        }

        for key in (
            "limit_train_batches",
            "limit_val_batches",
            "limit_test_batches",
            "gradient_clip_val",
            "check_val_every_n_epoch",
        ):
            if hasattr(self.configs, key):
                trainer_kwargs[key] = getattr(self.configs, key)

        print("\nLightning Trainer 配置:")
        for k, v in trainer_kwargs.items():
            if k != "callbacks":
                print(f"  {k}: {v}")
        print("")

        return l.Trainer(**trainer_kwargs)

    def train(self):
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_data,
            val_dataloaders=self.val_data,
            ckpt_path=getattr(self.configs, "ckpt_path", None),
        )

        best_path = self.checkpoint_callback.best_model_path
        last_path = self.checkpoint_callback.last_model_path

        print(f"best checkpoint: {best_path}")
        print(f"last checkpoint: {last_path}")

    def save_checkpoint(self, path):
        self.trainer.save_checkpoint(path)