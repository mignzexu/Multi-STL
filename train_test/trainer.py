import os

import lightning as l
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

        self.cur_epoch = 1
        self.best_epoch = 1
        self.best_loss = float("inf")

        self.ckpt_dir = os.path.join(self.configs.obj_dir, "model")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.train_data = Data.DataLoader(
                train_dataset,
                batch_size=self.configs.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4,  # 增加数据加载线程
                pin_memory=True,  # 使用固定内存加速数据传输
            )
        self.val_data = Data.DataLoader(
                val_dataset,
                batch_size=self.configs.batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=4,  # 增加数据加载线程
                pin_memory=True,  # 使用固定内存加速数据传输
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


    def _build_trainer(self):
        trainer_kwargs = {
            "default_root_dir": self.configs.obj_dir,
            "max_epochs": int(self.configs.epoch),
            "accelerator": "gpu" if self.configs.gpu_count > 0 else "cpu",
            "devices": self.configs.gpu_count,
            "logger": False,
            "callbacks": [self.checkpoint_callback],
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
            "log_every_n_steps": 50,
        }

        for key in ("limit_train_batches", "limit_val_batches", "limit_test_batches"):
            if hasattr(self.configs, key):
                trainer_kwargs[key] = getattr(self.configs, key)

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

    def save_checkpoint(self, path):
        self.trainer.save_checkpoint(path)
