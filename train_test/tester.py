import os
import lightning as l
import torch.utils.data as Data
from lightning.pytorch.callbacks import ModelCheckpoint


class Tester(object):
    def __init__(self, configs, model, save, test_dataset=None):
        if test_dataset is None:
            raise ValueError("测试数据集不能为空")

        self.configs = configs
        self.model = model(self.configs)
        self.test_dataset = test_dataset

        global_bs = int(self.configs.batch_size)
        trainer_devices = int(getattr(self.configs, "devices", 1))
        if trainer_devices > 1 and global_bs > 1:
            if global_bs % trainer_devices != 0:
                raise ValueError(
                    f"batch_size={global_bs} 无法被 Lightning devices={trainer_devices} 整除"
                )
            batch_size = global_bs // trainer_devices
        else:
            batch_size = global_bs

        self.test_data = Data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=int(getattr(self.configs, "num_workers", 4)),
                pin_memory=self.configs.accelerator == "gpu",
            )

        self.trainer = self._build_trainer()
        self.model_path = self.get_path()
        self.save = save

    def _build_trainer(self):
        trainer_kwargs = {
            "default_root_dir": self.configs.obj_dir,
            "accelerator": self.configs.accelerator,
            "devices": self.configs.devices,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": True,
            "enable_model_summary": False,
        }

        return l.Trainer(**trainer_kwargs)

    def get_path(self):
        model_dir = os.path.join(self.configs.obj_dir, "model")
        file_list = os.listdir(model_dir)
        model_path = ""
        for i in file_list:
            if i[:5] == "epoch":
                model_path = os.path.join(model_dir, i)
                break
        
        if model_path == "":
            raise ValueError("未找到模型")
        
        return model_path

    def test(self):

        # 先准备测试阶段依赖的数据索引
        self.model.test_prepare(self.test_dataset.data_idx, self.save)

        # 不接返回值，直接让 Lightning 跑完整个 test loop
        self.trainer.test(
            model=self.model,
            dataloaders=self.test_data,
            ckpt_path=self.model_path,
            verbose=True,
        )
