import torch
import torch.nn as nn
import lightning as l
import json
import os
import numpy as np

from utils import Recorder
from Instrument.standardizer import Load_Standardizer

#需要定义：get_model, forward, configure_optimizers, training_step, validation_step


class System(l.LightningModule):

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.pre_seq_length = self.configs.total_seq[0]
        self.aft_seq_length = self.configs.total_seq[1]
        self.label_idx = self.configs.label_idx
        self.test_seq = self.configs.test_seq
        self.batch_size = self.configs.batch_size

        self.model = self.get_model()
        self.criterion = nn.MSELoss()
        self.standardizer = Load_Standardizer(self.configs).standardizer
        self.standardizer.metric_params()

        self.recoder = Recorder(self.configs)
        self.train_epoch_loss = []
        self.valid_epoch_loss = []
        self.best_valid_loss = float("inf")
        self.best_valid_epoch = 0
        self.test_outputs = []

    def get_model(self):
        raise NotImplementedError


    def configure_optimizers(self):
        raise NotImplementedError
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         ...
        #     },
        # }

    def current_lr(self):
        optimizer = getattr(self, "_last_configured_optimizer", None)
        if optimizer is None:
            return self.opt_config["lr"]
        return optimizer.param_groups[0]["lr"]

    def forward(self, batch_x, batch_y=None, **kwargs):
        raise NotImplementedError
        # return pred_y #真实图片
    
    #训练步

    def on_train_epoch_start(self):

        epoch = self.current_epoch + 1
        self.recoder.register_epoch(epoch)
        self.train_epoch_loss.clear()
        self.valid_epoch_loss.clear()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError
        # self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # return {
        #     "loss": loss, 
        #     "train_loss": loss.detach().cpu().item()
        #     }

    def on_train_batch_end(self, outputs, batch, batch_idx):

        loss = outputs.get("train_loss", None)
        self.train_epoch_loss.append(loss)
    
    def on_train_epoch_end(self):

        avg_loss = torch.tensor(self.train_epoch_loss).mean().item()
        self.recoder.train_step(avg_loss, self.current_lr())
        

    #验证步

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # return {
        #     "val_loss": loss.detach().cpu().item(),
        #     "output": pred.detach(),
        #     "label": label.detach()
        #     }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx):

        loss = outputs["val_loss"]
        self.valid_epoch_loss.append(loss)
        output = self.standardizer.de_standardizing(outputs["output"],"metric")
        label = self.standardizer.de_standardizing(outputs["label"],"metric")
        self.recoder.within_the_epoch(output, label)

    def on_validation_epoch_end(self):

        avg_loss = torch.tensor(self.valid_epoch_loss).mean().item()
        self.recoder.valid_step(avg_loss)

        if avg_loss < self.best_valid_loss:
            self.best_valid_loss = avg_loss
            self.best_valid_epoch = self.current_epoch + 1
            print(f"当前最佳验证损失: {self.best_valid_loss:.6f}, 发生在第{self.best_valid_epoch}轮")
        
        self.recoder.save_process()

    #测试步

    def test_prepare(self, data_idx, is_save):
        self.data_idx = data_idx
        self.save_interval = self.configs.save_interval
        self.save_list = []
        self.label_list = []
        self.save = is_save

    def test_sample(self, batch_idx, output):
        start = batch_idx * self.batch_size
        end = min(start + output.shape[0], len(self.data_idx))
        for i, lab in enumerate(range(start, end)):
            if lab % self.save_interval == 0:
                self.save_list.append(output[i].cpu())
                self.label_list.append(self.data_idx[lab])
                break

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        output = self(batch_x)
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        output = self.standardizer.de_standardizing(output,"metric")
        label = self.standardizer.de_standardizing(label,"metric")
        self.recoder.within_the_epoch(output, label)
        if self.save == True:
            self.test_sample(batch_idx, output)
        
    def test_save(self, path):
        data = torch.stack(self.save_list, dim=0).numpy()
        save_path = os.path.join(path, "outputs")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        np.save(os.path.join(save_path, 'out_data.npy'), data)

        json_path = os.path.join(save_path, 'out_label.json')
        with open(json_path, 'w', encoding='utf-8') as f:   
            json.dump(self.label_list, f, ensure_ascii=False, indent=4)
        
    def on_test_epoch_end(self):
        self.recoder.test_step()
        if self.save == True:
            self.test_save(self.configs.obj_dir)
            print(f"测试完成，结果已保存到 {self.configs.obj_dir}。")

