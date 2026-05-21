import torch
import torch.nn as nn
import lightning as l
import json
import os
import numpy as np
import os.path as osp
import sys
from pathlib import Path
from types import SimpleNamespace
from torch import optim
from lightning.pytorch.utilities.types import OptimizerLRScheduler

try:
    from timm.scheduler.cosine_lr import CosineLRScheduler
except ImportError:
    CosineLRScheduler = None

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from gSTA_model import gSTA_Model
else:
    from .gSTA_model import gSTA_Model
from utils import Recorder
from Instrument.standardizer import Load_Standardizer
from ..Model_system import System



class Model(System):
    def _init__(self, configs):
        super().__init__(configs)

    def get_model(self):
        return gSTA_Model(self.configs)
    
    def forward(self, batch_x, batch_y=None, **kwargs):
        pred_y = None
        if self.test_seq == self.aft_seq_length:
            pred_y = self.model(batch_x)
        elif self.test_seq < self.aft_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.test_seq]
        elif self.test_seq > self.aft_seq_length:
            if len(self.configs.out_category) != len(self.configs.in_category):
                raise ValueError('输入输出通道一致才能滚动预测。')
            else:
                if self.pre_seq_length < self.aft_seq_length:
                    pred_y = []
                    d = self.test_seq // self.pre_seq_length
                    m = self.test_seq % self.pre_seq_length
                    
                    cur_seq = batch_x.clone() #[b, t_in, c, h, w]
                    for _ in range(d):
                        cur_seq = self.model(cur_seq) #[b, t_out, c, h, w]
                        cur_seq = cur_seq[:, :self.pre_seq_length] #[b, t_in, c, h, w]
                        pred_y.append(cur_seq)

                    if m != 0:
                        cur_seq = self.model(cur_seq)
                        pred_y.append(cur_seq[:, :m])
                    
                    pred_y = torch.cat(pred_y, dim=1) # [b, t_test, c, h, w]

                elif self.pre_seq_length > self.aft_seq_length:

                    differ = self.pre_seq_length - self.aft_seq_length

                    pred_y = []
                    d = self.test_seq // self.aft_seq_length
                    m = self.test_seq % self.aft_seq_length
                    
                    in_seq = batch_x.clone() #[b, t_in, c, h, w]
                    for i in range(d):
                        out_seq = self.model(in_seq) #[b, t_out, c, h, w]
                        pred_y.append(out_seq)
                        in_seq = torch.cat((in_seq[:, differ:], out_seq), dim=1) #[b, t_in, c, h, w]


                    if m != 0:
                        out_seq = self.model(in_seq)
                        pred_y.append(out_seq[:, :m])
                    
                    pred_y = torch.cat(pred_y, dim=1) # [b, t_test, c, h, w]
                
                elif self.pre_seq_length == self.aft_seq_length:
                    pred_y = []
                    d = self.test_seq // self.pre_seq_length
                    m = self.test_seq % self.pre_seq_length
                    
                    cur_seq = batch_x.clone() #[b, t_in, c, h, w]
                    for _ in range(d):
                        cur_seq = self.model(cur_seq) #[b, t_out, c, h, w]
                        pred_y.append(cur_seq)

                    if m != 0:
                        cur_seq = self.model(cur_seq)
                        pred_y.append(cur_seq[:, :m])
                    
                    pred_y = torch.cat(pred_y, dim=1) # [b, t_test, c, h, w]


        assert pred_y is not None
        return pred_y
        
    def configure_optimizers(self) -> OptimizerLRScheduler:

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )
        if CosineLRScheduler is None:
            raise ImportError("SimVP_gSTA requires timm CosineLRScheduler")

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.configs.epoch,
            lr_min=self.configs.lr_min,
            warmup_t=self.configs.warmup_t,
            t_in_epochs=True,
            k_decay=self.configs.k_decay,
        )

        self._last_configured_optimizer = optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
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
        pred = self.model(batch_x)
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        loss = self.criterion(pred, label)

        # 不在这里 self.log(sync_dist=True)。
        # loss 统计交给 System.on_train_batch_end + on_train_epoch_end 处理。
        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self.model(batch_x)
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        loss = self.criterion(pred, label)

        # 不在这里 self.log(sync_dist=True)。
        # val_loss 统计和 ModelCheckpoint 监控值由 System.on_validation_epoch_end 统一写入。
        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred.detach(),
            "label": label.detach(),
        }

    


# class Model(l.LightningModule):

#     def __init__(self, configs):
#         super().__init__()

#         self.configs = configs
#         self.pre_seq_length = self.configs.total_seq[0]
#         self.aft_seq_length = self.configs.total_seq[1]
#         self.label_idx = self.configs.label_idx
#         self.test_seq = self.configs.test_seq
#         self.batch_size = self.configs.batch_size

#         self.model = gSTA_Model(self.configs)
#         self.criterion = nn.MSELoss()
#         self.opt_config = self._build_opt_config()
#         self.standardizer = Load_Standardizer(self.configs).standardizer
#         self.standardizer.metric_params()

#         self.recoder = Recorder(self.configs)
#         self.train_epoch_loss = []
#         self.valid_epoch_loss = []
#         self.best_valid_loss = float("inf")
#         self.best_valid_epoch = 0
#         self.test_outputs = []


#     def _build_opt_config(self):
#         return {
#             "lr": float(getattr(self.configs, "learning_rate", 5e-3)),
#             "weight_decay": float(getattr(self.configs, "weight_decay", 0.0)),
#             "min_lr": float(getattr(self.configs, "lr_min", 1e-6)),
#             "warmup_lr": float(getattr(self.configs, "warmup_lr_init", 1e-5)),
#             "warmup_epoch": int(getattr(self.configs, "warmup_t", getattr(self.configs, "warmup_epoch", 0))),
#             "lr_k_decay": float(getattr(self.configs, "k_decay", 1.0)),
#             "epoch": max(1, int(getattr(self.configs, "epoch", 100))),
#         }
    
#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         if not hasattr(self, "opt_config"):
#             self.opt_config = self._build_opt_config()

#         optimizer = optim.Adam(
#             self.model.parameters(),
#             lr=self.opt_config["lr"],
#             weight_decay=self.opt_config["weight_decay"],
#         )
#         if CosineLRScheduler is None:
#             raise ImportError("SimVP_gSTA requires timm CosineLRScheduler")

#         scheduler = CosineLRScheduler(
#             optimizer,
#             t_initial=self.opt_config["epoch"],
#             lr_min=self.opt_config["min_lr"],
#             warmup_lr_init=self.opt_config["warmup_lr"],
#             warmup_t=self.opt_config["warmup_epoch"],
#             t_in_epochs=True,
#             k_decay=self.opt_config["lr_k_decay"],
#         )

#         self._last_configured_optimizer = optimizer
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "epoch",
#             },
#         }
    
#     def lr_scheduler_step(self, scheduler, metric):
#         if CosineLRScheduler is not None and isinstance(scheduler, CosineLRScheduler):
#             trainer = self.__dict__.get("_trainer")
#             current_epoch = getattr(trainer, "current_epoch", 0)
#             scheduler.step(epoch=current_epoch)
#             return

#         if metric is None:
#             scheduler.step()
#         else:
#             scheduler.step(metric)

#     def current_lr(self):
#         optimizer = getattr(self, "_last_configured_optimizer", None)
#         if optimizer is None:
#             return self.opt_config["lr"]
#         return optimizer.param_groups[0]["lr"]

#     def forward(self, batch_x, batch_y=None, **kwargs):
#         pred_y = None
#         if self.test_seq == self.aft_seq_length:
#             pred_y = self.model(batch_x)
#         elif self.test_seq < self.aft_seq_length:
#             pred_y = self.model(batch_x)
#             pred_y = pred_y[:, :self.test_seq]
#         elif self.test_seq > self.aft_seq_length:
#             if len(self.configs.out_category) != len(self.configs.in_category):
#                 raise ValueError('输入输出通道一致才能滚动预测。')
#             else:
#                 if self.pre_seq_length < self.aft_seq_length:
#                     pred_y = []
#                     d = self.test_seq // self.pre_seq_length
#                     m = self.test_seq % self.pre_seq_length
                    
#                     cur_seq = batch_x.clone() #[b, t_in, c, h, w]
#                     for _ in range(d):
#                         cur_seq = self.model(cur_seq) #[b, t_out, c, h, w]
#                         cur_seq = cur_seq[:, :self.pre_seq_length] #[b, t_in, c, h, w]
#                         pred_y.append(cur_seq)

#                     if m != 0:
#                         cur_seq = self.model(cur_seq)
#                         pred_y.append(cur_seq[:, :m])
                    
#                     pred_y = torch.cat(pred_y, dim=1) # [b, t_test, c, h, w]

#                 elif self.pre_seq_length > self.aft_seq_length:

#                     differ = self.pre_seq_length - self.aft_seq_length

#                     pred_y = []
#                     d = self.test_seq // self.aft_seq_length
#                     m = self.test_seq % self.aft_seq_length
                    
#                     in_seq = batch_x.clone() #[b, t_in, c, h, w]
#                     for i in range(d):
#                         out_seq = self.model(in_seq) #[b, t_out, c, h, w]
#                         pred_y.append(out_seq)
#                         in_seq = torch.cat((in_seq[:, differ:], out_seq), dim=1) #[b, t_in, c, h, w]


#                     if m != 0:
#                         out_seq = self.model(in_seq)
#                         pred_y.append(out_seq[:, :m])
                    
#                     pred_y = torch.cat(pred_y, dim=1) # [b, t_test, c, h, w]
                
#                 elif self.pre_seq_length == self.aft_seq_length:
#                     pred_y = []
#                     d = self.test_seq // self.pre_seq_length
#                     m = self.test_seq % self.pre_seq_length
                    
#                     cur_seq = batch_x.clone() #[b, t_in, c, h, w]
#                     for _ in range(d):
#                         cur_seq = self.model(cur_seq) #[b, t_out, c, h, w]
#                         pred_y.append(cur_seq)

#                     if m != 0:
#                         cur_seq = self.model(cur_seq)
#                         pred_y.append(cur_seq[:, :m])
                    
#                     pred_y = torch.cat(pred_y, dim=1) # [b, t_test, c, h, w]


#         assert pred_y is not None
#         return pred_y
    
#     #训练步

#     def on_train_epoch_start(self):

#         epoch = self.current_epoch + 1
#         self.recoder.register_epoch(epoch)
#         print(f">>>>>>>>>>>>>>>第{epoch}轮训练<<<<<<<<<<<<<<<")

#     def training_step(self, batch, batch_idx):
        
#         batch_x, batch_y = batch
#         pred = self.model(batch_x)
#         label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
#         loss = self.criterion(pred, label)
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
#         return {
#             "loss": loss, 
#             "train_loss": loss.detach().cpu().item()
#             }

#     def on_train_batch_end(self, outputs, batch, batch_idx):

#         loss = outputs.get("train_loss", None)
#         self.train_epoch_loss.append(loss)
    
#     def on_train_epoch_end(self):

#         avg_loss = torch.tensor(self.train_epoch_loss).mean().item()
#         self.recoder.train_step(avg_loss, self.current_lr())
#         self.train_epoch_loss.clear()

#     #验证步

#     def on_validation_start(self):

#         epoch = self.current_epoch + 1
#         print(f">>>>>>>>>>>>>>>>第{epoch}轮验证<<<<<<<<<<<<<<<")

#     def validation_step(self, batch, batch_idx):

#         batch_x, batch_y = batch
#         pred = self.model(batch_x)
#         label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
#         loss = self.criterion(pred, label)  
#         self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
#         return {
#             "loss": loss, 
#             "val_loss": loss.detach().cpu().item(),
#             "output": pred.detach(),
#             "label": label.detach()
#             }
    
#     def on_validation_batch_end(self, outputs, batch, batch_idx):

#         loss = outputs["val_loss"]
#         self.valid_epoch_loss.append(loss)
#         output = self.standardizer.de_standardizing(outputs["output"],"metric")
#         label = self.standardizer.de_standardizing(outputs["label"],"metric")
#         self.recoder.within_the_epoch(output, label)

#     def on_validation_epoch_end(self):

#         avg_loss = torch.tensor(self.valid_epoch_loss).mean().item()
#         self.recoder.valid_step(avg_loss)
#         self.valid_epoch_loss.clear()

#         if avg_loss < self.best_valid_loss:
#             self.best_valid_loss = avg_loss
#             self.best_valid_epoch = self.current_epoch + 1
#             print(f"当前最佳验证损失: {self.best_valid_loss:.6f}, 发生在第{self.best_valid_epoch}轮")
        
#         self.recoder.save_process()

#     #测试步

#     def test_prepare(self, data_idx, is_save):
#         self.data_idx = data_idx
#         print(len(data_idx))
#         self.save_interval = self.configs.save_interval
#         self.save_list = []
#         self.label_list = []
#         self.save = is_save

#     def test_sample(self, batch_idx, output):
#         start = batch_idx * self.batch_size
#         end = min(start + output.shape[0], len(self.data_idx))
#         for i, lab in enumerate(range(start, end)):
#             if lab % self.save_interval == 0:
#                 self.save_list.append(output[i].cpu())
#                 self.label_list.append(self.data_idx[lab])
#                 break

#     def test_step(self, batch, batch_idx):
#         batch_x, batch_y = batch
#         output = self(batch_x)
#         label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
#         output = self.standardizer.de_standardizing(output,"metric")
#         label = self.standardizer.de_standardizing(label,"metric")
#         self.recoder.within_the_epoch(output, label)
#         if self.save == True:
#             self.test_sample(batch_idx, output)
        
#     def test_save(self, path):
#         data = torch.stack(self.save_list, dim=0).numpy()
#         save_path = os.path.join(path, "outputs")
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
        
#         np.save(os.path.join(save_path, 'out_data.npy'), data)

#         json_path = os.path.join(save_path, 'out_label.json')
#         with open(json_path, 'w', encoding='utf-8') as f:   
#             json.dump(self.label_list, f, ensure_ascii=False, indent=4)
        
#     def on_test_epoch_end(self):
#         self.recoder.test_step()
#         if self.save == True:
#             self.test_save(self.configs.obj_dir)
#             print(f"测试完成，结果已保存到 {self.configs.obj_dir}。")


if __name__ == "__main__":
    torch.manual_seed(0)

    def build_debug_configs():
        debug_dir = Path(__file__).resolve().parent / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            total_seq=[2, 2],
            label_idx=[0, 1],
            in_category=["tp"],
            out_category=["tp"],
            img_size=[8, 8],
            N_S=2,
            N_T=2,
            hid_S=4,
            hid_T=8,
            mlp_ratio=2.0,
            drop=0.0,
            drop_path=0.0,
            spatio_kernel_enc=3,
            spatio_kernel_dec=3,
            learning_rate=5e-4,
            weight_decay=0.0,
            lr_min=1e-6,
            warmup_lr_init=1e-5,
            warmup_t=0,
            k_decay=1.0,
            epoch=1,
            std_method="z_score",
            std_params={
                "dataset": {
                    "mean": [[[[0.0]]]],
                    "std": [[[[1.0]]]],
                },
                "metric": {
                    "mean": [[[[[0.0]]]]],
                    "std": [[[[[1.0]]]]],
                },
            },
            threshold=[[0.5]],
            metrics=["mae"],
            obj_dir=str(debug_dir),
        )

    def bcthw_to_btchw(tensor):
        return tensor.permute(0, 2, 1, 3, 4).contiguous()

    def btchw_to_bcthw(tensor):
        return tensor.permute(0, 2, 1, 3, 4).contiguous()

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
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{tag} expected BCTHW shape {expected_shape}, but got {actual_shape}."
            )

    debug_configs = build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(debug_configs).to(device)
    debug_model.train()

    batch_size = 2
    channels = len(debug_configs.in_category)
    timesteps = debug_configs.total_seq[0]
    height, width = debug_configs.img_size
    input_bcthw = torch.randn(batch_size, channels, timesteps, height, width, device=device)
    expected_output_shape = (
        batch_size,
        len(debug_configs.out_category),
        debug_configs.total_seq[1],
        height,
        width,
    )

    debug_summary = {
        "device": str(device),
        "input_bcthw_shape": list(input_bcthw.shape),
    }

    debug_model.zero_grad(set_to_none=True)
    model_input_bcthw = input_bcthw.detach().clone().requires_grad_(True)
    model_output_btchw = debug_model.model(bcthw_to_btchw(model_input_bcthw))
    model_output_bcthw = btchw_to_bcthw(model_output_btchw)
    ensure_shape(model_output_bcthw, expected_output_shape, "self.model")
    model_loss = model_output_bcthw.square().mean()
    model_loss.backward()
    ensure_input_grad(model_input_bcthw, "self.model")
    model_grad_count, model_grad_abs_sum = summarize_gradients(debug_model.model, "self.model")
    debug_summary["self.model"] = {
        "output_bcthw_shape": list(model_output_bcthw.shape),
        "loss": float(model_loss.detach().cpu().item()),
        "grad_param_count": model_grad_count,
        "grad_abs_sum": model_grad_abs_sum,
        "input_grad_abs_sum": float(model_input_bcthw.grad.detach().abs().sum().item()),
    }

    debug_model.zero_grad(set_to_none=True)
    forward_input_bcthw = input_bcthw.detach().clone().requires_grad_(True)
    forward_output_btchw = debug_model(bcthw_to_btchw(forward_input_bcthw))
    forward_output_bcthw = btchw_to_bcthw(forward_output_btchw)
    ensure_shape(forward_output_bcthw, expected_output_shape, "forward")
    forward_loss = forward_output_bcthw.square().mean()
    forward_loss.backward()
    ensure_input_grad(forward_input_bcthw, "forward")
    forward_grad_count, forward_grad_abs_sum = summarize_gradients(debug_model.model, "forward")
    debug_summary["forward"] = {
        "output_bcthw_shape": list(forward_output_bcthw.shape),
        "loss": float(forward_loss.detach().cpu().item()),
        "grad_param_count": forward_grad_count,
        "grad_abs_sum": forward_grad_abs_sum,
        "input_grad_abs_sum": float(forward_input_bcthw.grad.detach().abs().sum().item()),
    }

    print(json.dumps(debug_summary, indent=2, ensure_ascii=False))

