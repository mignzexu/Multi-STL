import torch
import lightning as l
import json
import os
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from .MS_RadarFormer.MS_RadarFormer import MsRadarFormer
from .loss import WindWeightedLoss
from torch import optim
from utils import Recorder
from Instrument.standardizer import Load_Standardizer
try:
    from ..Model_system import System, distribute_model_layers
except ImportError:
    from models.Model_system import System, distribute_model_layers

class Model(System):

    def __init__(self, configs):
        super().__init__(configs)

        self.criterion = WindWeightedLoss("l2")

    def get_model(self):
        return MsRadarFormer(self.configs)

    def forward(self, batch_x, batch_y=None, **kwargs):
        distribute_model_layers(self.model, self.configs, batch_x.device)
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
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.configs.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.configs.pct_start,
        )

        self._last_configured_optimizer = optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        pred = self.model(batch_x)
        pred = self.standardizer.de_standardizing(pred, "metric")
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        label = self.standardizer.de_standardizing(label, "metric")
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
        pred = self.standardizer.de_standardizing(pred, "metric")
        label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
        label = self.standardizer.de_standardizing(label, "metric")
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
#         self.test_seq = self.configs.test_seq
#         self.label_idx = self.configs.label_idx

#         self.model = MsRadarFormer(self.configs)
#         self.batch_size = self.configs.batch_size
#         self.criterion = WindWeightedLoss("l2")
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
#             "lr": float(getattr(self.configs, "learning_rate", 1e-4)),
#             "weight_decay": float(getattr(self.configs, "weight_decay", 1e-5)),
#             "pct_start": float(getattr(self.configs, "pct_start", 0.3)),
#             "epoch": max(1, int(getattr(self.configs, "epoch", 100))),
#         }

#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         optimizer = optim.AdamW(
#             self.model.parameters(),
#             lr=self.opt_config["lr"],
#             weight_decay=self.opt_config["weight_decay"],
#         )

#         scheduler = optim.lr_scheduler.OneCycleLR(
#             optimizer,
#             max_lr=self.opt_config["lr"],
#             total_steps=self.trainer.estimated_stepping_batches,
#             pct_start=self.opt_config["pct_start"],
#         )

#         self._last_configured_optimizer = optimizer

#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "step",
#             },
#         }

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
#         self.recoder.register_epoch(self.current_epoch + 1)

#     def training_step(self, batch, batch_idx):
        
#         batch_x, batch_y = batch
        
#         pred = self.model(batch_x)
#         pred = self.standardizer.de_standardizing(pred,"metric")
#         label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
#         label = self.standardizer.de_standardizing(label,"metric")
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

#     def validation_step(self, batch, batch_idx):

#         batch_x, batch_y = batch
#         pred = self.model(batch_x)
#         pred = self.standardizer.de_standardizing(pred,"metric")
#         label = batch_y[:, :, self.label_idx[0] : self.label_idx[1], :, :]
#         label = self.standardizer.de_standardizing(label,"metric")
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
#         self.recoder.within_the_epoch(outputs["output"], outputs["label"])

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
#             if self.data_idx[lab][0] % self.save_interval == 0:
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
#         save_path = os.path.join(path, "output")
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

    class DebugConfigs(SimpleNamespace):
        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

    def build_debug_configs():
        debug_dir = Path(__file__).resolve().parent / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return DebugConfigs(
            total_seq=[6, 6],
            test_seq=6,
            label_idx=[0, 1],
            in_category=["wind", "aux", "radar"],
            out_category=["wind"],
            img_size=[16, 32],
            patch_size=1,
            model_patch_size=[3, 5, 5],
            window_size=[3, 5, 5],
            embed_dim=32,
            num_heads=4,
            depths=2,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            use_multi_resolution_branch=0,
            use_multi_scale_patch_embedding=0,
            learning_rate=1e-4,
            weight_decay=0.0,
            pct_start=0.3,
            epoch=1,
            test_interval=1,
            batch_size=2,
            std_method="msr_std",
            std_params={
                "dataset": {
                    "mean": [[[[0.0]], [[0.0]], [[0.0]]]],
                    "std": [[[[30.0]], [[1.0]], [[70.0]]]],
                },
                "metric": {
                    "mean": [[[[[0.0]]]]],
                    "std": [[[[[30.0]]]]],
                },
            },
            threshold=[[8.0]],
            metrics=["mae"],
            obj_dir=str(debug_dir),
        )

    def ensure_shape(tensor, expected_shape, tag):
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{tag} expected shape {expected_shape}, but got {actual_shape}."
            )

    def ensure_input_grad(tensor, tag):
        if tensor.grad is None:
            raise RuntimeError(f"{tag} backward did not produce input gradients.")
        if not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"{tag} input gradients contain non-finite values.")

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

    input_btchw = torch.randn(2, 6, 1, 16, 32, device=device)
    expected_output_shape = (2, 6, 1, 16, 32)
    debug_summary = {
        "device": str(device),
        "input_btchw_shape": list(input_btchw.shape),
    }

    debug_model.zero_grad(set_to_none=True)
    model_input = input_btchw.detach().clone().requires_grad_(True)
    model_output = debug_model.model(model_input)
    ensure_shape(model_output, expected_output_shape, "self.model")
    model_loss = model_output.square().mean()
    model_loss.backward()
    ensure_input_grad(model_input, "self.model")
    model_grad_count, model_grad_abs_sum = summarize_gradients(
        debug_model.model, "self.model"
    )
    debug_summary["self.model"] = {
        "output_btchw_shape": list(model_output.shape),
        "loss": float(model_loss.detach().cpu().item()),
        "grad_param_count": model_grad_count,
        "grad_abs_sum": model_grad_abs_sum,
        "input_grad_abs_sum": float(model_input.grad.detach().abs().sum().item()),
    }

    debug_model.zero_grad(set_to_none=True)
    forward_input = input_btchw.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_input)
    ensure_shape(forward_output, expected_output_shape, "forward")
    forward_loss = forward_output.square().mean()
    forward_loss.backward()
    ensure_input_grad(forward_input, "forward")
    forward_grad_count, forward_grad_abs_sum = summarize_gradients(
        debug_model.model, "forward"
    )
    debug_summary["forward"] = {
        "output_btchw_shape": list(forward_output.shape),
        "loss": float(forward_loss.detach().cpu().item()),
        "grad_param_count": forward_grad_count,
        "grad_abs_sum": forward_grad_abs_sum,
        "input_grad_abs_sum": float(forward_input.grad.detach().abs().sum().item()),
    }

    print(json.dumps(debug_summary, ensure_ascii=False, indent=2))
