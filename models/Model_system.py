"""
System.py 使用说明
==================

这个版本用于 Lightning 单卡、DDP、FSDP 多卡训练，核心目标是：

1. 保留你自己的 Recorder 统计体系
   - 不依赖 Lightning 的 self.log(sync_dist=True) 来做最终统计。
   - train_loss / val_loss 由 System 自己在 epoch 结束时统计。

2. 避免 NCCL collective 顺序错乱
   - 不使用 all_gather_object 收集 Python list。
   - 不在 training_step / validation_step 中使用 self.log(..., sync_dist=True)。
   - 统一使用 all_reduce(sum/count) 计算所有 rank 的全局平均 loss。

3. 多卡统计逻辑
   - 每个 rank 本地累计 loss_sum 和 loss_count。
   - epoch 结束时，所有 rank 按相同顺序执行：
       all_reduce(loss_sum)
       all_reduce(loss_count)
   - 得到全局平均 loss。

4. Recorder / print / save_process
   - 只在 global rank 0 上执行。
   - 避免多卡重复写文件、重复打印。

5. validation 指标
   - val_loss 是所有 rank 的全局平均。
   - Recorder.within_the_epoch 默认只使用 rank0 的 output/label 子集。
   - 如果要让 Recorder 的 MAE/MSE/RMSE 也严格覆盖所有 rank，
     建议后续把 Recorder 改成 sum/count 统计，再 all_reduce 指标；
     不建议 gather 全部 output/label，大 tensor all_gather 容易慢或死锁。

6. Model.py 需要配合修改：
   - training_step 不要 self.log(sync_dist=True)
   - validation_step 不要 self.log(sync_dist=True)
   - training_step 返回：
       {"loss": loss, "train_loss": loss.detach(), "batch_size": batch_x.shape[0]}
   - validation_step 返回：
       {"val_loss": loss.detach(), "batch_size": batch_x.shape[0], "output": ..., "label": ...}
"""

import os
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import lightning as l

from utils import Recorder
from Instrument.standardizer import Load_Standardizer


def resolve_manual_parallel_devices(configs, main_device, required_gpus=2):
    main_device = torch.device(main_device)
    devices = tuple(main_device for _ in range(required_gpus))

    if main_device.type != "cuda":
        return devices

    if isinstance(configs, dict):
        visible_gpus = int(configs.get("gpu_count", torch.cuda.device_count()))
        trainer_devices = int(configs.get("devices", 1))
    else:
        visible_gpus = int(getattr(configs, "gpu_count", torch.cuda.device_count()))
        trainer_devices = int(getattr(configs, "devices", 1))

    is_distributed_child = "LOCAL_RANK" in os.environ or "RANK" in os.environ

    if visible_gpus < required_gpus or trainer_devices > 1 or is_distributed_child:
        return devices

    return tuple(torch.device(f"cuda:{idx}") for idx in range(required_gpus))


def distribute_model_layers(inner_model, configs, main_device):
    if not hasattr(inner_model, '_get_layer_groups'):
        return

    main_device = torch.device(main_device)

    if isinstance(configs, dict):
        visible_gpus = int(configs.get("gpu_count", 0))
        trainer_devices = int(configs.get("devices", 1))
    else:
        visible_gpus = int(getattr(configs, "gpu_count", 0))
        trainer_devices = int(getattr(configs, "devices", 1))

    is_ddp = "LOCAL_RANK" in os.environ or "RANK" in os.environ

    if visible_gpus < 2 or trainer_devices > 1 or is_ddp:
        devices = (main_device, main_device)
    else:
        devices = resolve_manual_parallel_devices(configs, main_device)

    current_devices = getattr(inner_model, '_layer_devices', None)
    if current_devices == devices:
        return

    layer_groups = inner_model._get_layer_groups(devices)
    for module, device in layer_groups:
        module.to(device)
    inner_model._layer_devices = devices


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

        self.best_valid_loss = float("inf")
        self.best_valid_epoch = 0

        self.train_loss_sum = None
        self.train_loss_count = None
        self.valid_loss_sum = None
        self.valid_loss_count = None

        self.test_outputs = []
        self.save = False
        self.save_list = []
        self.label_list = []
        self.data_idx = []
        self.rank_offset = 0
        self.save_interval = getattr(self.configs, "save_interval", 1)

    # =========================
    # distributed helpers
    # =========================

    def _dist_is_ready(self):
        return dist.is_available() and dist.is_initialized()

    def _is_global_zero(self):
        if hasattr(self, "trainer") and self.trainer is not None:
            return self.trainer.is_global_zero

        if self._dist_is_ready():
            return dist.get_rank() == 0

        return True

    def _new_scalar(self, value=0.0):
        return torch.tensor(float(value), device=self.device, dtype=torch.float32)

    def _all_reduce_sum(self, tensor):
        """
        所有 rank 对 tensor 求和。

        注意：
            这是 collective 操作。
            所有 rank 必须以相同顺序调用。
        """
        if self._dist_is_ready():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _reset_train_stats(self):
        self.train_loss_sum = self._new_scalar(0.0)
        self.train_loss_count = self._new_scalar(0.0)

    def _reduce_recorder_accumulators(self):
        """
        跨所有 rank 汇总 Recorder 的 sum/count 累加器。

        所有 rank 必须同步调用此方法。
        调用后 recorder 的 scalar_sums / scalar_counts / cvm_counts
        在所有 rank 上变为全局汇总后的值。
        """
        if not self._dist_is_ready():
            return

        rec = self.recoder
        device = self.device

        flat_values = []
        for key in ("mae", "mse", "rmse"):
            for val in rec.scalar_sums[key]:
                flat_values.append(float(val))
            for val in rec.scalar_counts[key]:
                flat_values.append(float(val))
        for cat_counts in rec.cvm_counts:
            for thr_counts in cat_counts:
                for key in ("tp", "fp", "fn", "tn"):
                    flat_values.append(float(thr_counts[key]))

        tensor = torch.tensor(flat_values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        idx = 0
        for key in ("mae", "mse", "rmse"):
            for i in range(len(rec.scalar_sums[key])):
                rec.scalar_sums[key][i] = float(tensor[idx].item())
                idx += 1
            for i in range(len(rec.scalar_counts[key])):
                rec.scalar_counts[key][i] = int(tensor[idx].item())
                idx += 1
        for cat_counts in rec.cvm_counts:
            for thr_counts in cat_counts:
                for key in ("tp", "fp", "fn", "tn"):
                    thr_counts[key] = int(tensor[idx].item())
                    idx += 1

    def _reset_valid_stats(self):
        self.valid_loss_sum = self._new_scalar(0.0)
        self.valid_loss_count = self._new_scalar(0.0)

    def _update_loss_stats(self, loss_sum_attr, loss_count_attr, loss, batch_size):
        """
        本 rank 内部累计 loss sum / count。

        loss 一般是当前 batch 的平均 loss。
        因此这里使用 loss * batch_size 累加，epoch end 再除以总样本数。
        """
        if loss is None:
            return

        if isinstance(loss, torch.Tensor):
            loss = loss.detach().to(self.device).float()
        else:
            loss = torch.tensor(float(loss), device=self.device, dtype=torch.float32)

        batch_size = int(batch_size)
        if batch_size <= 0:
            batch_size = 1

        current_sum = getattr(self, loss_sum_attr)
        current_count = getattr(self, loss_count_attr)

        current_sum += loss * float(batch_size)
        current_count += float(batch_size)

    def _global_average(self, local_sum, local_count):
        """
        使用 all_reduce 计算所有 rank 的全局平均值。

        返回:
            avg tensor，所有 rank 上值相同。
        """
        global_sum = self._all_reduce_sum(local_sum.clone())
        global_count = self._all_reduce_sum(local_count.clone())

        return global_sum / global_count.clamp_min(1.0)

    # =========================
    # subclass interfaces
    # =========================

    def get_model(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def current_lr(self):
        optimizer = getattr(self, "_last_configured_optimizer", None)
        if optimizer is None:
            return self.opt_config["lr"]
        return optimizer.param_groups[0]["lr"]

    def forward(self, batch_x, batch_y=None, **kwargs):
        raise NotImplementedError

    # =========================
    # train
    # =========================


    def on_train_epoch_start(self):
        epoch = self.current_epoch + 1

        self._reset_train_stats()
        self._reset_valid_stats()

        if self._is_global_zero():
            self.recoder.register_epoch(epoch)

    def training_step(self, batch, batch_idx):
        raise NotImplemented

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs is None or not isinstance(outputs, dict):
            return

        self._update_loss_stats(
            loss_sum_attr="train_loss_sum",
            loss_count_attr="train_loss_count",
            loss=outputs.get("train_loss", None),
            batch_size=outputs.get("batch_size", 1),
        )

    def on_train_epoch_end(self):
        avg_loss_tensor = self._global_average(
            self.train_loss_sum,
            self.train_loss_count,
        )

        avg_loss = float(avg_loss_tensor.detach().cpu())

        # 这里不要 sync_dist=True。
        # avg_loss_tensor 已经是 all_reduce 后的全局平均。
        # 这里 log 是为了让 Lightning 进度条/内部结果系统知道这个值。
        self.log(
            "train_loss",
            avg_loss_tensor.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )

        if self._is_global_zero():
            self.recoder.train_step(avg_loss, self.current_lr())

    # =========================
    # validation
    # =========================

    def on_validation_epoch_start(self):
        self._reset_valid_stats()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        if outputs is None or not isinstance(outputs, dict):
            return

        self._update_loss_stats(
            loss_sum_attr="valid_loss_sum",
            loss_count_attr="valid_loss_count",
            loss=outputs.get("val_loss", None),
            batch_size=outputs.get("batch_size", 1),
        )

        if "output" in outputs and "label" in outputs:
            output = self.standardizer.de_standardizing(outputs["output"], "metric")
            label = self.standardizer.de_standardizing(outputs["label"], "metric")
            self.recoder.within_the_epoch(output, label)

    def on_validation_epoch_end(self):
        avg_loss_tensor = self._global_average(
            self.valid_loss_sum,
            self.valid_loss_count,
        )

        avg_loss = float(avg_loss_tensor.detach().cpu())

        # 这里不要 sync_dist=True。
        # avg_loss_tensor 已经是全 rank 平均值。
        # ModelCheckpoint(monitor="val_loss") 会读取这个 val_loss。
        self.log(
            "val_loss",
            avg_loss_tensor.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

        self._reduce_recorder_accumulators()

        if self._is_global_zero():
            self.recoder.valid_step(avg_loss)

            if avg_loss < self.best_valid_loss:
                self.best_valid_loss = avg_loss
                self.best_valid_epoch = self.current_epoch + 1
                print(
                    f"当前最佳验证损失: {self.best_valid_loss:.6f}, "
                    f"发生在第{self.best_valid_epoch}轮",
                    flush=True,
                )

            self.recoder.save_process()

    # =========================
    # test
    # =========================

    def test_prepare(self, data_idx, is_save):
        """
        多卡时，每个 rank 取自己的 data_idx 连续分片。
        DistributedSampler 把数据集等分成 N 份，按 rank 顺序分配，
        所以 data_idx 的切片与 sampler 的分配一致。
        """
        self.rank_offset = 0
        if self._dist_is_ready():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            total = len(data_idx)
            per_rank = math.ceil(total / world_size)
            self.rank_offset = rank * per_rank
            data_idx = data_idx[self.rank_offset : self.rank_offset + per_rank]

        self.data_idx = data_idx
        self.save_interval = self.configs.save_interval
        self.save_list = []
        self.label_list = []
        self.save = is_save

    def test_sample(self, batch_idx, output):
        start = batch_idx * self.batch_size
        end = min(start + output.shape[0], len(self.data_idx))

        for i, local_lab in enumerate(range(start, end)):
            global_lab = self.rank_offset + local_lab
            if global_lab % self.save_interval == 0:
                self.save_list.append(output[i].detach().cpu())
                self.label_list.append(self.data_idx[local_lab])
                break

    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        output = self(batch_x)
        label = batch_y[:, :, self.label_idx[0]: self.label_idx[1], :, :]

        output = self.standardizer.de_standardizing(output, "metric")
        label = self.standardizer.de_standardizing(label, "metric")

        self.recoder.within_the_epoch(output, label)

        if self.save is True:
            self.test_sample(batch_idx, output)

    def test_save(self, path):
        if len(self.save_list) == 0:
            return

        data = torch.stack(self.save_list, dim=0).numpy()

        save_path = os.path.join(path, "outputs")
        os.makedirs(save_path, exist_ok=True)

        if self._dist_is_ready():
            rank = dist.get_rank()
            data_file = os.path.join(save_path, f"out_data_rank{rank}.npy")
            label_file = os.path.join(save_path, f"out_label_rank{rank}.json")
        else:
            data_file = os.path.join(save_path, "out_data.npy")
            label_file = os.path.join(save_path, "out_label.json")

        np.save(data_file, data)

        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(self.label_list, f, ensure_ascii=False, indent=4)

    def on_test_epoch_end(self):
        self._reduce_recorder_accumulators()

        if self._is_global_zero():
            self.recoder.test_step()
            print(f"测试完成，结果已保存到 {self.configs.obj_dir}。", flush=True)

        if self.save is True:
            self.test_save(self.configs.obj_dir)
