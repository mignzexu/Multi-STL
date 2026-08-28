# Multi-STL 模型迁移指南

## 概述

本指南说明如何将独立的 PyTorch 模型迁移到 Multi-STL 框架的 **System-Model 双层架构**。迁移后，模型将自动获得分布式训练支持、标准化指标统计、Checkpoint 管理等能力，无需手动实现。

---

## 1. 架构对比

### 迁移前（旧版风格）

```python
import lightning as l

class Main(l.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.model = YourModel(configs)
        self.criterion = nn.MSELoss()
        self.recorder = Recorder(configs)
        self.train_epoch_loss = []
        self.valid_epoch_loss = []
        # ... 更多初始化

    def training_step(self, batch, batch_idx):
        # 前向计算
        loss = ...
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "train_loss": loss.detach().cpu().item()}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.train_epoch_loss.append(outputs["train_loss"])

    def on_train_epoch_end(self):
        avg_loss = torch.tensor(self.train_epoch_loss).mean().item()
        self.recorder.train_step(avg_loss, self.current_lr())
        self.train_epoch_loss.clear()

    # ... 验证、测试逻辑类似，全部手动实现
```

### 迁移后（System 架构风格）

```python
from ..Model_system import System

class Model(System):
    def __init__(self, configs):
        super().__init__(configs)
        # self.model, self.criterion, self.recorder 等由 System 基类自动初始化

    def get_model(self):
        return YourModel(self.configs)

    def forward(self, batch_x, batch_y=None, **kwargs):
        # 推理接口：测试时调用
        pred_y = self.model(batch_x)
        if self.test_seq < self.aft_seq_length:
            pred_y = pred_y[:, :self.test_seq]
        return pred_y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self.model(batch_x)  # 直接调用 self.model，不是 self.forward
        label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.criterion(pred, label)
        return {"loss": loss, "train_loss": loss.detach(), "batch_size": batch_x.shape[0]}

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred = self.model(batch_x)  # 直接调用 self.model，不是 self.forward
        label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.criterion(pred, label)
        return {"val_loss": loss.detach(), "batch_size": batch_x.shape[0], "output": pred.detach(), "label": label.detach()}
```

---

## 2. 迁移步骤

### 2.1 文件结构

迁移后的模型目录应包含以下文件：

```
models/YourModel/
├── __init__.py          # 导出 Model 和 Config
├── Model.py             # 核心：继承 System 的 Model 类
├── config.py            # 模型配置类
├── loss.py              # 损失函数（可选）
└── YourModel_model.py   # 模型实现（所有源码必须整合到此文件）
```

**重要**：`YourModel_model.py` 必须包含模型的所有实现代码，不能引用 `origin/` 目录中的模块。`origin/` 目录是 git ignore 的外部参考代码，不会被版本控制。

### 2.2 修改 `__init__.py`

**旧版**：
```python
from .YourModel_model import Main
from .config import YourModel_Config
from .loss import loss_fn

__all__ = ['Main', 'YourModel_Config', 'loss_fn']
```

**新版**：
```python
from .Model import Model
from .config import YourModel_Config

__all__ = ['Model', 'YourModel_Config']
```

**关键变化**：
- 导出类名从 `Main` 改为 `Model`
- 不再导出 `loss_fn`（损失函数在 `Model.py` 内部使用）

### 2.3 创建 `Model.py`

这是迁移的核心文件。以下是完整模板：

```python
import torch
from torch import optim
from ..Model_system import System

class Model(System):
    def __init__(self, configs):
        super().__init__(configs)

    def get_model(self):
        """返回原始模型实例"""
        return YourModel(self.configs)

    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        推理接口：测试时调用此方法获取预测结果。

        参数:
            batch_x: 输入张量 [B, T_in, C, H, W]

        返回:
            pred_y: 预测张量 [B, T_test, C, H, W]
        """
        pred_y = self.model(batch_x)

        if self.test_seq < self.aft_seq_length:
            pred_y = pred_y[:, :self.test_seq]

        return pred_y

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )
        self._last_configured_optimizer = optimizer
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        """训练步"""
        batch_x, batch_y = batch
        pred = self.model(batch_x)
        label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.criterion(pred, label)

        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
        }

    def validation_step(self, batch, batch_idx):
        """验证步"""
        batch_x, batch_y = batch
        pred = self.model(batch_x)
        label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
        loss = self.criterion(pred, label)

        return {
            "val_loss": loss.detach(),
            "batch_size": batch_x.shape[0],
            "output": pred.detach(),
            "label": label.detach(),
        }
```

---

## 3. 关键规则

### 3.1 必须实现的方法

| 方法 | 说明 | 返回值要求 |
|------|------|-----------|
| `get_model()` | 返回原始神经网络实例 | `nn.Module` |
| `forward()` | 推理接口（整合预测逻辑） | `torch.Tensor` |
| `configure_optimizers()` | 优化器配置 | `dict` 或 `OptimizerLRScheduler` |
| `training_step()` | 训练逻辑 | 见下方 |
| `validation_step()` | 验证逻辑 | 见下方 |

### 3.2 返回值格式（严格要求）

**`training_step` 必须返回**：
```python
{
    "loss": loss,                    # 用于反向传播的 loss（带梯度）
    "train_loss": loss.detach(),     # 用于统计的 loss（detach）
    "batch_size": batch_x.shape[0],  # 当前 batch 大小
}
```

**`validation_step` 必须返回**：
```python
{
    "val_loss": loss.detach(),       # 用于统计的 loss（detach）
    "batch_size": batch_x.shape[0],  # 当前 batch 大小
    "output": pred.detach(),         # 预测结果（用于计算指标）
    "label": label.detach(),         # 标签（用于计算指标）
}
```

### 3.3 禁止事项

1. **不要在 `training_step` / `validation_step` 中调用 `self.log(sync_dist=True)`**
   - loss 统计由 System 基类自动处理
   - 使用 `sync_dist=True` 可能导致 NCCL 集体操作顺序错乱

2. **不要手动实现 epoch 级别的统计逻辑**
   - 删除 `on_train_epoch_end`、`on_validation_epoch_end` 中的自定义逻辑
   - 删除 `self.train_epoch_loss`、`self.valid_epoch_loss` 等列表

3. **不要手动实现 Recorder 调用**
   - `recorder.train_step()`、`recorder.valid_step()` 由 System 自动调用

4. **不要手动实现分布式同步**
   - `all_reduce`、`all_gather` 等操作由 System 自动处理

5. **不要在训练/验证时调用 `self.forward()`**
   - 训练/验证时直接调用 `self.model(batch_x)`
   - `forward()` 是推理接口，由框架在测试时自动调用

6. **不要引用 `origin/` 目录中的模块**
   - `origin/` 是 git ignore 的外部参考代码
   - 所有实现代码必须整合到 `{model_name}_model.py` 中

### 3.4 可选覆盖的方法

如果需要特殊行为，可以覆盖以下方法：

```python
def on_train_epoch_start(self):
    """训练 epoch 开始时的自定义逻辑"""
    super().on_train_epoch_start()  # 必须调用父类
    # 自定义逻辑

def on_validation_epoch_start(self):
    """验证 epoch 开始时的自定义逻辑"""
    super().on_validation_epoch_start()  # 必须调用父类
    # 自定义逻辑
```

---

## 4. 推理接口（`forward`）

**`forward` 是模型在推理（测试）时的统一入口**，负责整合所有预测逻辑并返回最终结果。它不是训练时的前向传播——训练时的前向计算直接在 `training_step` / `validation_step` 中调用 `self.model(batch_x)` 完成。

### 4.1 核心职责

`forward` 需要处理 `test_seq`（测试需要输出的帧数）与 `aft_seq_length`（模型单次输出帧数）之间的关系：

- `test_seq == aft_seq_length`：直接返回模型输出
- `test_seq < aft_seq_length`：截取前 `test_seq` 帧返回
- `test_seq > aft_seq_length`：需要滚动预测（多次调用模型拼接结果）

### 4.2 标准实现

```python
def forward(self, batch_x, batch_y=None, **kwargs):
    pred_y = self.model(batch_x)

    if self.test_seq < self.aft_seq_length:
        pred_y = pred_y[:, :self.test_seq]

    return pred_y
```

当 `test_seq > aft_seq_length` 时，需要实现滚动预测逻辑（多次调用模型，逐步拼接结果）。可参考 `models/SimVP_gSTA/Model.py` 中的实现。

---

## 5. System 基类提供的属性和方法

继承 System 后，以下属性和方法可直接使用，无需手动定义：

### 5.1 常用属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.configs` | `Namespace` | 所有配置参数的集合 |
| `self.model` | `nn.Module` | 由 `get_model()` 返回的原始模型 |
| `self.criterion` | `nn.Module` | 损失函数，默认 `nn.MSELoss()` |
| `self.standardizer` | `Standardizer` | 数据标准化/反标准化工具 |
| `self.recoder` | `Recorder` | 指标统计记录器 |
| `self.pre_seq_length` | `int` | 输入序列长度（`total_seq[0]`） |
| `self.aft_seq_length` | `int` | 输出序列长度（`total_seq[1]`） |
| `self.label_idx` | `list` | 输出通道在输入中的索引范围 `[start, end]` |
| `self.test_seq` | `int` | 测试时需要输出的帧数 |
| `self.batch_size` | `int` | 批次大小 |

### 5.2 数据格式

框架统一使用 **5D 张量**格式：`[B, T, C, H, W]`

- `B`：批次大小
- `T`：时间步数（帧数）
- `C`：通道数
- `H`：高度
- `W`：宽度

输入 `batch_x` 和标签 `batch_y` 都是此格式。

### 5.3 前向调用说明

- **训练/验证时**：在 `training_step` / `validation_step` 中直接调用 `self.model(batch_x)`，不要调用 `self.forward()`
- **测试/推理时**：由框架自动调用 `self.forward()` 获取预测结果

```python
def training_step(self, batch, batch_idx):
    batch_x, batch_y = batch
    pred = self.model(batch_x)  # ✅ 正确：直接调用 self.model
    # pred = self.forward(batch_x)  # ❌ 错误：不要调用 self.forward
    ...
```

---

## 6. 配置管理

### 6.1 配置类定义

```python
# config.py
class YourModel_Config:
    def __init__(self):
        self.model_config = {
            "learning_rate": 5e-3,
            "weight_decay": 0.0,
            "lr_min": 1e-6,
            "warmup_t": 0,
            "warmup_lr_init": 1e-5,
            # 模型特定参数
            "hidden_dim": 256,
            "num_layers": 4,
            # ...
        }
```

### 6.2 配置注册

在 `Instrument/models.py` 中添加模型注册：

```python
elif self.model_name == 'yourmodel':
    from models.YourModel import Model, YourModel_Config

    if self.mode == 'train':
        model_config = YourModel_Config().model_config
        self.configs = config_update(self.configs, model_config)

    self.model = Model
```

---

## 7. 损失函数处理

### 7.1 使用默认 MSELoss

System 基类默认使用 `nn.MSELoss()`，无需额外配置。

### 7.2 使用自定义损失函数

在 `Model.__init__` 中覆盖 `self.criterion`：

```python
class Model(System):
    def __init__(self, configs):
        super().__init__(configs)
        self.criterion = YourCustomLoss()  # 覆盖默认损失函数
```

### 7.3 复杂损失函数（带中间输出）

如果损失函数需要返回中间结果（如 STPANet 的 `loss_bridge`）：

```python
def training_step(self, batch, batch_idx):
    batch_x, batch_y = batch
    loss, _ = self.loss_bridge(self, batch_x, batch_y)  # 损失函数返回 loss 和中间结果

    return {
        "loss": loss,
        "train_loss": loss.detach(),
        "batch_size": batch_x.shape[0],
    }
```

---

## 8. 优化器配置

### 8.1 标准配置

```python
def configure_optimizers(self):
    optimizer = optim.Adam(
        self.model.parameters(),
        lr=self.configs.learning_rate,
        weight_decay=self.configs.weight_decay,
    )
    self._last_configured_optimizer = optimizer
    return {"optimizer": optimizer}
```

### 8.2 带学习率调度器

```python
def configure_optimizers(self):
    optimizer = optim.Adam(
        self.model.parameters(),
        lr=self.configs.learning_rate,
        weight_decay=self.configs.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=self.configs.epoch,
        eta_min=self.configs.lr_min,
    )

    self._last_configured_optimizer = optimizer
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
        },
    }
```

### 8.3 自定义调度器步骤

如果使用 `timm` 的 `CosineLRScheduler`，需要覆盖 `lr_scheduler_step`：

```python
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
```

---

## 9. 验证清单

迁移完成后，请检查以下事项：

### 9.1 代码结构检查

- [ ] `Model` 类继承 `System`
- [ ] `__init__.py` 导出 `Model` 和 `Config`
- [ ] `get_model()` 返回原始模型实例
- [ ] `training_step` 返回 `{"loss", "train_loss", "batch_size"}`
- [ ] `validation_step` 返回 `{"val_loss", "batch_size", "output", "label"}`
- [ ] 没有调用 `self.log(sync_dist=True)`
- [ ] 没有手动实现 epoch 级别的统计逻辑
- [ ] 没有手动调用 Recorder
- [ ] 训练/验证时调用 `self.model(batch_x)` 而非 `self.forward(batch_x)`
- [ ] `forward` 方法能够实现推理逻辑
- [ ] 模型实现代码已整合到 `{model_name}_model.py`，不引用 `origin/` 目录
- [ ] 在 `Instrument/models.py` 中注册模型

### 9.2 调试测试（必须）

在 `Model.py` 文件末尾添加 `if __name__ == "__main__":` 调试代码，验证以下内容：

1. **前向传播**：`self.model(batch_x)` 能正常输出，形状正确
2. **反向传播**：梯度能正常回传，不掉梯度
3. **推理接口**：`forward()` 能正常调用，输出形状正确

```python
import json
from pathlib import Path
from types import SimpleNamespace

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. 构建调试配置
    def build_debug_configs():
        debug_dir = Path(__file__).resolve().parent / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            total_seq=[2, 2],          # [输入帧数, 输出帧数]
            test_seq=2,
            label_idx=[0, 1],
            in_category=["tp"],        # 输入通道
            out_category=["tp"],       # 输出通道
            img_size=[8, 8],           # 空间尺寸
            learning_rate=5e-4,
            weight_decay=0.0,
            epoch=1,
            std_method="z_score",
            std_params={
                "dataset": {"mean": [[[[0.0]]]], "std": [[[[1.0]]]]},
                "metric": {"mean": [[[[[0.0]]]]], "std": [[[[[1.0]]]]]},
            },
            threshold=[[0.5]],
            metrics=["mae"],
            obj_dir=str(debug_dir),
            # ... 其他模型特定参数
        )

    # 2. 辅助函数
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

    # 3. 测试 self.model（原始模型）
    configs = build_debug_configs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_model = Model(configs).to(device)
    debug_model.train()

    batch_size = 2
    channels = len(configs.in_category)
    timesteps = configs.total_seq[0]
    height, width = configs.img_size
    input_tensor = torch.randn(batch_size, channels, timesteps, height, width, device=device)

    debug_model.zero_grad(set_to_none=True)
    model_input = input_tensor.detach().clone().requires_grad_(True)
    model_output = debug_model.model(model_input)
    model_loss = model_output.square().mean()
    model_loss.backward()

    ensure_input_grad(model_input, "self.model")
    grad_count, grad_sum = summarize_gradients(debug_model.model, "self.model")
    print(f"[self.model] output shape: {model_output.shape}, grad params: {grad_count}, grad sum: {grad_sum:.4f}")

    # 4. 测试 forward（推理接口）
    debug_model.zero_grad(set_to_none=True)
    forward_input = input_tensor.detach().clone().requires_grad_(True)
    forward_output = debug_model(forward_input)
    forward_loss = forward_output.square().mean()
    forward_loss.backward()

    ensure_input_grad(forward_input, "forward")
    grad_count, grad_sum = summarize_gradients(debug_model.model, "forward")
    print(f"[forward] output shape: {forward_output.shape}, grad params: {grad_count}, grad sum: {grad_sum:.4f}")

    print("All checks passed!")
```

**运行方式**：
```bash
python models/YourModel/Model.py
```

如果输出 `All checks passed!` 且无报错，说明迁移基本正确。

---

## 10. 常见问题

### Q1: 为什么 `training_step` 返回的 `train_loss` 需要 `detach()`？

A: `loss` 用于反向传播，必须保持计算图。`train_loss` 用于统计，需要 `detach()` 断开计算图，避免内存泄漏。

### Q2: 为什么不能使用 `self.log(sync_dist=True)`？

A: `sync_dist=True` 会导致 NCCL 集体操作，可能与 System 基类的 `all_reduce` 操作顺序冲突，导致死锁或错误。System 基类使用更安全的 `all_reduce(sum/count)` 方式。

### Q3: 如何处理多通道输入输出？

A: 使用 `self.label_idx` 从 `batch_y` 中提取对应的通道：
```python
label = batch_y[:, :, self.label_idx[0]:self.label_idx[1], :, :]
```

### Q4: 如何在验证时计算额外指标？

A: `validation_step` 返回的 `output` 和 `label` 会被 System 基类自动用于计算 Recorder 中的指标（MAE、MSE、RMSE 等）。如需自定义指标，可覆盖 `on_validation_batch_end`。

### Q5: 迁移后模型无法被 `Model_Instrument` 加载？

A: 检查 `Instrument/models.py` 是否正确注册了模型名称。名称必须小写且与命令行参数一致。

### Q6: 为什么训练时要调用 `self.model(batch_x)` 而不是 `self.forward(batch_x)`？

A: `forward()` 是推理接口，主要用于测试时获取预测结果。训练时直接调用 `self.model(batch_x)` 更高效，且避免触发 `forward` 中可能存在的滚动预测等推理专用逻辑。

### Q7: 为什么不能引用 `origin/` 目录中的代码？

A: `origin/` 目录是 git ignore 的外部参考代码，不会被版本控制。迁移时必须将所有实现代码整合到 `{model_name}_model.py` 中，确保模型代码的完整性和可移植性。

---

## 11. 完整示例

参考 `models/SimVP_gSTA/Model.py`、`models/MSRadar/Model.py` 或 `models/STPANet/Model.py` 的实现。
