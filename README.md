# Multi-STL

多源数据融合降水临近预报模型 —— 基于 PyTorch Lightning 的时空序列预测框架。

## 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
  - [训练](#训练)
  - [测试](#测试)
  - [多卡训练](#多卡训练)
  - [可视化](#可视化)
- [核心架构](#核心架构)
- [已集成的模型](#已集成的模型)
- [数据集](#数据集)
- [配置说明](#配置说明)

## 项目简介

Multi-STL 是一个面向气象临近预报的深度学习框架，支持多源观测数据（雷达回波、降水、风场、温度等）的时空序列建模。框架采用 **System-Model 双层架构**，以 PyTorch Lightning 为训练引擎，原生支持单卡/多卡（DDP/FSDP/DeepSpeed）训练、混合精度和不限期续训。

核心设计理念：

- **System 层**（`models/Model_system.py`）：所有模型共享的训练基础设施 —— 分布式 loss 统计、指标记录、标准化管道、Checkpoint 管理；
- **Model 层**（各 `models/*/Model.py`）：具体模型的前向传播、损失函数、优化器配置。

## 目录结构

```
Multi-STL/
├── main.py                  # 入口脚本，解析 CLI 参数并分发到 train/test 流程
├── visualization.py         # 预测结果可视化工具，支持单图 / 对比图 / GIF 导出
├── look.py                  # 快速检查 .npy 文件形状的调试工具
│
├── models/                  # 模型仓库（17 个模型，17 个已适配 System 架构）
│   ├── Model_system.py      # ★ 所有模型的基类，继承 LightningModule
│   ｜                         定义 training_step / validation_step / test_step 标准流程
│   ├── SimVP_gSTA/          # gSTA（默认模型），基于 Transformer 的视频预测
│   ├── MSRadar/             # MS-RadarFormer，面向雷达数据的 Transformer
│   ├── STPANet/             # 时空渐进注意力网络
│   ├── TAU/                 # 时间注意力单元
│   ├── SwinLSTM/            # Swin + LSTM 混合架构
│   ├── ConvLSTM/            # 卷积 LSTM
│   ├── E3DLSTM/             # 3D 卷积 LSTM（含 PredRNN）
│   ├── MAU/                 # 运动感知单元
│   ├── MIM/                 # 记忆交互模块
│   ├── MMVP/                # 多模态视频预测
│   ├── PerdRNNpp/           # PredRNN++（含梯度高速网络）
│   ├── PhyDNet/             # 物理约束网络
│   ├── PoolFormer/          # PoolFormer（含 gSTA 变体）
│   ├── PreDiff/             # 预训练扩散模型
│   ├── PredRNN/             # PredRNN 原始版本
│   ├── PredRNNv2/           # PredRNN v2
│   └── SimVP_IncepU/        # SimVP + InceptionU
│
├── dataset/                 # 数据集加载与预处理
│   ├── SDweather/           # 站点观测天气数据
│   │   ├── dataloader_SD.py # SD 数据加载器
│   │   ├── configs/         # 数据集配置文件（config1~config5）
│   │   └── vis/             # 可视化画笔
│   └── WeatherBench/        # WeatherBench 气象基准数据
│       ├── dataloader_WB.py # WB 数据加载器
│       ├── configs/         # 数据集配置文件
│       └── vis_WB.py        # 可视化画笔
│
├── Instrument/              # 工厂 & 基础工具层
│   ├── models.py            # Model_Instrument：按模型名动态导入 Model 类并注入配置
│   ├── datasets.py          # Dataset_Instrument：按数据集名加载对应的 Dataloader
│   └── standardizer.py      # Load_Standardizer：管理数据标准化/反标准化方法
│
├── train_test/              # 训练 & 测试引擎
│   ├── trainer.py           # Trainer：封装 Lightning Trainer，处理 DataLoader / 策略 / Checkpoint
│   └── tester.py            # Tester：加载最佳 Checkpoint 并执行测试循环
│
├── utils/                   # 工具库
│   ├── __init__.py          # 统一导出接口
│   ├── cfg.py               # Logger：实验配置记录与训练/测试日志的格式化输出
│   ├── cfg_up.py            # config_update：运行时动态合并配置参数
│   ├── log.py               # train_config：GPU 显存管理器（防止 OOM）& save_loger
│   ├── metrics.py           # Recorder 旧版（基于 batch 累积）
│   ├── metrics_v2.py        # Recorder 新版（基于 sum/count 累加器，支持多卡 all_reduce）
│   └── std_method.py        # Z_Score / Z_Score_SD：数据标准化方法实现
│
├── origin/                  # 引用的外部研究代码
│   └── conditional-flow-matching/  # Conditional Flow Matching 实现
│
├── work_dirs/               # ★ 实验输出目录（每个实验一个子目录）
│   └── <experiment_name>/   # 由 --ex_name 指定
│       ├── obj_config.json  # 实验完整配置快照
│       ├── log.txt          # 训练日志（含指标表格）
│       ├── model/           # Checkpoint 文件
│       ├── process/         # 训练/验证指标 JSON
│       ├── outputs/         # 测试预测结果 (.npy)
│       └── vis/             # 可视化输出（可选）
│
└── dataset/                 # 外部数据目录
    └── data.txt             # 数据来源与目录路径说明
```

## 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Lightning ≥ 2.0
- NumPy, tqdm

### 最简训练

```bash
# 使用默认模型 (gSTA) + WeatherBench 数据集训练
python main.py --train -ex my_exp
```

### 切换模型

```bash
python main.py --train -ex msradar_exp -mod msradar          # MS-RadarFormer
python main.py --train -ex stpanet_exp -mod stpanet          # STPANet
python main.py --train -ex convlstm_exp -mod convlstm        # ConvLSTM
python main.py --train -ex tau_exp -mod tau                  # TAU
```

### 切换数据集

```bash
python main.py --train -ex sd_exp -ds SDweather -mod gsta    # 使用 SDweather 数据集
```

## 使用说明

### 训练

```bash
python main.py --train \
    -ex <实验名称> \           # 必填，实验标识
    -mod <模型名> \            # 可选，默认 gsta，见“已集成的模型”表
    -ds <数据集> \             # 可选，默认 weatherbench (SDweather)
    -b <批次大小> \            # 可选，默认 32
    -e <训练轮数> \            # 可选，默认 200
    -dc <数据配置> \           # 可选，数据集预定义配置名
    -mc <模型配置> \           # 可选，模型预定义配置名
    -sty <训练策略> \          # 可选，默认 auto (ddp | fsdp | deepspeed_stage_2 | ...)
    --precision 16-mixed \     # 可选，默认 32-true (bf16-mixed)
    --num_workers 4 \          # 可选，DataLoader 线程数
    --seed 42 \                # 可选，随机种子
    -sc                        # 可选，保存配置文件副本到实验目录
```

> **注意**：训练前会检测实验目录是否已存在。若存在，会提示是否覆盖。若需要续训，请在模型代码中配置 `ckpt_path` 参数。

### 测试

```bash
python main.py --test \
    -ex <已完成训练的实验名> \   # 必填
    -b <批次大小> \             # 可选，覆盖训练时的 batch_size
    --save                      # 可选，保存模型预测输出 (.npy)
```

测试模式会从 `work_dirs/<ex_name>/model/` 目录加载最佳 Checkpoint（按 `val_loss` 最低筛选），并计算 MAE / MSE / RMSE 等评估指标。

### 多卡训练

```bash
# 使用全部 GPU（自动 DDP）
python main.py --train -ex multi_gpu -mod gsta --device all

# 指定 GPU 并显式声明 DDP 策略
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --train -ex multi_gpu -mod gsta -sty ddp

# FSDP（大模型适用）
python main.py --train -ex fsdp_exp -mod stpanet --device all -sty fsdp

# DeepSpeed Stage 2
python main.py --train -ex ds_exp -mod msradar --device all -sty deepspeed_stage_2
```

多卡模式下，框架会自动：
- 将 `batch_size` 按 GPU 数量均分到每张卡
- 使用 `all_reduce` 汇总所有 rank 的 loss 和指标
- 只在 rank 0 输出日志和保存 Checkpoint

### 可视化

```bash
# 仅看预测结果
python visualization.py -ex <实验名>

# 预测 vs 真实值对比
python visualization.py -ex <实验名> --contrast

# 只看输入数据
python visualization.py -ex <实验名> --input

# 生成 GIF 动画
python visualization.py -ex <实验名> --gif

# 指定输出目录
python visualization.py -ex <实验名> -pd ./my_vis_output
```

## 核心架构

### 训练流程

```
main.py
  │
  ├─[args parse]────────────────────────→ configs (argparse.Namespace)
  │
  ├─[Dataset_Instrument.load_dataset]───→ train_dataset / valid_dataset
  │   │  根据 --dataset 选择数据加载器
  │   │  根据 --data_config 注入数据集参数（total_seq, img_size, in/out_category...）
  │   └─────────────────────────────────→ configs 被动态更新
  │
  ├─[Model_Instrument.load_model]───────→ Model 类
  │   │  根据 --load_model 选择模型
  │   │  根据 --model_config 注入模型参数
  │   └─────────────────────────────────→ configs 被进一步更新
  │
  ├─[Trainer.__init__]──────────────────→ Model 实例化 + DataLoader 构建
  │   │  DataLoader 自动处理 batch_size 分配 / num_workers / pin_memory
  │   │  ModelCheckpoint 监控 val_loss
  │   └─────────────────────────────────→ Lightning Trainer 创建
  │
  └─[Trainer.train]─────────────────────→ trainer.fit(model, train_data, val_data)
      │
      └─ Per Epoch ──→ System.training_step   → on_train_batch_end (累计 loss)
                    → System.validation_step → on_validation_batch_end (累计指标)
                    → on_validation_epoch_end → all_reduce 汇总 → Recorder 输出
```

### Model-System 双层架构

```
LightningModule (PyTorch Lightning)
    │
    └── System (models/Model_system.py)    ← 基类：定义训练/验证/测试标准流程
         │  ├── training_step / validation_step   → 子类实现具体前向计算
         │  ├── on_*_epoch_end                    → 分布式 loss 聚合 + 指标统计
         │  ├── Recorder                           → 指标累积与 JSON 持久化
         │  └── Load_Standardizer                  → 数据标准化/反标准化
         │
         └── Model (models/*/Model.py)      ← 具体模型：重写 forward / get_model / configure_optimizers
              │  ├── get_model()             → 返回神经网络实例
              │  ├── forward(batch_x)        → 前向推理（含滚动预测逻辑）
              │  └── configure_optimizers()  → 优化器 + 学习率调度
```

### 配置合并机制

配置通过三层叠加，后层覆盖前层：

1. **CLI 默认值**（`main.py` 中的 `argparse` 默认值）
2. **数据集配置**（`dataset/*/configs/` 中的预定义字典）—— 由 `Dataset_Instrument` 注入
3. **模型配置**（模型目录中的 `*_Config` 类）—— 由 `Model_Instrument` 注入

所有操作通过 `utils.cfg_up.config_update()` 执行，本质是对 `argparse.Namespace` 做动态 `setattr`。

### 分布式指标统计

多卡训练时，每个 rank 在 `on_validation_batch_end` 中本地累加 sum/count。epoch 结束时：

```
rank0: loss_sum=10, count=5
rank1: loss_sum=8,  count=3
      ↓ all_reduce(SUM)
total: loss_sum=18, count=8  →  avg = 18/8 = 2.25
```

此方式避免了 `sync_dist=True` 可能引发的 NCCL collective 错序和死锁问题。

## 已集成的模型

| 模型 | 命令行参数 | System 适配 | 简介 |
|------|-----------|------------|------|
| **SimVP_gSTA** | `-mod gsta` | ✓ | 基于 Transformer 的视频预测，默认模型 |
| **MSRadar** | `-mod msradar` | ✓ | MS-RadarFormer，面向雷达数据的专用架构 |
| **STPANet** | `-mod stpanet` | ✓ | 时空渐进注意力网络 |
| TAU | `-mod tau` | ✓ | 时间注意力单元 |
| SwinLSTM | `-mod swinlstm` | ✓ | Swin Transformer + LSTM |
| ConvLSTM | `-mod convlstm` | ✓ | 卷积 LSTM 基线 |
| E3DLSTM | `-mod e3dlstm` | ✓ | 3D 卷积 LSTM（Eidetic） |
| PredRNN | `-mod predrnn` | ✓ | 预测 RNN 原版 |
| PredRNNv2 | `-mod predrnnv2` | ✓ | 预测 RNN 第二版 |
| PredRNN++ | `-mod predrnnpp` | ✓ | 带梯度高速网络的预测 RNN |
| MAU | `-mod mau` | ✓ | 运动感知单元 |
| MIM | `-mod mim` | ✓ | 记忆交互模块 |
| MMVP | `-mod mmvp` | ✓ | 多模态视频预测 |
| PhyDNet | `-mod phydnet` | ✓ | 物理约束网络 |
| PoolFormer | `-mod poolformer` | ✓ | PoolFormer（含 gSTA 变体） |
| SimVP_IncepU | `-mod simvp_incepu` | ✓ | SimVP + InceptionU |
| PreDiff | `-mod prediff` | ✓ | 预训练扩散模型 |

> 标记 ✓ 的模型均已适配 System 架构（`models/Model_system.py`），可通过 `-mod` 命令行参数切换。

## 数据集

### SDweather（站点观测数据）

中国区域气象站点观测数据，包含降水、温度、风场等通道。数据存储为 `.npy` 格式。

```bash
python main.py --train -ex sd_exp -ds SDweather -mod gsta
```

预定义配置：
- `config1` ~ `config5`：不同的输入/输出通道组合和时空窗口配置

### WeatherBench（气象基准数据）

全球气象再分析/预报数据集，支持风速、位势高度、温度等变量的预测。

```bash
python main.py --train -ex wb_exp -ds weatherbench -mod gsta
```

预定义配置：
- `test`：基础配置
- `config1` ~ `config3`：不同的变量组合和预测时长

### 数据目录配置

数据默认从 `/shares/weather/Split_Data` 加载，可通过 `--data_dir` 覆盖。各数据集子目录结构请参考 `dataset/data.txt`。

## 配置说明

### 常用命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--ex_name` | `-ex` | `test` | 实验名称，输出到 `work_dirs/<名称>/` |
| `--load_model` | `-mod` | `gsta` | 模型选择 |
| `--dataset` | `-ds` | `weatherbench` | 数据集选择 |
| `--train` | | `False` | 训练模式 |
| `--test` | | `False` | 测试模式 |
| `--retrain` | | `False` | 续训模式 |
| `--batch_size` | `-b` | `32` | 批次大小 |
| `--epoch` | `-e` | `200` | 训练轮数 |
| `--device` | | `all` | GPU ID（`all`/`0,1`/`cpu`） |
| `--seed` | | `42` | 随机种子 |
| `--strategy` | `-sty` | `auto` | Lightning 训练策略 |
| `--precision` | | `32-true` | 训练精度（`16-mixed`/`bf16-mixed`） |
| `--data_config` | `-dc` | 数据集默认 | 数据集预定义配置名 |
| `--data_dir` | | `/shares/weather/Split_Data` | 数据根目录 |
| `--save` | | `False` | 测试时保存预测输出 |
| `--save_config` | `-sc` | `False` | 保存配置文件副本 |

### 动态配置键

以下参数由数据集/模型配置注入到 `configs` 对象中：

| 参数 | 来源 | 说明 |
|------|------|------|
| `total_seq` | 数据集 | `[输入帧数, 输出帧数]`，如 `[12, 12]` |
| `img_size` | 数据集 | 空间尺寸 `(H, W)` |
| `in_category` | 数据集 | 输入变量列表，如 `["tp", "t2m"]` |
| `out_category` | 数据集 | 输出变量列表 |
| `label_idx` | 数据集 | 输出通道在 `in_category` 中的起始/结束索引 |
| `test_seq` | 数据集 | 测试时输出的帧数 |
| `threshold` | 数据集 | 分类指标（CSI/POD/FAR/HSS）的阈值列表 |
| `metrics` | 数据集 | 评价指标列表 |
| `std_method` | 数据集 | 标准化方法（`z_score`/`z_score_sd`/`None`） |
| `std_params` | 数据集 | 运行时计算的标准化参数（mean/std） |

## 许可证

本项目仅供研究使用。
