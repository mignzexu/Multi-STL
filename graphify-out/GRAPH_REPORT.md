# Graph Report - .  (2026-07-16)

## Corpus Check
- 171 files · ~193,472 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1793 nodes · 3227 edges · 90 communities (56 shown, 34 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 276 edges (avg confidence: 0.58)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- MS-RadarFormer Architecture
- Model Debug Config Builders
- MIM Configuration
- Model System Base Class
- WeatherBench Config
- E3DLSTM Configuration
- STPANet Model Core
- PredRNN++ Configuration
- SBFlow Configuration
- SimVP IncepU Config
- SwinLSTM Configuration
- SDweather Data Loading
- MAU Configuration
- PredRNN Configuration
- PredRNNv2 Configuration
- PreDiff Configuration
- ConvLSTM Model Core
- TAU Model Core
- Drift Configuration
- GVBF Configuration
- SimVP gSTA Attention
- WeatherBench Data Loading
- GVBF Flow Analysis
- Plot Metrics Types
- E3DLSTM Model Core
- PhyDNet Model Core
- Report PPT Builders
- Metrics Utils v1
- SDweather Data Sequence
- Metrics Utils v2
- PhyDNet Components
- SDweather Dataloader
- PhyDNet Physics
- SDweather Config Loader
- Utils Log Rationale
- MMVP Model Core
- Editable PPT Builder
- WeatherBench Dataloader WB
- Instrument Model Registry
- PoolFormer Model Core
- PoolFormer Components
- PoolFormer Attention
- SimVP gSTA Model Core
- MMVP Components
- MMVP Model Wrapper
- PhyDNet Model Wrapper
- PhyDNet Network
- SimVP gSTA Network
- STPANet Tests
- MMVP Network
- PoolFormer Layers
- TAU Model Wrapper
- Trainer Engine
- Config Logger
- Standardizer Method
- SDweather Dataloader Copy
- Instrument Model Loader
- MMVP Blocks
- PoolFormer Config
- SimVP gSTA Components
- Tester Engine
- Z-Score Standardizer
- Z-Score SD Standardizer
- MMVP Configuration
- MMVP Model Blocks
- PhyDNet Configuration
- PoolFormer Model Wrapper
- SimVP IncepU Model
- TAU Loss
- MMVP Loss
- MMVP Network Blocks
- PhyDNet Blocks
- PoolFormer Loss
- PoolFormer Model
- SimVP gSTA Config
- SwinLSTM Loss
- TAU Config
- Utils Init Exports
- Standardizer Components
- MMVP Convs
- SDweather DataSeq Copy
- MMVP Sub-components
- PhyDNet Loss
- PreDiff Loss
- ConvLSTM Configuration
- Plot Metrics Tool
- Biflow Experiment Slides
- Flow Experiment Slides
- Condiff Experiment Slides

## God Nodes (most connected - your core abstractions)
1. `System` - 71 edges
2. `distribute_model_layers()` - 36 edges
3. `Load_Standardizer` - 25 edges
4. `SD_Dataloader` - 23 edges
5. `Mem_Dataloader` - 23 edges
6. `PreDiffModel` - 23 edges
7. `Recorder` - 22 edges
8. `WB_Dataloader` - 21 edges
9. `Model` - 19 edges
10. `Model` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Biflow Training Loss Curve` --conceptually_related_to--> `GVBF Flow Mode Analysis`  [INFERRED]
  reports/assets/biflow_loss_chart.png → models/GVBF/Flow分析.md
- `Condiff Training Loss Curve` --conceptually_related_to--> `GVBF Flow Mode Analysis`  [INFERRED]
  reports/assets/condiff_loss_chart.png → models/GVBF/Flow分析.md
- `Model_Instrument` --uses--> `Model`  [INFERRED]
  Instrument/models.py → models/ConvLSTM/Model.py
- `SD_Dataloader` --uses--> `Load_Standardizer`  [INFERRED]
  dataset/SDweather/dataloader_SD copy.py → Instrument/standardizer.py
- `Mem_Loader` --uses--> `Load_Standardizer`  [INFERRED]
  dataset/SDweather/dataloader_SD.py → Instrument/standardizer.py

## Import Cycles
- None detected.

## Communities (90 total, 34 thin omitted)

### Community 0 - "MS-RadarFormer Architecture"
Cohesion: 0.06
Nodes (29): Decoder, Encoder, clones(), compute_mask(), ConvOut, get_window_size(), Mlp, MultiScalePatchEmbed3D (+21 more)

### Community 1 - "Model Debug Config Builders"
Cohesion: 0.06
Nodes (17): _build_debug_configs(), _build_debug_configs(), _build_debug_configs(), STPANet_Config, loss_fn, MSE_Loss, _build_debug_configs(), Model (+9 more)

### Community 2 - "MIM Configuration"
Cohesion: 0.06
Nodes (14): MIM_Config, loss_fn, MSE_Loss, Main, MIM_Model, MIMBlock, MIMN, r"""MIM Model      Implementation of `Memory In Memory: A Predictive Neural Netw (+6 more)

### Community 3 - "Model System Base Class"
Cohesion: 0.06
Nodes (13): 所有 rank 对 tensor 求和。          注意：             这是 collective 操作。             所有 r, 跨所有 rank 汇总 Recorder 的 sum/count 累加器。          所有 rank 必须同步调用此方法。         调用后 re, 本 rank 内部累计 loss sum / count。          loss 一般是当前 batch 的平均 loss。         因此这里使用, 使用 all_reduce 计算所有 rank 的全局平均值。          返回:             avg tensor，所有 rank 上值相同, 多卡时，每个 rank 取自己的 data_idx 连续分片。         DistributedSampler 把数据集等分成 N 份，按 rank 顺序, System, MS_RadarFormer_Config, WindWeightedLoss (+5 more)

### Community 4 - "WeatherBench Config"
Cohesion: 0.06
Nodes (15): WeatherBench_config, Dataset, 检查当前 mode 的缓存是否完整存在。          不能只检查 /dev/shm/M/{data_config} 文件夹是否存在，         因为, 如果 /dev/shm/M/{data_config}/{mode}.npy 和 {mode}_idx.json 已经存在，         直接使用。, 懒加载 mmap。          注意：             np.load(..., mmap_mode='r') 不会把完整 .npy 读入进程内存, DataLoader 多 worker 时，避免把 mmap 句柄 pickle 到 worker。         worker 进程里会重新 _open_d, 标准化参数初始化。          train:             可以计算参数。          valid/test:             应, WB_Dataloader (+7 more)

### Community 5 - "E3DLSTM Configuration"
Cohesion: 0.06
Nodes (14): E3DLSTM_Config, E3DLSTM_Model, Eidetic3DLSTMCell, Main, r"""E3D-LSTM Model      Implementation of `EEidetic 3D LSTM: A Model for Video P, SpatioTemporalLSTMCell, tf_Conv3d, E3DLSTM_Loss (+6 more)

### Community 6 - "STPANet Model Core"
Cohesion: 0.06
Nodes (21): BasicConv2d, Channel_Pooling, ConvSC, Decoder, Encoder, GroupNorm, Main, MetaBlock (+13 more)

### Community 7 - "PredRNN++ Configuration"
Cohesion: 0.06
Nodes (13): PredRNNpp_Config, loss_fn, MSE_Loss, _as_config_dict(), _build_debug_configs(), Model, ScheduledSamplingWrapper, CausalLSTMCell (+5 more)

### Community 8 - "SBFlow Configuration"
Cohesion: 0.06
Nodes (19): SBFlow_Config, __getattr__(), Model, _pad_size(), _pad_tensor(), _unpad_tensor(), _flatten_batch(), lerp() (+11 more)

### Community 9 - "SimVP IncepU Config"
Cohesion: 0.05
Nodes (15): IncepU_Config, BasicConv2d, ConvSC, Decoder, Encoder, gInception_ST, GroupConv2d, IncepU_Model (+7 more)

### Community 10 - "SwinLSTM Configuration"
Cohesion: 0.06
Nodes (17): SwinLSTM_Config, build_debug_configs(), Model, DownSample, Main, PatchExpanding, PatchInflated, r""" Tensor to Patch Inflating      Args:         in_chans (int): Number of inpu (+9 more)

### Community 11 - "SDweather Data Loading"
Cohesion: 0.07
Nodes (7): Mem_Loader, Dataset, 从 data_list 中收集有效数据段。          data_list 格式: [[day, real_range, modalities], ..., 将 SDweather 拆分数据写入 /dev/shm 中的 .npy memmap 缓存。      输出：         /dev/shm/SD/{dat, SDweather 数据集，使用 mmap 软加载方式。, SD_Dataloader, SD_Painter

### Community 12 - "MAU Configuration"
Cohesion: 0.07
Nodes (14): MAU_Config, loss_fn, MSE_Loss, Main, MAU_Model, MAUCell, r"""MAU Model      Implementation of `MAU: A Motion-Aware Unit for Video Predict, _as_config_dict() (+6 more)

### Community 13 - "PredRNN Configuration"
Cohesion: 0.07
Nodes (12): PredRNN_Config, loss_fn, MSE_Loss, _as_config_dict(), _build_debug_configs(), Model, ScheduledSamplingWrapper, Main (+4 more)

### Community 14 - "PredRNNv2 Configuration"
Cohesion: 0.07
Nodes (12): PredRNNv2_Config, loss_fn, PredRNNv2_Loss, _as_config_dict(), _build_debug_configs(), Model, ScheduledSamplingWrapper, Main (+4 more)

### Community 15 - "PreDiff Configuration"
Cohesion: 0.08
Nodes (6): PreDiff_Config, build_debug_configs(), Model, build_optimizer(), Main, PreDiffModel

### Community 16 - "ConvLSTM Model Core"
Cohesion: 0.08
Nodes (12): ConvLSTM_Model, ConvLSTMCell, Main, loss_fn, MSE_Loss, _as_config_dict(), Model, ScheduledSamplingWrapper (+4 more)

### Community 17 - "TAU Model Core"
Cohesion: 0.07
Nodes (15): BasicConv2d, ConvSC, Decoder, DWConv, Encoder, MetaBlock, MidMetaNet, MixMlp (+7 more)

### Community 18 - "Drift Configuration"
Cohesion: 0.09
Nodes (14): Drift_Config, DriftUNetGenerator, get_model(), _pad_size(), _pad_tensor(), _resolve_drift_channels(), _unpad_tensor(), _cdist() (+6 more)

### Community 19 - "GVBF Configuration"
Cohesion: 0.11
Nodes (14): GVBF_Config, get_model(), _make_unet(), _make_unet_small(), MyUNet2DModel, _parse_config(), UNet2DModel, TwoModel (+6 more)

### Community 20 - "SimVP gSTA Attention"
Cohesion: 0.08
Nodes (12): AttentionModule, BasicConv2d, ConvSC, Decoder, DWConv, Encoder, MidMetaNet, MixMlp (+4 more)

### Community 21 - "WeatherBench Data Loading"
Cohesion: 0.11
Nodes (11): Mem_Dataloader, 将 WeatherBench 数据逐年写入 /dev/shm 中的 .npy memmap 缓存。      输出：         /dev/shm/M/{d, 当前每个变量默认都是 1 个 channel。         如果后续某个变量本身有多层 channel，可以在这里扩展。, 加载单年数据，返回 numpy.float32:             [T, C, H, W]          这个函数只会在内存中保留当前 year 的, 加载单个变量。          动态变量一般返回：             [T, H, W]          常量变量返回：             [H, 由 u10/v10 计算 w10，并做 gust 增强。, 统一为 [T, C, H, W]。          [T, H, W] -> [T, 1, H, W]         [H, W]    -> [1, 1,, 沿 channel 维拼接。          year_data: [T1, C1, H, W]         dataset:   [T2, C2, H, (+3 more)

### Community 22 - "GVBF Flow Analysis"
Cohesion: 0.08
Nodes (29): Autoregressive Multi-Step Prediction, Removal of clamp(-1,1) in Prediction, GVBF Flow Mode Analysis, Flow Matching Mathematics, ODE Solver (Heun + Adaptive Step), U-Net Architecture (1_1_64), Validation-Test Inference Gap, Visual Realism vs Pixel Metrics Tradeoff (+21 more)

### Community 23 - "Plot Metrics Types"
Cohesion: 0.22
Nodes (26): EpochItems, Metrics, PlotSeries, Series, create_args(), group_list_series(), group_scalar_series(), is_number() (+18 more)

### Community 24 - "E3DLSTM Model Core"
Cohesion: 0.15
Nodes (9): PredRNN_Model, r"""PredRNN      Implementation of `PredRNN: A Recurrent Neural Network for Spat, reshape_patch_back(), distribute_model_layers(), System.py 使用说明 ==================  这个版本用于 Lightning 单卡、DDP、FSDP 多卡训练，核心目标是：  1., resolve_manual_parallel_devices(), reshape_patch_back(), reshape_patch_back() (+1 more)

### Community 25 - "PhyDNet Model Core"
Cohesion: 0.19
Nodes (9): _apply_axis_left_dot(), _apply_axis_right_dot(), M2K, _MK, convert moment matrix to convolution kernel     Arguments:         shape (tuple, m (Tensor): torch.size=[...,*self.shape], k (Tensor): torch.size=[...,*self.shape], tensordot in PyTorch, see numpy.tensordot? (+1 more)

### Community 26 - "Report PPT Builders"
Cohesion: 0.24
Nodes (18): cover_image(), draw_header(), draw_loss_chart(), draw_metric_card(), draw_results_slide(), draw_setup_slide(), draw_text(), fit_image() (+10 more)

### Community 27 - "Metrics Utils v1"
Cohesion: 0.15
Nodes (4): object, 加载训练或测试过程中的指标, 用于继续训练, Categorical Verification Metrics, Recorder

### Community 28 - "SDweather Data Sequence"
Cohesion: 0.19
Nodes (6): DataSeq, 在每个参与交集的 channel 中各选一个文件，使最终交集长度最大。          这一步是真正替代原先“以 self.channels[0] 为参考”的, 将最终交集 [inter_start, inter_end] 转为该文件内部的裁剪范围。          返回格式兼容 Mem_Loader._tailor_, 处理单日的一个请求时间窗口。          req_start / req_end 是这一天内希望使用的时间范围，例如：             first, 返回某个 channel 在某一天的所有文件信息。          文件名沿用原逻辑：             CW20200101_0_143.npy, 根据给定日期范围生成数据段索引。      方案 A:     1. 常规动态模态，也就是不在 optional_channels 中的 channels，必须

### Community 30 - "PhyDNet Components"
Cohesion: 0.15
Nodes (6): dcgan_upconv, decoder_D, decoder_specific, encoder_E, encoder_specific, PhyD_EncoderRNN

### Community 31 - "SDweather Dataloader"
Cohesion: 0.21
Nodes (3): Dataset, 按 new_order 指定的通道名顺序进行选择和重排。, SD_Dataloader

### Community 32 - "PhyDNet Physics"
Cohesion: 0.17
Nodes (4): dcgan_conv, PhyCell_Cell, PhyD_ConvLSTM_Cell, input_shape: (int, int)             Height and width of input tensor as (height,

### Community 33 - "SDweather Config Loader"
Cohesion: 0.21
Nodes (6): SDweather_config, Dataset_Instrument, is_dist_child_process(), Lightning DDP 子进程通常会带 LOCAL_RANK / RANK 等环境变量。     父进程没有这些变量。, test(), train()

### Community 34 - "Utils Log Rationale"
Cohesion: 0.24
Nodes (4): Run one memory-check and allocation cycle., Unified external entry.          The caller only needs to call this method., Return current free GPU memory and total GPU memory in bytes., train_config

### Community 35 - "MMVP Model Core"
Cohesion: 0.17
Nodes (5): filter_block, ImageEnhancer, Residual in Residual Dense Block, RRDB, RRDBDecoder

### Community 36 - "Editable PPT Builder"
Cohesion: 0.40
Nodes (12): add_rect(), add_rrect(), build_future(), build_results(), build_setup(), gen_chart(), gif_to_avi(), hex_rgb() (+4 more)

### Community 37 - "WeatherBench Dataloader WB"
Cohesion: 0.26
Nodes (3): Dataset, 调整气象数据的地理视角 (PyTorch 版本)。, WB_Dataloader

### Community 38 - "Instrument Model Registry"
Cohesion: 0.21
Nodes (3): TAU_Config, build_debug_configs(), config_update()

### Community 39 - "PoolFormer Model Core"
Cohesion: 0.20
Nodes (4): BasicConv2d, Channel_Pooling, GroupNorm, Group Normalization with 1 group.     Input: tensor in shape [B, C, H, W]

### Community 40 - "PoolFormer Components"
Cohesion: 0.17
Nodes (5): MetaBlock, MidMetaNet, PoolFormerSubBlock, The hidden Translator of MetaFormer for SimVP, A block of PoolFormer.

### Community 41 - "PoolFormer Attention"
Cohesion: 0.17
Nodes (6): Mlp, PoolFormerBlock, Pooling, Implementation of one PoolFormer block.     --dim: embedding dim     --pool_size, Implementation of pooling for PoolFormer     --pool_size: pooling size, Implementation of MLP with 1*1 convolutions.     Input: tensor with shape [B, C,

### Community 42 - "SimVP gSTA Model Core"
Cohesion: 0.18
Nodes (3): gSTA_Model, Model, OptimizerLRScheduler

### Community 43 - "MMVP Components"
Cohesion: 0.27
Nodes (8): build_similarity_matrix(), cum_multiply(), cut_off_process(), MidMotionMatrix, :param emb_feats: a sequence of embeddings for every frame (N,T,c,h,w)     :retu, :param value_seq: (B,S,***), B - batch num; S- sequence len     :return: output, sim_matrix_interpolate(), sim_matrix_postprocess()

### Community 46 - "PhyDNet Network"
Cohesion: 0.22
Nodes (4): K2M, PhyCell, PhyD_ConvLSTM, convert convolution kernel to moment matrix     Arguments:         shape (tuple

### Community 48 - "STPANet Tests"
Cohesion: 0.25
Nodes (3): build_configs(), ConfigNode, STPANetSystemMigrationTests

### Community 49 - "MMVP Network"
Cohesion: 0.22
Nodes (4): MatrixPredictor3DConv, PredictModel, in_tensor: batch,c,h'w',H'W'         tempolate_tensor: batch,c,hw,HW         out, SimpleMatrixPredictor3DConv_direct

### Community 50 - "PoolFormer Layers"
Cohesion: 0.24
Nodes (4): ConvSC, Decoder, Encoder, sampling_generator()

### Community 52 - "Trainer Engine"
Cohesion: 0.27
Nodes (4): object, mmap + /dev/shm 数据读取建议：             num_workers 默认先设 0。             稳定后再尝试 1、2。, 自动适配：             CPU / 单卡：auto             多卡：默认 ddp             用户显式指定：使用用户指定, Trainer

### Community 56 - "Instrument Model Loader"
Cohesion: 0.31
Nodes (3): Model_Instrument, MigrationContractTests, Namespace

### Community 59 - "SimVP gSTA Components"
Cohesion: 0.22
Nodes (4): GASubBlock, MetaBlock, The hidden Translator of MetaFormer for SimVP, A GABlock (gSTA) for SimVP

### Community 61 - "Z-Score Standardizer"
Cohesion: 0.25
Nodes (3): 在 datasets 和 test 保存结果时被调用。, 使用 NumPy/memmap 分块计算 mean/std。          输入:             data: numpy.ndarray 或 nu, Z_Score

### Community 62 - "Z-Score SD Standardizer"
Cohesion: 0.25
Nodes (3): 将指定类别通道改成不归一化:         mean -> 0         std  -> 1, 在datasets和test保存结果时被调用, Z_Score_SD

### Community 64 - "MMVP Model Blocks"
Cohesion: 0.25
Nodes (4): ConvLayer, Upscaling then double conv, (convolution => [BN] => ReLU) * 2, Up

### Community 70 - "MMVP Network Blocks"
Cohesion: 0.38
Nodes (3): Compose, :param feats: [B,T,c,h,w]         :param sim_matrix: [B,T,h*w,h*w]         :retu, :param emb_feat_list: (scale_num, (B,T,c,h,w))         :param sim_matrix:  (B,T-

### Community 71 - "PhyDNet Blocks"
Cohesion: 0.29
Nodes (3): Main, PhyDNet_Model, r"""PhyDNet Model      Implementation of `Disentangling Physical Dynamics from U

## Knowledge Gaps
- **17 isolated node(s):** `SDweather Dataset`, `WeatherBench Dataset`, `PyTorch Lightning Training Engine`, `U-Net Architecture (1_1_64)`, `Optimal Transport (POT) Dependency` (+12 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **34 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `System` connect `Model System Base Class` to `Model Debug Config Builders`, `MIM Configuration`, `E3DLSTM Configuration`, `PredRNN++ Configuration`, `SBFlow Configuration`, `SimVP IncepU Config`, `SwinLSTM Configuration`, `MAU Configuration`, `PredRNN Configuration`, `PredRNNv2 Configuration`, `PreDiff Configuration`, `ConvLSTM Model Core`, `Drift Configuration`, `GVBF Configuration`, `E3DLSTM Model Core`, `Instrument Model Registry`, `SimVP gSTA Model Core`, `MMVP Model Wrapper`, `PhyDNet Model Wrapper`, `SimVP gSTA Network`, `TAU Model Wrapper`, `SDweather Dataloader Copy`, `PoolFormer Config`, `MMVP Configuration`, `PhyDNet Configuration`, `PoolFormer Model Wrapper`?**
  _High betweenness centrality (0.219) - this node is a cross-community bridge._
- **Why does `Load_Standardizer` connect `SDweather Dataloader Copy` to `Model System Base Class`, `WeatherBench Config`, `WeatherBench Dataloader WB`, `SimVP gSTA Model Core`, `SDweather Data Loading`, `SimVP gSTA Network`, `WeatherBench Data Loading`, `E3DLSTM Model Core`, `SDweather Dataloader`?**
  _High betweenness centrality (0.185) - this node is a cross-community bridge._
- **Why does `distribute_model_layers()` connect `E3DLSTM Model Core` to `MS-RadarFormer Architecture`, `Model Debug Config Builders`, `MIM Configuration`, `Model System Base Class`, `STPANet Model Core`, `PhyDNet Blocks`, `PoolFormer Model`, `SimVP gSTA Model Core`, `MMVP Components`, `MAU Configuration`, `SimVP IncepU Config`, `SwinLSTM Configuration`, `TAU Config`, `ConvLSTM Model Core`, `MMVP Network`, `TAU Model Core`, `PhyDNet Model Core`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `System` (e.g. with `Model` and `Model`) actually correct?**
  _`System` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `Path` (e.g. with `_build_debug_configs()` and `_build_debug_configs()`) actually correct?**
  _`Path` has 22 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `Load_Standardizer` (e.g. with `SD_Dataloader` and `Mem_Loader`) actually correct?**
  _`Load_Standardizer` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `SDweather Dataset`, `WeatherBench Dataset`, `PyTorch Lightning Training Engine` to the rest of the system?**
  _17 weakly-connected nodes found - possible documentation gaps or missing edges._