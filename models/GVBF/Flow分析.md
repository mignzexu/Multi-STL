# GVBF Flow 模型源码与数学原理超详细分析

本文专门分析 `models/GVBF` 中的 `flow` 模式。目标不是只讲“这个模型是什么”，而是尽量把代码里每一个动作为什么存在、数学上对应什么、直觉上可以怎样理解都讲清楚。

分析对象主要来自以下文件：

- `models/GVBF/Model.py`
- `models/GVBF/gvbf_network.py`
- `models/GVBF/config.py`
- `models/Model_system.py`
- `work_dirs/flow_w3s/result.json`

当前 `flow_w3s` 的测试结果为：

```json
{
  "w10": {
    "mae": 1.4597407510882683,
    "mse": 4.68258168291579,
    "rmse": 2.1639273746860797,
    "csi": [0.865841, 0.763662, 0.67591],
    "pod": [0.927139, 0.874812, 0.833801],
    "far": [0.070942, 0.142644, 0.218848],
    "hss": [0.794592, 0.822005, 0.794351]
  }
}
```

这里的 `w10` 是 10 米风速或近地面风速相关变量，阈值来自工程配置，一般为：

```text
8.0, 13.9, 20.8
```

从实际表现看，`flow` 模型虽然逐像素误差不一定总是最优，但它的可视化结果非常真实。这与它的数学结构有关：它不是直接“一口气画出下一帧”，而是学习一个连续的运动场，然后沿着这个场把当前帧推向下一帧。

---

## 1. 一句话说明 Flow 模式在做什么

`GVBF flow` 模式学习的是：

```text
从当前天气场 x0 到下一帧天气场 x1 的变化方向。
```

更数学一点说，它学习一个向量场：

```text
v_theta(x_t, t)
```

这个向量场告诉我们：

```text
如果当前处在 x_t 这个中间天气状态，并且时间位置是 t，
那么应该往哪个方向走，才能从 x0 走到 x1。
```

训练时，模型会随机选一个中间状态：

```text
x_t = x0 + t(x1 - x0)
```

然后要求 U-Net 输出：

```text
v_theta(x_t, t) ≈ x1 - x0
```

推理时，没有真实 `x1`。模型从 `x0` 出发，反复询问 U-Net：

```text
现在应该往哪里走？
```

然后用 ODE 求解器把这个方向场从 `t=0` 积分到 `t=1`，得到下一帧预测。

通俗类比：

```text
x0 是你现在的位置。
x1 是训练时已知的目标位置。
x_t 是从当前位置走向目标路上的某个点。
模型学习的 v_theta 是“导航箭头”。
训练时告诉模型：不管你在路上的哪个点，都应该知道目标方向。
测试时没有目标位置，只能靠学到的导航箭头一步步走。
```

---

## 2. Flow 模式的配置从哪里来

`models/GVBF/config.py` 中定义了 flow 模式：

```python
if name == "flow":
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "gradient_clip_val": 1.0,
        "gvbf_weights_path": None,
        "gvbf_max_size": 0,
        "gvbf_mode": "flow",
        "gvbf_model_config": "1_1_64",
        "gvbf_solver_atol": 1e-2,
        "gvbf_solver_rtol": 1e-2,
        "gvbf_solver_dt0": None,
    }
```

逐项解释：

### `gvbf_mode = "flow"`

表示 `Model.py` 中会走 `_flow_loss()` 和 `_predict_next_frame()` 的 flow 分支。

### `gvbf_model_config = "1_1_64"`

这个字符串非常关键。它会被 `gvbf_network.py` 里的 `_parse_config()` 解析：

```python
def _parse_config(config_str):
    class_cond = "cond" in config_str
    digits = [int(p) for p in config_str.split("_") if p.isdigit()]
    in_channels, out_channels, c = digits
    return in_channels, out_channels, c, class_cond
```

对于 `"1_1_64"`：

```text
in_channels  = 1
out_channels = 1
c            = 64
class_cond   = False
```

也就是说，flow 模型的 U-Net 是：

```text
输入 1 通道
输出 1 通道
基础通道数 64
没有 class condition
```

为什么输出也是 1 通道？

因为 flow 只预测一个量：

```text
v = x1 - x0
```

对于单变量 `w10`，输入是 1 个气象场，输出也是 1 个气象场大小的速度/变化量。

### `gvbf_solver_atol = 1e-2` 与 `gvbf_solver_rtol = 1e-2`

这两个是 ODE 自适应求解器的误差容忍度：

- `atol` 是绝对误差容忍度。
- `rtol` 是相对误差容忍度。

通俗理解：

```text
ODE 求解器每走一步都会估计自己走得准不准。
如果误差太大，就把步子迈小。
如果误差足够小，就可以迈大一点。
```

`1e-2` 是比较宽松的精度。好处是速度快，坏处是积分轨迹可能不够精细。

### `gvbf_solver_dt0 = None`

表示 flow 默认不用固定步长，而是用自适应步长控制器：

```python
controller = to.IntegralController(...)
```

与 biflow 不同，flow 没有固定 `dt0=0.1`。

通俗理解：

```text
flow 让求解器自己决定每一步走多远。
复杂区域走小步，简单区域走大步。
```

---

## 3. 模型初始化阶段：从配置到 U-Net

`models/GVBF/Model.py` 中：

```python
class Model(System):
    def __init__(self, configs):
        self._mode = getattr(configs, "gvbf_mode", "biflow")
        self._noise_level = float(getattr(configs, "gvbf_noise_level", 0.1))
        self._model_cfg_str = getattr(
            configs, "gvbf_model_config", _MODE_DEFAULTS.get(self._mode, "1_2_64_cond")
        )
        self._weights_path = getattr(configs, "gvbf_weights_path", None)
        self._max_size = int(getattr(configs, "gvbf_max_size", 0))
        self._solver_atol = float(getattr(configs, "gvbf_solver_atol", 1e-2))
        self._solver_rtol = float(getattr(configs, "gvbf_solver_rtol", 1e-2))
        self._solver_dt0 = getattr(configs, "gvbf_solver_dt0", None)
        if self._solver_dt0 is not None:
            self._solver_dt0 = float(self._solver_dt0)
        super().__init__(configs)
        self.criterion = torch.nn.MSELoss()
```

这里要注意一个顺序：

```text
先保存 GVBF 自己的参数
再调用 super().__init__(configs)
```

为什么重要？

因为 `System.__init__()` 里会调用：

```python
self.model = self.get_model()
```

而 `get_model()` 需要用到前面保存的：

```text
self._model_cfg_str
self._weights_path
```

如果顺序反过来，模型还没拿到配置就被创建，会出问题。

---

## 4. `get_model()` 如何创建模型

`Model.py` 中：

```python
def get_model(self):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        config=self._model_cfg_str, weights_path=self._weights_path
    )
    return _gvbf_get_model(cfg)
```

这里做了一个轻量包装：

```text
self._model_cfg_str = "1_1_64"
self._weights_path = None
```

被包装成：

```python
cfg.config = "1_1_64"
cfg.weights_path = None
```

然后交给 `gvbf_network.py` 的 `get_model()`。

`gvbf_network.py` 中：

```python
def get_model(model_config):
    config_str = model_config.config
    in_c, out_c, c, cond = _parse_config(config_str)

    _get_model = _make_unet_small if "small" in config_str else _make_unet

    if "two_model" in config_str:
        m1 = _get_model(in_c, out_c, c, cond)
        m2 = _get_model(in_c, out_c, c, cond)
        model = TwoModel(m1, m2)
    else:
        model = _get_model(in_c, out_c, c, cond)

    if model_config.weights_path:
        print(f"Loading model from {model_config.weights_path}")
        load_model(model, Path(model_config.weights_path))

    return model
```

对 flow 的 `"1_1_64"` 来说：

```text
不包含 small
不包含 two_model
不包含 cond
```

所以走的是：

```python
model = _make_unet(1, 1, 64, False)
```

---

## 5. Flow 模式使用的 U-Net 结构

`_make_unet()` 中：

```python
def _make_unet(in_channels, out_channels, c, cond):
    block_out_channels = (c, c, 2 * c, 2 * c, 2 * c, 4 * c, 4 * c)
    down_block_types = (
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
    )
    return MyUNet2DModel(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        up_block_types=up_block_types,
        down_block_types=down_block_types,
        add_attention=True,
        class_embed_type="timestep" if cond else None,
    )
```

代入 flow 的参数：

```text
c = 64
in_channels = 1
out_channels = 1
cond = False
```

得到：

```text
block_out_channels = (64, 64, 128, 128, 128, 256, 256)

down_block_types =
(
  DownBlock2D,
  DownBlock2D,
  DownBlock2D,
  DownBlock2D,
  DownBlock2D,
  AttnDownBlock2D,
  DownBlock2D
)

up_block_types =
(
  UpBlock2D,
  AttnUpBlock2D,
  UpBlock2D,
  UpBlock2D,
  UpBlock2D,
  UpBlock2D,
  UpBlock2D
)

class_embed_type = None
```

也就是说，flow 的 U-Net：

- 有 7 个下采样 block。
- 有 7 个上采样 block。
- 在较深层有 attention。
- 只接收时间步 `t`，不接收额外类别条件 `alpha`。
- 输入和输出都是 1 通道。

### 为什么 U-Net 适合做这个任务

天气场是一个空间图像：

```text
H × W 的二维网格
每个格点有风速值
```

U-Net 的优势是：

```text
编码器负责理解大尺度结构
解码器负责恢复空间细节
跳跃连接帮助保留局部边界、涡旋、锋面、强风带等细节
```

通俗类比：

```text
编码器像“缩小地图看全局天气形势”。
解码器像“重新放大地图，把每个地方的细节补回来”。
attention 像“远距离看关联”，比如一个海区的风带和另一个区域的环流结构可能有关。
```

---

## 6. `MyUNet2DModel` 为什么要重写 forward

`gvbf_network.py` 中：

```python
class MyUNet2DModel(UNet2DModel):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).sample
```

`diffusers.UNet2DModel` 默认 forward 返回的不是一个裸 tensor，而是一个带字段的对象，其中真正的输出在 `.sample` 里。

原始形式大概是：

```python
out = unet(x, t)
tensor = out.sample
```

这里重写后，就可以直接写：

```python
v_p = self.model(xt, t)
```

而不是：

```python
v_p = self.model(xt, t).sample
```

这个动作很小，但让后面的代码更简洁。

注意：

```text
这里的 sample 不是随机采样。
它只是 diffusers 返回对象里的字段名。
```

---

## 7. 训练数据如何进入 flow

`training_step()` 中：

```python
def training_step(self, batch, batch_idx):
    batch_x, batch_y = batch
    x0 = batch_x[:, -1]
    x1 = batch_y[:, 0]

    if self._mode == "flow":
        loss, _ = self._flow_loss(x0, x1)
```

假设数据形状为：

```text
batch_x: [B, T_in, C, H, W]
batch_y: [B, T_out, C, H, W]
```

在当前风速实验里通常是：

```text
B = batch size
T_in = 12
T_out = 12
C = 1
H = 128
W = 256
```

代码取：

```python
x0 = batch_x[:, -1]
```

含义是：

```text
输入序列最后一帧
```

形状变成：

```text
x0: [B, C, H, W]
```

然后：

```python
x1 = batch_y[:, 0]
```

含义是：

```text
标签序列第一帧，也就是下一帧
```

形状也是：

```text
x1: [B, C, H, W]
```

这说明 flow 的训练目标是一个**单步预测目标**：

```text
从最后一个输入帧 x0 预测第一个未来帧 x1。
```

虽然测试时会预测 12 帧，但训练时每次只学习一步。

---

## 8. `_flow_loss()` 是整个 flow 训练的核心

源码：

```python
def _flow_loss(self, x0, x1):
    B, device = x0.shape[0], x0.device
    x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
    x1_p, _ = _pad_tensor(x1, max_size=self._max_size)
    t = torch.rand(B, device=device)
    xt = _lerp(x0_p, x1_p, t.view(-1, 1, 1, 1))
    v_p = self.model(xt, t)
    v = _unpad_tensor(v_p, orig)
    loss = torch.mean((v_p - (x1_p - x0_p)) ** 2)
    return loss, v
```

逐行分析如下。

---

## 9. `B, device = x0.shape[0], x0.device`

```python
B, device = x0.shape[0], x0.device
```

`B` 是 batch size。

`device` 是数据所在设备，例如：

```text
cuda:0
cpu
```

为什么要取 device？

后面要创建随机数：

```python
t = torch.rand(B, device=device)
```

如果 `x0` 在 GPU，但 `t` 在 CPU，那么参与计算时会报错：

```text
Expected all tensors to be on the same device
```

所以这一行的作用是保证随机时间 `t` 和输入张量在同一设备上。

---

## 10. `_pad_tensor()` 为什么存在

```python
x0_p, orig = _pad_tensor(x0, max_size=self._max_size)
x1_p, _ = _pad_tensor(x1, max_size=self._max_size)
```

U-Net 里有多层下采样和上采样。每次下采样通常会把空间尺寸除以 2。经过多次下采样后，如果原始 H/W 不是合适的倍数，就可能出现尺寸对不上。

例如：

```text
原始宽度 W = 250
下采样一次 -> 125
下采样两次 -> 62.5
```

神经网络不能有半个像素，最终上采样回来时就可能出现拼接尺寸不一致。

所以代码用：

```python
def _pad_size(size, multiple=64):
    return ((size + multiple - 1) // multiple) * multiple
```

把 H/W 补齐到 64 的倍数。

### `_pad_size()` 的数学含义

```python
((size + multiple - 1) // multiple) * multiple
```

这是一个“向上取整到 multiple 的倍数”的公式。

例如：

```text
size = 128, multiple = 64
结果 = 128

size = 130, multiple = 64
结果 = 192

size = 256, multiple = 64
结果 = 256
```

通俗理解：

```text
如果尺寸已经合适，就不动。
如果尺寸不合适，就补到下一个能被 64 整除的尺寸。
```

### `_pad_tensor()` 做了什么

```python
def _pad_tensor(x, multiple=64, max_size=0):
    _, _, H, W = x.shape
    H_p = _pad_size(H, multiple)
    W_p = _pad_size(W, multiple)
```

取出原始 H/W，并算出补齐后的 H/W。

如果设置了 `max_size`，并且补齐后超过了最大尺寸：

```python
if max_size > 0 and max(H_p, W_p) > max_size:
    scale = max_size / max(H_p, W_p)
    H_r = _pad_size(round(H * scale), multiple)
    W_r = _pad_size(round(W * scale), multiple)
    return F.interpolate(x, (H_r, W_r), mode="bilinear"), (H, W)
```

这会先缩放图像，避免显存太大。

对于当前 flow 配置：

```text
gvbf_max_size = 0
```

所以这部分一般不会触发。

如果原始尺寸已经是 64 的倍数：

```python
if H == H_p and W == W_p:
    return x, None
```

当前 WeatherBench 配置：

```text
H = 128
W = 256
```

它们都是 64 的倍数，因此：

```text
orig = None
x0_p = x0
x1_p = x1
```

如果尺寸不合适：

```python
return F.pad(x, (0, W_p - W, 0, H_p - H), mode="replicate"), (H, W)
```

这里使用 `replicate` padding。

通俗理解：

```text
不是补 0，而是复制边缘值。
```

例如一行数据：

```text
[3, 4, 5]
```

向右 replicate 补两个值会变成：

```text
[3, 4, 5, 5, 5]
```

为什么不用 0？

因为气象场边界突然补 0 会制造人为断崖，模型可能误以为那里有强烈梯度。复制边缘更平滑。

---

## 11. 随机时间 `t` 是什么

```python
t = torch.rand(B, device=device)
```

这里为 batch 中每一个样本随机采一个时间：

```text
t_i ~ Uniform(0, 1)
```

形状是：

```text
t: [B]
```

例如 batch size 为 4：

```text
t = [0.13, 0.82, 0.47, 0.05]
```

每个样本的 t 不一样。

为什么要随机采？

因为模型要学习整个从 `x0` 到 `x1` 的路径，而不是只学习起点或终点。

通俗理解：

```text
如果你教一个人从家走到学校，
不能只教他“在家怎么走第一步”，
也不能只教他“快到学校时怎么走最后一步”。
你希望他在路上的任何位置，都知道该往哪里走。
```

随机采 `t` 就是在训练模型：

```text
无论你在 x0 到 x1 的哪一个中间点，都要知道正确方向。
```

---

## 12. `_lerp()` 的数学意义

源码：

```python
def _lerp(a, b, t):
    return a + t * (b - a)
```

`lerp` 是 linear interpolation，线性插值。

数学形式：

```text
lerp(a, b, t) = (1 - t)a + tb
```

代码写成：

```text
a + t(b - a)
```

二者等价：

```text
a + t(b-a)
= a + tb - ta
= (1-t)a + tb
```

在 flow 中：

```python
xt = _lerp(x0_p, x1_p, t.view(-1, 1, 1, 1))
```

即：

```text
x_t = x0 + t(x1 - x0)
```

### 为什么 `t.view(-1, 1, 1, 1)`

原始 `t` 是：

```text
[B]
```

而 `x0_p` 是：

```text
[B, C, H, W]
```

如果直接相乘，维度对不上。需要把 `t` 变成：

```text
[B, 1, 1, 1]
```

这样 PyTorch 会广播：

```text
每个样本一个 t
该样本内所有通道、所有像素共享这个 t
```

通俗理解：

```text
第 1 个样本取 30% 的路程位置。
第 2 个样本取 80% 的路程位置。
每张图内部所有像素都使用同一个时间进度。
```

---

## 13. `xt` 是什么：一张“中间天气图”

```python
xt = _lerp(x0_p, x1_p, t.view(-1, 1, 1, 1))
```

假设某个像素：

```text
x0 = 10
x1 = 18
t = 0.25
```

那么：

```text
xt = 10 + 0.25 * (18 - 10)
   = 12
```

如果：

```text
t = 0
xt = x0

t = 1
xt = x1

t = 0.5
xt = x0 和 x1 的中点
```

所以 `xt` 是一个训练构造出来的“中间天气状态”。

它不一定真实存在于数据集中，但它位于当前帧和下一帧之间。

---

## 14. Flow Matching 的核心监督信号

```python
v_p = self.model(xt, t)
loss = torch.mean((v_p - (x1_p - x0_p)) ** 2)
```

模型输入：

```text
xt, t
```

模型输出：

```text
v_p
```

监督目标：

```text
x1_p - x0_p
```

也就是说：

```text
模型预测的速度场 v_p，要接近真实帧差 x1 - x0。
```

数学写法：

```text
L = E_{x0,x1,t} || v_theta(x_t, t) - (x1 - x0) ||^2
```

这里的期望 `E` 表示：

```text
对训练集样本求平均
对随机 t 求平均
对所有像素求平均
```

### 为什么目标是 `x1 - x0`

因为选择的路径是直线：

```text
x_t = x0 + t(x1 - x0)
```

对 `t` 求导：

```text
dx_t / dt = x1 - x0
```

这就是直线路径上的真实速度。

所以模型学习的不是 `x1` 本身，而是路径导数：

```text
当前状态应该以什么速度变化。
```

通俗例子：

```text
当前位置是 10，目标是 18。
你设计了一条匀速路线。
从 t=0 到 t=1，总共要增加 8。
所以任何时刻的速度都是 +8。
模型要学的就是这个 +8。
```

对于图像而言，每个像素都有自己的速度：

```text
某些地方风速增加
某些地方风速减少
某些地方基本不变
```

因此 `v_p` 是一整张速度图。

---

## 15. 为什么这会产生真实感

普通直接预测模型通常学：

```text
f(x历史) -> x未来
```

它容易学到平均解。

例如某个区域未来可能有两种情况：

```text
情况 A：强风带偏北
情况 B：强风带偏南
```

MSE 模型可能预测成：

```text
强风带在中间变模糊
```

这样逐像素误差可能不大，但图会很假。

Flow 模型学的是：

```text
当前场如何被推进
```

它更像在做连续形变：

```text
把已有的风场纹理沿着速度方向推过去
```

所以它往往能保留：

- 涡旋结构
- 锋面边界
- 高低值相邻形成的强梯度
- 风带的连贯形态
- 更自然的空间纹理

这也是为什么你观察到：

```text
flow 指标略逊色，但可视化非常真。
```

它更像“物理演化”，而不是单纯“回归平均图”。

---

## 16. `v = _unpad_tensor(v_p, orig)` 是做什么

```python
v = _unpad_tensor(v_p, orig)
```

如果前面 `_pad_tensor()` 真的改变了尺寸，那么模型输出 `v_p` 也是 pad 后的尺寸。

例如：

```text
原图: 130 × 250
补齐: 192 × 256
模型输出: 192 × 256
```

此时需要变回原图尺寸。

`_unpad_tensor()` 的代码：

```python
def _unpad_tensor(x, original):
    if original is None:
        return x
    H, W = original
    if x.shape[2] == H and x.shape[3] == W:
        return x
    return F.interpolate(x, (H, W), mode="bilinear")
```

如果 `orig is None`，说明没有 pad，直接返回。

如果需要恢复尺寸，使用双线性插值。

当前 WeatherBench 的 `128 × 256` 本身就是 64 倍数，所以一般：

```text
orig = None
v = v_p
```

但代码写得更通用，可以支持其他尺寸。

---

## 17. 损失函数为什么是 MSE

```python
loss = torch.mean((v_p - (x1_p - x0_p)) ** 2)
```

这是标准均方误差：

```text
MSE = 平均 [(预测速度 - 真实速度)^2]
```

优点：

- 简单稳定。
- 对大错误惩罚更重。
- 容易优化。

缺点：

- 对稀有极值不特别友好。
- 会倾向于平均化。
- 对空间结构是否真实没有直接约束。

但在 flow 中，由于预测的是速度场而不是最终图像，它比直接预测 `x1` 更容易保留形态。

---

## 18. 训练时只学一步，测试时为什么能预测多步

训练中：

```python
x0 = batch_x[:, -1]
x1 = batch_y[:, 0]
loss, _ = self._flow_loss(x0, x1)
```

只学：

```text
最后输入帧 -> 第一个未来帧
```

测试中：

```python
def forward(self, batch_x, batch_y=None, **kwargs):
    x0 = batch_x[:, -1]
    n_pred = min(self.test_seq, self.aft_seq_length)

    frames = []
    current = x0
    for _ in range(n_pred):
        frame = self._predict_next_frame(current)
        frames.append(frame)
        current = frame
```

它会：

```text
第 1 步：x0 -> pred1
第 2 步：pred1 -> pred2
第 3 步：pred2 -> pred3
...
```

这叫自回归预测。

通俗理解：

```text
模型只学会了“看当前图，预测下一张”。
要预测 12 张，就把刚预测出来的下一张再当作当前图。
```

优点：

- 只需要训练一个单步模型。
- 可以滚动预测任意步数。
- 每一步都保持动态演化风格。

缺点：

- 误差会累积。
- 如果第一步有一点偏差，后面可能逐步放大。
- 模型在训练时看到的 `x0` 都是真实帧，但测试后几步看到的是自己生成的帧，这叫 exposure bias。

---

## 19. `_predict_next_frame()` 是推理核心

源码中 flow 分支：

```python
def _predict_next_frame(self, x0):
    device, B = x0.device, x0.shape[0]
    C, H, W = x0.shape[1:]
    x0_p, orig = _pad_tensor(x0, max_size=self._max_size)

    if self._mode == "flow":
        def _ode_f(t, y):
            y = y.reshape(-1, C, H, W)
            return self.model(y, t).flatten(start_dim=1)

        y0 = x0_p.flatten(start_dim=1)
```

逐步解释。

---

## 20. 推理时为什么要解 ODE

训练学到的是：

```text
v_theta(x_t, t)
```

推理时想从：

```text
y(0) = x0
```

走到：

```text
y(1) = x1_hat
```

这就是一个常微分方程：

```text
dy/dt = v_theta(y(t), t)
```

初值：

```text
y(0) = x0
```

解：

```text
y(1) = x0 + ∫_0^1 v_theta(y(t), t) dt
```

最终预测：

```text
x1_hat = y(1)
```

通俗类比：

```text
模型不是直接说终点在哪。
模型说：你每一刻应该往哪里走。
ODE 求解器负责按这些方向一步步走到终点。
```

---

## 21. `_ode_f(t, y)` 为什么要 reshape

ODE 求解器通常希望状态是二维：

```text
[B, D]
```

其中：

```text
D = C × H × W
```

所以初始状态：

```python
y0 = x0_p.flatten(start_dim=1)
```

把：

```text
[B, C, H, W]
```

变成：

```text
[B, C*H*W]
```

但是 U-Net 需要图像格式：

```text
[B, C, H, W]
```

所以 `_ode_f` 里要先 reshape 回去：

```python
y = y.reshape(-1, C, H, W)
```

然后送入模型：

```python
self.model(y, t)
```

输出仍是图像格式：

```text
[B, C, H, W]
```

再 flatten 回 ODE 求解器需要的格式：

```python
return self.model(y, t).flatten(start_dim=1)
```

这两个变形不改变数据，只改变视图/排列方式。

通俗理解：

```text
ODE 求解器喜欢把整张图摊平成一条长向量。
U-Net 喜欢看二维图像。
所以每一步都要在“长向量”和“图像”之间来回转换。
```

---

## 22. `torchode` 如何求解

代码：

```python
term = to.ODETerm(_ode_f)
step_method = to.Heun(term)
```

`ODETerm` 把 `_ode_f` 包装成一个 ODE 项。

`Heun` 是一种二阶 Runge-Kutta 方法。

### Heun 方法直觉

普通欧拉法是：

```text
先看当前位置的斜率
直接走一步
```

公式：

```text
y_{n+1} = y_n + h f(t_n, y_n)
```

Heun 方法更谨慎：

```text
先用当前位置斜率预测一个临时终点
再在临时终点看一次斜率
最后取两次斜率的平均
```

公式大致为：

```text
k1 = f(t_n, y_n)
y_temp = y_n + h k1
k2 = f(t_n + h, y_temp)
y_{n+1} = y_n + h * (k1 + k2) / 2
```

通俗例子：

```text
你开车按导航走。
欧拉法只看当前位置导航方向，一脚油门开到底。
Heun 会先估计开过去以后方向会不会变，再折中决定真正方向。
```

这比欧拉法更稳定。

---

## 23. 自适应步长控制器

flow 默认：

```python
if self._solver_dt0 is not None:
    controller = to.FixedStepController()
else:
    controller = to.IntegralController(
        atol=self._solver_atol, rtol=self._solver_rtol, term=term
    )
    dt0 = None
```

因为 `gvbf_solver_dt0 = None`，所以走自适应控制器。

它会根据误差估计自动决定步长。

通俗理解：

```text
天气场变化平滑的地方，大步走。
天气场变化复杂的地方，小步走。
```

好处：

- 减少不必要计算。
- 对复杂轨迹更稳。

风险：

- 不同样本可能实际使用不同步数。
- 结果可能受误差容忍度影响。
- 如果容忍度太宽，可能积分不够精细。

---

## 24. `t_eval` 为什么只有 `[0, 1]`

```python
t_eval = torch.linspace(0, 1, 2, device=device)[None, :].repeat(B, 1)
problem = to.InitialValueProblem(y0=y0, t_eval=t_eval)
```

`torch.linspace(0, 1, 2)` 得到：

```text
[0, 1]
```

也就是说，我们只关心：

```text
t=0 的初始状态
t=1 的最终状态
```

中间求解器实际可能走很多内部步，但不需要全部保存。

通俗理解：

```text
你从家走到学校。
我们只要起点照片和终点照片。
中间每一步脚印不保存。
```

---

## 25. 求解后为什么 reshape 成 `[B, 2, C, H, W]`

```python
sol = adjoint.solve(problem, dt0=dt0)
result = sol.ys.reshape(B, 2, C, H, W)[:, -1]
```

因为 `t_eval` 有两个时间点：

```text
0 和 1
```

所以 `sol.ys` 包含两个状态：

```text
y(0), y(1)
```

reshape 后：

```text
[B, 2, C, H, W]
```

其中第二维的长度 2 对应：

```text
index 0 -> t=0
index 1 -> t=1
```

所以：

```python
[:, -1]
```

取最后一个时间点：

```text
y(1)
```

也就是下一帧预测。

---

## 26. 为什么现在不应该再 clamp

之前代码曾经有：

```python
return _unpad_tensor(result, orig).clamp(-1, 1)
```

这会在 z-score 标准化空间里把输出限制到：

```text
[-1, 1]
```

对于 `w10` 的标准化参数：

```text
mean ≈ 10.55
std  ≈ 5.98
```

反标准化后最大值会变成：

```text
mean + std ≈ 16.53
```

这会直接导致：

```text
预测无法达到 20.8 高阈值
红色强风区域消失
低值也被抬高
```

现在已经去掉了 clamp：

```python
return _unpad_tensor(result, orig)
```

这是正确的。因为标准化空间的物理变量不应该被随意限制在 `[-1,1]`。

如果未来需要防止数值爆炸，也应该使用更合理的方式，例如：

- 在物理空间做范围限制。
- 只在可视化时做色标饱和。
- 加入损失约束，而不是硬截断预测。

---

## 27. 验证阶段和测试阶段不完全一致

验证阶段：

```python
if self._mode == "flow":
    loss, v = self._flow_loss(x0, x1)

with torch.no_grad():
    pred = (x0 + v).unsqueeze(1)[:, :, out_slice]
```

这里验证输出是：

```text
pred = x0 + v
```

而 `v` 来自 `_flow_loss()` 中随机采样的 `t` 对应的模型输出。

测试阶段：

```python
frame = self._predict_next_frame(current)
```

它会真正走 ODE 积分。

这说明：

```text
验证时不是完整 ODE 推理。
测试时是完整 ODE 推理。
```

这是一处需要注意的设计差异。

通俗理解：

```text
验证时像是“看一下模型预测的方向，然后直接走一大步”。
测试时像是“按模型方向一步步积分走到终点”。
```

如果要让验证指标更贴近测试，建议验证也调用 `_predict_next_frame()` 或 `forward()`。

---

## 28. Flow 为什么可视化更真实

你之前观察到：

```text
gSTA 指标更好，但图像有点假。
flow 指标稍逊色，但图像非常真。
```

这和模型机制高度一致。

### gSTA 的特点

gSTA 是直接从历史序列预测未来序列：

```text
batch_x -> pred_y
```

它的训练损失是直接的 MSE：

```python
loss = self.criterion(pred, label)
```

这种模型往往更擅长降低平均误差。

但是为了降低 MSE，它可能会倾向于：

- 平滑强梯度。
- 缩小局地极值。
- 让不确定区域变成平均状态。
- 减少误报，从而提升 FAR/HSS。

### Flow 的特点

flow 预测的是变化方向：

```text
v = x1 - x0
```

并通过 ODE 连续推进：

```text
dy/dt = v_theta(y,t)
```

它更像物理演化：

```text
当前风场被一个学到的速度场推动。
```

这使得它更容易保持：

- 风带结构
- 涡旋结构
- 连续纹理
- 高低值交界
- 局地强风区域

所以它看起来更真实。

### 统计证据

对当前保存输出的统计显示，flow 的空间梯度强度高于 gSTA。

直觉上：

```text
空间梯度越高，图像越有锋面、边界、纹理和局地结构。
空间梯度越低，图像越平滑。
```

flow 的视觉真实性来自更强的空间结构。

---

## 29. 为什么指标可能不如 gSTA

逐像素指标有一个特点：

```text
它要求每个位置都对。
```

如果真实强风带在这里：

```text
位置 A
```

flow 预测在旁边：

```text
位置 A + 2 个网格
```

肉眼看可能仍然很真实，因为形态对了。

但逐像素 MSE 会认为：

```text
A 位置漏报
A+2 位置误报
```

于是惩罚很重。

通俗例子：

```text
你画了一条非常真实的云带，但比真实位置偏了 50 公里。
人眼会说“这很像天气图”。
逐像素指标会说“这里错，那里也错”。
```

因此：

```text
视觉真实感 和 逐像素指标高低 并不完全一致。
```

flow 更像“结构正确但位置可能有偏差”的模型。

gSTA 更像“位置和数值更保守”的模型。

---

## 30. Flow 模型当前表现好的原因

结合代码和现象，flow 表现好主要有以下原因。

### 1. 任务本身适合“从上一帧演化”

`w10` 风速场在短时尺度上具有连续性。

未来 1 步通常不是凭空生成，而是：

```text
已有风场移动、增强、减弱、形变
```

flow 正好建模这种连续变化。

### 2. 训练目标简单直接

flow 不引入噪声，不引入额外条件，也不学复杂的双分支。

它只学：

```text
x1 - x0
```

这使得优化更稳定。

相比 biflow：

```text
biflow 同时学 v 和 noise
```

flow 的目标更单纯。

### 3. 推理无随机噪声

flow 推理：

```python
y0 = x0_p.flatten(start_dim=1)
```

没有随机噪声。

biflow 推理：

```python
y0 = x0_p + torch.randn_like(x0_p) * noise_level
```

有随机性。

flow 的测试更确定，batch size 改变或重复测试时波动更少。

### 4. 输出值域不再被 clamp 限制

去掉 `clamp(-1,1)` 后，flow 能产生更合理的高值和低值。

这让：

- 红色高风速区域可以出现。
- 低值浅蓝区域可以保留。
- 高阈值指标不再被硬性截断。

### 5. ODE 推理保留动态连续性

ODE 不是直接把输入映射成输出，而是持续沿速度场演化。

这给模型增加了“连续运动”的 inductive bias。

---

## 31. Flow 模型的风险

虽然 flow 当前表现最好，但它仍有一些需要注意的地方。

### 风险 1：训练单步，测试多步

训练只学：

```text
x0 -> x1
```

测试要滚动：

```text
x0 -> pred1 -> pred2 -> ... -> pred12
```

后面的输入来自模型自己，而不是真实数据。

这会造成误差累积。

### 风险 2：验证和测试不完全一致

验证用：

```text
pred = x0 + v
```

测试用：

```text
ODE solve
```

所以验证指标不一定完全代表测试表现。

### 风险 3：MSE 不关心空间结构

flow 虽然结构上更真实，但训练损失仍然是普通 MSE。

它没有显式要求：

- 梯度真实
- 频谱真实
- 强风区域面积真实
- 连通域真实
- 风带位置整体合理

### 风险 4：高阈值事件仍可能被误报

flow 更敢产生高值，这让图更真，也让 POD 可能更好。

但它可能提高 FAR。

当前 `flow_w3s` 的高阈值 FAR：

```text
FAR@20.8 = 0.218848
```

说明高值误报仍然不少。

---

## 32. Flow 和 Biflow 的差别

flow 训练：

```text
xt = lerp(x0, x1, t)
v = model(xt, t)
target = x1 - x0
loss = MSE(v, target)
```

biflow 训练：

```text
xt = lerp(x0, x1, t)
xta = xt + alpha * noise
output = model(xta, t, alpha)
v, d = chunk(output, 2)
loss = MSE(v, x1-x0) + MSE(d, noise)
```

flow 推理：

```text
y0 = x0
dy/dt = model(y,t)
```

biflow 推理：

```text
y0 = x0 + noise_level * noise
t: 0 -> 1
alpha: noise_level -> 0
dy/dk = v + d * (0 - noise_level)
```

flow 更简单、更确定。

biflow 更复杂，理论上可以同时建模运动和去噪，但也更难训练稳定。

当前现象中 flow 表现更好，合理原因是：

```text
风速短时预测本身更需要连续运动，而不是强随机生成。
```

---

## 33. Flow 和 Condiff 的差别

condiff 训练：

```text
noise = randn_like(x1)
xa = lerp(noise, x1, alpha)
d = model(concat(xa, x0), alpha)
target = x1 - noise
```

condiff 更像条件扩散：

```text
从噪声逐步生成目标帧，x0 作为条件。
```

flow 更像确定性演化：

```text
从当前帧沿速度场走到下一帧。
```

对于短期风场：

```text
下一帧和当前帧高度相关
```

所以从 `x0` 出发的 flow 可能比从噪声出发的 condiff 更自然。

---

## 34. 当前 Flow 的完整训练伪代码

```text
输入:
  batch_x: 历史序列 [B,T_in,C,H,W]
  batch_y: 未来序列 [B,T_out,C,H,W]

取:
  x0 = batch_x 最后一帧
  x1 = batch_y 第一帧

空间处理:
  x0_p = pad(x0)
  x1_p = pad(x1)

随机采样:
  t ~ Uniform(0,1)

构造中间状态:
  xt = x0_p + t * (x1_p - x0_p)

模型预测:
  v_pred = UNet(xt, t)

真实速度:
  v_true = x1_p - x0_p

损失:
  loss = mean((v_pred - v_true)^2)

反向传播:
  更新 UNet 参数
```

---

## 35. 当前 Flow 的完整推理伪代码

```text
输入:
  batch_x: 历史序列 [B,T_in,C,H,W]

初始化:
  current = batch_x 最后一帧
  frames = []

循环 n_pred 次:
  1. current_p = pad(current)
  2. y0 = flatten(current_p)
  3. 定义 ODE:
       dy/dt = flatten(UNet(reshape(y), t))
  4. 用 Heun 方法从 t=0 积分到 t=1
  5. 得到 frame = y(1)
  6. frame = unpad(frame)
  7. frames.append(frame)
  8. current = frame

输出:
  pred_y = stack(frames)
```

---

## 36. 小白版总结

如果完全不看数学，可以这样理解 flow：

```text
它不是直接画未来天气图。
它学的是天气图怎么动。
```

训练时：

```text
给它现在图 x0 和未来图 x1。
随机拿一张中间图 xt。
问模型：如果天气现在长这样，应该往哪里变？
正确答案就是 x1 - x0。
```

测试时：

```text
只有现在图 x0。
模型不断给出“下一小步该往哪里走”。
ODE 求解器按照这些方向走到 t=1。
得到下一帧。
然后把下一帧当成当前帧，继续预测。
```

为什么它看起来真？

```text
因为天气图确实像连续运动的东西。
flow 模型顺着这个特点建模，所以容易保留形态。
```

为什么指标不一定第一？

```text
因为逐像素指标很严格。
哪怕风带形状很真，只要位置偏一点，指标就会扣分。
```

---

## 37. 建议的后续改进

### 1. 让验证和测试路径一致

当前验证：

```text
x0 + v
```

测试：

```text
ODE solve
```

建议验证也用：

```python
pred = self._predict_next_frame(x0)
```

这样验证指标更接近测试。

### 2. 加入梯度损失

为了保留锋面、边界和风带结构，可以加入：

```text
L_grad = || ∇x pred - ∇x label || + || ∇y pred - ∇y label ||
```

总损失：

```text
L = L_flow + λ L_grad
```

### 3. 加入高阈值加权

对强风区域加权：

```text
weight = 1 + a * I(x1 >= 13.9) + b * I(x1 >= 20.8)
```

损失：

```text
L = mean(weight * (v_pred - v_true)^2)
```

这样模型会更重视强风。

### 4. 加入频谱损失

天气图的真实感和空间频率有关。

可以比较 FFT：

```text
L_spec = || |FFT(pred)| - |FFT(label)| ||
```

这能减少过平滑。

### 5. 使用 gSTA 作为 coarse prior

可以让 gSTA 先给粗预测：

```text
x_gsta
```

然后 flow 学残差：

```text
x_true - x_gsta
```

输入条件可以是：

```text
concat(current, x_gsta, x_gsta-current)
```

这样兼顾：

```text
gSTA 的指标稳定性
flow 的视觉真实性
```

---

## 38. 最终结论

`GVBF flow` 是一个基于 flow matching 思想的连续演化模型。

它的核心不是直接预测未来帧，而是学习：

```text
从当前帧到下一帧的速度场。
```

训练阶段：

```text
在 x0 和 x1 之间随机采中间状态 xt，
让 U-Net 预测真实速度 x1-x0。
```

推理阶段：

```text
从 x0 出发，用 U-Net 定义的速度场作为 ODE 右端项，
通过 Heun 方法积分到 t=1，得到下一帧。
```

它之所以可视化真实，是因为它带有强烈的动态演化先验：

```text
天气不是被重新画出来的，而是从上一帧连续变过去的。
```

它当前的主要问题是：

```text
训练单步，测试多步；
验证路径和测试路径不一致；
损失仍然是普通 MSE；
高阈值和结构真实性没有被显式优化。
```

但从现在的表现看，flow 是一个非常值得继续深入的方向。相比直接预测模型，它更适合作为“形态真实”的核心模块；如果后续结合 gSTA 粗预测、高阈值加权和梯度/频谱损失，有机会同时提升客观指标和视觉真实感。
