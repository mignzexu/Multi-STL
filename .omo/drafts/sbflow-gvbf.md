# Draft: SBFlow 与 GVBF 集成评估

## Requirements (confirmed)
- 用户希望评估 `origin/conditional-flow-matching` 中的 SBCFM，并判断是否能在 GVBF 中加入 SBflow 模式。
- 用户建议在 `models/` 下重新开一个 `SBFlow`，单独存放该模型。
- 用户关心两个风险：POT 依赖是否可单独使用；是否应在 GVBF `flow` 模式基础上只补充 SBCFM 与普通 CFM 的差异部分。

## Technical Decisions
- 倾向方案：新增独立 `models/SBFlow/` 模型目录，但第一版内部复用 GVBF 的网络构造与 ODE 推理思想。
- 第一版建议实现 `sbflow-lite`：先只加入 SBCFM 相对普通 CFM 的桥噪声与目标速度公式，OT/Sinkhorn 作为可选配置而非默认开启。
- 不建议直接大改 `models/GVBF/Model.py`，避免污染现有 `flow/biflow/condiff` 对照实验。

## Research Findings
- `SchrodingerBridgeConditionalFlowMatcher` 位于 `origin/conditional-flow-matching/torchcfm/conditional_flow_matching.py:397-512`。
- SBCFM 与普通 CFM 的核心差异：`sigma_t=sigma*sqrt(t*(1-t))`、`x_t=mu_t+sigma_t*eps`、目标速度增加桥噪声导数项。
- OT 配对依赖 `origin/conditional-flow-matching/torchcfm/optimal_transport.py:11-145` 中的 POT `ot` 包。
- GVBF 当前 mode 分发集中在 `models/GVBF/Model.py`，配置集中在 `models/GVBF/config.py`。

## Open Questions
- 新模型目录名是否统一用 `SBFlow`，命令行模型名是否用 `sbflow`。
- 第一版是否默认关闭 OT，仅保留 `gvbf_sb_use_ot=False` 或等价配置。

## Scope Boundaries
- INCLUDE: 方案评估、集成路径建议、风险权衡。
- EXCLUDE: 当前不修改源码、不安装依赖、不启动训练。
