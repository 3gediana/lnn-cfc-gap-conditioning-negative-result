# 论文协议说明（复刻口径与差异清单）

本仓库复现/扩展的论文：**arXiv:2605.27467（JCSSE 2026）**——"CfC vs LSTM 丢帧鲁棒性基准"
（论文代码与训练日志全开源；论文仓库 `experiments/_repo/` 下有 `lln_10.log`、
`lln_nmnist_10.log`、`lln_nmnist.py`、`run_lln_nmnist.sh` 等原件，本说明的论文侧常数均从
这些一手材料取证，非转述）。

## 1. 论文测了什么（与我们仓库的关系）

- 论文给的核心证据：QuickDraw / N-MNIST 上，LNN（CfC）比 LSTM 抗丢帧；压测方式是
  **推理期均匀随机掩码**（0/30/50/70%），且 **time-delta 只作为普通输入特征**，丢帧后时间
  仍按均匀网格处理（CfC 原生 timespans 接口不接真实 Δt）。
- 本仓库的缺口补充：把丢帧从「均匀随机」升级为「块状间隙（真实传感器故障形态）」，并把每步
  真实 Δt 喂进 CfC 原生 timespans 接口（B 臂），再加 gap 指示位与块状增强训练（C 臂）。
- 结论关系：本仓库的负结果（推理期时间条件化无增益）不否定论文「LNN 抗 LSTM 更抗丢帧」的
  主张，而是把该优势的来源指向训练期数据增强/结构而非推理期时间机制（见 README 第 5 节）。

## 2. 论文侧协议常数（取证来源双标注）

| 项 | QuickDraw | N-MNIST | 来源 |
|---|---|---|---|
| epoch | 30（LR 三段：1e-3 ep1-12 → 5e-4 ep13-25 → 2.5e-4 ep26-30） | 10 | `lln_10.log` / `lln_nmnist_10.log` 逐 epoch LR 行；`run_lln_nmnist.sh --epochs 10` |
| 终值精度（Table I） | Train 0.9978 / **Test 0.9577** | Train 0.9997 / **Test 0.9938** | Table I 与仓库日志逐位一致（双源） |
| Table III 掩码压测（LNN） | 0%=**92.69** / 30%=68.80 / 50%=41.47 / 70%=22.48 | 0%=98.71（LSTM 98.63） | Table III |
| 数据 | 10 类子集；stroke x,y+pen state 共 5 特征；85/15 split | 事件流；每样本跨度等分 10 个物理时 bin | 论文 §II.G/§V.C；`lln_nmnist.py` 帧化规则 |
| 架构 | 5→Linear+LN+ReLU→128；256-unit core（CfC）；Mean Pooling；分类头 MLP(Dropout+ReLU) | units128/CfC(4096,128,proj128)/双CNN+BN | 论文 §V.C / Table V |
| 训练次数 | 单次训练/配置 | 单次 | 论文 §VI.A |
| 每 epoch 步数 | 310 it/ep | — | `lln_10_full.log` tqdm 行 |
| 优化器/wd/batch | unknown（日志不打印；脚本遇路径墙未取到） | Table V: batch64/no-wd/no-clip/Cosine | 记 unknown，不开实验 |

> 论文自身有两处内部矛盾，本仓库双记不吞：①Table I N-MNIST 0.9938 vs Table III 0% 列 98.71；
> ②Table V N-MNIST 记 30ep/Cosine/units128 vs 训练日志与 run 脚本的 10ep。D7 主锚点取 Table I
> （与仓库日志逐位一致），Table III 值兼报。

## 3. 本仓冻结规格（跑前冻结，未随结果改动）

| 项 | 本仓（QuickDraw 与 N-MNIST 同规格） |
|---|---|
| 模型 | CfC(128,256,proj256,backbone_layers=1,backbone_units=128)（ncps 0.0.2）；单CNN+LN+ReLU |
| 优化 | AdamW lr 1e-3 恒定（S11 前）/ wd 0.01 / batch 128 / clip 1.0；3 种子 [0,1,2] |
| epoch | QuickDraw 30 / N-MNIST 10（与论文逐项一致） |
| 读表口径 | 主口径 meanpool（论文口径）；gapnb / final 兼报 |
| 评测 | QuickDraw 全量 test 7500；N-MNIST 全量 test 10000 |
| 掩码 | uniform30（逐时间步 Bernoulli 0.3）；block5-30/block15-30（块长 b 划分、每块 Bernoulli 0.3，期望掩码率 30%；3 固定种子化模式跨臂配对） |
| 数据切分 | QuickDraw 首 5000/类×10 类=50k，85/15 seeded split（SPLIT_SEED=12345）→ 42500/7500 |
| N-MNIST 帧化 | L=10；每样本按自身跨度等分 10 个物理时 bin；frames[bin,p,y,x] 计数 → log1p → (N,10,2,34,34) |

## 4. 协议差异逐项判定（本仓 vs 论文；证据在 results/d7_fidelity_analysis.json）

| # | 项 | 论文 | 本仓 | 判定 |
|---|---|---|---|---|
| 1 | epoch 数 | 30 / 10 | 30 / 10 | **一致** |
| 2 | LR 调度 | step decay 三段 | 恒定 1e-3（S11 前） | **偏离 #1** → S11 隔离复测：+0.90pp（inconclusive，次因） |
| 3 | 分类头 | MLP(Dropout+ReLU) | 单层 Linear(256,10) | **偏离 #2**（正则差；论文 dropout 率 unknown；禁改项，记 caveat） |
| 4 | 训练次数 | 单次 | 3 种子 mean±std | 偏离 #3（±0.86pp 量级，非系统性） |
| 5 | 每 epoch 步数 | 310 | ~332（42500/128） | 待核（it-ep 口径 unknown；脚本墙未确认论文 batch） |
| 6 | 数据切分/特征 | 5 特征，85/15 | 同左（seeded） | 一致 |
| 7 | core/读表 | Linear+LN+ReLU→128；256 core；Mean Pooling | 同左 | 一致 |
| 8 | 优化器/wd/batch | unknown（QuickDraw） | AdamW/wd0.01/batch128 | unknown（论文侧） |
| 9 | units/架构（N-MNIST） | units128/双CNN+BN/batch64/Cosine | units256/单CNN+LN/batch128/恒定 lr | 偏离（全部入 caveat；H1 主判据为同 ckpt 推理期配对，不受影响） |

**为什么这些偏离不威胁 H1 结论**：H1 主判据 B−A 是**同一个 checkpoint** 上两条前向路径的配对
比较（只差「推理期是否喂真实 timespans」），协议常数（epoch/LR/架构）在分子分母同时出现、
配对抵消；绝对水位（D7）才受协议差影响，而 D7 缺口 ≤3pp 已单独记账（README 第 6 节）。

## 5. 本仓库对论文方法的扩展点（模块级，无新数据集/新架构）

1. 丢帧形态：均匀随机 → 块状间隙（block5-30 / block15-30，期望掩码率同为 30%，与论文 30% 档可比）。
2. 时间条件化：论文式（dt 恒 1）→ 真实 timespans（B 臂，manual-loop 逐样本前向，
   与库 forward(timespans=None) 在 clean 上逐位恒等）→ +gap 指示位与块状增强训练（C 臂）。
3. 归因设计：D 臂（训练期激活时间通路、无 gap 位）+ C_no_dt 口径（C ckpt 以 A 口径前向）
   把「训练期 vs 推理期」两个因素干净分离。
