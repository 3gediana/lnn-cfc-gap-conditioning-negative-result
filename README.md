# 液态网络抗丢帧：把真实时间接进 CfC，到底有没有用？

> 一句话结论：**没用——这是个负结果仓库。** 在两个公开数据集上，把推理期真实时间差
> （timespans）喂进液态网络（CfC）的原生时间输入，在「一段一段丢帧」的真实传感器故障场景下
> 都没有提升测试精度（QuickDraw −1.22pp、N-MNIST +0.09pp，预注册证伪线 +1pp）。
> 论文宣称的抗丢帧优势，主要不来自推理期的时间机制。

> 仓库地址：https://github.com/3gediana/lnn-cfc-gap-conditioning-negative-result
> 主课题仓（完整研究过程：判读/复盘/协议取证）：https://github.com/3gediana/lnn-cfc-gap-conditioning

- 复现入口（一条命令，纯标准库、无需 GPU/数据/下载，秒级）：
  ```bash
  python stress_gap_test.py
  ```
  它会从 `results/` 的原始逐行 CSV 独立重算全部头条数字、机械执行预注册门线，
  并与已判读期望值逐项比对（本仓库交付时 147/147 硬检查通过；加 `--identity` 为 148/148）。
- 论文：arXiv:2605.27467（JCSSE 2026，代码/日志全开源）——CfC vs LSTM 丢帧鲁棒性基准。
  协议复刻口径与差异清单见 [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md)。
- 完整复现（数据 + GPU 重训）见 [REPRODUCE.md](REPRODUCE.md)。

---

## 1. 这个问题是哪来的（大白话）

2026 年那篇论文测「液态网络比 LSTM 抗丢帧」，给了部署级证据。但它有两个我们认为是缺口的地方：

1. **丢帧方式是均匀随机**：每个时间步独立地按 30%/50%/70% 概率丢弃。真实传感器故障往往是
   「一段一段地坏」（连续几秒没数据），不是均匀撒胡椒面。
2. **时间信息从不进模型的时间输入**：论文把 time-delta 当普通输入特征拼进去，丢帧之后时间
   仍按均匀网格处理——而液态网络的数学结构（CfC 的 ODE 形式）本来就能直接吃「真实经过的时间」。

液态网络的连续时间优势，恰好在最能兑现它的场景（成段丢失、不规则采样）没被接上。
我们这个仓库就是把这两件事补上：**块状间隙 + 每步真实 Δt 进 CfC 原生 timespans 接口**，
看论文宣称的优势是变强还是缩水。改进是纯模块级（换时间条件化方式），没有新数据集、没有新架构。

## 2. 实验怎么设计的（大白话）

同一个基准、同一批数据、三个「臂」对照（预注册，跑前冻结判据）：

| 臂 | 是什么 | 改了什么 |
|---|---|---|
| A | 论文式 | 丢帧置零，时间按均匀网格（dt 恒等于 1） |
| B | A 的同一个模型 | 只把前向换成手动循环 + 真实 timespans（**推理期唯一变量**） |
| C | 单独训练 | B + gap 指示位 + 训练也用块状增强（训练和推理都改） |
| D | 归因臂 | C 的同款训练但不带 gap 指示位（用于拆解 C 的增益来源） |

- 两个数据集：**QuickDraw**（论文同款 10 类子集，42500 训练 / 7500 测试）和
  **N-MNIST**（论文原文事件流数据集，60000/10000，10 个时间 bin）。
- 丢帧条件：clean（不丢）/ uniform30（逐时间步 30%）/ block5-30、block15-30（块长 5 或 15，
  每块 30% 概率整块丢，期望掩码率 30%），3 个固定种子化掩码模式跨臂配对。
- 主口径 meanpool（论文口径），meanpool/gapnb/final 三口径兼报；3 个随机种子。

**H1（待检验假设）**：30% 块状间隙下，B 或 C 相对 A 的测试精度提升 ≥2pp，且干净数据降幅 ≤0.5pp。
**预注册失败条件**：B/C 提升 <1pp 且干净精度不降 → 时间条件化无边际收益，负结果落盘、不硬撑。

## 3. 结果（结论级，3 种子 mean±std，% acc）

### QuickDraw（results/full_threearm.csv）

| 条件 | A | B | C | B−A | C−A |
|---|---|---|---|---|---|
| clean | 92.30±0.86 | 92.30±0.86 | 92.89±0.17 | +0.00 | +0.59 |
| uniform30 | 63.93±2.86 | 64.37±3.76 | 86.10±0.80 | +0.44 | +22.17 |
| block5-30 | 71.03±4.10 | 69.81±4.69 | 88.52±0.66 | **−1.22** | +17.49 |
| block15-30 | 72.83±2.80 | 72.03±3.22 | 83.55±1.08 | −0.80 | +10.72 |

### N-MNIST（论文原文数据集，results/nmnist_threearm.csv）

| 条件 | A | B | C | B−A | C−A |
|---|---|---|---|---|---|
| clean | 98.12±0.20 | 98.12±0.20 | 98.78±0.04 | +0.00 | +0.67 |
| uniform30 | 90.06±4.26 | 90.13±3.71 | 97.85±0.27 | +0.07 | +7.79 |
| block5-30 | 80.99±4.76 | 81.08±4.47 | 90.45±1.37 | **+0.09** | +9.47 |
| block15-30（L=10 退化为整段掩码，口径特性） | 71.34±1.83 | 71.34±1.83 | 71.44±1.94 | +0.00 | +0.10 |

## 4. 结论

- **H1 = falsified（结论级，双数据集）**：块状丢帧下推理期真实 timespans 条件化没有增益
  （QuickDraw B−A=−1.22pp、N-MNIST B−A=+0.09pp，均低于 +1pp 预注册证伪线），clean 不受损
  （B−A=0.00pp）。两个数据集、两种丢帧条件（uniform30 与 block5-30）同型。
- **H2 = 字面满足但科学空洞**：Δ=+0.42pp 但两档增益均为负，没有正增益可缩放。
- 按预注册转向路径：负结果落盘，不开二轮调参；转向回答「那 C 臂为什么高」。

## 5. C 臂为什么高？（归因，负结果仓库的必答题）

C 臂 +9.47pp（N-MNIST）的增益**几乎全部来自训练期**，不是推理期的时间机制：

| 证据 | 数字 | 说明 |
|---|---|---|
| S7 D 臂（同训练、训练期激活时间通路、无 gap 位） | D−A=+15.31pp，D−A/(C−A)=**0.8752** | QuickDraw 上训练期通路贡献 C−A 增益的 87.5% |
| S10-redesign 主探针（C ckpt 以 A 口径前向=去 dt） | C_no_dt−A=**+9.58pp**（≥+5pp 阈值） | N-MNIST 确认：增益在训练期（块增强+gap-bit） |
| 推理期 dt 净效应 | C_dt−C_no_dt=**−0.11pp** | 噪声级，与 B−A≈0 互证 |

即：**C 臂高是因为训练时见了块状丢帧（数据增强），不是因为推理期喂了真实时间。**

## 6. What was ruled out（我们排除了什么）

1. **复现保真度不构成解释**：本仓 A@clean 与论文锚点缺口 ≤3pp（QuickDraw −2.57pp / N-MNIST −1.26pp，
   S11 LR decay 重训后；论文 Table I 与 Table III 双锚点双报）。详见 `results/d7_fidelity_analysis.json`
   （含一起因记账：早期引用的「论文 QuickDraw≈98.6-98.7%」实为 Table III 的 N-MNIST 行，
   参照系引错曾产生 2.83pp 虚增缺口，已辨明更正）。
2. **实现缺陷已排除**：clean 上 A≡B 逐位 0.000pp（同 ckpt 两条前向路径）、ts 梯度非零、
   实现掩码率 ≈0.30、跨臂同掩码配对、恒等自检零漂移（探针表 vs 结论级表 max|diff|=0.000000）。
3. **推理期时间信息无用**：两个数据集 × 两种丢帧条件 × 两种增强条件（B/C）均无增益。
4. **LR 调度是次因非主因**：补上论文的 LR step decay 只收窄缺口 ~26%（+0.90pp，落在
   inconclusive 冻结带），不改变 H1。
5. **仍开放的项（都带 caveat，保留可证伪结构）**：
   - 帧化量化候选解释：dt*=20≥5，CfC 对 bin 级 dt 仍敏感（L=10≈30ms/bin 把 µs 时间量化约
     4 个数量级）。加强验法入口=L=34 重分箱（≈10ms/bin）复测，论文协议已精确复刻备好。
   - QuickDraw 剩余 ~2.6pp 保真度缺口候选=分类头 Dropout / it-ep 口径（论文 unknown）。

## 7. 怎么复现

```bash
# 一条命令：从原始 CSV 重算全部头条数字 + 机械执行预注册门线 + 与已判读期望值比对
python stress_gap_test.py            # 纯标准库，无需 GPU/数据/下载，秒级完成

# 可选：追加 torch 前向恒等自检（clean 上 A/B 两条前向路径数值恒等，随机初始化小模型，
# 只验证口径性质，不验证科学结论；需要 torch+ncps，见 requirements.txt）
python stress_gap_test.py --identity

# 打包代码导入级完整性冒烟（不训练不读数据）
python code/repo_code_import_smoke.py
```

`stress_gap_test.py` 退出码 0 = 全部硬检查通过。它检查：
原始 CSV 行数与键完整性 → 头条数字重算（mean±pstdev / 臂间差 / 逐种子）→ 与
`results/expected_verdicts.json`（已判读期望值，逐项带来源）比对 → H1/sanity/E2/探针/dt*/D7
门线机械执行 → 交叉文件一致性（探针表 vs S10 结论级表）。

**完整重训**（需要 QuickDraw + N-MNIST 数据与 GPU，约 2-3 小时单卡）：见 [REPRODUCE.md](REPRODUCE.md)。
本仓库不附带数据与 checkpoint（体积原因），但附带产生全部数字的原始代码（`code/`），
协议常数与论文的差异逐项列在 [PAPER_PROTOCOL.md](PAPER_PROTOCOL.md)。

## 8. 限制（诚实清单）

- QuickDraw 为论文 10 类子集（降级数据集）；N-MNIST 为论文原文数据集，结论以双数据集同型为准。
- N-MNIST L=10 下 block15-30 退化为整段掩码（每样本 30% 概率全掩码），该行仅作协议可比。
- C 臂增益含增强训练混杂（预注册 D3：门线对 C 为保守存活判据），归因由 S7/S10-redesign 回答。
- 本仓 3 种子 mean±std vs 论文单次训练（±0.86pp 量级，非系统性）；论文每 epoch 步数 310 vs
  本仓 ~332（it-ep 口径 unknown）；论文优化器/wd/batch 因脚本墙记 unknown。
- 协议差（LR 调度/分类头/units/架构等）全部入 caveat，未改冻结规格与判据——H1 主判据 B−A 是
  同 ckpt 推理期配对，不受绝对水位影响。
- clean 门线有「双侧 |Δ|≤0.5pp（实现约定）」与「单侧降幅 ≤0.5pp（预注册字面）」两个口径，
  按冻结纪律双约定并报不改判；两种约定都不改 H1 证伪与训练期主导归因。

## 9. 仓库结构

```
stress_gap_test.py          一条命令复现入口（本仓库的主交付物）
code/                       产生全部数字的实验代码（原样封包，未改判据）
  pilot_threearm.py         协议同源代码：模型/掩码/条件/评测（A/B/C 臂语义）
  full_protocol.py          QuickDraw 全量 30ep 训练+108 格评测
  nmnist_loader.py          N-MNIST 事件 loader（L=10 帧化，log1p）
  nmnist_threearm.py        N-MNIST 三臂结论级基准
  e2_btrain.py              D 臂（E2 归因臂）
  quickdraw_lrdecay.py      S11 LR step decay 重训
  nm_negative_probes.py     负结果归因两探针（纯推理）
  eval_patch_selftest.py    评测补丁自检
  repo_code_import_smoke.py 打包代码导入级冒烟
tools/make_final_tables.py  从原始 CSV 机械生成 results/final_results.{csv,md}
results/
  full_threearm.csv / nmnist_threearm.csv / e2_btrain.csv / quickdraw_lrdecay.csv /
  nm_negative_probes.csv    原始逐行结果（每行=一个 (数据集,种子,臂,条件,掩码模式) 实测）
  pilot_threearm.csv        pilot 子集结果（历史，结论以全量为准）
  nm_dt_response_spectrum.json / d7_fidelity_analysis.json
  expected_verdicts.json    已判读期望值锚点（stress_gap_test.py 的比对基准）
  final_results.csv / final_results.md   结论级汇总表（tools/ 生成）
```

## 10. 数据与许可

- 数据：QuickDraw（`.npy` 位图，Google 公开）与 N-MNIST（事件流，公开）均按论文同款预处理，
  获取与重建命令见 [REPRODUCE.md](REPRODUCE.md) 数据一节；本仓库不含数据文件。
- 论文：arXiv:2605.27467（JCSSE 2026）。本仓库是对其缺口的独立复现/扩展，结论（负结果）与
  论文主张（LNN 抗丢帧）不冲突——论文测的是均匀丢帧+无时间条件化，我们测的是块状间隙+真实
  timespans 条件化，并说明其优势来源。
- 代码：本仓库代码可自由使用；论文侧代码遵循其原始许可。
