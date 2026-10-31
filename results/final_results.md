# 最终结果表（结论级汇总）

> 本文件由 `python tools/make_final_tables.py` 从 results/ 原始逐行 CSV 机械生成（改原始 CSV 后重跑即可再生）；聚合口径与 stress_gap_test.py 一致。
> 单位：% acc（meanpool 主口径，3 种子 mean±pstdev）；pp=百分点。
> 臂：A=论文式（丢帧置零+均匀 dt=1）；B=同一 ckpt 换手动循环+真实 timespans（推理期唯一变量）；C=B+gap 指示位+块状增强训练；D=C 同训练但不带 gap 位。
> 条件：clean / uniform30（逐时间步 Bernoulli 0.3）/ block5-30、block15-30（块长 b 划分、每块 Bernoulli 0.3，期望掩码率 30%，3 固定种子化模式跨臂配对）。

## 1. QuickDraw 三臂（S5-full-protocol，results/full_threearm.csv）

| 条件 | A | B | C |
|---|---|---|---|
| clean | 92.30±0.86 | 92.30±0.86 | 92.89±0.17 |
| uniform30 | 63.93±2.86 | 64.37±3.76 | 86.10±0.80 |
| block5-30 | 71.03±4.10 | 69.81±4.69 | 88.52±0.66 |
| block15-30 | 72.83±2.80 | 72.03±3.22 | 83.55±1.08 |

| 条件 | B−A | C−A |
|---|---|---|
| clean | +0.00 | +0.59 |
| uniform30 | +0.44 | +22.17 |
| block5-30 | -1.22 | +17.49 |
| block15-30 | -0.80 | +10.72 |

**H1 主判据（block5-30）：B−A=−1.22pp < +1pp 且 clean B−A=0.00pp → 预注册负结果分支成立（falsified）**；C 臂 C−A=+17.49pp 幅值过线但 clean 逐种子 |Δ|=1.82/0.37/0.31 不全 ≤0.5pp（双侧实现约定）→ C 保守门 FAIL，增益归因转入 S7/S10-redesign。
三口径兼报（block5-30）：B−A meanpool=-1.22 / gapnb=+0.13 / final=+0.05pp；C−A meanpool=+17.49 / gapnb=+22.50 / final=+0.79pp。

## 2. N-MNIST 三臂（S10，论文原文数据集，results/nmnist_threearm.csv）

| 条件 | A | B | C |
|---|---|---|---|
| clean | 98.12±0.20 | 98.12±0.20 | 98.78±0.04 |
| uniform30 | 90.06±4.26 | 90.13±3.71 | 97.85±0.27 |
| block5-30 | 80.99±4.76 | 81.08±4.47 | 90.45±1.37 |
| block15-30 | 71.34±1.83 | 71.34±1.83 | 71.44±1.94 |

| 条件 | B−A | C−A |
|---|---|---|
| clean | +0.00 | +0.67 |
| uniform30 | +0.07 | +7.79 |
| block5-30 | +0.09 | +9.47 |
| block15-30 | +0.00 | +0.10 |

**H1 主判据（block5-30）：B−A=+0.09pp（逐种子 +0.53/+0.04/−0.29）< +1pp 且 clean B−A=0.00pp → falsified**——与 QuickDraw 同型，H1 双数据集结论级证伪。C−A=+9.47pp 但 clean 逐种子 +0.95/+0.43/+0.62 不全 ≤0.5pp → C 保守门 FAIL。
注：L=10 下 block15-30 退化为整段掩码（已知口径特性，行保留作协议可比）。

## 3. E2 归因臂 D（S7，QuickDraw，results/e2_btrain.csv）

| 条件 | D−A | D−C |
|---|---|---|
| clean | +0.24 | -0.34 |
| uniform30 | +20.48 | -1.69 |
| block5-30 | +15.31 | -2.18 |
| block15-30 | +7.96 | -2.75 |

**归因比例 D−A/(C−A) = 15.31/17.49 = 0.8752**（判据③部分归因）：训练期激活的时间通路贡献 C−A 增益的 87.5%，gap 指示位边际贡献仅 -2.18pp。clean 门双约定并报：单侧字面 PASS、双侧实现 FAIL（seed0 +1.94pp 超线），两种约定共同结论不变。

## 4. S10-redesign 负结果归因两探针（纯推理，results/nm_negative_probes.csv）

| 探针 | 重算值 | 判据（跑前冻结） | 机械输出 |
|---|---|---|---|
| 主探针 C_no_dt−A（block5-30，3 种子 mean） | +9.58pp | ≥+5pp→训练期主导 | **training_dominant_supported** |
| 推理期 dt 净效应 C_dt−C_no_dt | -0.11pp | 噪声级=无推理期增益 | 与 B−A≈0 互证 |
| clean 口径切换差 | 0.0000pp | <0.3pp | PASS |
| 次探针 dt*（block5-30） | 20（R_max=0.199260） | dt*≥5→帧化量化假设存活 | **supported** |
| 恒等自检 A_ref/C_dt vs S10 结论级表 | max|diff|=0.000000 | <5e-4 | PASS（口径零漂移） |

解读：C 臂 +9.47pp 增益几乎全部来自训练期（块增强+gap-bit），推理期时间信息两个数据集、两种增强条件下均无增益；dt*=20≥5 说明 CfC 对 bin 级 dt 仍敏感——帧化量化（L=10≈30ms/bin 把 µs 时间量化约 4 个数量级）是 H1 失败的协议侧候选解释（caveat，保留可证伪结构：L=34 重分箱≈10ms/bin 为加强验法入口）。

## 5. QuickDraw LR step decay 重训（S11，results/quickdraw_lrdecay.csv）

| 条件 | A | B |
|---|---|---|
| clean | 93.20±0.09 | 93.20±0.09 |
| uniform30 | 64.81±2.48 | 65.09±3.39 |
| block5-30 | 73.36±2.20 | 71.87±3.00 |
| block15-30 | 75.47±2.15 | 73.93±2.28 |

| 条件 | B−A |
|---|---|
| clean | +0.00 |
| uniform30 | +0.28 |
| block5-30 | -1.49 |
| block15-30 | -1.55 |

**判据①（跑前冻结）：A@clean=93.20±0.09 vs S5 基线 92.30±0.86 → 增益 +0.90pp → inconclusive**（+0.5~+1.5pp 冻结带；逐种子 +1.99/+0.20/+0.52）。
D7 缺口：vs 论文 Table I 0.9577 由 −3.47pp 收窄至 −2.57pp；vs Table III 0% 列 92.69 由 −0.39pp 转为 +0.51pp（已越过该锚点）。block5-30 B−A=−1.49pp（H1 结论不受影响）。

## 6. D7 复现保真度（论文锚点双报）

| 口径 | 本仓 A@clean | vs 论文 Table I LNN Test | vs 论文 Table III 0% 列 |
|---|---|---|---|
| QuickDraw（S5） | 92.30 | -3.47pp（0.9577） | -0.39pp（92.69） |
| QuickDraw（S11 后） | 93.20 | -2.57pp | +0.51pp |
| N-MNIST（S10） | 98.12 | -1.26pp（0.9938） | -0.59pp（98.71） |

两数据集保真度缺口均 ≤3pp（QuickDraw −2.57 / N-MNIST −1.26，S11 后）→ 复现保真度不构成 H1 负结果的竞争解释。S9 已辨明并记账：早期引用的「论文 QuickDraw≈98.6-98.7%」实为 Table III **N-MNIST** 行 0% 列，参照系引错产生 2.83pp 虚增缺口（d7_fidelity_analysis.json）。

## 7. 终判汇总（都经过课题 researcher 判读单元独立复算）

| 编号 | 结论 | 依据 |
|---|---|---|
| H1 | **falsified（结论级，双数据集）** | 推理期真实 timespans 条件化在块状丢帧下无增益：QuickDraw B−A=−1.22pp、N-MNIST B−A=+0.09pp（均 <+1pp 预注册证伪线），clean 均不受损 |
| H2 | 字面满足、科学空洞 | Δ=+0.42pp 但两档增益均负，无正增益可缩放 |
| E2 | **部分归因（训练期主导）** | D−A/(C−A)=0.8752；推理期 dt 净效应 −0.11pp |
| C 臂增益来源 | **训练期主导（N-MNIST 确认）** | C_no_dt−A=+9.58pp ≥+5pp |
| 帧化量化候选解释 | **supported（caveat）** | dt*=20≥5，CfC 对 bin 级 dt 仍敏感 |
| D7 保真度 | 不构成竞争解释 | 缺口 ≤3pp（双锚点双报），QuickDraw 已越过 Table III 0% 锚点 |
