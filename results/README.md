# results/ 说明（每个文件是什么、能复现什么）

| 文件 | 内容 | 行/结构 |
|---|---|---|
| `final_results.md` / `final_results.csv` | 结论级汇总表（`tools/make_final_tables.py` 从下列原始 CSV 机械生成） | 91 行 MD / 77 行 CSV |
| `full_threearm.csv` | QuickDraw 三臂原始逐行结果（S5 全量 30ep） | 108 行 = 3 臂 × 3 种子 × 4 条件 × 3 掩码模式 |
| `nmnist_threearm.csv` | N-MNIST 三臂原始逐行结果（S10，论文原文数据集） | 108 行，同上 |
| `e2_btrain.csv` | E2 归因臂 D + A/B/C 同跑对账（S7） | 144 行 = 4 臂 × 3 种子 × 4 条件 × 3 模式 |
| `quickdraw_lrdecay.csv` | QuickDraw LR step decay 重训（S11，隔离论文 LR 调度） | 72 行 = 2 臂 × 3 种子 × 4 条件 × 3 模式 |
| `nm_negative_probes.csv` | 负结果归因两探针（S10-redesign，纯推理） | 108 行 = 4 口径 × 3 种子 × 3 条件 × 3 模式 |
| `pilot_threearm.csv` | pilot 子集结果（历史存档；结论以全量为准） | 108 行 |
| `nm_dt_response_spectrum.json` | dt 粒度响应谱（次探针：常数 dt 扫描测 CfC 饱和点 dt*） | 3 条件 × 5 个 dt |
| `d7_fidelity_analysis.json` | D7 复现保真度归因（含参照系引错 2.83pp 记账） | S9 判读落账 |
| `expected_verdicts.json` | **已判读期望值锚点**：stress_gap_test.py 的比对基准，每项带 stage/判读来源 | 冻结于 2026-11-01 |

CSV 统一 schema：`dataset,seed,arm,condition,pattern,n_eval,masked_fraction_realized,acc_meanpool,acc_final,acc_gapnb`
（`nm_negative_probes.csv` 首列多一个 `probe` 口径列）。精度为 0-1 小数；meanpool 为主口径。
每行是一个 (数据集, 种子, 臂, 条件, 掩码模式) 的独立实测——**所有聚合数字都可以从这些行重算**
（这正是 `stress_gap_test.py` 做的事）。
