# 完整复现指南（数据 + GPU 重训）

`python stress_gap_test.py` 是**零依赖秒级**的结果复现（从已落盘原始 CSV 重算头条数字与门线）。
本文是**从零重训**的完整路径：需要 QuickDraw + N-MNIST 数据、一张 8G 显卡、约 2-3 小时。
两级复现都只使用 `code/` 下原样封包的脚本（协议常数见 PAPER_PROTOCOL.md 第 3 节）。

## 0. 环境

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # torch / ncps / numpy（实测 torch 2.10 + ncps 0.0.2）
python code/repo_code_import_smoke.py                   # 导入级冒烟：7 模块 + 14 项协议常量（应 PASS）
```

> ncps 版本注意：本仓库全部数字基于 **ncps 0.0.2**。ncps 0.0.7/1.0.x 的 timespans 行为
> （默认 ones、`time_factor=sigmoid(-ff_time*timespans)`、无尺度保护）与 0.0.2 有差异，
> 换版本需重跑 clean A≡B 恒等自检（`python stress_gap_test.py --identity`）确认口径未漂移。

## 1. 数据准备（都不入本仓库，按需下载/重建）

### QuickDraw（论文同款 10 类子集）

- 来源：Google QuickDraw `.npy` 位图（公开下载，10 个类目文件）。
- 放置：`data/quickdraw/`（`pilot_threearm.py` 的 `load_quickdraw()` 会解析并缓存
  `data/quickdraw/pilot_quickdraw.pt`；首 5000 条/类、5 特征 (nx,ny,ndx,ndy,p)、max_len=100、
  85/15 seeded split，SPLIT_SEED=12345）。
- 论文同款子集类目：airplane/apple/banana/basketball/bed/bee/bicycle/bird/book/butterfly。

### N-MNIST（事件流）

- 来源：N-MNIST 原始 zip（事件流，train/test 各 10000 文件内的 70000/70000 事件文件组）。
- 字节格式（已与论文 `lln_nmnist.py:37-40` 双向坐实）：每事件 5 字节 40-bit 位域——
  `x=b0, y=b1, p=(b2>>7)&1, t=((b2&0x7f)<<16)|(b3<<8)|b4`（t 单位 µs，低位双字节大端）。
- 一条命令物化可训练张量（G: 盘空间约 6.5GB）：
  ```bash
  python code/nmnist_loader.py build --split all     # → data/nmnist/cache/{train,test}_x.f32.npy
  python code/nmnist_loader.py verify                # 分布/锚点自检（应 PASS）
  ```
  帧化口径：L=10，每样本按自身跨度 [t_min,t_max] 等分 10 个物理时 bin，`frames[bin,p,y,x]`
  计数 → log1p → (N,10,2,34,34) float32。

## 2. 按阶段重训（顺序即课题 workflow 的科学线；每阶段都有 selftest/smoke 门）

```bash
# 阶段 1（pilot，可选）：1 万样本子集 2 epoch，30 分钟见第一组数
python code/pilot_threearm.py ab --dataset quickdraw --seed 0
python code/pilot_threearm.py c  --dataset quickdraw --seed 0

# 阶段 2（结论级 QuickDraw 三臂）：全量 42500 × 30ep × 6 次（A/B 共享 3 种子 + C 3 种子）
python code/full_protocol.py selftest                # 先跑：恒等性+续跑等价+冒烟
python code/full_protocol.py all                     # 自动跳过已完成 run；逐 epoch 原子 ckpt；齐 6/6 自动评测落表

# 阶段 3（N-MNIST 三臂）：全量 60000 × 10ep × 6 次
python code/nmnist_threearm.py selftest
python code/nmnist_threearm.py all

# 阶段 4（E2 归因臂 D）：QuickDraw，3 种子 × 30ep
python code/e2_btrain.py --seed 0                    # （脚本参数见 --help；逐 seed 跑 0/1/2）

# 阶段 5（QuickDraw LR step decay 重训，隔离论文 LR 调度变量）
python code/quickdraw_lrdecay.py --seed 0            # 逐 seed 跑 0/1/2

# 阶段 6（负结果归因两探针，纯推理禁训练，需要上面的 ckpt）
python code/nm_negative_probes.py all                # smoke→main→spectrum→判据→CSV/MD/progress
```

所有阶段产物落在 `results/`（CSV/MD/progress/timing）与 `code/ckpt/`（不入 git）。
重训完成后重跑 `python stress_gap_test.py`：如果新数字与 `expected_verdicts.json` 不一致，
脚本会逐项报 FAIL——那要么是环境/协议漂移（先查 ncps 版本与数据），要么是有意义的新发现
（请附带完整输出提 issue，不要静默改锚点）。

## 3. 已知工程注意点（都是课题实测踩过的）

- **逐 epoch 原子 ckpt**：所有训练脚本每个 epoch 落盘一次 ckpt，重跑同一命令自动跳过已完成
  run 并无损续跑（跨天/跨夜中断已验证多次无损）。
- **日志追加模式**：`experiments/*.log` 为追加写；读 summary 必须取最后一个 JSON 块。
- **timing 键名**：`data_prep_sec` 曾误标为总耗时（同款 bug 两次），以运行日志分段计时为准。
- **N-MNIST 掩码广播**：掩码须 view 成 (B,L,1,1,1) 再与 (B,L,2,34,34) 广播。
- **L=10 下 block15-30 退化**：每样本 30% 概率整段被掩码，是口径特性不是 bug；该行保留作协议可比。
- **库 forward 的 timespans>1 崩溃**：ncps 0.0.2 库函数在 B>1 的 timespans 上会崩
  （输入须为 backbone 特征），因此 B/D 臂用 manual-loop 逐样本前向；clean A≡B 恒等由
  `--identity` 自检持续担保。

## 4. 预计耗时（单卡 8G，实测参考）

| 阶段 | 耗时 |
|---|---|
| QuickDraw 全量三臂 6 训练（30ep） | ~2.5-3.5h（46-58s/ep/run，跨课题 CPU 争抢时 +18%） |
| N-MNIST 三臂 6 训练（10ep） | ~11s/ep/run，合计 ~11min |
| E2 D 臂 3 训练 | ~41s/ep/run |
| S11 LR decay 3 训练 | ~40s/ep/run |
| 评测（108 格/阶段） | 65-130s |
| 探针（纯推理） | ~70s |
