#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pilot_threearm.py — S3-pilot-threearm：三臂 pilot（S1 训练 + S2 评测矩阵）。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S3-pilot-threearm（design.md §四 S1/S2、§4.0 D1-D7、§五门线；S2.result.s3_entry_conditions）

协议锁定（EV-06 逐行核验 stress_test.py + admission real_validation_path）：
  - QuickDraw：10 类×首 5000 条、5 特征(nx,ny,ndx,ndy,p)、max_len=100；85/15 random_split
  - N-MNIST：.bin→(10,2,34,34) log1p 帧（10 时 bins）
  - 模型 DrawingClassifier：quickdraw=Linear(5,128)+LN+ReLU；nmnist=CNN(2→32→64)+Linear→128；
    core=CfC(128,256,proj_size=256,batch_first=True,backbone_layers=1,backbone_units=128)；
    读表=core 输出时序 mean pooling→Linear(256,10)（论文 Table III 口径）
  - 训练：AdamW lr=1e-3 wd=0.01、grad clip 1.0、batch 128、CE；掩码只在推理期施加（论文协议）

臂语义（design.md §4.0）：
  D1 A/B 共享 checkpoint 与同一掩码后输入张量，唯一差异=timespans；
  D2 B 臂 timespans：掩码帧 dt=0、间隙后首真实帧 dt=gap_len+1、其余 dt=1（零填充保序）；
  D3 C 臂=B+可学习 gap 指示位（输入维+1）+块状间隙增强训练（单独训练，含增强混杂，门线保守）；
  D4 条件={clean, uniform30, block5-30, block15-30}×3 固定种子化模式，跨臂同掩码配对；
  D5 种子{0,1,2}，mean±std，逐种子明细；
  D7 pilot 忠实性判据=A 臂 clean ≥97%（N-MNIST）/~99% 严格复现归 S5。

读表双口径（S2 caveat：B−A 差异被收缩性衰减、集中在间隙附近时间步）：
  acc_meanpool=论文口径（全时序均值）；acc_final=末帧；acc_gapnb=各间隙后 K 帧均值（无间隙样本回退全时序均值）。

执行路径（S2.result.s3_entry_conditions）：
  - A 臂=库 forward(timespans=None)（断言①：None≡全1，B>1 不受影响）；
  - B/C 臂=manual-loop（u2_timespans_probe.py::cfc_forward_ts_b1 同款，ts 以 (B,1) 进 cell）；
  - verify 子命令在真实尺寸复验 manual-loop 与库 B=1 一致并记录 B>1 崩溃；
  - clean A≡B<0.1pp 与 A@uniform30 方向核对为强制 sanity。

数据回退（F3 式降级，如实记录）：本机无 N-MNIST 本地数据，执行期 5 通道获取均被 SSL 墙挡
（garrickorchard Cloudflare / Mendeley public-files EOF / PyPI EOF / HF EOF / GitHub 无资产，
证据见 results/fetch_assets_report.json + data/nmnist/download_report.json），QuickDraw
（同论文仓库协议、storage.googleapis.com 可达）为 pilot 降级数据集；N-MNIST 移交下一单元。

用法：
  python experiments/pilot_threearm.py verify                 # 真实尺寸复验 manual-loop + B>1 崩溃记录
  python experiments/pilot_threearm.py train --seed 0 --arm ab
  python experiments/pilot_threearm.py train --seed 0 --arm c
  python experiments/pilot_threearm.py eval                    # 评测矩阵→CSV/MD/timing
  python experiments/pilot_threearm.py all                     # verify+6 训练+eval（断点续跑）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from ncps.torch import CfC

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RESULTS = os.path.join(PROJECT, "results")
CKPT_DIR = os.path.join(HERE, "ckpt")
DATA_QD = os.path.join(PROJECT, "data", "quickdraw")
DATA_NM = os.path.join(PROJECT, "data", "nmnist", "NMNIST")
PROGRESS = os.path.join(RESULTS, "pilot_progress.json")
TIMING = os.path.join(RESULTS, "pilot_timing.json")
CSV_OUT = os.path.join(RESULTS, "pilot_threearm.csv")
MD_OUT = os.path.join(RESULTS, "pilot_threearm.md")

# ---- 协议常数（EV-06 / stress_test.py / design.md D4-D6）----
SEEDS = [0, 1, 2]
UNITS = 256
BATCH = 128
LR = 1e-3
WD = 0.01
CLIP = 1.0
EPOCHS = 2
PILOT_TRAIN_N = 10000        # D6：1 万样本子集
PILOT_EVAL_N = 3000          # pilot 评测子集（论文 test split 内 seeded 抽样）
CONDITIONS = ["clean", "uniform30", "block5-30", "block15-30"]
DROP_RATE = 0.3              # 30%
N_PATTERNS = 3               # D4：3 固定种子化掩码模式，跨臂配对
BLOCK_LENS = {"block5-30": 5, "block15-30": 15}
GAPNB_K = 5                  # 间隙邻域读表窗
CATEGORIES = ["airplane", "apple", "banana", "basketball", "bed",
              "bee", "bicycle", "bird", "book", "butterfly"]
MAX_LEN_QD = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 数据 =================

def load_quickdraw():
    """stress_test.py QuickDrawDataset 同款解析（首 5000 条/类，5 特征，max_len=100）。
    返回 (seqs Tensor[N,L,5], labels Tensor[N])；85/15 split 由 stress_test.py main 复刻。"""
    cache = os.path.join(DATA_QD, "pilot_quickdraw.pt")
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        return d["x"], d["y"]
    seqs, labels = [], []
    for idx, cat in enumerate(CATEGORIES):
        path = os.path.join(DATA_QD, f"{cat}.ndjson")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 5000:
                    break
                drawing = json.loads(line)["drawing"]
                seq, last_x, last_y = [], drawing[0][0][0], drawing[0][1][0]
                for stroke in drawing:
                    xs, ys = stroke[0], stroke[1]
                    for j in range(len(xs)):
                        nx, ny = xs[j] / 255.0, ys[j] / 255.0
                        ndx, ndy = (xs[j] - last_x) / 255.0, (ys[j] - last_y) / 255.0
                        p = 1.0 if j == len(xs) - 1 else 0.0
                        seq.append([nx, ny, ndx, ndy, p])
                        last_x, last_y = xs[j], ys[j]
                seq = torch.tensor(seq[:MAX_LEN_QD], dtype=torch.float32)
                if seq.numel() == 0:
                    seq = torch.zeros(1, 5)
                seqs.append(seq)
                labels.append(idx)
    lens = [s.shape[0] for s in seqs]
    L = max(lens)
    x = torch.zeros(len(seqs), L, 5)
    for i, s in enumerate(seqs):
        x[i, : s.shape[0]] = s
    y = torch.tensor(labels)
    torch.save({"x": x, "y": y}, cache)
    return x, y


def pilot_splits(dataset: str = "quickdraw"):
    """论文 split（85/15 seeded）+ pilot 子集（train 10k seeded / eval 3k seeded）。"""
    x, y = load_quickdraw()
    g = torch.Generator().manual_seed(12345)
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(0.85 * n)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    g2 = torch.Generator().manual_seed(999)
    tr_sub = tr_idx[torch.randperm(len(tr_idx), generator=g2)[:PILOT_TRAIN_N]]
    te_sub = te_idx[torch.randperm(len(te_idx), generator=g2)[:PILOT_EVAL_N]]
    return x[tr_sub], y[tr_sub], x[te_sub], y[te_sub]


# ================= 掩码与 timespans（D2/D4） =================

def make_mask(cond: str, B: int, L: int, seed: int, pattern: int) -> torch.Tensor:
    """返回 mask (B,L,1)：1=该帧被掩码（零填充）。
    uniform30：逐时间步 Bernoulli(0.3)（EV-06 字面）；
    block*：把 L 划分为块长 b 的块（末块截断），每块 i.i.d. Bernoulli(0.3)（期望掩码率 30%）。
    3 个 pattern=3 个种子化实现，跨臂复用同一 mask（配对比较）。"""
    g = torch.Generator().manual_seed(seed * 1000 + CONDITIONS.index(cond) * 10 + pattern)
    if cond == "clean":
        return torch.zeros(B, L, 1)
    if cond == "uniform30":
        return (torch.rand(B, L, 1, generator=g) < DROP_RATE).float()
    b = BLOCK_LENS[cond]
    mask = torch.zeros(B, L)
    nblocks = (L + b - 1) // b
    drop = (torch.rand(B, nblocks, generator=g) < DROP_RATE).float()
    for j in range(nblocks):
        mask[:, j * b:(j + 1) * b] = drop[:, j:j + 1]
    return mask.unsqueeze(2)


def d2_timespans(mask: torch.Tensor) -> torch.Tensor:
    """D2 主语义：掩码帧 dt=0；间隙后首真实帧 dt=gap_len+1；其余 dt=1。
    mask: (B,L,1) → ts: (B,L)。"""
    B, L, _ = mask.shape
    m = mask[:, :, 0]
    ts = torch.ones(B, L)
    for i in range(B):
        j = 0
        while j < L:
            if m[i, j] > 0.5:
                k = j
                while k < L and m[i, k] > 0.5:
                    ts[i, k] = 0.0
                    k += 1
                if k < L:
                    ts[i, k] = float(k - j + 1)
                j = k
            else:
                j += 1
    return ts


def gap_neighborhood_index(mask: torch.Tensor, K: int = GAPNB_K):
    """每个样本：各间隙结束后的前 K 帧索引（无间隙→None 回退全时序）。"""
    B, L, _ = mask.shape
    m = mask[:, :, 0]
    out = []
    for i in range(B):
        idx = []
        j = 0
        while j < L:
            if m[i, j] > 0.5:
                k = j
                while k < L and m[i, k] > 0.5:
                    k += 1
                idx.extend(range(k, min(k + K, L)))
                j = k
            else:
                j += 1
        out.append(idx if idx else None)
    return out


def gap_neighborhood_weights(mask: torch.Tensor, K: int = GAPNB_K) -> torch.Tensor:
    """gapnb 读表权重（B,L，float）：各间隙结束后的前 K 帧各计 1 次（跨间隙重叠按次累加）；
    无间隙样本回退全时序均匀权重。与 gap_neighborhood_index+逐样本 mean 数值等价
    （自验 experiments/eval_patch_selftest.py：hit 计数逐例相等、pooled max abs diff=0.0）。
    2026-10-26T02:1x researcher 判读补丁：原逐样本 Python 循环+.item() 同步为 GPU 利用率
    6% 的主因（单臂 12 次调用实测 265s，其中 gapnb 约占 15-21s/次）。"""
    B, L, _ = mask.shape
    m = mask[:, :, 0]
    nxt = torch.cat([m[:, 1:], torch.zeros(B, 1, device=m.device)], dim=1)
    gap_end = m * (1.0 - nxt)                       # 掩码 run 的最后一帧（1=是）
    w = torch.zeros_like(m)
    for j in range(1, K + 1):                       # 间隙后第 j 帧 +1（j=1..K）
        if j >= L:
            break
        w[:, j:] += gap_end[:, :L - j]
    empty = w.sum(dim=1, keepdim=True) == 0         # 无间隙→全时序均值回退
    w = torch.where(empty, torch.ones_like(w), w)
    return w


# ================= 模型（stress_test.py 同款） =================

class DrawingClassifier(nn.Module):
    def __init__(self, dataset_type: str, with_gap_bit: bool = False):
        super().__init__()
        self.dataset_type = dataset_type
        self.with_gap_bit = with_gap_bit
        if dataset_type == "quickdraw":
            self.backbone = nn.Sequential(nn.Linear(5, 128), nn.LayerNorm(128), nn.ReLU())
            inp = 128
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                nn.Linear(32 * 17 * 17, 128), nn.LayerNorm(128), nn.ReLU())
            inp = 128
        core_in = inp + (1 if with_gap_bit else 0)
        self.core = CfC(core_in, UNITS, proj_size=UNITS, batch_first=True,
                        backbone_layers=1, backbone_units=128)
        self.classifier = nn.Linear(UNITS, 10)

    def features(self, x):
        b, t = x.shape[0], x.shape[1]
        flat = x.view(b * t, *x.shape[2:])
        return self.backbone(flat).view(b, t, -1)

    def forward_lib(self, x):
        """A 臂：库 forward，timespans=None（均匀网格 dt=1；断言①坐实论文语义）。"""
        feats = self.features(x)
        out, _ = self.core(feats)
        return self.classifier(torch.mean(out, dim=1)), out

    def forward_ts(self, x, ts):
        """B/C 臂：manual-loop（u2 探针同款，ts 以 (B,1) 进 cell），另返回逐步输出。"""
        feats = self.features(x)
        if self.with_gap_bit:
            raise RuntimeError("gap 指示位需先拼接进 feats（见 forward_ts_gap）")
        b, l = feats.shape[0], feats.shape[1]
        h = torch.zeros((b, self.core.state_size), device=feats.device)
        outs = []
        for t in range(l):
            ts_t = ts[:, t].unsqueeze(1)
            h_out, h = self.core.rnn_cell.forward(feats[:, t], h, ts_t)
            outs.append(self.core.fc(h_out))
        out = torch.stack(outs, dim=1)
        return self.classifier(torch.mean(out, dim=1)), out

    def forward_ts_gap(self, x, ts, gapbit):
        """C 臂：feats 拼接 gap 指示位后走 manual-loop。"""
        feats = self.features(x)
        feats = torch.cat([feats, gapbit], dim=2)
        b, l = feats.shape[0], feats.shape[1]
        h = torch.zeros((b, self.core.state_size), device=feats.device)
        outs = []
        for t in range(l):
            ts_t = ts[:, t].unsqueeze(1)
            h_out, h = self.core.rnn_cell.forward(feats[:, t], h, ts_t)
            outs.append(self.core.fc(h_out))
        out = torch.stack(outs, dim=1)
        return self.classifier(torch.mean(out, dim=1)), out


# ================= 训练（S1） =================

def train_one(seed: int, arm: str, dataset: str, x_tr, y_tr):
    """arm='ab'：论文协议干净训练（A/B 共享）；arm='c'：块状间隙增强训练（D3）。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = DrawingClassifier(dataset, with_gap_bit=(arm == "c")).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss()
    n = x_tr.shape[0]
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)
        perm = torch.randperm(n, generator=g)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_tr[idx].to(DEVICE), y_tr[idx].to(DEVICE)
            opt.zero_grad()
            if arm == "ab":
                logits, _ = model.forward_lib(xb)
            else:
                # C 臂增强：块长5、每块 Bernoulli(0.3)（与 block5-30 同分布的合成间隙）
                mask = make_mask("block5-30", xb.shape[0], xb.shape[1], seed * 77 + ep, 0)
                mask = mask.to(DEVICE)
                xb = xb * (1.0 - mask)   # 2026-10-27T01:3x 修正：mask=1=掩码(清零)（原倒挂，证据同 eval 侧）
                ts = d2_timespans(mask).to(DEVICE)
                logits, _ = model.forward_ts_gap(xb, ts, mask)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
        print(f"[train seed={seed} arm={arm} ep={ep + 1}] loss={tot_loss / seen:.4f} "
              f"acc={correct / seen:.4f} ({time.time() - t0:.0f}s)", flush=True)
    # checkpoint（D6：每次训练完成即写）
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt = os.path.join(CKPT_DIR, f"{dataset}_seed{seed}_{arm}.pt")
    torch.save({"state_dict": model.state_dict(), "seed": seed, "arm": arm,
                "dataset": dataset, "units": UNITS, "with_gap_bit": arm == "c",
                "train_acc": correct / seen, "train_loss": tot_loss / seen,
                "epochs": EPOCHS, "train_n": n, "train_sec": round(time.time() - t0, 1)}, ckpt)
    return ckpt, correct / seen, tot_loss / seen, time.time() - t0


# ================= 评测（S2） =================

def eval_arm_condition(model, arm: str, x_te, y_te, cond: str, seed: int, pattern: int):
    """返回该 (seed, arm, condition, pattern) 的三读口径 acc + 实现掩码率。"""
    model.eval()
    n = x_te.shape[0]
    correct = {"meanpool": 0, "final": 0, "gapnb": 0}
    masked_frames, total_frames = 0, 0
    with torch.no_grad():
        for i in range(0, n, 256):
            xb = x_te[i:i + 256].to(DEVICE)
            yb = y_te[i:i + 256].to(DEVICE)
            mask = make_mask(cond, xb.shape[0], xb.shape[1], seed, pattern).to(DEVICE)
            xin = xb * (1.0 - mask)   # 2026-10-27T01:3x 修正：mask=1=掩码(清零)，与 D2/D4 预注册一致
                                     # （原 xb*mask 语义倒挂：clean 时输入全清零→chance；
                                     #   证据 experiments/s5_mask_convention_probe.py T1-T6）
            if arm == "A":
                logits, out = model.forward_lib(xin)
            elif arm == "B":
                ts = d2_timespans(mask).to(DEVICE)
                logits, out = model.forward_ts(xin, ts)
            else:  # C
                ts = d2_timespans(mask).to(DEVICE)
                logits, out = model.forward_ts_gap(xin, ts, mask)
            correct["meanpool"] += (logits.argmax(1) == yb).sum().item()
            # final/gapnb 读表：先过分类头再 argmax（2026-10-26T02:1x 判读修复——
            # 首派实现对 256 维隐状态直接 argmax 与 10 类标签比较，两列恒 ≈0 语义破损；
            # 隐空间池化的数值等价性不受影响，见 eval_patch_selftest.py）
            correct["final"] += (model.classifier(out[:, -1]).argmax(1) == yb).sum().item()
            # gapnb 读表：向量化权重散点（替代逐样本 Python 循环，数值等价见
            # gap_neighborhood_weights  docstring 与 eval_patch_selftest.py 自验）
            w = gap_neighborhood_weights(mask)                      # (B, L)
            pooled = (out * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)
            correct["gapnb"] += (model.classifier(pooled).argmax(1) == yb).sum().item()
            masked_frames += mask.sum().item()
            total_frames += mask.numel()
    return {k: v / n for k, v in correct.items()}, masked_frames / max(total_frames, 1)


def load_model(seed: int, arm: str, dataset: str):
    ckpt = os.path.join(CKPT_DIR, f"{dataset}_seed{seed}_{arm}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = DrawingClassifier(dataset, with_gap_bit=payload.get("with_gap_bit", arm == "c"))
    model.load_state_dict(payload["state_dict"])
    return model.to(DEVICE).eval(), payload


# ================= verify（S2 入口条件④：真实尺寸复验） =================

def verify(dataset: str, x_te) -> dict:
    torch.manual_seed(0)
    model = DrawingClassifier(dataset, with_gap_bit=False).to(DEVICE).eval()
    xb = x_te[:4].to(DEVICE)
    mask = make_mask("block5-30", 4, xb.shape[1], 0, 0).to(DEVICE)
    xin = xb * (1.0 - mask)   # 2026-10-27T02:1x 判读补修：verify 内同款倒挂遗留（identity 类对比 convention 无关，指标不变）
    ts = d2_timespans(mask).to(DEVICE)
    rep = {"real_size": {"batch": 4, "L": xb.shape[1], "units": UNITS}}
    with torch.no_grad():
        # B>1 库函数崩溃记录（输入须为 backbone 特征，与库 forward 语义一致）
        feats_all = model.features(xin)
        crash = None
        try:
            model.core(feats_all, timespans=ts)
        except RuntimeError as exc:
            crash = f"{type(exc).__name__}: {exc}"
        rep["library_B_gt_1_timespans_crash"] = crash
        # manual-loop vs 库 B=1 逐样本一致
        max_d = 0.0
        for r in range(xb.shape[0]):
            xin_r = xb[r:r + 1] * (1.0 - mask[r:r + 1])   # 2026-10-27T02:1x 判读补修：同上（manual-loop vs 库 identity 对比，convention 无关）
            feats_r = model.features(xin_r)
            out_lib, _ = model.core(feats_r, timespans=ts[r:r + 1])
            _, out_man = model.forward_ts(xin_r, ts[r:r + 1])
            max_d = max(max_d, (out_lib.double() - out_man.double()).abs().max().item())
        rep["manual_loop_vs_library_B1_max_abs_diff"] = max_d
        # clean A≡B 数值一致（logits 级）
        _, out_a = model.forward_lib(xb)
        ts_clean = torch.ones(xb.shape[0], xb.shape[1], device=DEVICE)
        _, out_b = model.forward_ts(xb, ts_clean)
        rep["clean_armB_vs_armA_output_max_abs_diff"] = (out_a.double() - out_b.double()).abs().max().item()
    rep["verify_pass"] = (rep["library_B_gt_1_timespans_crash"] is not None
                          and rep["manual_loop_vs_library_B1_max_abs_diff"] < 1e-4
                          and rep["clean_armB_vs_armA_output_max_abs_diff"] < 1e-4)
    return rep


# ================= progress / timing =================

def load_progress() -> dict:
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as fh:
            return json.load(fh)
    return {"runs": {}, "eval": [], "sanity": {}, "verdict": None}


def save_progress(p: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(p, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, PROGRESS)          # 原子替换：kill 不留半截 JSON（断点续跑前提）


def save_timing(p: dict, extra: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    payload = {"schema": "pilot-timing/v1", "stage": "S3-pilot-threearm", "u5_runtime": extra}
    with open(TIMING, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ================= 汇总（CSV/MD + 门线） =================

def summarize(rows: list[dict], dataset: str) -> dict:
    """rows: eval 行（含 arm/condition/pattern/seed 与三读口径 acc）。"""
    import statistics as st

    def pick(arm, cond, key="acc_meanpool"):
        vals = [r[key] for r in rows if r["arm"] == arm and r["condition"] == cond]
        return vals

    summary = {"conditions": {}, "sanity": {}, "verdict": {}}
    for cond in CONDITIONS:
        entry = {}
        for arm in ["A", "B", "C"]:
            vals = pick(arm, cond)
            if vals:
                entry[arm] = {"mean": st.mean(vals), "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
                              "n": len(vals), "per_seed": vals}
        if "A" in entry and "B" in entry:
            entry["B_minus_A"] = entry["B"]["mean"] - entry["A"]["mean"]
        if "A" in entry and "C" in entry:
            entry["C_minus_A"] = entry["C"]["mean"] - entry["A"]["mean"]
        summary["conditions"][cond] = entry

    a_clean = summary["conditions"]["clean"].get("A", {}).get("mean")
    b_clean = summary["conditions"]["clean"].get("B", {}).get("mean")
    c_clean = summary["conditions"]["clean"].get("C", {}).get("mean")
    summary["sanity"]["clean_A_mean"] = a_clean
    summary["sanity"]["clean_B_mean"] = b_clean
    summary["sanity"]["clean_A_minus_B"] = (a_clean - b_clean) if (a_clean is not None and b_clean is not None) else None
    summary["sanity"]["clean_A_equals_B_lt_0p1pp"] = (
        summary["sanity"]["clean_A_minus_B"] is not None
        # 2026-10-26T03:1x researcher 判读修正：acc 为分数单位，design.md 预注册判据
        # 「数值差 <0.1pp」= 分数 <0.001；原写 <0.1 实为 10pp，宽 100 倍会误放行
        and abs(summary["sanity"]["clean_A_minus_B"]) < 0.001)
    a_u30 = summary["conditions"]["uniform30"].get("A", {}).get("mean")
    summary["sanity"]["A_uniform30_mean"] = a_u30
    # design.md §五预注册判据="LNN@30% 显著高于随机水平，pilot 预算内只做方向性核对"。
    # 2026-10-26T03:1x researcher 判读修正：原实现 >=0.60 与预注册判据不符（60% 接近
    # 论文 clean 数量级，pilot 预算内不可能达到，会把方向 PASS 误报为 FAIL）→ 改为对
    # chance=0.10 的单边二项 z 检验（z>2 即显著高于随机），n 取 A 臂 uniform30 行。
    n_eval_u30 = max([r.get("n_eval", 0) for r in rows
                      if r["arm"] == "A" and r["condition"] == "uniform30"] or [0])
    if a_u30 is not None and n_eval_u30 > 0:
        z_u30 = (a_u30 - 0.10) / (0.10 * 0.90 / n_eval_u30) ** 0.5
        summary["sanity"]["A_uniform30_z_vs_chance"] = round(z_u30, 2)
        summary["sanity"]["A_uniform30_direction_ok"] = z_u30 > 2.0
    else:
        summary["sanity"]["A_uniform30_z_vs_chance"] = None
        summary["sanity"]["A_uniform30_direction_ok"] = None
    if a_clean is not None:
        summary["sanity"]["A_clean_fidelity_97"] = a_clean >= 0.97
        summary["sanity"]["A_clean_fidelity_99"] = a_clean >= 0.99

    b5 = summary["conditions"]["block5-30"]
    b15 = summary["conditions"]["block15-30"]
    b_minus_a = b5.get("B_minus_A")
    c_minus_a = b5.get("C_minus_A")
    summary["verdict"]["B_minus_A_block5"] = b_minus_a
    summary["verdict"]["C_minus_A_block5"] = c_minus_a
    summary["verdict"]["B_clean_drop_pp"] = (b_clean - a_clean) * 100 if (b_clean is not None and a_clean is not None) else None
    summary["verdict"]["C_clean_drop_pp"] = (c_clean - a_clean) * 100 if (c_clean is not None and a_clean is not None) else None
    summary["verdict"]["H2_block15_vs_block5"] = (
        (b15.get("B_minus_A", 0) - b5.get("B_minus_A", 0)) if b15.get("B_minus_A") is not None else None)
    # 门线（design.md §五）
    gate_arm = None
    if b_minus_a is not None and b_minus_a >= 0.02 and summary["verdict"]["B_clean_drop_pp"] is not None \
            and abs(summary["verdict"]["B_clean_drop_pp"]) <= 0.5:
        gate_arm = "B"
    if c_minus_a is not None and c_minus_a >= 0.02 and summary["verdict"]["C_clean_drop_pp"] is not None \
            and abs(summary["verdict"]["C_clean_drop_pp"]) <= 0.5:
        gate_arm = gate_arm or "C"
    summary["verdict"]["gate_pass_arm"] = gate_arm
    if gate_arm:
        summary["verdict"]["decision"] = "pass"
        summary["verdict"]["next_action"] = "replicate"
    elif b_minus_a is not None and b_minus_a < 0.01 and summary["verdict"]["B_clean_drop_pp"] is not None \
            and summary["verdict"]["B_clean_drop_pp"] >= -0.5:
        summary["verdict"]["decision"] = "falsified_negative_branch"
        summary["verdict"]["next_action"] = "redesign"
    else:
        summary["verdict"]["decision"] = "inconclusive"
        summary["verdict"]["next_action"] = "continue"
    return summary


def write_csv_md(rows: list[dict], summary: dict, dataset: str, verify_rep: dict, timing: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    cols = ["dataset", "seed", "arm", "condition", "pattern", "n_eval",
            "masked_fraction_realized", "acc_meanpool", "acc_final", "acc_gapnb"]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    def fmt(entry, arm):
        e = entry.get(arm)
        return f"{e['mean'] * 100:.2f}±{e['std'] * 100:.2f}" if e else "—"

    lines = []
    lines.append("# S3-pilot-threearm 三臂 pilot 结果（% acc，3 种子 mean±std，逐种子明细见 CSV）\n")
    lines.append(f"- 数据集：{dataset}（F3 式降级：N-MNIST 数据通道被 SSL 墙挡，证据见 "
                 f"results/fetch_assets_report.json、data/nmnist/download_report.json；N-MNIST 移交下一单元）")
    lines.append(f"- 协议：stress_test.py 逐字同款（EV-06）；pilot 子集 train 10k×{EPOCHS} epoch、eval {PILOT_EVAL_N}；"
                 f"读表三口径=meanpool(论文)/final/gapnb(间隙邻域K={GAPNB_K})")
    lines.append(f"- 臂：A=库 forward(timespans=None,均匀dt=1)；B=manual-loop+D2 真实 timespans（与 A 共享 checkpoint/掩码）；"
                 f"C=B+gap 指示位+块状增强训练（单独 checkpoint）")
    lines.append(f"- 条件：clean / uniform30(逐时间步 Bernoulli 0.3) / block5-30、block15-30（块长 b 划分、每块 Bernoulli 0.3，"
                 f"期望掩码率 30%；3 固定种子化模式跨臂配对）\n")
    lines.append("| 条件 | A | B | C | B−A(pp) | C−A(pp) |")
    lines.append("|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        e = summary["conditions"][cond]
        bma = f"{e['B_minus_A'] * 100:+.2f}" if e.get("B_minus_A") is not None else "—"
        cma = f"{e['C_minus_A'] * 100:+.2f}" if e.get("C_minus_A") is not None else "—"
        lines.append(f"| {cond} | {fmt(e, 'A')} | {fmt(e, 'B')} | {fmt(e, 'C')} | {bma} | {cma} |")
    lines.append("")
    s = summary["sanity"]
    lines.append("## 强制 sanity（S2 入口条件）")
    lines.append(f"- clean A≡B：|A−B|={s['clean_A_minus_B'] * 100:.3f}pp → {'PASS(<0.1pp)' if s['clean_A_equals_B_lt_0p1pp'] else 'FAIL'}"
                 if s["clean_A_minus_B"] is not None else "- clean A≡B：无数据")
    lines.append(f"- A@uniform30={s['A_uniform30_mean'] * 100:.2f}% → 方向核对 "
                 f"{'PASS(z=' + str(s.get('A_uniform30_z_vs_chance')) + '>2 vs chance)' if s['A_uniform30_direction_ok'] else '未显著高于随机水平'}"
                 if s["A_uniform30_mean"] is not None else "- A@uniform30：无数据")
    lines.append(f"- A@clean={s['clean_A_mean'] * 100:.2f}% → D7 pilot 忠实性(≥97%) {'PASS' if s['A_clean_fidelity_97'] else '未达（按 D7 查实现，不算方向失败）'}"
                 if s["clean_A_mean"] is not None else "- A@clean：无数据")
    lines.append(f"- manual-loop 真实尺寸复验：verify_pass={verify_rep.get('verify_pass')}，"
                 f"vs 库 B=1 max diff={verify_rep.get('manual_loop_vs_library_B1_max_abs_diff')}")
    lines.append("")
    v = summary["verdict"]
    lines.append("## 门线与假设判读（design.md §五）")
    lines.append(f"- H1（block5-30 下 B 或 C ≥+2pp 且该臂 clean 降幅 ≤0.5pp）：B−A={v['B_minus_A_block5'] * 100:+.2f}pp、"
                 f"C−A={v['C_minus_A_block5'] * 100:+.2f}pp；B clean 降幅={v['B_clean_drop_pp']:+.2f}pp、"
                 f"C clean 降幅={v['C_clean_drop_pp']:+.2f}pp → gate_pass_arm={v['gate_pass_arm']}")
    lines.append(f"- H2（block15 增益 ≥ block5 增益，mean 意义）：Δ={v['H2_block15_vs_block5'] * 100:+.2f}pp"
                 if v["H2_block15_vs_block5"] is not None else "- H2：无数据")
    lines.append(f"- 判读：decision={v['decision']}，next_action={v['next_action']}")
    lines.append("")
    lines.append("## 限制")
    lines.append("- pilot 为内部方向性结果（1 万样本 1-2 epoch、3 种子、评测子集 3000）；结论级归 S5 全量协议")
    lines.append("- C−A 含增强训练混杂（D3 预注册：门线对 C 为保守存活判据，归因细化归 S5）")
    lines.append("- QuickDraw L=100（block15 为字面块长）；N-MNIST L=10 时 block15 将退化为整段掩码，故 pilot 数据集选择本身影响 H2 可检验性")
    lines.append(f"- 计时（U5）：{json.dumps(timing, ensure_ascii=False)}")
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ================= 评测矩阵（断点续跑，2026-10-26T02:1x researcher 判读补丁） =================

EVAL_TOTAL = len(SEEDS) * 3 * len(CONDITIONS) * N_PATTERNS   # 108 行


def run_eval_matrix(args, progress: dict, x_te, y_te):
    """三臂评测矩阵：逐行落盘（progress["eval_rows"]）+ 已完成 (seed,arm,cond,pattern) 跳过。
    返回 (rows, eval_sec)。kill/超后重跑同一命令即续跑，不重算已出行。
    注意：CSV/MD 仅在 108 行齐全后写出（由调用方检查），半程不落表。"""
    rows = list(progress.get("eval_rows", []))
    done = {(r.get("dataset"), r.get("seed"), r.get("arm"), r.get("condition"), r.get("pattern"))
            for r in rows}
    t_eval = time.time()
    for seed in SEEDS:
        models = {a: load_model(seed, a, args.dataset)[0] for a in ["ab", "c"]}
        for arm_name, arm_key in [("A", "ab"), ("B", "ab"), ("C", "c")]:
            for cond in CONDITIONS:
                for pattern in range(N_PATTERNS):
                    key = (args.dataset, seed, arm_name, cond, pattern)
                    if key in done:
                        continue
                    accs, mf = eval_arm_condition(models[arm_key], arm_name, x_te, y_te, cond, seed, pattern)
                    rows.append({"dataset": args.dataset, "seed": seed, "arm": arm_name,
                                 "condition": cond, "pattern": pattern, "n_eval": x_te.shape[0],
                                 "masked_fraction_realized": round(mf, 4),
                                 "acc_meanpool": round(accs["meanpool"], 4),
                                 "acc_final": round(accs["final"], 4),
                                 "acc_gapnb": round(accs["gapnb"], 4)})
                    progress["eval_rows"] = rows
                    save_progress(progress)      # 逐行持久化：超时/kill 不丢已出行
            print(f"[eval] seed={seed} arm={arm_name} done "
                  f"({time.time() - t_eval:.0f}s cum, {len(rows)}/{EVAL_TOTAL} rows)", flush=True)
    return rows, round(time.time() - t_eval, 1)


# ================= 主流程 =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "train", "eval", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["ab", "c"], default="ab")
    ap.add_argument("--dataset", default="quickdraw")
    args = ap.parse_args()

    t_start = time.time()
    x_tr, y_tr, x_te, y_te = pilot_splits(args.dataset)
    progress = load_progress()
    timing = {"data_prep_sec": round(time.time() - t_start, 1), "train_runs": {}, "eval_sec": None,
              "total_sec": None}

    if args.cmd == "verify":
        rep = verify(args.dataset, x_te)
        progress["sanity"]["verify_real_size"] = rep
        save_progress(progress)
        print(json.dumps(rep, ensure_ascii=False))
        return 0

    if args.cmd == "train":
        key = f"seed{args.seed}_{args.arm}"
        if key in progress["runs"] and os.path.exists(os.path.join(CKPT_DIR, f"{args.dataset}_{key}.pt")):
            print(f"skip {key} (checkpoint exists)")
            return 0
        ckpt, acc, loss, sec = train_one(args.seed, args.arm, args.dataset, x_tr, y_tr)
        progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT), "train_acc": round(acc, 4),
                                 "train_loss": round(loss, 4), "sec": round(sec, 1)}
        save_progress(progress)
        return 0

    if args.cmd == "all":
        # verify（入口条件④）
        if "verify_real_size" not in progress["sanity"]:
            progress["sanity"]["verify_real_size"] = verify(args.dataset, x_te)
            save_progress(progress)
            print("[all] verify:", json.dumps(progress["sanity"]["verify_real_size"], ensure_ascii=False), flush=True)
        # S1：6 次训练（A/B 共享 3 种子 + C-aug 3 种子），每次 checkpoint
        for seed in SEEDS:
            for arm in ["ab", "c"]:
                key = f"seed{seed}_{arm}"
                ckpt_path = os.path.join(CKPT_DIR, f"{args.dataset}_{key}.pt")
                if key in progress["runs"] and os.path.exists(ckpt_path):
                    print(f"[all] skip {key}", flush=True)
                    continue
                ckpt, acc, loss, sec = train_one(seed, arm, args.dataset, x_tr, y_tr)
                progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT), "train_acc": round(acc, 4),
                                         "train_loss": round(loss, 4), "sec": round(sec, 1)}
                save_progress(progress)
        # S2：评测矩阵（断点续跑：逐行落盘，已完成行跳过）
        rows, eval_sec = run_eval_matrix(args, progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"], CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        if len(rows) < EVAL_TOTAL:
            print(f"[all] eval incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            return 0
        summary = summarize(rows, args.dataset)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(progress, timing)
        write_csv_md(rows, summary, args.dataset, progress["sanity"].get("verify_real_size", {}), timing)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "eval":
        rows, eval_sec = run_eval_matrix(args, progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"], CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        if len(rows) < EVAL_TOTAL:
            print(f"[eval] incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            return 0
        summary = summarize(rows, args.dataset)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(progress, timing)
        write_csv_md(rows, summary, args.dataset, progress["sanity"].get("verify_real_size", {}), timing)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
