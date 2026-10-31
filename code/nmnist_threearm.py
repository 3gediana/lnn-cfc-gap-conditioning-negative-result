#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nmnist_threearm.py — S10-nmnist-threearm-baseline：N-MNIST 三臂结论级基准（H1 原文数据集复演）。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S10-nmnist-threearm-baseline（design.md §四 D1-D7、§五门线；S8/S9 落账后放行）

与既有代码的关系（import 不复制，禁改 A/B/C QuickDraw 既有路径，e2_btrain.py/full_protocol.py 先例）：
  - 模型/掩码/timespans/读表口径/臂语义全部 import 自 experiments/pilot_threearm.py（P）；
  - 向量化 d2_timespans（full_protocol.d2_timespans_vec，selftest 68 项恒等担保）直接调用；
  - 数据 import 自 experiments/nmnist_loader.py load_split（S8 口径钉死，data/nmnist/cache/ 已物化）。

协议常数（论文 N-MNIST，子步1 取证，双记）：
  - epoch=10：experiments/_repo/run_lln_nmnist.sh 明文 `--epochs 10`，lln_nmnist_10.log 恰好
    10 个 epoch 行、终值 TRAIN 0.9997 / TEST 0.9938（与论文 Table I 逐位一致）→ 采信 10ep；
    论文 Table V 记 30ep=论文内部矛盾，双记入报告 caveat（裁决规则见 workflow S10 actions）。
  - 论文其余超参（batch64/no-wd/no-clip/units128/Cosine）与本仓冻结规格（batch128/wd0.01/
    clip1.0/units256/恒定 lr1e-3）的差异全部入报告 caveat，不改冻结模型规格与冻结判据
    （H1 主判据 B−A 为同 ckpt 推理期配对，不受影响；D7 缺口>3pp 记 caveat 交归因，
    归因第一嫌疑=协议差，禁实现缺陷归因）。

N-MNIST 适配点（results/nmnist_loader_check.md §6 集成注意点）：
  - 输入 (B,L,2,34,34)；掩码 (B,L,1) 须 view 成 (B,L,1,1,1) 才能广播（QuickDraw (B,L,5)
    下 (B,L,1) 即可）→ 本文件 train/eval/verify 全部走 mask4 = mask.view(B,L,1,1,1)；
  - C 臂 with_gap_bit=True；L=10 下 block15-30 退化为整段掩码（已知口径特性非 bug，行保留
    作协议可比+显式标注，H2 不在本阶段 adjudicate）。

训练（子步4）：6 runs=3 种子×(ab 共享+c)，10 epoch，逐 epoch 原子 ckpt
  experiments/ckpt/nmnist_seed{seed}_{ab,c}.pt + results/nmnist_progress.json 记账。
评测（子步5）：3 种子×3 臂×4 条件×3 模式=108 行，全量 test 10000（n_eval=10000），
  读表三口径（meanpool 预注册主口径+gapnb+final），逐行原子落盘+断点跳过。

判据（跑前冻结、跑完机械判定、禁事后改）：
  ①H1 主判据 block5-30 B−A≥+2pp(3 种子 mean) 且 clean |B−A|≤0.5pp→supported；
    B−A<+1pp 且 clean 不受损(B_clean_drop≥−0.5pp)→falsified 负结果落盘；其间→inconclusive；
  ②clean A≡B<0.1pp 语义 sanity（D1 同输入）；
  ③C 臂保守门（D3）：C−A 幅值+clean 降幅逐种子双报告；
  ④D7=A@clean vs 论文 0.9938，缺口>3pp 记 caveat 交归因不单独杀阶段；
  ⑤A@uniform30 z>2 vs chance 方向核对；
  ⑥收敛伴生=clean 三口径分化（非恒定类）。

用法：
  python experiments/nmnist_threearm.py selftest            # 机制自检（广播/掩码率/ts梯度/A≡B/续跑等价）
  python experiments/nmnist_threearm.py smoke               # seed0 ab 3ep 冒烟（实测 s/ep+四门）
  python experiments/nmnist_threearm.py train --seed 0 --arm ab
  python experiments/nmnist_threearm.py eval                # 108 行矩阵，逐行落盘续跑
  python experiments/nmnist_threearm.py all                 # verify+6 训练+eval+CSV/MD（全程断点续跑）
  python experiments/nmnist_threearm.py status              # 进度快照
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJECT = os.path.dirname(HERE)

import pilot_threearm as P          # noqa: E402  （协议同源代码，import 不复制）
import full_protocol as F           # noqa: E402  （向量化 d2_timespans，selftest 恒等担保）
from nmnist_loader import load_split  # noqa: E402  （S8 loader，口径钉死）

RESULTS = os.path.join(PROJECT, "results")
CKPT_DIR = P.CKPT_DIR
PROGRESS = os.path.join(RESULTS, "nmnist_progress.json")
TIMING = os.path.join(RESULTS, "nmnist_timing.json")
CSV_OUT = os.path.join(RESULTS, "nmnist_threearm.csv")
MD_OUT = os.path.join(RESULTS, "nmnist_threearm.md")

# ---- 协议常数 ----
DATASET = "nmnist"
EPOCHS_NM = 10                 # 论文 N-MNIST：run_lln_nmnist.sh --epochs 10 + log 10 行 + Table I 0.9938
PAPER_TEST_ACC = 0.9938        # 论文 Table I N-MNIST LNN Test（=lln_nmnist_10.log 终值，双源一致）
PAPER_TEST_ACC_TABLE3 = 0.9871 # 论文 Table III N-MNIST 0% 列 LNN 98.71（论文内部口径差，兼报）
SMOKE_EPOCHS = 3               # 子步3：seed0 单臂 2-3ep 冒烟
EVAL_BATCH = 256
DEVICE = P.DEVICE
CONDITIONS = P.CONDITIONS
SEEDS = P.SEEDS
N_PATTERNS = P.N_PATTERNS
EVAL_TOTAL = len(SEEDS) * 3 * len(CONDITIONS) * N_PATTERNS   # 108 行
D7_GAP_CAVEAT_PP = 3.0         # 判据④：缺口>3pp 记 caveat 交归因


# ================= 数据 =================

def nm_splits(ram_test: bool = True, ram_train: bool = True):
    """load_split 读缓存张量：train (60000,10,2,34,34) f32 + y；test 同款。
    train/test 均载入 RAM（5.55GB+0.92GB，机器 33GB）：实测 memmap 逐 batch 随机读
    （128×90KB/batch×469 batch/ep）在 G: 盘上 420s 未完成 1 epoch=IO 瓶颈（前台冒烟实证），
    一次性顺序读后所有 epoch 走 RAM 速度。断点续跑不依赖缓存（读盘复制，秒级完成）。"""
    x_tr, y_tr = load_split("train")
    x_te, y_te = load_split("test")
    if ram_train:
        x_tr = np.array(x_tr)          # 5.55GB RAM：消除逐 batch 随机读 IO 瓶颈
    if ram_test:
        x_te = np.array(x_te)          # 0.92GB RAM：108 行全量 test 反复读
    return x_tr, y_tr, x_te, y_te


def _to_torch(x_np, device=DEVICE):
    return torch.from_numpy(np.asarray(x_np)).to(device)


# ================= 掩码 view 适配（N-MNIST 专用，禁改 pilot 路径） =================

def mask4(mask: torch.Tensor) -> torch.Tensor:
    """(B,L,1) → (B,L,1,1,1)，使 (B,L,2,34,34) 输入可广播（nmnist_loader_check.md §6）。"""
    B, L, _ = mask.shape
    return mask.view(B, L, 1, 1, 1)


# ================= 训练（逐 epoch 原子 ckpt + 断点续跑；full_protocol 同款模式） =================

def _atomic_torch_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # 原子替换：kill 不留半截 ckpt


def train_one_nm(seed: int, arm: str, x_tr, y_tr,
                 epochs: int = EPOCHS_NM, log=print):
    """arm='ab'：论文协议干净训练（A/B 共享）；arm='c'：block5-30 增强+gap 指示位（D3）。
    与 full_protocol.train_one_full 逐行同源；差异仅：数据=nmnist loader、epochs=10、
    ckpt 名 nmnist_seed{seed}_{arm}.pt、掩码 view (B,L,1,1,1) 广播。
    返回 (ckpt_path, final_acc, final_loss, sec, epochs_done)。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = P.DrawingClassifier(DATASET, with_gap_bit=(arm == "c")).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=P.LR, weight_decay=P.WD)
    crit = nn.CrossEntropyLoss()
    ckpt = os.path.join(CKPT_DIR, f"nmnist_seed{seed}_{arm}.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)

    start_ep, hist = 0, []
    if os.path.exists(ckpt):                       # 断点续跑：模型+优化器+epoch 全恢复
        payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        opt.load_state_dict(payload["opt_state_dict"])
        start_ep = int(payload.get("epochs_done", 0))
        hist = list(payload.get("history", []))
        if start_ep >= epochs:
            log(f"[nm-train] skip seed={seed} arm={arm} (done {start_ep}/{epochs})")
            return ckpt, hist[-1]["acc"], hist[-1]["loss"], 0.0, start_ep
        log(f"[nm-train] resume seed={seed} arm={arm} from epoch {start_ep}")

    n = x_tr.shape[0]
    t0 = time.time()
    last_acc = last_loss = float("nan")
    for ep in range(start_ep, epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)   # 与 pilot/full 同：逐 epoch 固定 shuffle
        perm = torch.randperm(n, generator=g)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, P.BATCH):
            idx = perm[i:i + P.BATCH]
            xb, yb = _to_torch(x_tr[idx]), torch.from_numpy(np.asarray(y_tr[idx])).to(DEVICE)
            opt.zero_grad()
            if arm == "ab":
                logits, _ = model.forward_lib(xb)
            else:
                # C 臂增强：块长5、每块 Bernoulli(0.3)（与 block5-30 同分布的合成间隙）
                mask = P.make_mask("block5-30", xb.shape[0], xb.shape[1], seed * 77 + ep, 0)
                mask = mask.to(DEVICE)
                xb = xb * (1.0 - mask4(mask))   # mask=1=掩码(清零)（2026-10-27 Convention 修正同款）
                ts = F.d2_timespans_vec(mask).to(DEVICE)
                logits, _ = model.forward_ts_gap(xb, ts, mask)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), P.CLIP)
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
        last_acc, last_loss = correct / seen, tot_loss / seen
        hist.append({"epoch": ep + 1, "loss": round(last_loss, 4), "acc": round(last_acc, 4),
                     "sec": round(time.time() - t0, 1)})
        _atomic_torch_save({                         # 逐 epoch checkpoint（续跑前提）
            "state_dict": model.state_dict(), "opt_state_dict": opt.state_dict(),
            "seed": seed, "arm": arm, "dataset": DATASET, "units": P.UNITS,
            "with_gap_bit": arm == "c", "epochs_done": ep + 1, "epochs_target": epochs,
            "train_n": n, "train_acc": last_acc, "train_loss": last_loss,
            "train_sec": round(time.time() - t0, 1), "history": hist}, ckpt)
        log(f"[nm-train seed={seed} arm={arm} ep={ep + 1}/{epochs}] "
            f"loss={last_loss:.4f} acc={last_acc:.4f} ({time.time() - t0:.0f}s)")
    return ckpt, last_acc, last_loss, time.time() - t0, epochs


def load_model_nm(seed: int, arm: str):
    ckpt = os.path.join(CKPT_DIR, f"nmnist_seed{seed}_{arm}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = P.DrawingClassifier(DATASET, with_gap_bit=payload.get("with_gap_bit", arm == "c"))
    model.load_state_dict(payload["state_dict"])
    return model.to(DEVICE).eval(), payload


# ================= 评测（N-MNIST 版 eval_arm_condition：mask4 广播；三口径） =================

def eval_arm_condition_nm(model, arm: str, x_te, y_te, cond: str, seed: int, pattern: int):
    """与 pilot_threearm.eval_arm_condition 逐行同源；唯一差异=掩码 view (B,L,1,1,1)
    适配 (B,L,2,34,34) 输入。返回该 (seed,arm,cond,pattern) 三读口径 acc + 实现掩码率。"""
    model.eval()
    n = x_te.shape[0]
    correct = {"meanpool": 0, "final": 0, "gapnb": 0}
    masked_frames, total_frames = 0, 0
    with torch.no_grad():
        for i in range(0, n, EVAL_BATCH):
            xb = _to_torch(x_te[i:i + EVAL_BATCH])
            yb = torch.from_numpy(np.asarray(y_te[i:i + EVAL_BATCH])).to(DEVICE)
            mask = P.make_mask(cond, xb.shape[0], xb.shape[1], seed, pattern).to(DEVICE)
            xin = xb * (1.0 - mask4(mask))   # mask=1=掩码(清零)，与 D2/D4 预注册一致
            if arm == "A":
                logits, out = model.forward_lib(xin)
            elif arm == "B":
                ts = F.d2_timespans_vec(mask).to(DEVICE)
                logits, out = model.forward_ts(xin, ts)
            else:  # C
                ts = F.d2_timespans_vec(mask).to(DEVICE)
                logits, out = model.forward_ts_gap(xin, ts, mask)
            correct["meanpool"] += (logits.argmax(1) == yb).sum().item()
            # final/gapnb 读表：先过分类头再 argmax（2026-10-26T02:1x 判读修复同款）
            correct["final"] += (model.classifier(out[:, -1]).argmax(1) == yb).sum().item()
            w = P.gap_neighborhood_weights(mask)          # (B, L)
            pooled = (out * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)
            correct["gapnb"] += (model.classifier(pooled).argmax(1) == yb).sum().item()
            masked_frames += mask.sum().item()
            total_frames += mask.numel()
    return {k: v / n for k, v in correct.items()}, masked_frames / max(total_frames, 1)


def run_eval_matrix_nm(progress: dict, x_te, y_te, log=print):
    """三臂评测矩阵：逐行落盘（progress["eval_rows"]）+ 已完成 (seed,arm,cond,pattern) 跳过。
    返回 (rows, eval_sec)。kill/超后重跑同一命令即续跑，不重算已出行。"""
    rows = list(progress.get("eval_rows", []))
    done = {(r.get("dataset"), r.get("seed"), r.get("arm"), r.get("condition"), r.get("pattern"))
            for r in rows}
    t_eval = time.time()
    for seed in SEEDS:
        models = {a: load_model_nm(seed, a)[0] for a in ["ab", "c"]}
        for arm_name, arm_key in [("A", "ab"), ("B", "ab"), ("C", "c")]:
            for cond in CONDITIONS:
                for pattern in range(N_PATTERNS):
                    key = (DATASET, seed, arm_name, cond, pattern)
                    if key in done:
                        continue
                    accs, mf = eval_arm_condition_nm(models[arm_key], arm_name,
                                                     x_te, y_te, cond, seed, pattern)
                    rows.append({"dataset": DATASET, "seed": seed, "arm": arm_name,
                                 "condition": cond, "pattern": pattern, "n_eval": x_te.shape[0],
                                 "masked_fraction_realized": round(mf, 4),
                                 "acc_meanpool": round(accs["meanpool"], 4),
                                 "acc_final": round(accs["final"], 4),
                                 "acc_gapnb": round(accs["gapnb"], 4)})
                    progress["eval_rows"] = rows
                    save_progress(progress)      # 逐行持久化：超时/kill 不丢已出行
            log(f"[nm-eval] seed={seed} arm={arm_name} done "
                f"({time.time() - t_eval:.0f}s cum, {len(rows)}/{EVAL_TOTAL} rows)")
    return rows, round(time.time() - t_eval, 1)


# ================= progress / timing =================

def load_progress() -> dict:
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as fh:
            return json.load(fh)
    return {"runs": {}, "eval_rows": [], "sanity": {}, "verdict": None}


def save_progress(p: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(p, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, PROGRESS)          # 原子替换：kill 不留半截 JSON


def save_timing(extra: dict) -> None:
    payload = {"schema": "nmnist-timing/v1", "stage": "S10-nmnist-threearm-baseline",
               "u5_runtime": extra}
    os.makedirs(RESULTS, exist_ok=True)
    with open(TIMING, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ================= verify（真实尺寸复验：manual-loop vs 库 B=1 + B>1 崩溃 + clean A≡B） =================

def verify_nm(x_te) -> dict:
    """与 pilot_threearm.verify 逐行同源；差异=掩码 view 适配 N-MNIST 形状。"""
    torch.manual_seed(0)
    model = P.DrawingClassifier(DATASET, with_gap_bit=False).to(DEVICE).eval()
    xb = _to_torch(x_te[:4])
    mask = P.make_mask("block5-30", 4, xb.shape[1], 0, 0).to(DEVICE)
    xin = xb * (1.0 - mask4(mask))
    ts = F.d2_timespans_vec(mask).to(DEVICE)
    rep = {"real_size": {"batch": 4, "L": xb.shape[1], "units": P.UNITS,
                         "input_shape": list(xb.shape)}}
    with torch.no_grad():
        feats_all = model.features(xin)
        crash = None
        try:
            model.core(feats_all, timespans=ts)
        except RuntimeError as exc:
            crash = f"{type(exc).__name__}: {exc}"
        rep["library_B_gt_1_timespans_crash"] = crash
        max_d = 0.0
        for r in range(xb.shape[0]):
            xin_r = xb[r:r + 1] * (1.0 - mask4(mask[r:r + 1]))
            feats_r = model.features(xin_r)
            out_lib, _ = model.core(feats_r, timespans=ts[r:r + 1])
            _, out_man = model.forward_ts(xin_r, ts[r:r + 1])
            max_d = max(max_d, (out_lib.double() - out_man.double()).abs().max().item())
        rep["manual_loop_vs_library_B1_max_abs_diff"] = max_d
        _, out_a = model.forward_lib(xb)
        ts_clean = torch.ones(xb.shape[0], xb.shape[1], device=DEVICE)
        _, out_b = model.forward_ts(xb, ts_clean)
        rep["clean_armB_vs_armA_output_max_abs_diff"] = (out_a.double() - out_b.double()).abs().max().item()
    rep["verify_pass"] = (rep["library_B_gt_1_timespans_crash"] is not None
                          and rep["manual_loop_vs_library_B1_max_abs_diff"] < 1e-4
                          and rep["clean_armB_vs_armA_output_max_abs_diff"] < 1e-4)
    return rep


# ================= 汇总（CSV/MD + 判据机械输出；S5/S7 同款门线结构） =================

def summarize_nm(rows: list[dict]) -> dict:
    """预注册判据机械输出（跑前冻结、禁事后改）：
    ①H1 主判据 / ②clean A≡B sanity / ③C 臂保守门 / ④D7 vs 0.9938 / ⑤uniform30 z / ⑥三口径分化。"""
    import statistics as st

    def pick(arm, cond, key="acc_meanpool"):
        return [r[key] for r in rows if r["arm"] == arm and r["condition"] == cond]

    def per_seed(arm, cond, key="acc_meanpool"):
        out = {}
        for s in SEEDS:
            vals = [r[key] for r in rows if r["arm"] == arm and r["condition"] == cond
                    and r["seed"] == s]
            if vals:
                out[s] = st.mean(vals)
        return out

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

    # ---- sanity ----
    a_clean = summary["conditions"]["clean"].get("A", {}).get("mean")
    b_clean = summary["conditions"]["clean"].get("B", {}).get("mean")
    c_clean = summary["conditions"]["clean"].get("C", {}).get("mean")
    s = summary["sanity"]
    s["clean_A_mean"] = a_clean
    s["clean_B_mean"] = b_clean
    s["clean_C_mean"] = c_clean
    s["clean_A_minus_B"] = (a_clean - b_clean) if (a_clean is not None and b_clean is not None) else None
    # ②clean A≡B<0.1pp（分数单位 <0.001，2026-10-26T03:1x 判读修正同款口径）
    s["clean_A_equals_B_lt_0p1pp"] = (s["clean_A_minus_B"] is not None
                                      and abs(s["clean_A_minus_B"]) < 0.001)
    # ⑥收敛伴生：clean 下 A 臂三口径分化（pilot 恒定类消失判据）
    a_clean_cal = {k: st.mean(pick("A", "clean", f"acc_{k}")) if pick("A", "clean", f"acc_{k}") else None
                   for k in ["meanpool", "final", "gapnb"]}
    s["A_clean_three_calibers"] = a_clean_cal
    vals = [v for v in a_clean_cal.values() if v is not None]
    s["A_clean_caliber_spread_pp"] = (max(vals) - min(vals)) * 100 if len(vals) == 3 else None
    s["A_clean_constant_class_gone"] = (s["A_clean_caliber_spread_pp"] is not None
                                        and s["A_clean_caliber_spread_pp"] > 0.1)
    # ⑤A@uniform30 z>2 vs chance=0.10
    a_u30 = summary["conditions"]["uniform30"].get("A", {}).get("mean")
    s["A_uniform30_mean"] = a_u30
    n_eval_u30 = max([r.get("n_eval", 0) for r in rows
                      if r["arm"] == "A" and r["condition"] == "uniform30"] or [0])
    if a_u30 is not None and n_eval_u30 > 0:
        z_u30 = (a_u30 - 0.10) / (0.10 * 0.90 / n_eval_u30) ** 0.5
        s["A_uniform30_z_vs_chance"] = round(z_u30, 2)
        s["A_uniform30_direction_ok"] = z_u30 > 2.0

    # ---- verdict（主口径=meanpool 论文口径；三口径兼报） ----
    v = summary["verdict"]
    b5, b15 = summary["conditions"]["block5-30"], summary["conditions"]["block15-30"]
    v["B_minus_A_block5"] = b5.get("B_minus_A")
    v["C_minus_A_block5"] = b5.get("C_minus_A")
    v["three_calibers_block5"] = {
        "B_minus_A": {k: (st.mean(pick("B", "block5-30", f"acc_{k}"))
                          - st.mean(pick("A", "block5-30", f"acc_{k}"))) for k in
                      ["meanpool", "final", "gapnb"]},
        "C_minus_A": {k: (st.mean(pick("C", "block5-30", f"acc_{k}"))
                          - st.mean(pick("A", "block5-30", f"acc_{k}"))) for k in
                      ["meanpool", "final", "gapnb"]},
    }
    # clean 降幅：mean + 逐种子（同种子配对，73fb33f2 判读补刀口径）
    v["B_clean_drop_pp"] = (b_clean - a_clean) * 100 if (a_clean is not None and b_clean is not None) else None
    v["C_clean_drop_pp"] = (c_clean - a_clean) * 100 if (a_clean is not None and c_clean is not None) else None
    a_clean_ps = per_seed("A", "clean")
    v["B_clean_drop_pp_per_seed"] = {str(k): round((pv - a_clean_ps[k]) * 100, 2)
                                     for k, pv in per_seed("B", "clean").items()
                                     if k in a_clean_ps} if a_clean is not None else {}
    v["C_clean_drop_pp_per_seed"] = {str(k): round((pv - a_clean_ps[k]) * 100, 2)
                                     for k, pv in per_seed("C", "clean").items()
                                     if k in a_clean_ps} if a_clean is not None else {}
    a_b5_ps = per_seed("A", "block5-30")
    v["B_minus_A_block5_per_seed"] = {str(k): round((pv - a_b5_ps[k]) * 100, 2)
                                      for k, pv in per_seed("B", "block5-30").items()
                                      if k in a_b5_ps}
    v["C_minus_A_block5_per_seed"] = {str(k): round((pv - a_b5_ps[k]) * 100, 2)
                                      for k, pv in per_seed("C", "block5-30").items()
                                      if k in a_b5_ps}
    v["C_clean_drop_all_seeds_le_0p5pp"] = (
        bool(v["C_clean_drop_pp_per_seed"])
        and all(abs(x) <= 0.5 for x in v["C_clean_drop_pp_per_seed"].values()))
    v["H2_block15_vs_block5"] = (
        (b15.get("B_minus_A", 0) - b5.get("B_minus_A", 0)) if b15.get("B_minus_A") is not None else None)

    # ④D7=A@clean vs 论文 0.9938（双锚点：Table I 0.9938 主、Table III 0.9871 兼报）
    if a_clean is not None:
        s["A_clean_gap_to_paper_table1_pp"] = round((a_clean - PAPER_TEST_ACC) * 100, 2)
        s["A_clean_gap_to_paper_table3_pp"] = round((a_clean - PAPER_TEST_ACC_TABLE3) * 100, 2)
        s["D7_gap_gt_3pp_caveat"] = (s["A_clean_gap_to_paper_table1_pp"] < -D7_GAP_CAVEAT_PP)

    # ①H1 主判据（跑前冻结操作化，源自 design.md §五 + S5 实现先例）：
    #   supported：block5-30 B−A≥+2pp 且 clean |B−A|≤0.5pp；
    #   falsified：B−A<+1pp 且 clean 不受损（B_clean_drop≥−0.5pp）；
    #   其间：inconclusive。
    b_minus_a = v["B_minus_A_block5"]
    b_drop = v["B_clean_drop_pp"]
    if b_minus_a is not None and b_drop is not None:
        if b_minus_a >= 0.02 and abs(b_drop) <= 0.5:
            v["H1_verdict"] = "supported"
        elif b_minus_a < 0.01 and b_drop >= -0.5:
            v["H1_verdict"] = "falsified"
        else:
            v["H1_verdict"] = "inconclusive"
    else:
        v["H1_verdict"] = None

    # ③C 臂保守门（D3：幅值+clean 逐种子双报告；门线对 C 为保守存活判据）
    c_minus_a = v["C_minus_A_block5"]
    c_drop = v["C_clean_drop_pp"]
    v["C_gate_pass"] = bool(c_minus_a is not None and c_minus_a >= 0.02
                            and c_drop is not None and abs(c_drop) <= 0.5
                            and v["C_clean_drop_all_seeds_le_0p5pp"])

    # 总 decision（与 S5 门线结构同款；H1 裁决交 researcher，本阶段不代判）
    if v["H1_verdict"] == "supported":
        v["decision"], v["next_action"] = "pass", "replicate"
    elif v["H1_verdict"] == "falsified":
        v["decision"], v["next_action"] = "falsified_negative_branch", "redesign"
    else:
        v["decision"], v["next_action"] = "inconclusive", "continue"
    return summary


def write_csv_md_nm(rows: list[dict], summary: dict, verify_rep: dict, timing: dict,
                    protocol: dict) -> None:
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

    L = []
    L.append("# S10-nmnist-threearm-baseline N-MNIST 三臂结论级基准（% acc，3 种子 mean±std，逐种子明细见 CSV）\n")
    L.append("- 数据集：N-MNIST（论文原文数据集；S8 loader 口径：L=10 每样本跨度等分+log1p，"
             "Train 60000 / Test 10000；评测用全量 test n_eval=10000）")
    L.append(f"- 训练：全量 60000 × {EPOCHS_NM} epoch × 6 次（A/B 共享 3 种子 + C-aug 3 种子），逐 epoch 原子 ckpt")
    L.append("- 臂：A=库 forward(timespans=None,均匀dt=1)；B=manual-loop+D2 真实 timespans（与 A 共享 checkpoint/掩码）；"
             "C=B+gap 指示位+块状增强训练（单独 checkpoint）")
    L.append("- 条件：clean / uniform30(逐时间步 Bernoulli 0.3,EV-06 逐字一致) / block5-30、block15-30"
             "（块长 b 划分、每块 Bernoulli 0.3，期望掩码率 30%；3 固定种子化模式跨臂配对）")
    L.append("- 读表口径：预注册主口径=meanpool（论文口径），三口径 meanpool/gapnb/final 兼报\n")
    L.append("| 条件 | A | B | C | B−A(pp) | C−A(pp) |")
    L.append("|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        e = summary["conditions"][cond]
        bma = f"{e['B_minus_A'] * 100:+.2f}" if e.get("B_minus_A") is not None else "—"
        cma = f"{e['C_minus_A'] * 100:+.2f}" if e.get("C_minus_A") is not None else "—"
        note = "（L=10 退化为整段掩码=已知口径特性，行保留作协议可比）" if cond == "block15-30" else ""
        L.append(f"| {cond}{note} | {fmt(e, 'A')} | {fmt(e, 'B')} | {fmt(e, 'C')} | {bma} | {cma} |")
    L.append("")
    s = summary["sanity"]
    L.append("## 强制 sanity")
    if s["clean_A_minus_B"] is not None:
        L.append(f"- ②clean A≡B：|A−B|={s['clean_A_minus_B'] * 100:.3f}pp → "
                 f"{'PASS(<0.1pp)' if s['clean_A_equals_B_lt_0p1pp'] else 'FAIL'}")
    if s["A_uniform30_mean"] is not None:
        L.append(f"- ⑤A@uniform30={s['A_uniform30_mean'] * 100:.2f}% → 方向核对 "
                 f"{'PASS(z=' + str(s.get('A_uniform30_z_vs_chance')) + '>2 vs chance)' if s.get('A_uniform30_direction_ok') else '未显著高于随机水平'}")
    if s["clean_A_mean"] is not None:
        caveat_txt = ("；缺口>3pp 记 caveat 交归因，归因第一嫌疑=协议差"
                      if s["A_clean_gap_to_paper_table1_pp"] < -D7_GAP_CAVEAT_PP
                      else "（缺口≤3pp，未触发 caveat 条款）")
        L.append(f"- ④D7：A@clean={s['clean_A_mean'] * 100:.2f}% vs 论文 Table I N-MNIST LNN "
                 f"{PAPER_TEST_ACC * 100:.2f}% → 缺口 {s['A_clean_gap_to_paper_table1_pp']:+.2f}pp"
                 f"（兼报 vs Table III 0% 列 {PAPER_TEST_ACC_TABLE3 * 100:.2f}%："
                 f"{s['A_clean_gap_to_paper_table3_pp']:+.2f}pp{caveat_txt}）")
        L.append(f"- ⑥收敛伴生：A 臂 clean 三口径={ {k: round(vv * 100, 2) for k, vv in s['A_clean_three_calibers'].items()} }，"
                 f"分化={s['A_clean_caliber_spread_pp']:.2f}pp → {'已分化（非恒定类）' if s['A_clean_constant_class_gone'] else '仍近似恒定类'}")
    L.append(f"- manual-loop 真实尺寸复验：verify_pass={verify_rep.get('verify_pass')}，"
             f"vs 库 B=1 max diff={verify_rep.get('manual_loop_vs_library_B1_max_abs_diff')}，"
             f"B>1 崩溃记录={'有' if verify_rep.get('library_B_gt_1_timespans_crash') else '无'}")
    L.append("")
    v = summary["verdict"]
    L.append("## 判据机械输出（跑前冻结；H1 终判交 researcher）")
    L.append(f"- ①H1 主判据（block5-30，meanpool，3 种子 mean）：B−A={v['B_minus_A_block5'] * 100:+.2f}pp"
             f"（逐种子 {v['B_minus_A_block5_per_seed']}）、clean B−A={v['B_clean_drop_pp']:+.2f}pp"
             f"（逐种子 {v['B_clean_drop_pp_per_seed']}）→ **H1_verdict={v['H1_verdict']}**")
    L.append(f"- ③C 臂保守门（D3）：C−A={v['C_minus_A_block5'] * 100:+.2f}pp（逐种子 {v['C_minus_A_block5_per_seed']}）、"
             f"C clean 降幅={v['C_clean_drop_pp']:+.2f}pp（逐种子 {v['C_clean_drop_pp_per_seed']}）→ "
             f"C_gate_pass={v['C_gate_pass']}")
    tc = v["three_calibers_block5"]
    L.append(f"- 三口径兼报（block5-30）：B−A meanpool={tc['B_minus_A']['meanpool'] * 100:+.2f} / "
             f"gapnb={tc['B_minus_A']['gapnb'] * 100:+.2f} / final={tc['B_minus_A']['final'] * 100:+.2f}pp；"
             f"C−A meanpool={tc['C_minus_A']['meanpool'] * 100:+.2f} / "
             f"gapnb={tc['C_minus_A']['gapnb'] * 100:+.2f} / final={tc['C_minus_A']['final'] * 100:+.2f}pp")
    if v["H2_block15_vs_block5"] is not None:
        L.append(f"- H2（block15 增益 ≥ block5 增益）：Δ={v['H2_block15_vs_block5'] * 100:+.2f}pp"
                 f"（H2 已由 S5 在 QuickDraw 结论级驳倒且 L=10 下 block15 退化不可检验，本阶段不 adjudicate，仅协议可比记录）")
    L.append(f"- decision={v['decision']}，next_action={v['next_action']}")
    L.append("")
    L.append("## 协议常数与 caveat（子步1 取证，双记）")
    for k, val in protocol.items():
        L.append(f"- {k}：{val}")
    L.append("")
    L.append("## 限制")
    L.append("- C−A 含增强训练混杂（D3 预注册：门线对 C 为保守存活判据；E2 已由 S7 在 QuickDraw 结论级解答："
             "训练期激活时间通路贡献 C−A 增益 87.5%，N-MNIST 不重开 D 臂除非 B−A 转正）")
    L.append("- block15-30 在 L=10 退化为整段掩码（每样本 30% 概率全掩码），为已知口径特性非 bug，行保留作协议可比")
    L.append(f"- 计时（U5）：{json.dumps(timing, ensure_ascii=False)}")
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ================= selftest（N-MNIST 机制门：广播/掩码率/ts梯度/A≡B/续跑等价） =================

def selftest() -> int:
    ok = True
    print("[selftest] N-MNIST mechanics gates ...")

    # 1) 掩码 view 广播 + 实现掩码率（四条件；B=256 使大数定律生效——
    #    B=8 时 block5-30 仅 16 个块、block15-30 仅 8 个样本，实现率方差大可误报）
    torch.manual_seed(0)
    x = torch.randn(256, 10, 2, 34, 34)
    mask_ok = True
    for cond in CONDITIONS:
        m = P.make_mask(cond, 256, 10, 0, 0)
        xin = x * (1.0 - mask4(m))
        rate = m.mean().item()
        expect = 0.0 if cond == "clean" else 0.30
        good = abs(rate - expect) < 0.06 and xin.shape == x.shape
        mask_ok &= good
        print(f"  [mask] {cond}: shape={tuple(xin.shape)} rate={rate:.3f} (expect≈{expect}) "
              f"{'PASS' if good else 'FAIL'}")
    ok &= mask_ok

    # 2) d2 向量化 vs 循环版恒等（full_protocol selftest 同款用例集，抽 12 例）
    d2_ok = True
    n_case = 0
    for m, name in F._d2_cases()[::7]:
        ref = F._ORIG_D2(m)
        got = F.d2_timespans_vec(m)
        same = torch.equal(got, ref)
        d2_ok &= same
        n_case += 1
        if not same:
            print(f"  [d2] DIFF case={name}")
    print(f"  [d2] vec vs loop identity: {n_case} cases {'PASS' if d2_ok else 'FAIL'}")
    ok &= d2_ok

    # 3) ts 梯度非零 + clean A≡B（随机权重模型即可验机制；真实训练冒烟另测）
    model = P.DrawingClassifier(DATASET, with_gap_bit=False).to(DEVICE).eval()
    xb = _to_torch(torch.randn(32, 10, 2, 34, 34))
    yb = torch.randint(0, 10, (32,), device=DEVICE)
    m = P.make_mask("block5-30", 32, 10, 0, 0).to(DEVICE)
    xin = xb * (1.0 - mask4(m))
    ts = F.d2_timespans_vec(m).to(DEVICE)
    ts.requires_grad_(True)
    logits, _ = model.forward_ts(xin, ts)
    nn.CrossEntropyLoss()(logits, yb).backward()
    ts_grad = float(ts.grad.abs().sum().item())
    ts_ok = ts_grad > 0.0
    ok &= ts_ok
    print(f"  [ts-grad] abs_sum={ts_grad:.6f} → {'PASS(非零)' if ts_ok else 'FAIL(零)'}")
    with torch.no_grad():
        la, _ = model.forward_lib(xb)
        lb, _ = model.forward_ts(xb, torch.ones(32, 10, device=DEVICE))
    ab = float((la - lb).abs().max().item())
    ab_ok = ab < 1e-4
    ok &= ab_ok
    print(f"  [clean A≡B] logits max|diff|={ab:.2e} → {'PASS' if ab_ok else 'FAIL'}")

    # 4) 逐 epoch ckpt 续跑等价（小数据 2 臂，与 full_protocol selftest 同款）
    print("[selftest] per-epoch ckpt resume equivalence (smoke subset) ...")
    x_tr, y_tr, x_te, y_te = nm_splits()
    sub = 256
    xs, ys = x_tr[:sub], y_tr[:sub]
    for arm in ["ab", "c"]:
        pth = os.path.join(CKPT_DIR, f"nmnist_smoke_seed0_{arm}.pt")
        for f in [pth]:
            if os.path.exists(f):
                os.remove(f)
        #  monkeypatch ckpt 名：临时改 train_one_nm 的命名不可行→用 tag 参数化
        _, acc1, _, _, _ = train_one_nm_tagged(0, arm, xs, ys, epochs=2, tag="smoke_")
        sd_one = torch.load(os.path.join(CKPT_DIR, f"nmnist_smoke_seed0_{arm}.pt"),
                            map_location="cpu", weights_only=False)["state_dict"]
        os.remove(os.path.join(CKPT_DIR, f"nmnist_smoke_seed0_{arm}.pt"))
        train_one_nm_tagged(0, arm, xs, ys, epochs=1, tag="smoke_")
        _, acc2, _, _, _ = train_one_nm_tagged(0, arm, xs, ys, epochs=2, tag="smoke_")
        sd_two = torch.load(os.path.join(CKPT_DIR, f"nmnist_smoke_seed0_{arm}.pt"),
                            map_location="cpu", weights_only=False)["state_dict"]
        max_d = max((sd_one[k].double() - sd_two[k].double()).abs().max().item()
                    for k in sd_one if sd_one[k].is_floating_point())
        arm_ok = max_d < 1e-6
        ok &= arm_ok
        print(f"  [resume] arm={arm}: max|Δstate|={max_d:.2e} acc_1shot={acc1:.4f} "
              f"acc_resume={acc2:.4f} → {'PASS' if arm_ok else 'FAIL'}")
        os.remove(os.path.join(CKPT_DIR, f"nmnist_smoke_seed0_{arm}.pt"))

    # 5) eval 机制冒烟：clean A≡B（acc 级）+ block5-30 实现掩码率≈0.30
    model = P.DrawingClassifier(DATASET, with_gap_bit=False).to(DEVICE).eval()
    xe, ye = x_te[:64], y_te[:64]
    accA, _ = eval_arm_condition_nm(model, "A", xe, ye, "clean", 0, 0)
    accB, _ = eval_arm_condition_nm(model, "B", xe, ye, "clean", 0, 0)
    _, mf5 = eval_arm_condition_nm(model, "B", xe, ye, "block5-30", 0, 0)
    ev_ok = (accA["meanpool"] == accB["meanpool"] and abs(mf5 - 0.3) < 0.06)
    ok &= ev_ok
    print(f"  [eval smoke] A@clean={accA['meanpool']:.4f} B@clean={accB['meanpool']:.4f} "
          f"(equal={accA['meanpool'] == accB['meanpool']}), block5-30 masked={mf5:.3f} → "
          f"{'PASS' if ev_ok else 'FAIL'}")

    print(f"[selftest] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def train_one_nm_tagged(seed: int, arm: str, x_tr, y_tr, epochs: int, tag: str = ""):
    """train_one_nm 的 tag 参数化版（selftest 用独立 smoke ckpt 名，不碰正式 nmnist_seed*）。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = P.DrawingClassifier(DATASET, with_gap_bit=(arm == "c")).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=P.LR, weight_decay=P.WD)
    crit = nn.CrossEntropyLoss()
    ckpt = os.path.join(CKPT_DIR, f"nmnist_{tag}seed{seed}_{arm}.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)
    n = x_tr.shape[0]
    t0 = time.time()
    last_acc = last_loss = float("nan")
    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)
        perm = torch.randperm(n, generator=g)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, P.BATCH):
            idx = perm[i:i + P.BATCH]
            xb, yb = _to_torch(x_tr[idx]), torch.from_numpy(np.asarray(y_tr[idx])).to(DEVICE)
            opt.zero_grad()
            if arm == "ab":
                logits, _ = model.forward_lib(xb)
            else:
                mask = P.make_mask("block5-30", xb.shape[0], xb.shape[1], seed * 77 + ep, 0)
                mask = mask.to(DEVICE)
                xb = xb * (1.0 - mask4(mask))
                ts = F.d2_timespans_vec(mask).to(DEVICE)
                logits, _ = model.forward_ts_gap(xb, ts, mask)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), P.CLIP)
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
        last_acc, last_loss = correct / seen, tot_loss / seen
        _atomic_torch_save({
            "state_dict": model.state_dict(), "opt_state_dict": opt.state_dict(),
            "seed": seed, "arm": arm, "dataset": DATASET, "units": P.UNITS,
            "with_gap_bit": arm == "c", "epochs_done": ep + 1, "epochs_target": epochs,
            "train_n": n, "train_acc": last_acc, "train_loss": last_loss,
            "train_sec": round(time.time() - t0, 1)}, ckpt)
    return ckpt, last_acc, last_loss, time.time() - t0, epochs


# ================= 主流程 =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "smoke", "train", "eval", "all", "status", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["ab", "c"], default="ab")
    ap.add_argument("--epochs", type=int, default=EPOCHS_NM)
    ap.add_argument("--smoke-epochs", type=int, default=SMOKE_EPOCHS)
    ap.add_argument("--train-budget", type=float, default=2700.0,
                    help="训练总预算硬顶（秒）；超限停止启动新 run 并交断点（预注册纪律）")
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()

    t_start = time.time()
    t_data = time.time()
    x_tr, y_tr, x_te, y_te = nm_splits()
    data_prep_sec = round(time.time() - t_data, 1)   # 数据载入单独计量（勿与总耗时混记）
    progress = load_progress()
    print(f"[nm] data: train={x_tr.shape[0]} test={x_te.shape[0]} "
          f"epochs={args.epochs} device={DEVICE} (data_prep {data_prep_sec}s)", flush=True)

    if args.cmd == "status":
        runs = {k: v.get("epochs", v.get("epochs_done", "?")) for k, v in progress.get("runs", {}).items()}
        print(json.dumps({"runs": runs, "eval_rows": len(progress.get("eval_rows", [])),
                          "eval_total": EVAL_TOTAL, "verdict": progress.get("verdict")},
                         ensure_ascii=False))
        return 0

    if args.cmd == "smoke":
        # 子步3：seed0 ab 单臂 SMOKE_EPOCHS ep（正式 ckpt 落盘，可续跑至 10ep）
        rep = {}
        # 3a) ts 梯度非零 + clean A≡B + 实测掩码率（真实数据前 64 样本）
        torch.manual_seed(0)
        model = P.DrawingClassifier(DATASET, with_gap_bit=False).to(DEVICE).eval()
        xb = _to_torch(x_te[:64])
        yb = torch.from_numpy(np.asarray(y_te[:64])).to(DEVICE)
        m = P.make_mask("block5-30", 64, 10, 0, 0).to(DEVICE)
        xin = xb * (1.0 - mask4(m))
        ts = F.d2_timespans_vec(m).to(DEVICE)
        ts.requires_grad_(True)
        logits, _ = model.forward_ts(xin, ts)
        nn.CrossEntropyLoss()(logits, yb).backward()
        rep["ts_grad_abs_sum"] = float(ts.grad.abs().sum().item())
        rep["block5_mask_rate_realized"] = round(float(m.mean().item()), 4)
        with torch.no_grad():
            la, _ = model.forward_lib(xb)
            lb, _ = model.forward_ts(xb, torch.ones(64, 10, device=DEVICE))
            acc_a = (la.argmax(1) == yb).float().mean().item()
            acc_b = (lb.argmax(1) == yb).float().mean().item()
        rep["clean_A_acc"] = round(acc_a, 4)
        rep["clean_B_acc"] = round(acc_b, 4)
        rep["clean_A_minus_B_pp"] = round((acc_a - acc_b) * 100, 4)
        # 3b) 训练冒烟：seed0 ab，SMOKE_EPOCHS ep，实测 s/ep
        t0 = time.time()
        ckpt, acc, loss, sec, ep_done = train_one_nm(args.seed, "ab", x_tr, y_tr,
                                                     epochs=args.smoke_epochs)
        smoke_sec = time.time() - t0
        rep["smoke_epochs"] = ep_done
        rep["smoke_train_acc"] = round(acc, 4)
        rep["smoke_train_loss"] = round(loss, 4)
        rep["smoke_sec"] = round(smoke_sec, 1)
        rep["sec_per_epoch_measured"] = round(smoke_sec / max(ep_done, 1), 2)
        rep["projected_total_train_sec"] = round(rep["sec_per_epoch_measured"] * 6 * args.epochs, 1)
        # 3c) 四门判定（预注册失败条件）
        gates = {
            "ts_grad_nonzero": rep["ts_grad_abs_sum"] > 0.0,
            "loss_decreasing": True,   # 由 history 首末 loss 判定（下方）
            "mask_rate_near_0p30": abs(rep["block5_mask_rate_realized"] - 0.30) <= 0.20 * 0.30,
            "clean_A_equals_B_lt_0p1pp": abs(rep["clean_A_minus_B_pp"]) < 0.1,
        }
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        hist = payload.get("history", [])
        if len(hist) >= 2:
            gates["loss_decreasing"] = hist[-1]["loss"] < hist[0]["loss"]
        rep["gates"] = gates
        rep["gates_all_pass"] = all(gates.values())
        rep["history"] = hist
        progress["sanity"]["smoke"] = rep
        save_progress(progress)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0 if rep["gates_all_pass"] else 2

    if args.cmd == "train":
        key = f"seed{args.seed}_{args.arm}"
        ckpt, acc, loss, sec, ep_done = train_one_nm(args.seed, args.arm, x_tr, y_tr,
                                                     epochs=args.epochs)
        if ep_done >= args.epochs:
            progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                     "train_acc": round(acc, 4), "train_loss": round(loss, 4),
                                     "epochs": ep_done, "sec": round(sec, 1)}
            save_progress(progress)
        return 0

    if args.cmd == "all":
        # verify（入口条件④：真实尺寸 manual-loop 复验，N-MNIST 适配版）
        if "verify_real_size" not in progress["sanity"]:
            progress["sanity"]["verify_real_size"] = verify_nm(x_te)
            save_progress(progress)
            print("[all] verify:", json.dumps(progress["sanity"]["verify_real_size"],
                                              ensure_ascii=False), flush=True)
        # 6 次训练（A/B 共享 3 种子 + C-aug 3 种子），逐 epoch ckpt；训练预算硬顶 2700s
        for seed in SEEDS:
            for arm in ["ab", "c"]:
                key = f"seed{seed}_{arm}"
                ckpt_path = os.path.join(CKPT_DIR, f"nmnist_seed{seed}_{arm}.pt")
                done_ep = 0
                if os.path.exists(ckpt_path):
                    done_ep = int(torch.load(ckpt_path, map_location="cpu",
                                             weights_only=False).get("epochs_done", 0))
                if key in progress["runs"] and done_ep >= args.epochs:
                    print(f"[all] skip {key} (done {done_ep}/{args.epochs})", flush=True)
                    continue
                elapsed = time.time() - t_start
                if elapsed > args.train_budget:
                    print(f"[all] train budget {args.train_budget:.0f}s exceeded "
                          f"(elapsed {elapsed:.0f}s); stop launching new runs → checkpoint handoff",
                          flush=True)
                    save_progress(progress)
                    return 0
                ckpt, acc, loss, sec, ep_done = train_one_nm(seed, arm, x_tr, y_tr,
                                                             epochs=args.epochs)
                if ep_done >= args.epochs:
                    progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                             "train_acc": round(acc, 4), "train_loss": round(loss, 4),
                                             "epochs": ep_done, "sec": round(sec, 1)}
                    save_progress(progress)
        # 评测矩阵（断点续跑：逐行落盘，已完成行跳过）
        rows, eval_sec = run_eval_matrix_nm(progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing = {"data_prep_sec": data_prep_sec, "train_runs":
                  {k: v.get("sec") for k, v in progress["runs"].items()},
                  "eval_sec": eval_sec, "total_sec": round(time.time() - t_start, 1)}
        if len(rows) < EVAL_TOTAL:
            print(f"[all] eval incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(timing)
            return 0
        summary = summarize_nm(rows)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(timing)
        protocol = {
            "epoch": f"{EPOCHS_NM}ep（论文 run_lln_nmnist.sh --epochs 10 + lln_nmnist_10.log 10 行 + "
                     f"Table I 0.9938 三源一致；Table V 记 30ep=论文内部矛盾，双记）",
            "论文架构": "units128/CfC(4096,128,proj128)/双CNN+BN/batch64/no-wd/no-clip/Cosine",
            "本仓冻结规格": f"units{P.UNITS}/CfC(128,256,proj256,backbone_layers=1,backbone_units=128)/"
                            f"单CNN+LN+ReLU/batch{P.BATCH}/wd{P.WD}/clip{P.CLIP}/AdamW lr{P.LR} 恒定",
            "协议差 caveat": "batch/wd/clip/units/架构/lr-schedule 与论文 Table V/代码默认值的差异全部记入；"
                             "不改冻结规格与判据（H1 主判据 B−A 为同 ckpt 推理期配对不受影响）；"
                             "D7 缺口>3pp 归因第一嫌疑=协议差，禁实现缺陷归因",
            "评测集": "全量 test 10000（QuickDraw S5 为 7500 子集，差异入报告）",
        }
        write_csv_md_nm(rows, summary, progress["sanity"].get("verify_real_size", {}),
                        timing, protocol)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "eval":
        missing = []
        for seed in SEEDS:
            for arm in ["ab", "c"]:
                pth = os.path.join(CKPT_DIR, f"nmnist_seed{seed}_{arm}.pt")
                if not os.path.exists(pth):
                    missing.append(os.path.basename(pth))
                else:
                    ep = int(torch.load(pth, map_location="cpu", weights_only=False)
                             .get("epochs_done", 0))
                    if ep < args.epochs:
                        missing.append(f"{os.path.basename(pth)}({ep}/{args.epochs})")
        if missing:
            print(f"[eval] training incomplete, {len(missing)} run(s) not ready: {missing}; "
                  f"run `all` (or `train`) first — resume is lossless from per-epoch ckpt",
                  flush=True)
            return 0
        rows, eval_sec = run_eval_matrix_nm(progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing = {"data_prep_sec": data_prep_sec,
                  "eval_sec": eval_sec, "total_sec": round(time.time() - t_start, 1)}
        if len(rows) < EVAL_TOTAL:
            print(f"[eval] incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(timing)
            return 0
        summary = summarize_nm(rows)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(timing)
        protocol = {
            "epoch": f"{EPOCHS_NM}ep（论文 run_lln_nmnist.sh --epochs 10 + lln_nmnist_10.log 10 行 + "
                     f"Table I 0.9938 三源一致；Table V 记 30ep=论文内部矛盾，双记）",
            "论文架构": "units128/CfC(4096,128,proj128)/双CNN+BN/batch64/no-wd/no-clip/Cosine",
            "本仓冻结规格": f"units{P.UNITS}/CfC(128,256,proj256,backbone_layers=1,backbone_units=128)/"
                            f"单CNN+LN+ReLU/batch{P.BATCH}/wd{P.WD}/clip{P.CLIP}/AdamW lr{P.LR} 恒定",
            "协议差 caveat": "batch/wd/clip/units/架构/lr-schedule 与论文 Table V/代码默认值的差异全部记入；"
                             "不改冻结规格与判据（H1 主判据 B−A 为同 ckpt 推理期配对不受影响）；"
                             "D7 缺口>3pp 归因第一嫌疑=协议差，禁实现缺陷归因",
            "评测集": "全量 test 10000（QuickDraw S5 为 7500 子集，差异入报告）",
        }
        write_csv_md_nm(rows, summary, progress["sanity"].get("verify_real_size", {}),
                        timing, protocol)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "report":
        # 从盘上 progress.eval_rows 确定性重建 CSV/MD/timing（不重算、不重训）。
        # 用途=脚本 bug 修复后再生 artifacts（本轮：timing data_prep_sec 误标修正），
        # 行数据逐字节不变（同 rows 同排序同舍入），summary 由 rows 重新机械计算。
        rows = sorted(progress.get("eval_rows", []),
                      key=lambda r: (r["seed"], r["arm"],
                                     CONDITIONS.index(r["condition"]), r["pattern"]))
        if len(rows) < EVAL_TOTAL:
            print(f"[report] incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; nothing written",
                  flush=True)
            return 0
        old_timing = {}
        if os.path.exists(TIMING):
            with open(TIMING, encoding="utf-8") as fh:
                old_timing = json.load(fh).get("u5_runtime", {})
        timing = {"data_prep_sec": None,
                  "data_prep_note": "report 模式未重载数据；原始 all-run 的 data_prep 曾误标为"
                                    "总耗时（键名 bug，本轮修正），此处诚实留空",
                  "train_runs": {k: v.get("sec") for k, v in progress["runs"].items()},
                  "eval_sec": old_timing.get("eval_sec"),
                  "total_sec": old_timing.get("total_sec"),
                  "regenerated_from": "results/nmnist_progress.json"}
        summary = summarize_nm(rows)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(timing)
        protocol = {
            "epoch": f"{EPOCHS_NM}ep（论文 run_lln_nmnist.sh --epochs 10 + lln_nmnist_10.log 10 行 + "
                     f"Table I 0.9938 三源一致；Table V 记 30ep=论文内部矛盾，双记）",
            "论文架构": "units128/CfC(4096,128,proj128)/双CNN+BN/batch64/no-wd/no-clip/Cosine",
            "本仓冻结规格": f"units{P.UNITS}/CfC(128,256,proj256,backbone_layers=1,backbone_units=128)/"
                            f"单CNN+LN+ReLU/batch{P.BATCH}/wd{P.WD}/clip{P.CLIP}/AdamW lr{P.LR} 恒定",
            "协议差 caveat": "batch/wd/clip/units/架构/lr-schedule 与论文 Table V/代码默认值的差异全部记入；"
                             "不改冻结规格与判据（H1 主判据 B−A 为同 ckpt 推理期配对不受影响）；"
                             "D7 缺口>3pp 归因第一嫌疑=协议差，禁实现缺陷归因",
            "评测集": "全量 test 10000（QuickDraw S5 为 7500 子集，差异入报告）",
        }
        write_csv_md_nm(rows, summary, progress["sanity"].get("verify_real_size", {}),
                        timing, protocol)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
