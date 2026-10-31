#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""full_protocol.py — S5-full-protocol 第一单元：QuickDraw 全量保真度复现主线。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S5-full-protocol（design.md §四 S1/S2、§4.0 D1-D7、§五门线；S4 verdict=replicate）

与 pilot_threearm.py 的关系（训练路径复用，协议常量升级）：
  - 模型/掩码/timespans/读表/臂语义（D1-D3）全部 import 自 experiments/pilot_threearm.py，
    逐行同源码，不复制不改写（防协议漂移）；
  - 升级点仅三项：①数据=pilot 子集(10k train/3k eval)→论文协议全量
    （首 5000/类×10 类=50k，85/15 seeded split → train 42500 / test 7500）；
    ②EPOCHS 2→30（design.md §五预注册「全量数据 30 epoch」）；
    ③新增逐 epoch checkpoint + 断点续跑（30 epoch×6 训练跨夜窗，kill 不丢）。
  - ckpt 命名 full_{dataset}_seed{seed}_{arm}.pt（与 pilot_* 分离，不覆盖 pilot 证据）；
    progress/timing/CSV/MD 全部落 results/full_*（不碰 pilot 产物）。

性能补丁（数值恒等，自验 experiments 内 `selftest` 子命令）：
  - d2_timespans 原实现为 CPU 双层 Python 循环，实测 867ms/batch（B=128），
    是 C 臂训练主瓶颈（fwd+bwd 仅 188ms）。本文件提供向量化 d2_timespans_vec
    （GPU 张量运算，整数算术）并 monkeypatch pilot_threearm.d2_timespans，
    selftest 以 torch.equal 逐例验证与循环版恒等（含边缘 mask）。
    C 臂 30 epoch×333 batch 预计 175.7min→约 16min；eval 108 行同步加速。

S5 判读须知（S4 verdict §六 复算带出，已落入 summarize/write_csv_md）：
  - 读表口径预注册主口径=meanpool（论文口径），三口径 meanpool/gapnb/final 兼报；
  - C clean 降幅改逐种子门控（pilot 散布 ±0.47pp 贴近 0.5pp 门线）；
  - A/B 恒定类消失（clean 三口径分化）作训练收敛伴生判据；
  - D7 结论级：A 臂 clean 复现论文 Table III QuickDraw ≈98.6-98.7% 数量级。

用法：
  python experiments/full_protocol.py selftest            # 恒等性+续跑等价+冒烟（先跑这个）
  python experiments/full_protocol.py train --seed 0 --arm ab
  python experiments/full_protocol.py eval                # 108 格矩阵，逐行落盘续跑
  python experiments/full_protocol.py all                 # verify+6 训练+eval+CSV/MD（全程断点续跑）
  python experiments/full_protocol.py status              # 打印进度快照（轮询用）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJECT = os.path.dirname(HERE)

import pilot_threearm as P  # noqa: E402  （协议同源代码，import 不复制）

RESULTS = os.path.join(PROJECT, "results")
CKPT_DIR = P.CKPT_DIR
PROGRESS = os.path.join(RESULTS, "full_progress.json")
TIMING = os.path.join(RESULTS, "full_timing.json")
CSV_OUT = os.path.join(RESULTS, "full_threearm.csv")
MD_OUT = os.path.join(RESULTS, "full_threearm.md")

# ---- 协议常数（升级点②：30 epoch；其余与 pilot 一致）----
EPOCHS_FULL = 30
SPLIT_SEED = 12345            # 与 pilot_splits 同一 split 种子（可比性）
FULL_TRAIN_N = None           # None=全量 42500
FULL_EVAL_N = None            # None=全量 7500
DEVICE = P.DEVICE
CONDITIONS = P.CONDITIONS
SEEDS = P.SEEDS
N_PATTERNS = P.N_PATTERNS
EVAL_TOTAL = len(SEEDS) * 3 * len(CONDITIONS) * N_PATTERNS   # 108 行


# ================= 性能补丁：向量化 d2_timespans（数值恒等，selftest 验证） =================

def d2_timespans_vec(mask: torch.Tensor) -> torch.Tensor:
    """D2 主语义（与 pilot_threearm.d2_timespans 逐值恒等，整数算术）：
    掩码帧 dt=0；间隙后首真实帧 dt=gap_len+1；其余 dt=1。mask: (B,L,1) → ts: (B,L)。

    向量化实现（替代 CPU 双层 Python 循环，实测 867ms→亚毫秒级/B=128）：
      run 起点 = m[t]=1 且 m[t-1]=0；cummax(start_idx) 给出每个位置所属 run 的起点；
      run_len[t] = t - run_start[t] + 1（仅掩码位）；
      gap_end[e]（m[e]=1,m[e+1]=0）→ ts[e+1] = run_len[e]+1（e+1<L）。
    """
    m = mask[:, :, 0]                                  # (B,L)
    B, L = m.shape
    dev = m.device
    mk = m > 0.5
    ts = torch.ones(B, L, device=dev)
    ts = torch.where(mk, torch.zeros(B, L, device=dev), ts)
    idx = torch.arange(L, device=dev).expand(B, L)
    prev = torch.cat([torch.zeros(B, 1, dtype=m.dtype, device=dev), m[:, :-1]], dim=1)
    is_start = mk & (prev <= 0.5)
    start_idx = torch.where(is_start, idx, torch.full_like(idx, -1))
    run_start = torch.cummax(start_idx, dim=1).values   # 每个位置所属 run 的起点（-1=不在 run 内）
    run_len = torch.where(mk, idx - run_start + 1, torch.zeros_like(idx))
    nxt = torch.cat([m[:, 1:], torch.zeros(B, 1, dtype=m.dtype, device=dev)], dim=1)
    gap_end = mk & (nxt <= 0.5)
    val = torch.where(gap_end, run_len + 1, torch.zeros_like(idx))   # 位于 run 末帧 e 的值
    shifted = torch.cat([torch.zeros(B, 1, dtype=val.dtype, device=dev), val[:, :-1]], dim=1)
    ts = torch.where(shifted > 0, shifted.to(ts.dtype), ts)
    return ts


def _patch_pilot_d2() -> None:
    """让 pilot_threearm.eval_arm_condition 内的 d2_timespans 走向量化实现
    （函数体内引用模块全局名，运行时查找，patch 即生效）。恒等性由 selftest 担保。"""
    P.d2_timespans = d2_timespans_vec


# ================= 数据（升级点①：全量 50k，同 split 种子） =================

def full_splits(dataset: str = "quickdraw"):
    """论文协议全量：load_quickdraw() 的 50k（首 5000/类）×85/15 seeded split。
    与 pilot_splits 同一 perm 种子 12345 → pilot 子集是全量子集（可比）。"""
    x, y = P.load_quickdraw()
    g = torch.Generator().manual_seed(SPLIT_SEED)
    n = x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(0.85 * n)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    return x[tr_idx], y[tr_idx], x[te_idx], y[te_idx]


# ================= 训练（升级点③：逐 epoch checkpoint + 断点续跑） =================

def _atomic_torch_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # 原子替换：kill 不留半截 ckpt


def train_one_full(seed: int, arm: str, dataset: str, x_tr, y_tr,
                   epochs: int = EPOCHS_FULL, tag: str = ""):
    """arm='ab'：论文协议干净训练（A/B 共享）；arm='c'：块状间隙增强训练（D3）。
    与 pilot train_one 同一优化器/批量/语义；差异=全量数据+30 epoch+逐 epoch checkpoint。
    返回 (ckpt_path, final_train_acc, final_train_loss, sec, epochs_done)。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = P.DrawingClassifier(dataset, with_gap_bit=(arm == "c")).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=P.LR, weight_decay=P.WD)
    crit = nn.CrossEntropyLoss()
    ckpt = os.path.join(CKPT_DIR, f"full_{tag}{dataset}_seed{seed}_{arm}.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)

    start_ep, hist = 0, []
    if os.path.exists(ckpt):                       # 断点续跑：模型+优化器+epoch 全恢复
        payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        opt.load_state_dict(payload["opt_state_dict"])
        start_ep = int(payload.get("epochs_done", 0))
        hist = list(payload.get("history", []))
        if start_ep >= epochs:
            print(f"[full-train] skip seed={seed} arm={arm} (done {start_ep}/{epochs})", flush=True)
            return ckpt, hist[-1]["acc"], hist[-1]["loss"], 0.0, start_ep
        print(f"[full-train] resume seed={seed} arm={arm} from epoch {start_ep}", flush=True)

    n = x_tr.shape[0]
    t0 = time.time()
    last_acc = last_loss = float("nan")
    for ep in range(start_ep, epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)   # 与 pilot 同：逐 epoch 固定 shuffle
        perm = torch.randperm(n, generator=g)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, P.BATCH):
            idx = perm[i:i + P.BATCH]
            xb, yb = x_tr[idx].to(DEVICE), y_tr[idx].to(DEVICE)
            opt.zero_grad()
            if arm == "ab":
                logits, _ = model.forward_lib(xb)
            else:
                # C 臂增强：块长5、每块 Bernoulli(0.3)（与 block5-30 同分布的合成间隙）
                mask = P.make_mask("block5-30", xb.shape[0], xb.shape[1], seed * 77 + ep, 0)
                mask = mask.to(DEVICE)
                xb = xb * (1.0 - mask)   # 2026-10-27T01:3x 修正：mask=1=掩码(清零)（原倒挂，证据 s5_mask_convention_probe.py）
                ts = d2_timespans_vec(mask).to(DEVICE)
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
            "seed": seed, "arm": arm, "dataset": dataset, "units": P.UNITS,
            "with_gap_bit": arm == "c", "epochs_done": ep + 1, "epochs_target": epochs,
            "train_n": n, "train_acc": last_acc, "train_loss": last_loss,
            "train_sec": round(time.time() - t0, 1), "history": hist}, ckpt)
        print(f"[full-train seed={seed} arm={arm} ep={ep + 1}/{epochs}] "
              f"loss={last_loss:.4f} acc={last_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return ckpt, last_acc, last_loss, time.time() - t0, epochs


def load_model_full(seed: int, arm: str, dataset: str):
    ckpt = os.path.join(CKPT_DIR, f"full_{dataset}_seed{seed}_{arm}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = P.DrawingClassifier(dataset, with_gap_bit=payload.get("with_gap_bit", arm == "c"))
    model.load_state_dict(payload["state_dict"])
    return model.to(DEVICE).eval(), payload


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


def save_timing(p: dict, extra: dict) -> None:
    payload = {"schema": "full-timing/v1", "stage": "S5-full-protocol",
               "u5_runtime": extra}
    with open(TIMING, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ================= 评测矩阵（断点续跑，与 pilot 同机制） =================

def run_eval_matrix_full(args, progress: dict, x_te, y_te):
    """三臂评测矩阵：逐行落盘 + 已完成 (seed,arm,cond,pattern) 跳过。返回 (rows, eval_sec)。"""
    rows = list(progress.get("eval_rows", []))
    done = {(r.get("dataset"), r.get("seed"), r.get("arm"), r.get("condition"), r.get("pattern"))
            for r in rows}
    t_eval = time.time()
    for seed in SEEDS:
        models = {a: load_model_full(seed, a, args.dataset)[0] for a in ["ab", "c"]}
        for arm_name, arm_key in [("A", "ab"), ("B", "ab"), ("C", "c")]:
            for cond in CONDITIONS:
                for pattern in range(N_PATTERNS):
                    key = (args.dataset, seed, arm_name, cond, pattern)
                    if key in done:
                        continue
                    accs, mf = P.eval_arm_condition(models[arm_key], arm_name,
                                                    x_te, y_te, cond, seed, pattern)
                    rows.append({"dataset": args.dataset, "seed": seed, "arm": arm_name,
                                 "condition": cond, "pattern": pattern, "n_eval": x_te.shape[0],
                                 "masked_fraction_realized": round(mf, 4),
                                 "acc_meanpool": round(accs["meanpool"], 4),
                                 "acc_final": round(accs["final"], 4),
                                 "acc_gapnb": round(accs["gapnb"], 4)})
                    progress["eval_rows"] = rows
                    save_progress(progress)      # 逐行持久化：超时/kill 不丢已出行
            print(f"[full-eval] seed={seed} arm={arm_name} done "
                  f"({time.time() - t_eval:.0f}s cum, {len(rows)}/{EVAL_TOTAL} rows)", flush=True)
    return rows, round(time.time() - t_eval, 1)


# ================= 汇总（CSV/MD + 门线；S5 判读须知全量落入） =================

def summarize_full(rows: list[dict]) -> dict:
    """在 pilot summarize 门线基础上增加：三口径兼报、C clean 逐种子门控、
    clean 三口径分化（恒定类消失伴生判据）、D7 结论级（≈98.6-98.7%）。"""
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
    s["clean_A_equals_B_lt_0p1pp"] = (s["clean_A_minus_B"] is not None
                                      and abs(s["clean_A_minus_B"]) < 0.001)
    # 恒定类消失伴生判据：clean 下 A 臂三口径是否分化（pilot 9/9 行恒等=恒定类）
    a_clean_cal = {k: st.mean(pick("A", "clean", f"acc_{k}")) if pick("A", "clean", f"acc_{k}") else None
                   for k in ["meanpool", "final", "gapnb"]}
    s["A_clean_three_calibers"] = a_clean_cal
    vals = [v for v in a_clean_cal.values() if v is not None]
    s["A_clean_caliber_spread_pp"] = (max(vals) - min(vals)) * 100 if len(vals) == 3 else None
    s["A_clean_constant_class_gone"] = (s["A_clean_caliber_spread_pp"] is not None
                                        and s["A_clean_caliber_spread_pp"] > 0.1)
    a_u30 = summary["conditions"]["uniform30"].get("A", {}).get("mean")
    s["A_uniform30_mean"] = a_u30
    n_eval_u30 = max([r.get("n_eval", 0) for r in rows
                      if r["arm"] == "A" and r["condition"] == "uniform30"] or [0])
    if a_u30 is not None and n_eval_u30 > 0:
        z_u30 = (a_u30 - 0.10) / (0.10 * 0.90 / n_eval_u30) ** 0.5
        s["A_uniform30_z_vs_chance"] = round(z_u30, 2)
        s["A_uniform30_direction_ok"] = z_u30 > 2.0
    if a_clean is not None:
        s["A_clean_fidelity_97"] = a_clean >= 0.97
        s["A_clean_fidelity_99"] = a_clean >= 0.99
        # D7 结论级：论文 Table III QuickDraw clean≈98.6-98.7%
        s["A_clean_fidelity_986_paper_magnitude"] = a_clean >= 0.985
        s["A_clean_gap_to_paper_pp"] = (a_clean - 0.986) * 100

    # ---- verdict（主口径=meanpool 论文口径；三口径兼报）----
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
    # clean 降幅：mean + 逐种子（S5 须知：改逐种子门控）
    # 73fb33f2 判读补刀（2026-10-31）：逐种子必须同种子配对（原代码误配 A 臂总 mean，
    # 产出 {'0':0.61,'1':0.37,'2':0.78} 之类跨种子值）；block5 差值原码漏 ×100（分数当 pp）。
    # 门线结论不受影响（两种逐种子值下 C clean 逐种子门控均 FAIL），仅报告数字错误。
    v["B_clean_drop_pp"] = (b_clean - a_clean) * 100 if (b_clean is not None and a_clean is not None) else None
    v["C_clean_drop_pp"] = (c_clean - a_clean) * 100 if (c_clean is not None and a_clean is not None) else None
    a_clean_ps = per_seed("A", "clean")
    v["B_clean_drop_pp_per_seed"] = {str(k): round((pv - a_clean_ps[k]) * 100, 2)
                                     for k, pv in per_seed("B", "clean").items()
                                     if k in a_clean_ps} if a_clean is not None else {}
    v["C_clean_drop_pp_per_seed"] = {str(k): round((pv - a_clean_ps[k]) * 100, 2)
                                     for k, pv in per_seed("C", "clean").items()
                                     if k in a_clean_ps} if a_clean is not None else {}
    a_b5_ps = per_seed("A", "block5-30")
    v["C_minus_A_block5_per_seed"] = {str(k): round((pv - a_b5_ps[k]) * 100, 2)
                                      for k, pv in per_seed("C", "block5-30").items()
                                      if k in a_b5_ps}
    v["C_clean_drop_all_seeds_le_0p5pp"] = (
        bool(v["C_clean_drop_pp_per_seed"])
        and all(abs(x) <= 0.5 for x in v["C_clean_drop_pp_per_seed"].values()))
    v["H2_block15_vs_block5"] = (
        (b15.get("B_minus_A", 0) - b5.get("B_minus_A", 0)) if b15.get("B_minus_A") is not None else None)
    # 门线（design.md §五；主口径 meanpool；C clean 逐种子门控）
    b_minus_a, c_minus_a = v["B_minus_A_block5"], v["C_minus_A_block5"]
    c_clean_ok = (v["C_clean_drop_pp"] is not None and abs(v["C_clean_drop_pp"]) <= 0.5
                  and v["C_clean_drop_all_seeds_le_0p5pp"])
    b_clean_ok = (v["B_clean_drop_pp"] is not None and abs(v["B_clean_drop_pp"]) <= 0.5)
    gate_arm = None
    if b_minus_a is not None and b_minus_a >= 0.02 and b_clean_ok:
        gate_arm = "B"
    if c_minus_a is not None and c_minus_a >= 0.02 and c_clean_ok:
        gate_arm = gate_arm or "C"
    v["gate_pass_arm"] = gate_arm
    if gate_arm:
        v["decision"], v["next_action"] = "pass", "replicate"
    elif b_minus_a is not None and b_minus_a < 0.01 and b_clean_ok:
        v["decision"], v["next_action"] = "falsified_negative_branch", "redesign"
    else:
        v["decision"], v["next_action"] = "inconclusive", "continue"
    return summary


def write_csv_md_full(rows: list[dict], summary: dict, dataset: str,
                      verify_rep: dict, timing: dict) -> None:
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
    L.append("# S5-full-protocol QuickDraw 全量三臂结果（% acc，3 种子 mean±std，逐种子明细见 CSV）\n")
    L.append(f"- 数据集：{dataset}（论文协议全量：首 5000/类×10 类=50k，85/15 seeded split → "
             f"train 42500 / test 7500；N-MNIST 仍 SSL 墙，结论级基准留后续单元）")
    L.append(f"- 训练：全量 {EPOCHS_FULL} epoch × 6 次（A/B 共享 3 种子 + C-aug 3 种子），"
             f"逐 epoch checkpoint；臂语义 D1-D3 与 pilot 完全一致（同源代码 import）")
    L.append("- 读表口径：预注册主口径=meanpool（论文口径），三口径 meanpool/gapnb/final 兼报")
    L.append("- 臂：A=库 forward(timespans=None,均匀dt=1)；B=manual-loop+D2 真实 timespans（与 A 共享 checkpoint/掩码）；"
             "C=B+gap 指示位+块状增强训练（单独 checkpoint）")
    L.append("- 条件：clean / uniform30(逐时间步 Bernoulli 0.3) / block5-30、block15-30（块长 b 划分、每块 Bernoulli 0.3，"
             "期望掩码率 30%；3 固定种子化模式跨臂配对）\n")
    L.append("| 条件 | A | B | C | B−A(pp) | C−A(pp) |")
    L.append("|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        e = summary["conditions"][cond]
        bma = f"{e['B_minus_A'] * 100:+.2f}" if e.get("B_minus_A") is not None else "—"
        cma = f"{e['C_minus_A'] * 100:+.2f}" if e.get("C_minus_A") is not None else "—"
        L.append(f"| {cond} | {fmt(e, 'A')} | {fmt(e, 'B')} | {fmt(e, 'C')} | {bma} | {cma} |")
    L.append("")
    s = summary["sanity"]
    L.append("## 强制 sanity（S2 入口条件同款）")
    if s["clean_A_minus_B"] is not None:
        L.append(f"- clean A≡B：|A−B|={s['clean_A_minus_B'] * 100:.3f}pp → "
                 f"{'PASS(<0.1pp)' if s['clean_A_equals_B_lt_0p1pp'] else 'FAIL'}")
    if s["A_uniform30_mean"] is not None:
        L.append(f"- A@uniform30={s['A_uniform30_mean'] * 100:.2f}% → 方向核对 "
                 f"{'PASS(z=' + str(s.get('A_uniform30_z_vs_chance')) + '>2 vs chance)' if s['A_uniform30_direction_ok'] else '未显著高于随机水平'}")
    if s["clean_A_mean"] is not None:
        gap_pp = s["A_clean_gap_to_paper_pp"]
        L.append(f"- A@clean={s['clean_A_mean'] * 100:.2f}% → D7 结论级（论文 Table III QuickDraw ≈98.6-98.7%）"
                 + ("PASS 数量级" if s["A_clean_fidelity_986_paper_magnitude"]
                    else f"未达（缺口 {gap_pp:+.2f}pp）"))
        L.append(f"- 恒定类消失伴生判据：A 臂 clean 三口径={ {k: round(vv * 100, 2) for k, vv in s['A_clean_three_calibers'].items()} }，"
                 f"分化={s['A_clean_caliber_spread_pp']:.2f}pp → {'已分化（非恒定类）' if s['A_clean_constant_class_gone'] else '仍近似恒定类'}")
    L.append(f"- manual-loop 真实尺寸复验：verify_pass={verify_rep.get('verify_pass')}，"
             f"vs 库 B=1 max diff={verify_rep.get('manual_loop_vs_library_B1_max_abs_diff')}")
    L.append("")
    v = summary["verdict"]
    L.append("## 门线与假设判读（design.md §五；主口径 meanpool）")
    L.append(f"- H1（block5-30 下 B 或 C ≥+2pp 且该臂 clean 降幅 ≤0.5pp）：B−A={v['B_minus_A_block5'] * 100:+.2f}pp、"
             f"C−A={v['C_minus_A_block5'] * 100:+.2f}pp；B clean 降幅={v['B_clean_drop_pp']:+.2f}pp、"
             f"C clean 降幅={v['C_clean_drop_pp']:+.2f}pp → gate_pass_arm={v['gate_pass_arm']}")
    L.append(f"- C clean 逐种子降幅（pp）：{v['C_clean_drop_pp_per_seed']} → 逐种子门控(全 ≤0.5pp) "
             f"{'PASS' if v['C_clean_drop_all_seeds_le_0p5pp'] else 'FAIL'}")
    L.append(f"- block5-30 逐种子 C−A（pp）：{v['C_minus_A_block5_per_seed']}")
    tc = v["three_calibers_block5"]
    L.append(f"- 三口径兼报（block5-30）：B−A meanpool={tc['B_minus_A']['meanpool'] * 100:+.2f} / "
             f"gapnb={tc['B_minus_A']['gapnb'] * 100:+.2f} / final={tc['B_minus_A']['final'] * 100:+.2f}pp；"
             f"C−A meanpool={tc['C_minus_A']['meanpool'] * 100:+.2f} / "
             f"gapnb={tc['C_minus_A']['gapnb'] * 100:+.2f} / final={tc['C_minus_A']['final'] * 100:+.2f}pp")
    if v["H2_block15_vs_block5"] is not None:
        L.append(f"- H2（block15 增益 ≥ block5 增益，mean 意义）：Δ={v['H2_block15_vs_block5'] * 100:+.2f}pp")
    L.append(f"- 判读：decision={v['decision']}，next_action={v['next_action']}")
    L.append("")
    L.append("## 限制")
    L.append("- C−A 含增强训练混杂（D3 预注册：门线对 C 为保守存活判据，归因细化=E2 B-train 对照臂，留后续单元）")
    L.append("- QuickDraw L=100（block15 为字面块长）；N-MNIST L=10 场景 block15 退化为整段掩码，结论级须回 N-MNIST 复现")
    L.append(f"- 计时（U5）：{json.dumps(timing, ensure_ascii=False)}")
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ================= selftest =================

def _d2_cases():
    """(mask, 说明) 边缘用例集。"""
    cases = []
    torch.manual_seed(0)
    for B, L in [(1, 1), (1, 2), (2, 3), (3, 10), (7, 100), (128, 100), (256, 100)]:
        for p in [0.0, 0.3, 0.5, 1.0]:
            m = (torch.rand(B, L, 1) < p).float()
            cases.append((m, f"rand B={B} L={L} p={p}"))
    cases.append((torch.zeros(4, 20, 1), "all-clean"))
    cases.append((torch.ones(4, 20, 1), "all-masked"))
    alt = torch.zeros(2, 20, 1)
    alt[:, ::2] = 1.0
    cases.append((alt, "alternating"))
    end = torch.zeros(2, 20, 1)
    end[:, 15:] = 1.0
    cases.append((end, "run-at-end"))
    st = torch.zeros(2, 20, 1)
    st[:, :8] = 1.0
    cases.append((st, "run-at-start"))
    single = torch.zeros(2, 20, 1)
    single[:, 5] = 1.0
    single[:, 11] = 1.0
    cases.append((single, "single-frame-runs"))
    return cases


def selftest() -> int:
    ok = True

    # 1) d2 向量化 vs 循环版逐值恒等（CPU/GPU 双设备）
    print("[selftest] d2_timespans_vec vs loop identity ...")
    n_case = 0
    for m, name in _d2_cases():
        ref = _ORIG_D2(m)          # patch 前快照的原循环版（恒等基准）
        for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            got = d2_timespans_vec(m.to(dev)).cpu()
            if not torch.equal(got, ref):
                print(f"  DIFF case={name} dev={dev} max={ (got - ref).abs().max().item() }")
                ok = False
            n_case += 1
    print(f"[selftest] d2 identity: {n_case} case-device checks, {'PASS' if ok else 'FAIL'}")

    # 2) full_splits 形状/确定性
    print("[selftest] full_splits shape/determinism ...")
    x_tr, y_tr, x_te, y_te = full_splits("quickdraw")
    x_tr2, y_tr2, x_te2, y_te2 = full_splits("quickdraw")
    shape_ok = (x_tr.shape[0] == 42500 and x_te.shape[0] == 7500
                and torch.equal(x_tr, x_tr2) and torch.equal(x_te, x_te2))
    # pilot 子集应为全量子集（同 perm 种子）：pilot train 10k ⊆ full train
    px_tr, py_tr, px_te, py_te = P.pilot_splits("quickdraw")
    subset_ok = all((x_tr == px_tr[i]).all(dim=1).any().item() for i in range(0, 200))
    print(f"[selftest] full_splits: train={tuple(x_tr.shape)} test={tuple(x_te.shape)} "
          f"deterministic={shape_ok} pilot_subset_of_full(sampled200)={subset_ok} → "
          f"{'PASS' if (shape_ok and subset_ok) else 'FAIL'}")
    ok = ok and shape_ok and subset_ok

    # 3) 逐 epoch checkpoint 续跑等价：1+1 epoch 续跑 vs 2 epoch 一次跑
    print("[selftest] per-epoch checkpoint resume equivalence ...")
    smoke_n = 384
    xs, ys = x_tr[:smoke_n], y_tr[:smoke_n]
    for arm in ["ab", "c"]:
        tag = "smoke_"
        for f in [f"full_{tag}quickdraw_seed0_{arm}.pt"]:
            pth = os.path.join(CKPT_DIR, f)
            if os.path.exists(pth):
                os.remove(pth)
        # 一次跑 2 epoch
        _, acc1, _, _, _ = train_one_full(0, arm, "quickdraw", xs, ys, epochs=2, tag=tag)
        sd_one = torch.load(os.path.join(CKPT_DIR, f"full_{tag}quickdraw_seed0_{arm}.pt"),
                            map_location="cpu", weights_only=False)["state_dict"]
        os.remove(os.path.join(CKPT_DIR, f"full_{tag}quickdraw_seed0_{arm}.pt"))
        # 1 epoch + 续跑 1 epoch
        train_one_full(0, arm, "quickdraw", xs, ys, epochs=1, tag=tag)
        _, acc2, _, _, _ = train_one_full(0, arm, "quickdraw", xs, ys, epochs=2, tag=tag)
        sd_two = torch.load(os.path.join(CKPT_DIR, f"full_{tag}quickdraw_seed0_{arm}.pt"),
                            map_location="cpu", weights_only=False)["state_dict"]
        max_d = max((sd_one[k].double() - sd_two[k].double()).abs().max().item()
                    for k in sd_one if sd_one[k].is_floating_point())
        arm_ok = max_d < 1e-6
        print(f"[selftest] resume arm={arm}: max|Δstate|={max_d:.2e} acc_1shot={acc1:.4f} "
              f"acc_resume={acc2:.4f} → {'PASS' if arm_ok else 'FAIL'}")
        ok = ok and arm_ok
        os.remove(os.path.join(CKPT_DIR, f"full_{tag}quickdraw_seed0_{arm}.pt"))

    # 4) eval 机制冒烟：clean A≡B + block5-30 实现掩码率≈0.3（64 样本，随机初始化模型即可验机制）
    print("[selftest] eval mechanics smoke ...")
    model = P.DrawingClassifier("quickdraw", with_gap_bit=False).to(DEVICE).eval()
    xe, ye = x_te[:64], y_te[:64]
    accA, mfA = P.eval_arm_condition(model, "A", xe, ye, "clean", 0, 0)
    accB, mfB = P.eval_arm_condition(model, "B", xe, ye, "clean", 0, 0)
    _, mf5 = P.eval_arm_condition(model, "B", xe, ye, "block5-30", 0, 0)
    ev_ok = (accA["meanpool"] == accB["meanpool"] and abs(mf5 - 0.3) < 0.05)
    print(f"[selftest] eval smoke: A@clean={accA['meanpool']:.4f} B@clean={accB['meanpool']:.4f} "
          f"(equal={accA['meanpool'] == accB['meanpool']}), block5-30 masked={mf5:.3f} → "
          f"{'PASS' if ev_ok else 'FAIL'}")
    ok = ok and ev_ok

    print(f"[selftest] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


_ORIG_D2 = P.d2_timespans          # patch 前快照原循环版（selftest 恒等基准）


# ================= 主流程 =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "train", "eval", "all", "status"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=["ab", "c"], default="ab")
    ap.add_argument("--dataset", default="quickdraw")
    ap.add_argument("--epochs", type=int, default=EPOCHS_FULL)
    args = ap.parse_args()

    _patch_pilot_d2()

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "status":
        prog = load_progress()
        runs = {k: v.get("epochs", v.get("epochs_done", "?")) for k, v in prog.get("runs", {}).items()}
        print(json.dumps({"runs": runs, "eval_rows": len(prog.get("eval_rows", [])),
                          "eval_total": EVAL_TOTAL, "verdict": prog.get("verdict")},
                         ensure_ascii=False))
        return 0

    t_start = time.time()
    x_tr, y_tr, x_te, y_te = full_splits(args.dataset)
    progress = load_progress()
    timing = {"data_prep_sec": round(time.time() - t_start, 1), "train_runs": {},
              "eval_sec": None, "total_sec": None}
    print(f"[full] data: train={x_tr.shape[0]} test={x_te.shape[0]} "
          f"epochs={args.epochs} device={DEVICE}", flush=True)

    if args.cmd == "train":
        key = f"seed{args.seed}_{args.arm}"
        ckpt, acc, loss, sec, ep_done = train_one_full(args.seed, args.arm, args.dataset,
                                                       x_tr, y_tr, epochs=args.epochs)
        if ep_done >= args.epochs:
            progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                     "train_acc": round(acc, 4), "train_loss": round(loss, 4),
                                     "epochs": ep_done, "sec": round(sec, 1)}
            save_progress(progress)
        return 0

    if args.cmd == "all":
        # verify（入口条件④：真实尺寸 manual-loop 复验，与 pilot 同款）
        if "verify_real_size" not in progress["sanity"]:
            progress["sanity"]["verify_real_size"] = P.verify(args.dataset, x_te)
            save_progress(progress)
            print("[all] verify:", json.dumps(progress["sanity"]["verify_real_size"],
                                              ensure_ascii=False), flush=True)
        # S1：6 次训练（A/B 共享 3 种子 + C-aug 3 种子），逐 epoch checkpoint
        for seed in SEEDS:
            for arm in ["ab", "c"]:
                key = f"seed{seed}_{arm}"
                ckpt_path = os.path.join(CKPT_DIR, f"full_{args.dataset}_{key}.pt")
                done_ep = 0
                if os.path.exists(ckpt_path):
                    done_ep = int(torch.load(ckpt_path, map_location="cpu",
                                             weights_only=False).get("epochs_done", 0))
                if key in progress["runs"] and done_ep >= args.epochs:
                    print(f"[all] skip {key} (done {done_ep}/{args.epochs})", flush=True)
                    continue
                ckpt, acc, loss, sec, ep_done = train_one_full(seed, arm, args.dataset,
                                                               x_tr, y_tr, epochs=args.epochs)
                if ep_done >= args.epochs:
                    progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                             "train_acc": round(acc, 4),
                                             "train_loss": round(loss, 4),
                                             "epochs": ep_done, "sec": round(sec, 1)}
                    timing["train_runs"][key] = round(sec, 1)
                    save_progress(progress)
        # S2：评测矩阵（断点续跑：逐行落盘，已完成行跳过）
        rows, eval_sec = run_eval_matrix_full(args, progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        if len(rows) < EVAL_TOTAL:
            print(f"[all] eval incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(progress, timing)
            return 0
        summary = summarize_full(rows)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(progress, timing)
        write_csv_md_full(rows, summary, args.dataset,
                          progress["sanity"].get("verify_real_size", {}), timing)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "eval":
        missing = []
        for seed in SEEDS:
            for arm in ["ab", "c"]:
                pth = os.path.join(CKPT_DIR, f"full_{args.dataset}_seed{seed}_{arm}.pt")
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
        rows, eval_sec = run_eval_matrix_full(args, progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        if len(rows) < EVAL_TOTAL:
            print(f"[eval] incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(progress, timing)
            return 0
        summary = summarize_full(rows)
        progress["verdict"] = summary["verdict"]
        save_progress(progress)
        save_timing(progress, timing)
        write_csv_md_full(rows, summary, args.dataset,
                          progress["sanity"].get("verify_real_size", {}), timing)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
