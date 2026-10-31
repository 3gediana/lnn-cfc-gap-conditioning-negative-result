#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""e2_btrain.py — S7-e2-btrain-arm：E2 归因臂 = D 臂（B-train）。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S7-e2-btrain-arm（73fb33f2 判读放行，2026-10-31）

科学问题（E2 竞争解释的判决性实验）：
  S5 结论级表（results/full_threearm.csv）：block5-30 C−A=+17.49pp（meanpool），
  但 C 臂=增强训练+gap 指示位+timespans 训练期激活三者合体（D3 预注册：归因交 E2）。
  E2 字面主张：「timespans 通路训练期未激活，推理期喂变 dt 用不上」。
  D 臂=与 C 臂完全同训练（同 block5-30 增强调度/同 30 epoch/同种子/同优化器批量 CLIP/
  全量 QuickDraw 42500/逐 epoch 原子 ckpt），唯一差异=不带 gap 指示位
  （DrawingClassifier(with_gap_bit=False)，前向 model.forward_ts(xb, ts)，
  ts=d2_timespans_vec(mask)——timespans 通路训练期激活）。
  C−D=gap 指示位贡献；D−A=训练期激活的时间通路+增强贡献。

与 full_protocol.py 的关系（禁改 A/B/C 既有代码路径 → 本文件 import 其 helpers）：
  - 数据/模型/掩码/timespans/读表全部 import 自 pilot_threearm（P）与 full_protocol（F）；
  - train_one_d 与 F.train_one_full 的 arm='c' 分支逐行同源，差异仅两处（下文标「差异①/②」）；
  - 评测直接复用 P.eval_arm_condition 的 "B" 分支（manual-loop+真实 timespans），
    D 模型 with_gap_bit=False 与该分支语义 exactly 匹配；
  - ckpt 落 experiments/ckpt/full_quickdraw_seed{seed}_d.pt（S7 stage 规定名）；
  - 结果落 results/e2_btrain.{csv,md}+results/e2_btrain_progress.json（不碰 S5 产物）。

预注册判据（跑前写死、跑完机械判定，禁事后改；主口径=meanpool 论文口径，三口径兼报）：
  ① D−A(block5-30) ≥+2pp 且 D clean 降幅 ≤0.5pp
     → C 增益主要来自训练期激活的时间通路+增强（E2 字面主张被反驳：通路训练期激活即可用）；
  ② D−A <+1pp 而 C−A=+17.49pp
     → 增益必需 gap 指示位（输入端信息），时间通路即使训练期激活也无用 → 强化 H1 负结果；
  ③ 居中 → 部分归因，按幅值比例 D−A/(C−A) 陈述。
  clean 降幅门控同时按两种约定报告（S5 已记录的实现双侧 |Δ|≤0.5pp+逐种子 vs 预注册单侧降幅）；
  冒烟=tsprobe（ts 梯度非零+loss 有限+掩码率≈0.30）+ 单种子前 5ep loss 下降，再全量；
  失败条件=D 臂训练不收敛（train acc <C 臂同种子 0.89 的 80%）→ 查实现差异落 partial 不硬撑。

用法：
  python experiments/e2_btrain.py tsprobe                      # 冒烟①：ts 梯度非零（10s）
  python experiments/e2_btrain.py train --seed 0 --epochs 5    # 冒烟②：单种子 5ep（计入正跑）
  python experiments/e2_btrain.py train --seed 0               # 续跑满 30ep（逐 epoch 原子 ckpt）
  python experiments/e2_btrain.py eval                         # 144 格（A/B/C/D），逐行落盘续跑
  python experiments/e2_btrain.py all                          # 3 训练+eval+对账+CSV/MD+判据
  python experiments/e2_btrain.py status                       # 进度快照
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

import pilot_threearm as P          # noqa: E402  （协议同源代码，import 不复制）
import full_protocol as F           # noqa: E402  （full_protocol helpers：vec d2/splits/load_model）

RESULTS = os.path.join(PROJECT, "results")
CKPT_DIR = P.CKPT_DIR
PROGRESS = os.path.join(RESULTS, "e2_btrain_progress.json")
TIMING = os.path.join(RESULTS, "e2_btrain_timing.json")
CSV_OUT = os.path.join(RESULTS, "e2_btrain.csv")
MD_OUT = os.path.join(RESULTS, "e2_btrain.md")
S5_CSV = os.path.join(RESULTS, "full_threearm.csv")   # A/B/C 复现对账基准（S5 结论级表）

EPOCHS_D = 30
ARM = "d"
DEVICE = P.DEVICE
CONDITIONS = P.CONDITIONS
SEEDS = P.SEEDS
N_PATTERNS = P.N_PATTERNS
# 评测臂表：(CSV row 的 arm 名, eval_arm_condition 的 arm 参数, ckpt arm 键)
# A/B/C 入表只为同跑配对（D−A/D−C）+ 与 S5 逐位对账；D 的 36 格才是本阶段新增评测。
EVAL_ARMS = [("A", "A", "ab"), ("B", "B", "ab"), ("C", "C", "c"), ("D", "B", "d")]
EVAL_TOTAL = len(SEEDS) * len(EVAL_ARMS) * len(CONDITIONS) * N_PATTERNS   # 144
D_CELLS = len(SEEDS) * len(CONDITIONS) * N_PATTERNS                       # 36（阶段要求）
# S5 结论级常数（results/full_threearm.csv，meanpool 3 种子 mean；判据②的 C−A 锚点）
C_MINUS_A_BLOCK5_REF_PP = 17.49
C_TRAIN_ACC = {0: 0.8907, 1: 0.8898, 2: 0.9047}   # S5 C 臂收敛参考（failure_conditions 用）


# ================= 训练（与 F.train_one_full arm='c' 逐行同源；差异仅①②） =================

def train_one_d(seed: int, dataset: str, x_tr, y_tr,
                epochs: int = EPOCHS_D, tag: str = ""):
    """D 臂（B-train）：与 C 臂完全同训练，唯一差异=不带 gap 指示位+forward_ts。

    逐行对照 F.train_one_full(seed,'c',...)（full_protocol.py:137-201）：
      同：torch.manual_seed/random.seed(seed)、AdamW(lr=P.LR,wd=P.WD)、CE、
          逐 epoch shuffle g=Generator().manual_seed(seed*100+ep)、P.BATCH、
          mask=P.make_mask("block5-30",B,L,seed*77+ep,0)、xb=xb*(1.0-mask)、
          ts=F.d2_timespans_vec(mask)、clip_grad_norm_(P.CLIP)、逐 epoch 原子 ckpt+续跑。
      差异①：DrawingClassifier(dataset, with_gap_bit=False)（C 臂为 arm=="c"→True）；
      差异②：model.forward_ts(xb, ts)（C 臂为 model.forward_ts_gap(xb, ts, mask)）。
    返回 (ckpt_path, final_train_acc, final_train_loss, sec, epochs_done)。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = P.DrawingClassifier(dataset, with_gap_bit=False).to(DEVICE)      # 差异①
    opt = optim.AdamW(model.parameters(), lr=P.LR, weight_decay=P.WD)
    crit = nn.CrossEntropyLoss()
    ckpt = os.path.join(CKPT_DIR, f"full_{tag}{dataset}_seed{seed}_{ARM}.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)

    start_ep, hist = 0, []
    if os.path.exists(ckpt):                       # 断点续跑：模型+优化器+epoch 全恢复
        payload = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        opt.load_state_dict(payload["opt_state_dict"])
        start_ep = int(payload.get("epochs_done", 0))
        hist = list(payload.get("history", []))
        if start_ep >= epochs:
            print(f"[d-train] skip seed={seed} (done {start_ep}/{epochs})", flush=True)
            return ckpt, hist[-1]["acc"], hist[-1]["loss"], 0.0, start_ep
        print(f"[d-train] resume seed={seed} from epoch {start_ep}", flush=True)

    n = x_tr.shape[0]
    t0 = time.time()
    last_acc = last_loss = float("nan")
    for ep in range(start_ep, epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)   # 与 C 臂同：逐 epoch 固定 shuffle
        perm = torch.randperm(n, generator=g)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, P.BATCH):
            idx = perm[i:i + P.BATCH]
            xb, yb = x_tr[idx].to(DEVICE), y_tr[idx].to(DEVICE)
            opt.zero_grad()
            # 与 C 臂同增强调度：块长5、每块 Bernoulli(0.3)（与 block5-30 同分布的合成间隙）
            mask = P.make_mask("block5-30", xb.shape[0], xb.shape[1], seed * 77 + ep, 0)
            mask = mask.to(DEVICE)
            xb = xb * (1.0 - mask)   # mask=1=掩码(清零)（2026-10-27 修正 Convention，与 C 臂一致）
            ts = F.d2_timespans_vec(mask).to(DEVICE)
            logits, _ = model.forward_ts(xb, ts)                               # 差异②
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
        F._atomic_torch_save({                        # 逐 epoch checkpoint（续跑前提，同 C 臂）
            "state_dict": model.state_dict(), "opt_state_dict": opt.state_dict(),
            "seed": seed, "arm": ARM, "dataset": dataset, "units": P.UNITS,
            "with_gap_bit": False, "epochs_done": ep + 1, "epochs_target": epochs,
            "train_n": n, "train_acc": last_acc, "train_loss": last_loss,
            "train_sec": round(time.time() - t0, 1), "history": hist}, ckpt)
        print(f"[d-train seed={seed} ep={ep + 1}/{epochs}] "
              f"loss={last_loss:.4f} acc={last_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return ckpt, last_acc, last_loss, time.time() - t0, epochs


# ================= 冒烟①：tsprobe（时间通路训练期激活的直接证据） =================

def tsprobe(dataset: str = "quickdraw") -> dict:
    """单 batch 前向+反向，验证 D 臂前向路径：
      (a) loss 有限；(b) ts 梯度非零（timespans 通路训练期激活=E2 反驳/强化的判决量）；
      (c) 实现掩码率≈0.30（Convention 修正后 sanity）；(d) rnn_cell 参数梯度有限且非零。"""
    x_tr, y_tr, _, _ = F.full_splits(dataset)
    torch.manual_seed(0)
    model = P.DrawingClassifier(dataset, with_gap_bit=False).to(DEVICE)
    model.train()
    xb = x_tr[:P.BATCH].to(DEVICE)
    yb = y_tr[:P.BATCH].to(DEVICE)
    mask = P.make_mask("block5-30", xb.shape[0], xb.shape[1], 0 * 77 + 0, 0).to(DEVICE)
    xin = xb * (1.0 - mask)
    ts = F.d2_timespans_vec(mask).to(DEVICE)
    ts.requires_grad_(True)
    logits, _ = model.forward_ts(xin, ts)
    loss = nn.CrossEntropyLoss()(logits, yb)
    loss.backward()
    ts_grad_abs_sum = float(ts.grad.detach().abs().sum().item())
    ts_grad_max = float(ts.grad.detach().abs().max().item())
    cell = {n: float(p.grad.detach().abs().sum().item())
            for n, p in model.named_parameters() if "rnn_cell" in n and p.grad is not None}
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                 if p.grad is not None)
    masked_fraction = float(mask.mean().item())
    rep = {
        "loss": round(float(loss.item()), 6),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "ts_grad_abs_sum": ts_grad_abs_sum,
        "ts_grad_max": ts_grad_max,
        "ts_grad_nonzero": ts_grad_abs_sum > 0.0,
        "rnn_cell_grads": cell,
        "rnn_cell_grads_all_nonzero": all(v > 0 for v in cell.values()) and len(cell) > 0,
        "all_grads_finite": finite,
        "masked_fraction": round(masked_fraction, 4),
        "mask_rate_ok": abs(masked_fraction - 0.30) < 0.05,
        "with_gap_bit": False,
        "forward": "model.forward_ts(xb, ts) (B 式 manual-loop, 无 gap 位)",
    }
    ok = (rep["loss_finite"] and rep["ts_grad_nonzero"] and rep["rnn_cell_grads_all_nonzero"]
          and finite and rep["mask_rate_ok"])
    rep["pass"] = bool(ok)
    print("[tsprobe] " + json.dumps(rep, ensure_ascii=False), flush=True)
    return rep


# ================= 评测矩阵（144 格：A/B/C 同跑配对+对账，D=新增 36 格） =================

def run_eval_matrix_e2(args, progress: dict, x_te, y_te):
    """逐行落盘 + 已完成 (dataset,seed,arm,condition,pattern) 跳过（同 F 机制）。
    D 行走 eval_arm_condition 的 "B" 分支（manual-loop+真实 timespans，无 gap 位）。"""
    rows = list(progress.get("eval_rows", []))
    done = {(r.get("dataset"), r.get("seed"), r.get("arm"), r.get("condition"), r.get("pattern"))
            for r in rows}
    t_eval = time.time()
    for seed in SEEDS:
        models = {row_arm: F.load_model_full(seed, ck, args.dataset)[0]
                  for row_arm, _, ck in EVAL_ARMS}
        for row_arm, eval_arm, ck_key in EVAL_ARMS:
            for cond in CONDITIONS:
                for pattern in range(N_PATTERNS):
                    key = (args.dataset, seed, row_arm, cond, pattern)
                    if key in done:
                        continue
                    accs, mf = P.eval_arm_condition(models[row_arm], eval_arm,
                                                    x_te, y_te, cond, seed, pattern)
                    rows.append({"dataset": args.dataset, "seed": seed, "arm": row_arm,
                                 "condition": cond, "pattern": pattern, "n_eval": x_te.shape[0],
                                 "masked_fraction_realized": round(mf, 4),
                                 "acc_meanpool": round(accs["meanpool"], 4),
                                 "acc_final": round(accs["final"], 4),
                                 "acc_gapnb": round(accs["gapnb"], 4)})
                    progress["eval_rows"] = rows
                    save_progress(progress)      # 逐行持久化：超时/kill 不丢已出行
            print(f"[e2-eval] seed={seed} arm={row_arm} done "
                  f"({time.time() - t_eval:.0f}s cum, {len(rows)}/{EVAL_TOTAL} rows)", flush=True)
    return rows, round(time.time() - t_eval, 1)


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
    payload = {"schema": "e2-btrain-timing/v1", "stage": "S7-e2-btrain-arm",
               "u5_runtime": extra}
    with open(TIMING, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ================= 汇总 + 预注册判据机械输出 =================

def summarize_e2(rows: list[dict], progress: dict) -> dict:
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
        for arm in ["A", "B", "C", "D"]:
            vals = pick(arm, cond)
            if vals:
                entry[arm] = {"mean": st.mean(vals),
                              "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
                              "n": len(vals), "per_seed": vals}
        a, c, d = entry.get("A", {}).get("mean"), entry.get("C", {}).get("mean"), \
            entry.get("D", {}).get("mean")
        if a is not None and d is not None:
            entry["D_minus_A"] = d - a
        if c is not None and d is not None:
            entry["D_minus_C"] = d - c
        summary["conditions"][cond] = entry

    s = summary["sanity"]
    # ---- 与 S5 结论级表逐位对账（A/B/C 同 ckpt 同路径同跑，必须零差）----
    s5 = {}
    if os.path.exists(S5_CSV):
        with open(S5_CSV, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                s5[(r["seed"], r["arm"], r["condition"], r["pattern"])] = r
    mism = []
    for r in rows:
        if r["arm"] == "D":
            continue
        k = (str(r["seed"]), r["arm"], r["condition"], str(r["pattern"]))
        ref = s5.get(k)
        if ref is None:
            mism.append({"key": list(k), "issue": "missing_in_s5"})
            continue
        for col in ["acc_meanpool", "acc_final", "acc_gapnb"]:
            if abs(float(ref[col]) - float(r[col])) > 1e-9:
                mism.append({"key": list(k), "col": col,
                             "s5": ref[col], "e2": r[col]})
    s["s5_reproduction_mismatches"] = mism
    s["s5_reproduction_identical"] = (len(mism) == 0)
    # ---- D 与 B 同 eval 路径 sanity：D 行走 "B" 分支；clean 下 B 行对 S5 逐位一致即证路径无漂移 ----
    s["D_eval_path"] = "P.eval_arm_condition(model_D, 'B', ...) = forward_ts manual-loop（与 B 臂逐行同分支）"
    s["clean_D_minus_B_pp"] = (
        (summary["conditions"]["clean"].get("D", {}).get("mean", 0)
         - summary["conditions"]["clean"].get("B", {}).get("mean", 0)) * 100
        if summary["conditions"]["clean"].get("D") and summary["conditions"]["clean"].get("B")
        else None)
    # ---- 掩码率 sanity（全 D 行）----
    d_mf = [r["masked_fraction_realized"] for r in rows if r["arm"] == "D"]
    s["D_masked_fraction_mean"] = round(sum(d_mf) / len(d_mf), 4) if d_mf else None
    # ---- 收敛伴生判据：D train acc vs C 臂同种子（failure_conditions）----
    s["train_acc_D_vs_C"] = {}
    for seed in SEEDS:
        d_run = progress.get("runs", {}).get(f"seed{seed}_d", {})
        if d_run:
            ratio = d_run.get("train_acc", 0) / C_TRAIN_ACC.get(seed, 1e-9)
            s["train_acc_D_vs_C"][str(seed)] = {
                "D_train_acc": d_run.get("train_acc"),
                "C_train_acc_ref": C_TRAIN_ACC.get(seed),
                "ratio_vs_C": round(ratio, 4),
                "converged_ge_80pct_C": bool(d_run.get("train_acc", 0) >= 0.8 * C_TRAIN_ACC.get(seed, 1))}

    # ---- 预注册判据（主口径 meanpool；跑前写死）----
    v = summary["verdict"]
    b5 = summary["conditions"]["block5-30"]
    v["D_minus_A_block5_pp"] = b5.get("D_minus_A") * 100 if b5.get("D_minus_A") is not None else None
    v["D_minus_C_block5_pp"] = b5.get("D_minus_C") * 100 if b5.get("D_minus_C") is not None else None
    v["C_minus_A_block5_pp_in_run"] = b5.get("C_minus_A") * 100 if b5.get("C_minus_A") is not None else None
    v["C_minus_A_block5_pp_ref_S5"] = C_MINUS_A_BLOCK5_REF_PP
    v["D_minus_A_block5_per_seed_pp"] = {
        str(k): round((pv - a_b5) * 100, 2)
        for k, (pv, a_b5) in
        ((k, (pv, per_seed("A", "block5-30").get(k, float("nan"))))
         for k, pv in per_seed("D", "block5-30").items())
        if k in per_seed("A", "block5-30")}
    # 三口径兼报（block5-30）
    v["three_calibers_block5_pp"] = {}
    for k in ["meanpool", "final", "gapnb"]:
        a_v = st.mean(pick("A", "block5-30", f"acc_{k}")) if pick("A", "block5-30", f"acc_{k}") else None
        d_v = st.mean(pick("D", "block5-30", f"acc_{k}")) if pick("D", "block5-30", f"acc_{k}") else None
        c_v = st.mean(pick("C", "block5-30", f"acc_{k}")) if pick("C", "block5-30", f"acc_{k}") else None
        v["three_calibers_block5_pp"][k] = {
            "D_minus_A": round((d_v - a_v) * 100, 2) if (d_v is not None and a_v is not None) else None,
            "D_minus_C": round((d_v - c_v) * 100, 2) if (d_v is not None and c_v is not None) else None}
    # clean 降幅（mean + 逐种子同种子配对）
    clean = summary["conditions"]["clean"]
    a_clean_ps = per_seed("A", "clean")
    d_clean_ps = per_seed("D", "clean")
    v["D_clean_drop_pp"] = clean.get("D_minus_A") * 100 if clean.get("D_minus_A") is not None else None
    v["D_clean_drop_pp_per_seed"] = {str(k): round((pv - a_clean_ps[k]) * 100, 2)
                                     for k, pv in d_clean_ps.items() if k in a_clean_ps}
    drops = list(v["D_clean_drop_pp_per_seed"].values())
    # 约定甲（S5 实现口径）：双侧 |Δ|≤0.5pp（mean 且逐种子全过）
    v["D_clean_drop_ok_two_sided"] = bool(
        v["D_clean_drop_pp"] is not None and abs(v["D_clean_drop_pp"]) <= 0.5
        and drops and all(abs(x) <= 0.5 for x in drops))
    # 约定乙（预注册字面）：单侧降幅 ≤0.5pp（mean）
    v["D_clean_drop_ok_literal_one_sided"] = bool(
        v["D_clean_drop_pp"] is not None and v["D_clean_drop_pp"] <= 0.5)
    # ---- 判据①②③ 机械输出 ----
    dma = v["D_minus_A_block5_pp"]
    if dma is None:
        v["criterion"], v["decision"], v["next_action"] = "n/a", "incomplete", "continue"
    elif dma >= 2.0 and v["D_clean_drop_ok_two_sided"]:
        v["criterion"] = "①"
        v["decision"] = "E2_refuted__timespans_pathway_active_in_training_suffices"
        v["next_action"] = "continue"
        v["attribution_ratio_D_over_C"] = round(dma / C_MINUS_A_BLOCK5_REF_PP, 4)
    elif dma < 1.0:
        v["criterion"] = "②"
        v["decision"] = "gap_bit_required__strengthens_H1_negative"
        v["next_action"] = "continue"
        v["attribution_ratio_D_over_C"] = round(dma / C_MINUS_A_BLOCK5_REF_PP, 4)
    else:
        v["criterion"] = "③"
        v["decision"] = "partial_attribution"
        v["next_action"] = "continue"
        v["attribution_ratio_D_over_C"] = round(dma / C_MINUS_A_BLOCK5_REF_PP, 4)
    v["criterion_note"] = ("①D−A≥+2pp 且 clean 降幅≤0.5pp→E2 被反驳（训练期激活即可用）；"
                           "②D−A<+1pp 且 C−A=+17.49pp→必需 gap 指示位，强化 H1 负结果；"
                           "③居中→按 D−A/(C−A) 幅值比例部分归因。主口径 meanpool。")
    # 训练收敛失败条件（D train acc <C 臂 80%）
    conv = s.get("train_acc_D_vs_C", {})
    v["D_convergence_failure"] = any(not x.get("converged_ge_80pct_C", True) for x in conv.values()) \
        if conv else None
    return summary


def write_csv_md_e2(rows: list[dict], summary: dict, dataset: str, timing: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    cols = ["dataset", "seed", "arm", "condition", "pattern", "n_eval",
            "masked_fraction_realized", "acc_meanpool", "acc_final", "acc_gapnb"]
    rows_sorted = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                              CONDITIONS.index(r["condition"]), r["pattern"]))
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({c: r.get(c) for c in cols})

    def fmt(entry, arm):
        e = entry.get(arm)
        return f"{e['mean'] * 100:.2f}±{e['std'] * 100:.2f}" if e else "—"

    L = []
    L.append("# S7-e2-btrain-arm：E2 归因臂 = D 臂（B-train）结果（% acc，3 种子 mean±std）\n")
    L.append(f"- 数据集：{dataset}（论文协议全量：train 42500 / test 7500，与 S5 同 split 种子 12345）")
    L.append(f"- 训练：D 臂 3 种子×{EPOCHS_D} epoch，与 C 臂完全同训练（同 block5-30 增强调度 "
             f"P.make_mask(\"block5-30\",B,L,seed*77+ep,0)、同优化器/批量/CLIP、逐 epoch 原子 ckpt），"
             f"唯一差异=DrawingClassifier(with_gap_bit=False)+forward_ts（timespans 通路训练期激活）")
    L.append("- 评测：D=36 格（3 种子×4 条件×3 模式，B 式 manual-loop+真实 timespans、无 gap 位）；"
             "A/B/C 同跑入表仅用于 D−A/D−C 配对与 S5 对账（非本阶段新增）")
    L.append("- 读表口径：预注册主口径=meanpool（论文口径），三口径 meanpool/gapnb/final 兼报")
    L.append("- 臂：A=库 forward(timespans=None,均匀dt=1)；B=A 共享 ckpt+manual-loop D2 timespans；"
             "C=B+gap 指示位+增强训练；D=C 同训练但不带 gap 指示位（timespans 训练期激活）")
    L.append("- 条件：clean / uniform30(逐时间步 Bernoulli 0.3) / block5-30、block15-30（块长 b 划分、"
             "每块 Bernoulli 0.3，期望掩码率 30%；3 固定种子化模式跨臂配对）\n")
    L.append("| 条件 | A | B | C | D | D−A(pp) | D−C(pp) |")
    L.append("|---|---|---|---|---|---|---|")
    for cond in CONDITIONS:
        e = summary["conditions"][cond]
        dma = f"{e['D_minus_A'] * 100:+.2f}" if e.get("D_minus_A") is not None else "—"
        dmc = f"{e['D_minus_C'] * 100:+.2f}" if e.get("D_minus_C") is not None else "—"
        L.append(f"| {cond} | {fmt(e, 'A')} | {fmt(e, 'B')} | {fmt(e, 'C')} | {fmt(e, 'D')} "
                 f"| {dma} | {dmc} |")
    L.append("")
    s = summary["sanity"]
    L.append("## Sanity")
    L.append(f"- A/B/C 与 S5 结论级表（results/full_threearm.csv）同跑逐位对账："
             f"identical={s.get('s5_reproduction_identical')}，"
             f"mismatches={len(s.get('s5_reproduction_mismatches', []))}"
             + ("（0=确定性+路径无漂移）" if s.get("s5_reproduction_identical") else ""))
    L.append(f"- D 与 B 同 eval 路径：{s.get('D_eval_path')}；clean 下 D−B="
             f"{s.get('clean_D_minus_B_pp')}pp"
             f"（不同 ckpt 属预期；路径同一由 B 行对 S5 逐位一致证明）")
    L.append(f"- D 行实现掩码率 mean={s.get('D_masked_fraction_mean')}（名义 0.30）")
    if s.get("train_acc_D_vs_C"):
        L.append(f"- 收敛伴生判据（D train acc vs C 臂同种子 80% 门）：{json.dumps(s['train_acc_D_vs_C'], ensure_ascii=False)}")
    L.append("")
    v = summary["verdict"]
    L.append("## 预注册判据机械输出（跑前写死；主口径 meanpool）")
    L.append(f"- D−A(block5-30) = {v['D_minus_A_block5_pp']:+.2f}pp（逐种子 pp：{v['D_minus_A_block5_per_seed_pp']}）"
             if v["D_minus_A_block5_pp"] is not None else "- D−A(block5-30) = —")
    L.append(f"- C−A(block5-30) = 同跑 {v['C_minus_A_block5_pp_in_run']:+.2f}pp / S5 结论级锚点 "
             f"{v['C_minus_A_block5_pp_ref_S5']:+.2f}pp"
             if v["C_minus_A_block5_pp_in_run"] is not None else
             f"- C−A(block5-30) = S5 结论级锚点 {v['C_minus_A_block5_pp_ref_S5']:+.2f}pp")
    L.append(f"- D−C(block5-30) = {v['D_minus_C_block5_pp']:+.2f}pp（=gap 指示位贡献的配对估计）"
             if v["D_minus_C_block5_pp"] is not None else "- D−C(block5-30) = —")
    L.append(f"- D clean 降幅 mean={v['D_clean_drop_pp']:+.2f}pp、逐种子={v['D_clean_drop_pp_per_seed']}；"
             f"双侧约定(实现口径)PASS={v['D_clean_drop_ok_two_sided']}、"
             f"单侧字面(预注册)PASS={v['D_clean_drop_ok_literal_one_sided']}"
             if v["D_clean_drop_pp"] is not None else "- D clean 降幅 = —")
    L.append(f"- 三口径兼报（block5-30）：{json.dumps(v['three_calibers_block5_pp'], ensure_ascii=False)}")
    L.append(f"- **判据={v['criterion']}：decision={v['decision']}，next_action={v['next_action']}**"
             + (f"（幅值比例 D−A/(C−A)={v['attribution_ratio_D_over_C']}）"
                if v.get("attribution_ratio_D_over_C") is not None else ""))
    L.append(f"- 判据字面：{v['criterion_note']}")
    if v.get("D_convergence_failure"):
        L.append(f"- **训练收敛失败条件触发**：D train acc <C 臂 80% → {json.dumps(s.get('train_acc_D_vs_C'), ensure_ascii=False)}")
    L.append("")
    L.append("## 限制")
    L.append("- QuickDraw 降级数据集（N-MNIST 结论级基准=S8 后续）；L=100 使 block15 为字面块长")
    L.append("- D7 保真度缺口（S5：A@clean 92.30% vs 论文 98.6-98.7%）未在本阶段处理（S9 归因）")
    L.append(f"- 计时（U5）：{json.dumps(timing, ensure_ascii=False)}")
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ================= 主流程 =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tsprobe", "train", "eval", "all", "status"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="quickdraw")
    ap.add_argument("--epochs", type=int, default=EPOCHS_D)
    args = ap.parse_args()

    F._patch_pilot_d2()   # 与 S5 eval 同：向量化 d2（恒等，selftest 68 项担保）

    if args.cmd == "tsprobe":
        rep = tsprobe(args.dataset)
        return 0 if rep["pass"] else 1

    if args.cmd == "status":
        prog = load_progress()
        runs = {k: {"epochs": v.get("epochs", v.get("epochs_done", "?")),
                    "train_acc": v.get("train_acc")}
                for k, v in prog.get("runs", {}).items()}
        print(json.dumps({"runs": runs, "eval_rows": len(prog.get("eval_rows", [])),
                          "eval_total": EVAL_TOTAL, "verdict": prog.get("verdict")},
                         ensure_ascii=False))
        return 0

    t_start = time.time()
    x_tr, y_tr, x_te, y_te = F.full_splits(args.dataset)
    progress = load_progress()
    timing = {"data_prep_sec": round(time.time() - t_start, 1), "train_runs": {},
              "eval_sec": None, "total_sec": None}
    print(f"[e2] data: train={x_tr.shape[0]} test={x_te.shape[0]} "
          f"epochs={args.epochs} device={DEVICE}", flush=True)

    if args.cmd == "train":
        key = f"seed{args.seed}_{ARM}"
        ckpt, acc, loss, sec, ep_done = train_one_d(args.seed, args.dataset,
                                                    x_tr, y_tr, epochs=args.epochs)
        if ep_done >= args.epochs:
            progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                     "train_acc": round(acc, 4), "train_loss": round(loss, 4),
                                     "epochs": ep_done, "sec": round(sec, 1)}
            save_progress(progress)
        return 0

    if args.cmd == "all":
        for seed in SEEDS:
            key = f"seed{seed}_{ARM}"
            ckpt_path = os.path.join(CKPT_DIR, f"full_{args.dataset}_seed{seed}_{ARM}.pt")
            done_ep = 0
            if os.path.exists(ckpt_path):
                done_ep = int(torch.load(ckpt_path, map_location="cpu",
                                         weights_only=False).get("epochs_done", 0))
            if key in progress["runs"] and done_ep >= args.epochs:
                print(f"[all] skip {key} (done {done_ep}/{args.epochs})", flush=True)
                continue
            ckpt, acc, loss, sec, ep_done = train_one_d(seed, args.dataset,
                                                        x_tr, y_tr, epochs=args.epochs)
            if ep_done >= args.epochs:
                progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                         "train_acc": round(acc, 4),
                                         "train_loss": round(loss, 4),
                                         "epochs": ep_done, "sec": round(sec, 1)}
                timing["train_runs"][key] = round(sec, 1)
                save_progress(progress)
        # 评测门：3 个 D ckpt 必须 30/30 才评（未齐拒评，同 S5 纪律）
        missing = []
        for seed in SEEDS:
            pth = os.path.join(CKPT_DIR, f"full_{args.dataset}_seed{seed}_{ARM}.pt")
            if not os.path.exists(pth):
                missing.append(os.path.basename(pth))
            else:
                ep = int(torch.load(pth, map_location="cpu", weights_only=False)
                         .get("epochs_done", 0))
                if ep < args.epochs:
                    missing.append(f"{os.path.basename(pth)}({ep}/{args.epochs})")
        if missing:
            print(f"[all] D training incomplete, {len(missing)} run(s) not ready: {missing}; "
                  f"resume with the same command (per-epoch ckpt, lossless)", flush=True)
            save_progress(progress)
            save_timing(progress, timing)
            return 0
        rows, eval_sec = run_eval_matrix_e2(args, progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        d_rows = [r for r in rows if r["arm"] == "D"]
        if len(d_rows) < D_CELLS or len(rows) < EVAL_TOTAL:
            print(f"[all] eval incomplete: D {len(d_rows)}/{D_CELLS}, total {len(rows)}/{EVAL_TOTAL}; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(progress, timing)
            return 0
        summary = summarize_e2(rows, progress)
        progress["verdict"] = summary["verdict"]
        progress["sanity"] = summary["sanity"]
        save_progress(progress)
        save_timing(progress, timing)
        write_csv_md_e2(rows, summary, args.dataset, timing)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "eval":
        rows, eval_sec = run_eval_matrix_e2(args, progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        d_rows = [r for r in rows if r["arm"] == "D"]
        if len(d_rows) < D_CELLS or len(rows) < EVAL_TOTAL:
            print(f"[eval] incomplete: D {len(d_rows)}/{D_CELLS}, total {len(rows)}/{EVAL_TOTAL}; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(progress, timing)
            return 0
        summary = summarize_e2(rows, progress)
        progress["verdict"] = summary["verdict"]
        progress["sanity"] = summary["sanity"]
        save_progress(progress)
        save_timing(progress, timing)
        write_csv_md_e2(rows, summary, args.dataset, timing)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
