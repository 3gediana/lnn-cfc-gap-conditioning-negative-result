#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quickdraw_lrdecay.py — S11-quickdraw-lrdecay-retrain：QuickDraw A/B 共享 ckpt ×3 种子 ×30ep，
唯一变量=LR 调度（论文 step decay），隔离 D7 保真度缺口归因。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S11-quickdraw-lrdecay-retrain（S9 判读立项；workflow.json 判据跑前冻结）

与 S5（experiments/full_protocol.py）的关系（import-not-copy，禁协议漂移）：
  - 模型/掩码/timespans(D2)/读表三口径/臂语义(D1/D4)/数据切分(SPLIT_SEED=12345, 42500/7500)/
    batch128/AdamW(wd0.01)/clip1.0/meanpool 主口径——全部 import 自 pilot_threearm(P) 与
    full_protocol(F)，零复制零改写；
  - 唯一差异=LR 有效曲线：本仓 S5 恒定 1e-3（full_protocol.py:145 无 scheduler）→ 本阶段按论文
    仓库日志 experiments/_repo/0.06_result/lnn_10/lln_10.log（S9 判读独立取证：12×1e-3/
    13×5e-4/5×2.5e-4=30ep，TEST 0.9577）实施 step decay：
        1e-3（ep1-12）→ 5e-4（ep13-25）→ 2.5e-4（ep26-30）
    实现=epoch 边界手写 step（论文 Table V scheduler 列『—』=手动衰减非 torch scheduler，
    2f8565ee 判读增注：手写 step 或 lr_scheduler 均可，隔离变量语义不变）；
  - 只训 A/B 共享臂（arm='ab'，论文协议干净训练；B 臂推理期 manual-loop+D2，与 A 共享 ckpt/掩码）；
    C 臂不属于本阶段（F1 同款不适用，不做臂间新判据）。
  - ckpt 命名 quickdraw_lrdecay_seed{seed}_ab.pt（与 S5 full_* / pilot_* 分离，不覆盖证据）；
    progress/timing/CSV/MD 全部落 results/quickdraw_lrdecay_*（不碰 S5 产物）。

判据（跑前冻结，workflow.json S11.actions，禁事后改）：
  ① A@clean(3 种子 mean, meanpool) vs S5 恒定 LR 基线 92.30±0.86：
     增益 ≥+1.5pp → D7 缺口主因=LR 调度成立（supported）；
     增益 <+0.5pp → LR 非主因（falsified，负结果落盘不开第二轮调参）；
     其间 → inconclusive（留 dropout 头/数据量 30min 验法选项）。
  ② clean A≡B <0.1pp sanity。
  ③ A@uniform30 兼报 vs 论文 Table III QuickDraw 30%=68.80。

禁改分类头（Dropout 留待判据①后再定）/禁改 S5 冻结口径/禁手写并行分片/禁 silent 延长 epoch。

用法：
  python experiments/quickdraw_lrdecay.py selftest   # LR 调度表+续跑等价（轻量，无 GPU 大开销）
  python experiments/quickdraw_lrdecay.py smoke      # 子步1：seed0 单 run 2ep——loss 下降+clean A≡B
  python experiments/quickdraw_lrdecay.py train --seed 0
  python experiments/quickdraw_lrdecay.py eval       # 72 行矩阵（3 种子×A/B×4 条件×3 模式），逐行落盘续跑
  python experiments/quickdraw_lrdecay.py all        # verify+3 训练+eval+CSV/MD（全程断点续跑）
  python experiments/quickdraw_lrdecay.py status     # 进度快照（轮询用）
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
import full_protocol as F           # noqa: E402  （S5 同代码路径：数据切分/向量化 d2/原子写）

RESULTS = os.path.join(PROJECT, "results")
CKPT_DIR = P.CKPT_DIR
PROGRESS = os.path.join(RESULTS, "quickdraw_lrdecay_progress.json")
TIMING = os.path.join(RESULTS, "quickdraw_lrdecay_timing.json")
CSV_OUT = os.path.join(RESULTS, "quickdraw_lrdecay.csv")
MD_OUT = os.path.join(RESULTS, "quickdraw_lrdecay.md")

# ---- 协议常数（与 S5 逐项一致；唯一变量=LR 调度）----
EPOCHS_LRD = 30                     # 与 S5 EPOCHS_FULL=30、论文 30ep 一致（S9 差异表第 1 行）
SEEDS = P.SEEDS                     # [0,1,2]
CONDITIONS = P.CONDITIONS           # clean/uniform30/block5-30/block15-30
N_PATTERNS = P.N_PATTERNS           # 3（D4 固定种子化模式，跨臂同掩码配对）
EVAL_TOTAL = len(SEEDS) * 2 * len(CONDITIONS) * N_PATTERNS   # 72 行（每种子 24 行）

# ---- 论文 LR step decay（lln_10.log 逐 epoch LR 行，S9 判读独立取证双源确认）----
LR_BASE = P.LR                      # 1e-3（Table V 与本仓一致）
LR_STEP1_AT_EPOCH = 13              # ep13 起 5e-4
LR_STEP2_AT_EPOCH = 26              # ep26 起 2.5e-4
LR_MID, LR_LOW = 5e-4, 2.5e-4
LR_SCHEDULE_DESC = ("paper step decay: 1e-3(ep1-12) -> 5e-4(ep13-25) -> 2.5e-4(ep26-30) "
                    "(lln_10.log; S9 judge re-extract 12x1e-3/13x5e-4/5x2.5e-4=30ep)")


def lr_at_epoch(ep_idx: int) -> float:
    """ep_idx=0-based → 该 epoch 有效 LR（epoch 边界手动 step）。"""
    ep = ep_idx + 1
    if ep < LR_STEP1_AT_EPOCH:
        return LR_BASE
    if ep < LR_STEP2_AT_EPOCH:
        return LR_MID
    return LR_LOW


def lr_schedule_table(epochs: int = EPOCHS_LRD) -> list:
    return [lr_at_epoch(e) for e in range(epochs)]


# ---- S5 恒定 LR 基线（results/full_threearm.csv 实算，meanpool 主口径；判据①参照系）----
S5_BASE = {
    "A_clean_mean": 0.9230, "A_clean_std_pp": 0.86,
    "A_clean_per_seed": {"0": 0.9109, "1": 0.9304, "2": 0.9277},
    "A_uniform30_mean": 0.6393, "A_uniform30_std_pp": 2.86,
    "source": "results/full_threearm.csv (S5-full-protocol, constant LR 1e-3, 3 seeds)",
}
# ---- 论文锚点（PDF Table I/III；S9 判读直读）----
PAPER = {
    "tableI_quickdraw_lnn_test": 0.9577,
    "tableIII_quickdraw_0pct": 0.9269, "tableIII_quickdraw_30pct": 0.6880,
}


# ================= 训练（唯一变量=LR step decay；其余与 F.train_one_full 逐行同源） =================

def _atomic_torch_save(payload: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # 原子替换：kill 不留半截 ckpt


def train_one_lrdecay(seed: int, x_tr, y_tr, epochs: int = EPOCHS_LRD, tag: str = ""):
    """A/B 共享臂干净训练（arm='ab'），唯一差异 vs S5=每 epoch 边界按论文 step decay 设 LR。
    与 full_protocol.train_one_full 同：逐 epoch 固定 shuffle(seed*100+ep)、batch128、clip1.0、
    AdamW(wd0.01)、forward_lib。逐 epoch 原子 ckpt + 断点续跑。
    返回 (ckpt_path, final_train_acc, final_train_loss, sec, epochs_done)。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = P.DrawingClassifier("quickdraw", with_gap_bit=False).to(F.DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=P.WD)
    crit = nn.CrossEntropyLoss()
    ckpt = os.path.join(CKPT_DIR, f"quickdraw_lrdecay_{tag}seed{seed}_ab.pt")
    os.makedirs(CKPT_DIR, exist_ok=True)

    start_ep, hist = 0, []
    if os.path.exists(ckpt):                       # 断点续跑：模型+优化器+epoch 全恢复
        payload = torch.load(ckpt, map_location=F.DEVICE, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        opt.load_state_dict(payload["opt_state_dict"])
        start_ep = int(payload.get("epochs_done", 0))
        hist = list(payload.get("history", []))
        if start_ep >= epochs:
            print(f"[lrdecay-train] skip seed={seed} (done {start_ep}/{epochs})", flush=True)
            return ckpt, hist[-1]["acc"], hist[-1]["loss"], 0.0, start_ep
        print(f"[lrdecay-train] resume seed={seed} from epoch {start_ep}", flush=True)

    n = x_tr.shape[0]
    t0 = time.time()
    last_acc = last_loss = float("nan")
    for ep in range(start_ep, epochs):
        lr_now = lr_at_epoch(ep)                    # ← 唯一变量：epoch 边界手动 step
        for g in opt.param_groups:
            g["lr"] = lr_now
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + ep)   # 与 S5/pilot 同：逐 epoch 固定 shuffle
        perm = torch.randperm(n, generator=g)
        tot_loss, correct, seen = 0.0, 0, 0
        for i in range(0, n, P.BATCH):
            idx = perm[i:i + P.BATCH]
            xb, yb = x_tr[idx].to(F.DEVICE), y_tr[idx].to(F.DEVICE)
            opt.zero_grad()
            logits, _ = model.forward_lib(xb)       # A/B 共享：论文协议干净训练（不经掩码）
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), P.CLIP)
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
        last_acc, last_loss = correct / seen, tot_loss / seen
        hist.append({"epoch": ep + 1, "loss": round(last_loss, 4), "acc": round(last_acc, 4),
                     "lr": lr_now, "sec": round(time.time() - t0, 1)})
        _atomic_torch_save({                        # 逐 epoch checkpoint（续跑前提）
            "state_dict": model.state_dict(), "opt_state_dict": opt.state_dict(),
            "seed": seed, "arm": "ab", "dataset": "quickdraw", "units": P.UNITS,
            "with_gap_bit": False, "epochs_done": ep + 1, "epochs_target": epochs,
            "train_n": n, "train_acc": last_acc, "train_loss": last_loss,
            "train_sec": round(time.time() - t0, 1), "history": hist,
            "lr_schedule": LR_SCHEDULE_DESC, "lr_final": lr_now}, ckpt)
        print(f"[lrdecay-train seed={seed} ep={ep + 1}/{epochs}] lr={lr_now:.1e} "
              f"loss={last_loss:.4f} acc={last_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return ckpt, last_acc, last_loss, time.time() - t0, epochs


def load_model_lrdecay(seed: int):
    ckpt = os.path.join(CKPT_DIR, f"quickdraw_lrdecay_seed{seed}_ab.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    payload = torch.load(ckpt, map_location=F.DEVICE, weights_only=False)
    model = P.DrawingClassifier("quickdraw", with_gap_bit=False)
    model.load_state_dict(payload["state_dict"])
    return model.to(F.DEVICE).eval(), payload


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
    payload = {"schema": "lrdecay-timing/v1", "stage": "S11-quickdraw-lrdecay-retrain",
               "u5_runtime": extra}
    with open(TIMING, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ================= 评测矩阵（A/B ×4 条件×3 模式×3 种子=72 行；逐行落盘续跑） =================

def run_eval_matrix(progress: dict, x_te, y_te):
    """A/B 共享同一 ab ckpt（D1）；B=manual-loop+D2 真实 timespans；掩码 D4 固定 3 模式跨臂配对
    （make_mask 对 (seed,cond,pattern) 确定性，A/B 逐位同掩码）。全量 test 7500。"""
    rows = list(progress.get("eval_rows", []))
    done = {(r.get("seed"), r.get("arm"), r.get("condition"), r.get("pattern")) for r in rows}
    t_eval = time.time()
    for seed in SEEDS:
        model, _ = load_model_lrdecay(seed)
        for arm_name in ["A", "B"]:
            for cond in CONDITIONS:
                for pattern in range(N_PATTERNS):
                    key = (seed, arm_name, cond, pattern)
                    if key in done:
                        continue
                    accs, mf = P.eval_arm_condition(model, arm_name, x_te, y_te, cond, seed, pattern)
                    rows.append({"dataset": "quickdraw", "seed": seed, "arm": arm_name,
                                 "condition": cond, "pattern": pattern, "n_eval": x_te.shape[0],
                                 "masked_fraction_realized": round(mf, 4),
                                 "acc_meanpool": round(accs["meanpool"], 4),
                                 "acc_final": round(accs["final"], 4),
                                 "acc_gapnb": round(accs["gapnb"], 4)})
                    progress["eval_rows"] = rows
                    save_progress(progress)      # 逐行持久化：超时/kill 不丢已出行
            print(f"[lrdecay-eval] seed={seed} arm={arm_name} done "
                  f"({time.time() - t_eval:.0f}s cum, {len(rows)}/{EVAL_TOTAL} rows)", flush=True)
    return rows, round(time.time() - t_eval, 1)


# ================= 汇总（CSV/MD + 判据机械输出） =================

def summarize_lrdecay(rows: list[dict]) -> dict:
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

    summary = {"conditions": {}, "sanity": {}, "criteria": {}}
    for cond in CONDITIONS:
        entry = {}
        for arm in ["A", "B"]:
            vals = pick(arm, cond)
            if vals:
                entry[arm] = {"mean": st.mean(vals), "std": st.pstdev(vals) if len(vals) > 1 else 0.0,
                              "n": len(vals), "per_seed": vals}
        if "A" in entry and "B" in entry:
            entry["B_minus_A"] = entry["B"]["mean"] - entry["A"]["mean"]
        summary["conditions"][cond] = entry

    a_clean = summary["conditions"]["clean"].get("A", {}).get("mean")
    b_clean = summary["conditions"]["clean"].get("B", {}).get("mean")
    a_u30 = summary["conditions"]["uniform30"].get("A", {}).get("mean")
    s = summary["sanity"]
    s["clean_A_mean"] = a_clean
    s["clean_B_mean"] = b_clean
    s["clean_A_minus_B_pp"] = (a_clean - b_clean) * 100 if (a_clean is not None and b_clean is not None) else None
    s["clean_A_equals_B_lt_0p1pp"] = (s["clean_A_minus_B_pp"] is not None
                                      and abs(s["clean_A_minus_B_pp"]) < 0.1)
    s["A_uniform30_mean"] = a_u30
    n_eval_u30 = max([r.get("n_eval", 0) for r in rows
                      if r["arm"] == "A" and r["condition"] == "uniform30"] or [0])
    if a_u30 is not None and n_eval_u30 > 0:
        z_u30 = (a_u30 - 0.10) / (0.10 * 0.90 / n_eval_u30) ** 0.5
        s["A_uniform30_z_vs_chance"] = round(z_u30, 2)
        s["A_uniform30_direction_ok"] = z_u30 > 2.0
    # 恒定类消失伴生判据（S5 须知同款）：clean 下 A 臂三口径分化
    a_clean_cal = {k: st.mean(pick("A", "clean", f"acc_{k}")) if pick("A", "clean", f"acc_{k}") else None
                   for k in ["meanpool", "final", "gapnb"]}
    s["A_clean_three_calibers"] = a_clean_cal
    vals = [v for v in a_clean_cal.values() if v is not None]
    s["A_clean_caliber_spread_pp"] = (max(vals) - min(vals)) * 100 if len(vals) == 3 else None

    # ---- 判据机械输出（跑前冻结）----
    c = summary["criteria"]
    c["s5_baseline"] = S5_BASE
    c["paper_anchors"] = PAPER
    c["A_clean_mean"] = a_clean
    c["A_clean_std_pp"] = (summary["conditions"]["clean"]["A"]["std"] * 100
                           if a_clean is not None else None)
    c["gain_vs_s5_pp"] = (a_clean - S5_BASE["A_clean_mean"]) * 100 if a_clean is not None else None
    gain = c["gain_vs_s5_pp"]
    if gain is None:
        c["criterion1"] = "incomplete"
    elif gain >= 1.5:
        c["criterion1"] = "supported"        # D7 缺口主因=LR 调度成立
    elif gain < 0.5:
        c["criterion1"] = "falsified"        # LR 非主因（负结果落盘不开第二轮调参）
    else:
        c["criterion1"] = "inconclusive"     # 留 dropout 头/数据量 30min 验法选项
    c["criterion2_clean_A_equals_B_pass"] = s["clean_A_equals_B_lt_0p1pp"]
    c["A_uniform30_gap_vs_paper_68_80_pp"] = ((a_u30 - PAPER["tableIII_quickdraw_30pct"]) * 100
                                              if a_u30 is not None else None)
    c["d7_gap_vs_paper_tableI_pp"] = ((a_clean - PAPER["tableI_quickdraw_lnn_test"]) * 100
                                      if a_clean is not None else None)
    c["d7_gap_vs_paper_tableIII_0pct_pp"] = ((a_clean - PAPER["tableIII_quickdraw_0pct"]) * 100
                                             if a_clean is not None else None)
    aps = per_seed("A", "clean")
    c["A_clean_per_seed"] = {str(k): v for k, v in aps.items()}
    c["gain_vs_s5_pp_per_seed"] = {
        str(k): round((v - S5_BASE["A_clean_per_seed"][str(k)]) * 100, 2)
        for k, v in aps.items() if str(k) in S5_BASE["A_clean_per_seed"]}
    b5 = summary["conditions"]["block5-30"]
    c["B_minus_A_block5_pp"] = b5.get("B_minus_A") * 100 if b5.get("B_minus_A") is not None else None
    c["note_H1"] = ("本阶段不做臂间新判据（F1 同款不适用）；B−A 为 LR 变更后复演值兼报，"
                    "H1 结论级判定以 S5/S10 为准不变")
    return summary


def write_csv_md(rows: list[dict], summary: dict, timing: dict) -> None:
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
    L.append("# S11-quickdraw-lrdecay-retrain：QuickDraw LR step decay 重训结果"
             "（% acc，3 种子 mean±std，逐种子明细见 CSV）\n")
    L.append("- 阶段动机（S9 归因）：D7 缺口真实锚点=论文 Table I QuickDraw LNN 0.9577（-3.47pp）/"
             "Table III 0% 列 92.69（-0.39pp）；候选主因=LR 调度偏离（论文 step decay vs 本仓恒定 1e-3，"
             "估计 -1~-3pp）。本阶段唯一变量=LR 调度，其余协议与 S5 逐项一致（import-not-copy）。")
    L.append(f"- LR 调度：{LR_SCHEDULE_DESC}")
    L.append("- 数据/模型/优化器：全量 50k（85/15 seeded split→train 42500/test 7500，SPLIT_SEED=12345）、"
             "DrawingClassifier(quickdraw)、AdamW lr_base=1e-3/wd0.01/batch128/clip1.0、30ep、"
             "meanpool 主口径三口径兼报；A/B 共享 ckpt（D1），B=manual-loop+D2 真实 timespans")
    L.append("- 条件：clean / uniform30(逐时间步 Bernoulli 0.3) / block5-30、block15-30（块长 b 划分、"
             "每块 Bernoulli 0.3；3 固定种子化模式跨臂同掩码配对）；评测全量 test 7500\n")
    L.append("| 条件 | A | B | B−A(pp) |")
    L.append("|---|---|---|---|")
    for cond in CONDITIONS:
        e = summary["conditions"][cond]
        bma = f"{e['B_minus_A'] * 100:+.2f}" if e.get("B_minus_A") is not None else "—"
        L.append(f"| {cond} | {fmt(e, 'A')} | {fmt(e, 'B')} | {bma} |")
    L.append("")
    s = summary["sanity"]
    L.append("## 强制 sanity")
    if s["clean_A_minus_B_pp"] is not None:
        L.append(f"- clean A≡B：|A−B|={abs(s['clean_A_minus_B_pp']):.3f}pp → "
                 f"{'PASS(<0.1pp)' if s['clean_A_equals_B_lt_0p1pp'] else 'FAIL'}")
    if s["A_uniform30_mean"] is not None:
        L.append(f"- A@uniform30={s['A_uniform30_mean'] * 100:.2f}% → 方向核对 "
                 f"{'PASS(z=' + str(s.get('A_uniform30_z_vs_chance')) + '>2 vs chance)' if s.get('A_uniform30_direction_ok') else '未显著高于随机水平'}")
    if s.get("A_clean_caliber_spread_pp") is not None:
        L.append(f"- 恒定类消失伴生判据：A 臂 clean 三口径="
                 f"{ {k: round(vv * 100, 2) for k, vv in s['A_clean_three_calibers'].items()} }，"
                 f"分化={s['A_clean_caliber_spread_pp']:.2f}pp")
    L.append("")
    c = summary["criteria"]
    L.append("## 判据机械输出（跑前冻结；主口径 meanpool）")
    L.append(f"- 判据① A@clean(3 种子 mean)={c['A_clean_mean'] * 100:.2f}±{c['A_clean_std_pp']:.2f}% "
             f"vs S5 恒定 LR 基线 92.30±0.86 → 增益 {c['gain_vs_s5_pp']:+.2f}pp → "
             f"**criterion1={c['criterion1']}**"
             + ("（≥+1.5pp：D7 缺口主因=LR 调度成立）" if c["criterion1"] == "supported"
                else "（<+0.5pp：LR 非主因，负结果落盘不开第二轮调参）" if c["criterion1"] == "falsified"
                else "（+0.5~+1.5pp：inconclusive，留 dropout 头/数据量 30min 验法选项）"))
    L.append(f"- 逐种子 A@clean vs S5 基线配对（pp）：{c['gain_vs_s5_pp_per_seed']}"
             f"（S5 逐种子 {S5_BASE['A_clean_per_seed']}）")
    L.append(f"- 判据② clean A≡B<0.1pp：{'PASS' if c['criterion2_clean_A_equals_B_pass'] else 'FAIL'}")
    L.append(f"- 判据③ A@uniform30={s['A_uniform30_mean'] * 100:.2f}% 兼报 vs 论文 Table III "
             f"QuickDraw 30%=68.80 → 缺口 {c['A_uniform30_gap_vs_paper_68_80_pp']:+.2f}pp")
    L.append(f"- D7 缺口（本阶段后）：vs Table I 0.9577 = {c['d7_gap_vs_paper_tableI_pp']:+.2f}pp；"
             f"vs Table III 0% 列 92.69 = {c['d7_gap_vs_paper_tableIII_0pct_pp']:+.2f}pp")
    L.append(f"- block5-30 B−A={c['B_minus_A_block5_pp']:+.2f}pp（兼报；{c['note_H1']}）")
    L.append("")
    L.append("## 限制")
    L.append("- 论文单次训练/配置 vs 本仓 3 种子 mean（±0.86pp 量级，非系统性）；论文每 epoch 步数"
             "310 vs 本仓 ~332（it-ep 口径 unknown，脚本墙未确认）")
    L.append("- 分类头差异仍在（论文 MLP+Dropout vs 本仓单 Linear；本阶段禁改分类头，Dropout 留待"
             "判据①后由 researcher 定 30min 验法）")
    L.append(f"- 计时（U5）：{json.dumps(timing, ensure_ascii=False)}")
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ================= selftest（LR 调度表+续跑等价，轻量） =================

def selftest() -> int:
    ok = True
    print("[selftest] LR schedule table ...")
    tab = lr_schedule_table(30)
    exp = [1e-3] * 12 + [5e-4] * 13 + [2.5e-4] * 5
    sched_ok = (len(tab) == 30 and tab == exp)
    print(f"[selftest] lr schedule: 12x1e-3/13x5e-4/5x2.5e-4 = {sched_ok} → "
          f"{'PASS' if sched_ok else 'FAIL'}")
    ok = ok and sched_ok

    print("[selftest] per-epoch checkpoint resume equivalence (LR-decay path, 384 samples) ...")
    x_tr_full, y_tr_full, x_te_full, y_te_full = F.full_splits("quickdraw")
    xs, ys = x_tr_full[:384], y_tr_full[:384]
    for f in ["quickdraw_lrdecay_selftest_seed0_ab.pt"]:
        pth = os.path.join(CKPT_DIR, f)
        if os.path.exists(pth):
            os.remove(pth)
    _, acc1, _, _, _ = train_one_lrdecay(0, xs, ys, epochs=2, tag="selftest_")
    sd_one = torch.load(os.path.join(CKPT_DIR, "quickdraw_lrdecay_selftest_seed0_ab.pt"),
                        map_location="cpu", weights_only=False)["state_dict"]
    os.remove(os.path.join(CKPT_DIR, "quickdraw_lrdecay_selftest_seed0_ab.pt"))
    train_one_lrdecay(0, xs, ys, epochs=1, tag="selftest_")
    _, acc2, _, _, _ = train_one_lrdecay(0, xs, ys, epochs=2, tag="selftest_")
    sd_two = torch.load(os.path.join(CKPT_DIR, "quickdraw_lrdecay_selftest_seed0_ab.pt"),
                        map_location="cpu", weights_only=False)["state_dict"]
    max_d = max((sd_one[k].double() - sd_two[k].double()).abs().max().item()
                for k in sd_one if sd_one[k].is_floating_point())
    res_ok = max_d < 1e-6
    print(f"[selftest] resume: max|Δstate|={max_d:.2e} acc_1shot={acc1:.4f} acc_resume={acc2:.4f} → "
          f"{'PASS' if res_ok else 'FAIL'}")
    ok = ok and res_ok
    os.remove(os.path.join(CKPT_DIR, "quickdraw_lrdecay_selftest_seed0_ab.pt"))

    print("[selftest] eval mechanics smoke (clean A≡B + block5-30 masked≈0.3) ...")
    model = P.DrawingClassifier("quickdraw", with_gap_bit=False).to(F.DEVICE).eval()
    xe, ye = x_te_full[:64], y_te_full[:64]
    accA, _ = P.eval_arm_condition(model, "A", xe, ye, "clean", 0, 0)
    accB, _ = P.eval_arm_condition(model, "B", xe, ye, "clean", 0, 0)
    _, mf5 = P.eval_arm_condition(model, "B", xe, ye, "block5-30", 0, 0)
    ev_ok = (accA["meanpool"] == accB["meanpool"] and abs(mf5 - 0.3) < 0.05)
    print(f"[selftest] eval smoke: A@clean={accA['meanpool']:.4f} B@clean={accB['meanpool']:.4f} "
          f"(equal={accA['meanpool'] == accB['meanpool']}), block5-30 masked={mf5:.3f} → "
          f"{'PASS' if ev_ok else 'FAIL'}")
    ok = ok and ev_ok

    print(f"[selftest] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ================= 冒烟（子步1：seed0 单 run 2ep，全量数据） =================

def smoke() -> int:
    x_tr, y_tr, x_te, y_te = F.full_splits("quickdraw")
    ckpt, acc, loss, sec, ep_done = train_one_lrdecay(0, x_tr, y_tr, epochs=2, tag="smoke_")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    hist = payload["history"]
    loss_drop = len(hist) >= 2 and hist[1]["loss"] < hist[0]["loss"]
    lrs = [h["lr"] for h in hist]
    lr_ok = lrs == [1e-3, 1e-3]
    m = P.DrawingClassifier("quickdraw", with_gap_bit=False)
    m.load_state_dict(payload["state_dict"])
    m = m.to(F.DEVICE).eval()
    xe, ye = x_te[:256], y_te[:256]
    accA, _ = P.eval_arm_condition(m, "A", xe, ye, "clean", 0, 0)
    accB, _ = P.eval_arm_condition(m, "B", xe, ye, "clean", 0, 0)
    ab_equal = accA["meanpool"] == accB["meanpool"]
    print(f"[smoke] 2ep hist={[(h['epoch'], h['loss'], h['acc'], h['lr']) for h in hist]}")
    print(f"[smoke] loss_drop={loss_drop} (ep1 {hist[0]['loss']:.4f} → ep2 {hist[1]['loss']:.4f}), "
          f"lr_seq_ok={lr_ok}, clean A≡B: A={accA['meanpool']:.4f} B={accB['meanpool']:.4f} "
          f"equal={ab_equal}")
    ok = loss_drop and lr_ok and ab_equal
    print(f"[smoke] OVERALL: {'PASS' if ok else 'FAIL'}")
    os.remove(ckpt)      # 冒烟 ckpt 不污染正式链（证据=日志+本输出）
    return 0 if ok else 1


# ================= 主流程 =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "smoke", "train", "eval", "all", "status"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS_LRD)
    args = ap.parse_args()

    F._patch_pilot_d2()                 # eval_arm_condition 内 d2 走向量化实现（恒等性 S5 selftest 担保）

    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "smoke":
        return smoke()

    if args.cmd == "status":
        prog = load_progress()
        runs = {k: v.get("epochs", v.get("epochs_done", "?")) for k, v in prog.get("runs", {}).items()}
        print(json.dumps({"runs": runs, "eval_rows": len(prog.get("eval_rows", [])),
                          "eval_total": EVAL_TOTAL, "verdict": prog.get("verdict")},
                         ensure_ascii=False))
        return 0

    t_start = time.time()
    x_tr, y_tr, x_te, y_te = F.full_splits("quickdraw")
    progress = load_progress()
    timing = {"data_prep_sec": round(time.time() - t_start, 1), "train_runs": {},
              "eval_sec": None, "total_sec": None}
    print(f"[lrdecay] data: train={x_tr.shape[0]} test={x_te.shape[0]} "
          f"epochs={args.epochs} device={F.DEVICE} lr_schedule={LR_SCHEDULE_DESC}", flush=True)

    if args.cmd == "train":
        key = f"seed{args.seed}_ab"
        ckpt, acc, loss, sec, ep_done = train_one_lrdecay(args.seed, x_tr, y_tr, epochs=args.epochs)
        if ep_done >= args.epochs:
            progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                     "train_acc": round(acc, 4), "train_loss": round(loss, 4),
                                     "epochs": ep_done, "sec": round(sec, 1)}
            timing["train_runs"][key] = round(sec, 1)
            save_progress(progress)
        return 0

    if args.cmd == "all":
        # 入口 sanity（S2 条件④同款：真实尺寸 manual-loop 复验）
        if "verify_real_size" not in progress["sanity"]:
            progress["sanity"]["verify_real_size"] = P.verify("quickdraw", x_te)
            save_progress(progress)
            print("[all] verify:", json.dumps(progress["sanity"]["verify_real_size"],
                                              ensure_ascii=False), flush=True)
        # S1：3 次训练（A/B 共享 3 种子），逐 epoch checkpoint
        for seed in SEEDS:
            key = f"seed{seed}_ab"
            ckpt_path = os.path.join(CKPT_DIR, f"quickdraw_lrdecay_{key}.pt")
            done_ep = 0
            if os.path.exists(ckpt_path):
                done_ep = int(torch.load(ckpt_path, map_location="cpu",
                                         weights_only=False).get("epochs_done", 0))
            if key in progress["runs"] and done_ep >= args.epochs:
                print(f"[all] skip {key} (done {done_ep}/{args.epochs})", flush=True)
                continue
            ckpt, acc, loss, sec, ep_done = train_one_lrdecay(seed, x_tr, y_tr, epochs=args.epochs)
            if ep_done >= args.epochs:
                progress["runs"][key] = {"ckpt": os.path.relpath(ckpt, PROJECT),
                                         "train_acc": round(acc, 4), "train_loss": round(loss, 4),
                                         "epochs": ep_done, "sec": round(sec, 1)}
                timing["train_runs"][key] = round(sec, 1)
                save_progress(progress)
        # S2：评测矩阵（逐行落盘续跑）
        rows, eval_sec = run_eval_matrix(progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        if len(rows) < EVAL_TOTAL:
            print(f"[all] eval incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(timing)
            return 0
        summary = summarize_lrdecay(rows)
        progress["verdict"] = summary["criteria"]
        save_progress(progress)
        save_timing(timing)
        write_csv_md(rows, summary, timing)
        print(json.dumps(summary["criteria"], ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "eval":
        missing = []
        for seed in SEEDS:
            pth = os.path.join(CKPT_DIR, f"quickdraw_lrdecay_seed{seed}_ab.pt")
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
        rows, eval_sec = run_eval_matrix(progress, x_te, y_te)
        rows = sorted(rows, key=lambda r: (r["seed"], r["arm"],
                                           CONDITIONS.index(r["condition"]), r["pattern"]))
        timing["eval_sec"] = eval_sec
        timing["total_sec"] = round(time.time() - t_start, 1)
        if len(rows) < EVAL_TOTAL:
            print(f"[eval] incomplete: {len(rows)}/{EVAL_TOTAL} rows on disk; "
                  f"resume with the same command (checkpointed rows are skipped)", flush=True)
            save_progress(progress)
            save_timing(timing)
            return 0
        summary = summarize_lrdecay(rows)
        progress["verdict"] = summary["criteria"]
        save_progress(progress)
        save_timing(timing)
        write_csv_md(rows, summary, timing)
        print(json.dumps(summary["criteria"], ensure_ascii=False, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
