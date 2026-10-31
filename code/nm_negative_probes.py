#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nm_negative_probes.py — S10-nmnist-threearm-baseline-redesign / nmnist-negative-result-probes。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S10-nmnist-threearm-baseline-redesign（researcher 判读 eaea7487 重定义；
      H1 已双数据集结论级证伪，本阶段=负结果归因探针，纯推理、禁训练、禁改 ckpt）

背景（为何做）：
  S10 结论级表：block5-30 A=80.99 / B=81.08 / C=90.45（3 种子 mean，meanpool 主口径），
  C−A=+9.47pp 但 C_gate_pass=False（clean 逐种子降幅 0.95/0.43/0.62 不全 ≤0.5pp）。
  负结果仓库必须回答「C 臂 +9.47pp 从哪来」：训练期（块增强+gap-bit）还是推理期 dt？

两因素分离设计（预注册，跑前冻结）：
  因素1=训练期块增强+gap-bit；因素2=推理期 dt。
  - C_dt      = C 臂既有口径（manual-loop + D2 真实 timespans + gap 指示位）——应与 S10 CSV
                的 C 行逐位一致（本脚本内置恒等自检，防口径漂移）。
  - C_no_dt   = A 臂口径 forward(timespans=None)（无 dt）作用于同一 C ckpt。
                【口径说明（显式记录，禁 silent 改口径）】：C 模型 core_in=129（128+gap-bit），
                library forward 必须填满 129 维；gap-bit 通道填**真实掩码**（与 C 臂推理期
                完全相同），唯一改动=去掉 dt（timespans=None → 均匀 dt=1）。这样
                C_no_dt−A 隔离出的正是因素1（训练期+gap-bit），C−C_no_dt=纯推理期 dt 效应。
                （备选口径 gap-bit=0 记为诊断行 C_no_dt_nogap，不进冻结判据。）
  - A_ref     = A 臂口径作用于 ab ckpt（本进程重算，应与 S10 CSV 的 A 行逐位一致=恒等自检）。
  - B 臂（S10）：block5-30 B−A=+0.09pp ≈ 0（推理期 dt 无增益的事实锚点）。

判据（跑前冻结、禁事后改；源自 notes/researcher-interventions/eaea7487...md 第五节）：
  ①主探针 C_no_dt−A（block5-30，meanpool，3 种子 mean）：
    ≥+5pp → 训练期主导（E2 的 N-MNIST 确认，叙事闭合）；
    <+2pp → 推理 dt 主导（与 B−A≈0 矛盾→查实现落 partial 不硬撑）；
    其间   → inconclusive（两因素兼有）。
  ②次探针 dt 粒度响应谱饱和点 dt*：
    dt*≥5 → 帧化量化假设存活（H1 失败的协议侧候选解释成立，入机制讨论）；
    dt*<5 → 量化假设证伪（dt 粒度非瓶颈）；
    响应谱完全平坦（R_max<1e-3）→ 两解释都不成立，如实记 inconclusive。
  ③sanity：clean 口径切换差 |C_no_dt−C_dt| < 0.3pp（clean 无间隙 dt≡1，两口径应≈同值）。

次探针口径（预注册）：
  seed0_ab ckpt，B 臂口径（manual-loop），固定输入 batch=test 前 256 样本、
  mask=block5-30(seed0,pattern0)（H1 主条件；uniform30/clean 为稳健性诊断），
  扫 timespans 常数 dt∈{1,2,5,10,20}（同输入 batch），记录 CfC 输出 hidden（out，即
  core.fc(h_out) 逐步投影输出，分类头读取的对象）的 L2 响应谱：
    R(dt) = mean_i ‖out_i(dt)−out_i(1)‖_F / mean_i ‖out_i(1)‖_F（相对 L2 响应）
    R_max = max_dt R(dt)；dt* = min{dt∈{2,5,10,20}: R(dt) ≥ 0.95·R_max}，无则 ">20"。

失败条件（预注册）：
  冒烟不过（ckpt 加载失败 / clean 口径切换差 ≥0.3pp / dt 扫描 hidden 非有限 /
  A_ref 或 C_dt 与 S10 CSV 不一致）→ 落 partial+断点，不硬撑；
  主探针落 inconclusive 区间 → 如实记不硬撑成结论；
  总量超 600s 硬顶 → checkpoint 移交；
  禁训练、禁改 ckpt、禁 silent 改口径、禁手写并行分片。

用法：
  python experiments/nm_negative_probes.py smoke      # 子步1：seed0_c 两口径 clean + dt 扫描有限性
  python experiments/nm_negative_probes.py main       # 子步2：主探针全量（逐行原子落盘+断点续跑）
  python experiments/nm_negative_probes.py spectrum   # 子步3：dt 响应谱
  python experiments/nm_negative_probes.py all        # smoke→main→spectrum→判据→CSV/MD/progress
  python experiments/nm_negative_probes.py report     # 从 progress 确定性重建 CSV/MD（不重算）
  python experiments/nm_negative_probes.py status     # 进度快照
"""

import argparse
import csv
import json
import os
import statistics as st
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PROJECT = os.path.dirname(HERE)

import pilot_threearm as P          # noqa: E402  （协议同源代码：模型/掩码/条件，import 不复制）
import full_protocol as F           # noqa: E402  （向量化 d2_timespans，selftest 恒等担保）
import nmnist_threearm as N         # noqa: E402  （S10：load_model_nm/mask4/EVAL_BATCH/DATASET 口径）
from nmnist_loader import load_split  # noqa: E402  （S8 loader，口径钉死）

RESULTS = os.path.join(PROJECT, "results")
CSV_OUT = os.path.join(RESULTS, "nm_negative_probes.csv")
MD_OUT = os.path.join(RESULTS, "nm_negative_probes.md")
PROGRESS = os.path.join(RESULTS, "nm_negative_probes_progress.json")
SPECTRUM_OUT = os.path.join(RESULTS, "nm_dt_response_spectrum.json")
S10_CSV = os.path.join(RESULTS, "nmnist_threearm.csv")

STAGE = "S10-nmnist-threearm-baseline-redesign"
SUBSTAGE = "nmnist-negative-result-probes"

# ---- 预注册常数（跑前冻结）----
MAIN_CONDS = ["clean", "uniform30", "block5-30"]      # 主探针三条件（H1 主条件=block5-30）
MAIN_PROBES = ["C_dt", "C_no_dt"]                     # 冻结两口径（18 行=3 种子×3 条件×2 口径）
DIAG_PROBES = ["A_ref", "C_no_dt_nogap"]              # 诊断口径（恒等自检+gap-bit 拆解，不进判据）
DT_SCAN = [1, 2, 5, 10, 20]                           # 次探针 dt 扫描点
DT_SAT_FRAC = 0.95                                    # 饱和判定：R(dt)>=0.95*R_max
FLAT_EPS = 1e-3                                       # 平坦谱阈值（相对 L2）
CLEAN_SWITCH_TOL_PP = 0.3                             # 判据③：clean 口径切换差容忍
TRAIN_DOMINANT_PP = 5.0                               # 判据①上界
INFER_DOMINANT_PP = 2.0                               # 判据①下界
HARD_CAP_SEC = 600.0                                  # 总量硬顶（预注册）
SPECTRUM_BATCH = 256                                  # 次探针固定 batch


# ================= progress（原子写 + 断点续跑）=================

def load_progress() -> dict:
    if os.path.exists(PROGRESS):
        with open(PROGRESS, encoding="utf-8") as fh:
            return json.load(fh)
    return {"schema": "nm-negative-probes/v1", "stage": STAGE, "substage": SUBSTAGE,
            "rows": [], "smoke": None, "spectrum": None, "criteria": None,
            "status": "running"}


def save_progress(p: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(p, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, PROGRESS)          # 原子替换：kill 不留半截 JSON


# ================= 数据与模型 =================

def load_test():
    """只载 test（0.92GB）；主探针/次探针均不需要 train。"""
    x_te, y_te = load_split("test")
    return np.array(x_te), np.asarray(y_te)


def load_models(seeds):
    out = {}
    for s in seeds:
        out[s] = {a: N.load_model_nm(s, a)[0] for a in ["ab", "c"]}
    return out


# ================= 主探针：单 (probe, seed, cond, pattern) 行 =================

def eval_probe_row(models, probe: str, x_te, y_te, cond: str, seed: int, pattern: int):
    """四口径评测（N-MNIST 版；与 nmnist_threearm.eval_arm_condition_nm 同源代码路径）：
      A_ref        : ab ckpt + forward_lib（A 臂口径；恒等自检 vs S10 CSV A 行）
      C_dt         : c  ckpt + forward_ts_gap（C 臂既有口径；恒等自检 vs S10 CSV C 行）
      C_no_dt      : c  ckpt + core(timespans=None) + gap-bit=真实掩码（冻结口径：只去 dt）
      C_no_dt_nogap: c  ckpt + core(timespans=None) + gap-bit=0（诊断：再去 gap 信息）
    返回 (三口径 acc dict, 实现掩码率)。"""
    model = models[seed]["c"] if probe.startswith("C") else models[seed]["ab"]
    model.eval()
    n = x_te.shape[0]
    correct = {"meanpool": 0, "final": 0, "gapnb": 0}
    masked_frames, total_frames = 0, 0
    with torch.no_grad():
        for i in range(0, n, N.EVAL_BATCH):
            xb = N._to_torch(x_te[i:i + N.EVAL_BATCH])
            yb = torch.from_numpy(np.asarray(y_te[i:i + N.EVAL_BATCH])).to(N.DEVICE)
            mask = P.make_mask(cond, xb.shape[0], xb.shape[1], seed, pattern).to(N.DEVICE)
            xin = xb * (1.0 - N.mask4(mask))          # mask=1=掩码(清零)，D2/D4 预注册
            if probe == "A_ref":
                logits, out = model.forward_lib(xin)
            elif probe == "C_dt":
                ts = F.d2_timespans_vec(mask).to(N.DEVICE)
                logits, out = model.forward_ts_gap(xin, ts, mask)
            elif probe == "C_no_dt":
                feats = torch.cat([model.features(xin), mask], dim=2)
                out, _ = model.core(feats)            # timespans=None：无 dt（A 臂口径）
                logits = model.classifier(out.mean(dim=1))
            elif probe == "C_no_dt_nogap":
                feats = torch.cat([model.features(xin), torch.zeros_like(mask)], dim=2)
                out, _ = model.core(feats)
                logits = model.classifier(out.mean(dim=1))
            else:
                raise ValueError(probe)
            correct["meanpool"] += (logits.argmax(1) == yb).sum().item()
            correct["final"] += (model.classifier(out[:, -1]).argmax(1) == yb).sum().item()
            w = P.gap_neighborhood_weights(mask)
            pooled = (out * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)
            correct["gapnb"] += (model.classifier(pooled).argmax(1) == yb).sum().item()
            masked_frames += mask.sum().item()
            total_frames += mask.numel()
    return {k: v / n for k, v in correct.items()}, masked_frames / max(total_frames, 1)


def run_main(models, x_te, y_te, progress: dict, t0: float, log=print):
    """主探针全量：3 种子×3 条件×3 模式×(2 冻结口径+2 诊断口径)，逐行原子落盘+断点跳过。
    返回 (rows, elapsed)。HARD_CAP_SEC 硬顶：超限保存并交 checkpoint。"""
    rows = list(progress.get("rows", []))
    done = {(r["probe"], r["seed"], r["condition"], r["pattern"]) for r in rows}
    total = len(N.SEEDS) * len(MAIN_CONDS) * N.N_PATTERNS * (len(MAIN_PROBES) + len(DIAG_PROBES))
    for seed in N.SEEDS:
        for probe in MAIN_PROBES + DIAG_PROBES:
            for cond in MAIN_CONDS:
                for pattern in range(N.N_PATTERNS):
                    key = (probe, seed, cond, pattern)
                    if key in done:
                        continue
                    el = time.time() - t0
                    if el > HARD_CAP_SEC:
                        log(f"[main] HARD CAP {HARD_CAP_SEC:.0f}s exceeded "
                            f"(elapsed {el:.0f}s); checkpoint handoff")
                        progress["rows"] = rows
                        progress["status"] = "checkpointed"
                        save_progress(progress)
                        return rows, el
                    accs, mf = eval_probe_row(models, probe, x_te, y_te, cond, seed, pattern)
                    rows.append({"probe": probe, "dataset": N.DATASET, "seed": seed,
                                 "condition": cond, "pattern": pattern, "n_eval": x_te.shape[0],
                                 "masked_fraction_realized": round(mf, 4),
                                 "acc_meanpool": round(accs["meanpool"], 4),
                                 "acc_final": round(accs["final"], 4),
                                 "acc_gapnb": round(accs["gapnb"], 4)})
                    progress["rows"] = rows
                    save_progress(progress)      # 逐行原子落盘
                    log(f"[main] {probe} seed={seed} {cond} p={pattern} "
                        f"meanpool={accs['meanpool'] * 100:.2f}% "
                        f"({len(rows)}/{total}, {time.time() - t0:.0f}s cum)")
    progress["rows"] = rows
    save_progress(progress)
    return rows, time.time() - t0


# ================= 次探针：dt 粒度响应谱 =================

def dt_response_spectrum(model, x_te, y_te, cond: str, seed: int = 0, pattern: int = 0):
    """固定 batch + 固定掩码，扫常数 timespans dt∈DT_SCAN，返回每 dt 的响应记录。
    model 须为 with_gap_bit=False（B 臂口径 forward_ts）。"""
    L = x_te.shape[1]
    xb = N._to_torch(x_te[:SPECTRUM_BATCH])
    yb = torch.from_numpy(np.asarray(y_te[:SPECTRUM_BATCH])).to(N.DEVICE)
    mask = P.make_mask(cond, SPECTRUM_BATCH, L, seed, pattern).to(N.DEVICE)
    xin = xb * (1.0 - N.mask4(mask))
    rec = {"condition": cond, "mask_seed": seed, "mask_pattern": pattern,
           "batch": SPECTRUM_BATCH, "L": L,
           "masked_fraction_realized": round(float(mask.mean().item()), 4),
           "per_dt": {}, "finite_all": True}
    base_out = base_norm = base_logits = None
    with torch.no_grad():
        for dt in DT_SCAN:
            ts = torch.full((SPECTRUM_BATCH, L), float(dt), device=N.DEVICE)
            logits, out = model.forward_ts(xin, ts)
            finite = bool(torch.isfinite(out).all())
            rec["finite_all"] &= finite
            if base_out is None:
                base_out = out.clone()
                base_norm = float(out.norm(dim=2).mean().item())   # mean_i ‖out_i‖_F
                base_logits = logits.clone()
            diff = float((out.double() - base_out.double()).norm(dim=2).mean().item())
            acc = float((logits.argmax(1) == yb).float().mean().item())
            dl = float((logits.double() - base_logits.double()).abs().max().item())
            rec["per_dt"][str(dt)] = {
                "mean_hidden_L2": round(float(out.norm(dim=2).mean().item()), 6),
                "rel_L2_response_vs_dt1": round(diff / max(base_norm, 1e-12), 6),
                "meanpool_acc_on_batch": round(acc, 4),
                "logits_max_abs_diff_vs_dt1": round(dl, 6),
                "finite": finite,
            }
    # 饱和点 dt*（预注册机械定义）
    rvals = {int(k): v["rel_L2_response_vs_dt1"] for k, v in rec["per_dt"].items()}
    r_max = max(rvals.values())
    rec["R_max"] = round(r_max, 6)
    rec["flat"] = bool(r_max < FLAT_EPS)
    dt_star = None
    if not rec["flat"]:
        for dt in DT_SCAN[1:]:
            if rvals[dt] >= DT_SAT_FRAC * r_max:
                dt_star = dt
                break
        if dt_star is None:
            dt_star = ">20"
    rec["dt_star"] = dt_star
    return rec


# ================= 判据（跑前冻结、机械输出）=================

def _agg(rows, probe, cond, key="acc_meanpool"):
    """3 种子 mean（每种子对 3 模式取均值）+ 逐种子明细 + 跨 9 行 pstdev。"""
    per_seed = {}
    for s in N.SEEDS:
        vals = [r[key] for r in rows if r["probe"] == probe and r["condition"] == cond
                and r["seed"] == s]
        if vals:
            per_seed[s] = st.mean(vals)
    allv = [r[key] for r in rows if r["probe"] == probe and r["condition"] == cond]
    mean = st.mean(allv) if allv else None
    std = st.pstdev(allv) if len(allv) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(allv),
            "per_seed": {str(k): round(v, 4) for k, v in per_seed.items()}}


def s10_reference():
    """从 S10 结论级 CSV 读 A/B/C 锚点（不复算训练，只读盘上行）。"""
    with open(S10_CSV, encoding="utf-8") as fh:
        data = [ln.strip().split(",") for ln in fh if ln.strip()][1:]
    out = {}
    for arm in ["A", "B", "C"]:
        for cond in MAIN_CONDS:
            vals = [float(r[7]) for r in data if r[2] == arm and r[3] == cond]
            out[f"{arm}_{cond}"] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def s10_seed_anchor(arm: str, cond: str, seed: int):
    """S10 CSV 中同 seed 同条件的 3 个 pattern 级 meanpool 值（逐模式配对锚点）。"""
    with open(S10_CSV, encoding="utf-8") as fh:
        data = [ln.strip().split(",") for ln in fh if ln.strip()][1:]
    return [float(r[7]) for r in data
            if r[2] == arm and r[3] == cond and int(r[1]) == seed]


def compute_criteria(rows, spectrum):
    """冻结判据机械输出（禁事后改）。"""
    crit = {"criterion1_main_probe": {}, "criterion2_dt_spectrum": {},
            "criterion3_clean_switch_sanity": {}, "crosschecks": {}}
    s10 = s10_reference()
    # ---- ①主探针：C_no_dt-A（block5-30，meanpool，3 种子 mean）----
    cnd = _agg(rows, "C_no_dt", "block5-30")
    a = _agg(rows, "A_ref", "block5-30")
    cdt = _agg(rows, "C_dt", "block5-30")
    diff_pp = (cnd["mean"] - a["mean"]) * 100
    per_seed_pp = {str(s): round((cnd["per_seed"][s] - a["per_seed"][s]) * 100, 2)
                   for s in cnd["per_seed"] if s in a["per_seed"]}
    c1 = crit["criterion1_main_probe"]
    c1["C_no_dt_block5_mean_pct"] = round(cnd["mean"] * 100, 2)
    c1["A_ref_block5_mean_pct"] = round(a["mean"] * 100, 2)
    c1["C_dt_block5_mean_pct"] = round(cdt["mean"] * 100, 2)
    c1["C_no_dt_minus_A_pp"] = round(diff_pp, 2)
    c1["C_no_dt_minus_A_pp_per_seed"] = per_seed_pp
    c1["C_minus_A_pp"] = round((cdt["mean"] - a["mean"]) * 100, 2)
    c1["inference_dt_effect_pp"] = round((cdt["mean"] - cnd["mean"]) * 100, 2)
    c1["S10_anchor_C_minus_A_pp"] = round((s10["C_block5-30"] - s10["A_block5-30"]) * 100, 2)
    c1["frozen_thresholds_pp"] = {"training_dominant_ge": TRAIN_DOMINANT_PP,
                                  "inference_dt_dominant_lt": INFER_DOMINANT_PP}
    if diff_pp >= TRAIN_DOMINANT_PP:
        c1["verdict"] = "training_dominant_supported"
        c1["reading"] = ("C 臂增益几乎全部来自训练期（块增强+gap-bit）——E2 的 N-MNIST 确认，"
                         "叙事闭合")
    elif diff_pp < INFER_DOMINANT_PP:
        c1["verdict"] = "inference_dt_dominant_contradiction"
        c1["reading"] = ("增益似来自推理期 dt，与 B-A≈0 矛盾——按预注册查实现落 partial 不硬撑"
                         "（恒等自检见 crosschecks）")
    else:
        c1["verdict"] = "inconclusive"
        c1["reading"] = "两因素兼有，落 inconclusive 区间，如实记不硬撑成结论"
    # ---- ②次探针：dt* ----
    c2 = crit["criterion2_dt_spectrum"]
    sp = spectrum["block5-30"]
    c2["R_per_dt"] = {k: v["rel_L2_response_vs_dt1"] for k, v in sp["per_dt"].items()}
    c2["R_max"] = sp["R_max"]
    c2["dt_star"] = sp["dt_star"]
    c2["flat"] = sp["flat"]
    c2["finite_all"] = sp["finite_all"]
    c2["frozen_threshold"] = "dt*>=5 -> 量化假设存活; dt*<5 -> 证伪; 平坦 -> inconclusive"
    if sp["flat"]:
        c2["verdict"] = "inconclusive_flat"
        c2["reading"] = "响应谱完全平坦（R_max<1e-3）——两解释都不成立，如实记 inconclusive"
    elif sp["dt_star"] == 2:
        c2["verdict"] = "falsified"
        c2["reading"] = "dt*<5——帧化量化（bin 粒度）非瓶颈，量化假设证伪，H1 失败归模型侧"
    else:
        c2["verdict"] = "supported"
        c2["reading"] = ("dt*>=5——模型对 bin 级 dt 粒度仍敏感，帧化量化是 H1 失败的协议侧"
                         "候选解释（入机制讨论）")
    c2["robustness_other_conditions"] = {
        cond: {"dt_star": r["dt_star"], "R_max": r["R_max"], "flat": r["flat"]}
        for cond, r in spectrum.items() if cond != "block5-30"}
    # ---- ③sanity：clean 口径切换差 ----
    c3 = crit["criterion3_clean_switch_sanity"]
    cnd_c = _agg(rows, "C_no_dt", "clean")
    cdt_c = _agg(rows, "C_dt", "clean")
    switch_pp = (cnd_c["mean"] - cdt_c["mean"]) * 100
    c3["C_no_dt_clean_pct"] = round(cnd_c["mean"] * 100, 2)
    c3["C_dt_clean_pct"] = round(cdt_c["mean"] * 100, 2)
    c3["abs_switch_diff_pp"] = round(abs(switch_pp), 4)
    c3["frozen_threshold_pp"] = CLEAN_SWITCH_TOL_PP
    c3["pass"] = bool(abs(switch_pp) < CLEAN_SWITCH_TOL_PP)
    c3["reading"] = ("clean 无间隙 dt≡1，两口径应≈同值；差<0.3pp=PASS（口径切换生效）"
                     if c3["pass"] else "clean 口径切换差异超限——查实现")
    # ---- 恒等自检（失败条件项，防口径漂移）----
    cc = crit["crosschecks"]
    ident_ok = True
    for probe, arm in [("A_ref", "A"), ("C_dt", "C")]:
        worst = 0.0
        for cond in MAIN_CONDS:
            got = _agg(rows, probe, cond)["mean"]
            ref = s10[f"{arm}_{cond}"]
            worst = max(worst, abs(got - ref))
        cc[f"{probe}_vs_S10_{arm}_max_abs_diff"] = round(worst, 6)
        ident_ok &= worst < 5e-4
    cc["identity_pass"] = bool(ident_ok)
    cc["note"] = ("A_ref/C_dt 为本进程重算，应与 S10 结论级 CSV 对应臂逐位一致"
                  "（容差 5e-4=舍入级）；不一致=口径漂移，按失败条件处理")
    return crit


# ================= 冒烟（子步1）=================

def run_smoke(models, x_te, y_te, log=print) -> dict:
    """seed0_c 单 ckpt 两口径 clean 评测 + seed0_ab dt 扫描 hidden 有限性 + S10 锚点比对。"""
    rep = {"stage": SUBSTAGE, "gates": {}, "rows": {}}
    t0 = time.time()
    s10 = s10_reference()
    # 门1：seed0 两口径 clean（全量 test）
    accs_dt, _ = eval_probe_row(models, "C_dt", x_te, y_te, "clean", 0, 0)
    accs_ndt, _ = eval_probe_row(models, "C_no_dt", x_te, y_te, "clean", 0, 0)
    switch_pp = (accs_ndt["meanpool"] - accs_dt["meanpool"]) * 100
    rep["rows"]["seed0_clean_C_dt_meanpool"] = round(accs_dt["meanpool"], 4)
    rep["rows"]["seed0_clean_C_no_dt_meanpool"] = round(accs_ndt["meanpool"], 4)
    rep["rows"]["seed0_clean_switch_pp"] = round(switch_pp, 4)
    rep["gates"]["ckpt_load"] = True                       # 走到此处即加载成功
    rep["gates"]["clean_switch_lt_0p3pp"] = abs(switch_pp) < CLEAN_SWITCH_TOL_PP
    # 门2：A_ref/C_dt(seed0,clean,逐 pattern) 与 S10 同 seed 同行锚点逐位一致
    #     （修 2026-10-31 冒烟首跑门控 bug：原把 seed0 单值与 S10 三种子均值比，
    #      apples-to-oranges；改为同 (seed,cond,pattern) 配对，容差 5e-4=舍入级）
    anchor_worst = 0.0
    anchor_detail = {}
    for probe, arm in [("A_ref", "A"), ("C_dt", "C")]:
        for pattern in range(N.N_PATTERNS):
            got, _ = eval_probe_row(models, probe, x_te, y_te, "clean", 0, pattern)
            ref = s10_seed_anchor(arm, "clean", 0)[pattern]
            anchor_worst = max(anchor_worst, abs(got["meanpool"] - ref))
            anchor_detail[f"{probe}_p{pattern}"] = {"got": round(got["meanpool"], 4),
                                                    "S10": round(ref, 4)}
    rep["rows"]["seed0_clean_anchor_detail"] = anchor_detail
    rep["rows"]["seed0_clean_anchor_max_abs_diff"] = round(anchor_worst, 6)
    rep["gates"]["probe_matches_S10_same_seed"] = anchor_worst < 5e-4
    # 门3：seed0_ab dt 扫描 hidden 有限 + 响应非退化
    sp = dt_response_spectrum(models[0]["ab"], x_te, y_te, "block5-30", 0, 0)
    rep["spectrum_smoke"] = sp
    rep["gates"]["dt_scan_hidden_finite"] = bool(sp["finite_all"])
    rep["gates"]["dt_scan_response_nonzero"] = bool(sp["R_max"] > 0.0)
    rep["smoke_sec"] = round(time.time() - t0, 1)
    rep["gates_all_pass"] = all(rep["gates"].values())
    log("[smoke] " + json.dumps({k: v for k, v in rep.items() if k != "spectrum_smoke"},
                                ensure_ascii=False))
    log("[smoke] spectrum: " + json.dumps(sp, ensure_ascii=False))
    return rep


# ================= CSV / MD =================

CSV_COLS = ["probe", "dataset", "seed", "condition", "pattern", "n_eval",
            "masked_fraction_realized", "acc_meanpool", "acc_final", "acc_gapnb"]


def write_csv(rows) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    order = {"C_dt": 0, "C_no_dt": 1, "A_ref": 2, "C_no_dt_nogap": 3}
    rows = sorted(rows, key=lambda r: (order.get(r["probe"], 9), r["seed"],
                                       MAIN_CONDS.index(r["condition"]), r["pattern"]))
    tmp = CSV_OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in CSV_COLS})
    os.replace(tmp, CSV_OUT)


def write_md(rows, spectrum, crit, timing, smoke) -> None:
    L = []
    L.append("# nmnist-negative-result-probes — S10-redesign 负结果归因探针（纯推理，禁训练）\n")
    L.append("- 课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a")
    L.append("- 阶段：S10-nmnist-threearm-baseline-redesign（researcher 判读 eaea7487 重定义；"
             "H1 已双数据集结论级证伪，本阶段回答负结果仓库必答题「C 臂 +9.47pp 从哪来」）")
    L.append("- 输入：experiments/ckpt/nmnist_seed{0,1,2}_{ab,c}.pt（S10 收口 ckpt，10/10 epochs，"
             "verify exit=0；本阶段只读不写）+ results/nmnist_threearm.csv（S10 结论级表锚点）")
    L.append("- 口径：A 臂=forward(timespans=None)（无 dt）；C 臂=manual-loop+D2 真实 timespans"
             "+gap 指示位；C_no_dt=A 臂口径作用于 C ckpt（gap-bit 通道保留真实掩码，唯一改动=去 dt）")
    L.append("- 判据：跑前冻结（源自 notes/researcher-interventions/eaea7487ddfc479c83fe016c90d89c8f.md "
             "第五节），禁事后改\n")

    L.append("## 表1 主探针（% acc，meanpool 主口径；逐 seed×条件，每格=3 模式 mean±pstdev）")
    L.append("| seed | 条件 | C_dt（带 dt） | C_no_dt（无 dt） | C_no_dt−C_dt(pp) | "
             "A_ref（S10 A 臂重算） | C_no_dt−A(pp) |")
    L.append("|---|---|---|---|---|---|---|")
    for seed in N.SEEDS:
        for cond in MAIN_CONDS:
            def cell(probe):
                per = [r["acc_meanpool"] for r in rows if r["probe"] == probe
                       and r["condition"] == cond and r["seed"] == seed]
                return (st.mean(per), st.pstdev(per)) if per else (float("nan"), 0.0)
            cd, cds = cell("C_dt")
            cn, cns = cell("C_no_dt")
            ar, ars = cell("A_ref")
            # 与判据①同 rounding 路径（先 round 到 4 位再差），防显示与判据差 0.01pp
            cnr, arr = round(cn, 4), round(ar, 4)
            L.append(f"| {seed} | {cond} | {cd * 100:.2f}±{cds * 100:.2f} | "
                     f"{cn * 100:.2f}±{cns * 100:.2f} | {(cn - cd) * 100:+.2f} | "
                     f"{ar * 100:.2f}±{ars * 100:.2f} | {(cnr - arr) * 100:+.2f} |")
    L.append("")
    L.append("三口径汇总（3 种子 mean±pstdev，跨 9 行）")
    L.append("| 口径 | clean | uniform30 | block5-30 |")
    L.append("|---|---|---|---|")
    for probe in ["C_dt", "C_no_dt", "A_ref", "C_no_dt_nogap"]:
        cells = []
        for cond in MAIN_CONDS:
            ag = _agg(rows, probe, cond)
            cells.append(f"{ag['mean'] * 100:.2f}±{ag['std'] * 100:.2f}"
                         if ag["mean"] is not None else "—")
        tag = "（冻结）" if probe in MAIN_PROBES else ("（恒等自检）" if probe == "A_ref" else "（诊断）")
        L.append(f"| {probe}{tag} | " + " | ".join(cells) + " |")
    L.append("")

    c1 = crit["criterion1_main_probe"]
    L.append("## 判据①（跑前冻结）：C_no_dt−A（block5-30，meanpool，3 种子 mean）")
    L.append(f"- C_no_dt={c1['C_no_dt_block5_mean_pct']:.2f}%  A_ref={c1['A_ref_block5_mean_pct']:.2f}%  "
             f"C_dt={c1['C_dt_block5_mean_pct']:.2f}%")
    L.append(f"- **C_no_dt−A={c1['C_no_dt_minus_A_pp']:+.2f}pp**（逐种子 "
             f"{c1['C_no_dt_minus_A_pp_per_seed']}）；C−A={c1['C_minus_A_pp']:+.2f}pp"
             f"（S10 锚点 {c1['S10_anchor_C_minus_A_pp']:+.2f}pp，对账）；"
             f"推理期 dt 净效应 C−C_no_dt={c1['inference_dt_effect_pp']:+.2f}pp")
    L.append(f"- 冻结阈值：≥+{c1['frozen_thresholds_pp']['training_dominant_ge']:.0f}pp→训练期主导；"
             f"<+{c1['frozen_thresholds_pp']['inference_dt_dominant_lt']:.0f}pp→推理 dt 主导"
             f"（矛盾→查实现）；其间→inconclusive")
    L.append(f"- **verdict={c1['verdict']}**：{c1['reading']}\n")

    c3 = crit["criterion3_clean_switch_sanity"]
    L.append("## 判据③（sanity）：clean 口径切换差")
    L.append(f"- C_no_dt@clean={c3['C_no_dt_clean_pct']:.2f}%  C_dt@clean={c3['C_dt_clean_pct']:.2f}%  "
             f"|差|={c3['abs_switch_diff_pp']:.4f}pp（容忍 <{c3['frozen_threshold_pp']}pp）→ "
             f"**{'PASS' if c3['pass'] else 'FAIL'}**：{c3['reading']}\n")

    c2 = crit["criterion2_dt_spectrum"]
    L.append("## 判据②（跑前冻结）：dt 粒度响应谱（seed0_ab，B 臂口径，固定 batch=test 前 256，"
             "mask=block5-30(seed0,pattern0)，扫常数 timespans dt）")
    L.append("| dt | mean hidden L2 | 相对 L2 响应 R(dt) vs dt=1 | batch meanpool acc | "
             "logits max|Δ| vs dt=1 |")
    L.append("|---|---|---|---|---|")
    for dt in DT_SCAN:
        d = spectrum["block5-30"]["per_dt"][str(dt)]
        L.append(f"| {dt} | {d['mean_hidden_L2']:.4f} | {d['rel_L2_response_vs_dt1']:.6f} | "
                 f"{d['meanpool_acc_on_batch'] * 100:.2f}% | {d['logits_max_abs_diff_vs_dt1']:.4f} |")
    L.append(f"- R_max={c2['R_max']:.6f}（平坦阈值 {FLAT_EPS}），hidden 全有限={c2['finite_all']}")
    L.append(f"- **饱和点 dt*={c2['dt_star']}**（预注册定义：min dt∈{DT_SCAN[1:]} 使 "
             f"R(dt)≥{DT_SAT_FRAC}·R_max）")
    L.append("- 冻结阈值：dt*≥5→量化假设存活；dt*<5→证伪；平坦→inconclusive")
    L.append(f"- **verdict={c2['verdict']}**：{c2['reading']}")
    L.append(f"- 稳健性（诊断，非判据输入）："
             f"{json.dumps(c2['robustness_other_conditions'], ensure_ascii=False)}\n")

    cc = crit["crosschecks"]
    L.append("## 恒等自检（失败条件项，防口径漂移）")
    L.append(f"- A_ref vs S10 A 行 max|diff|={cc['A_ref_vs_S10_A_max_abs_diff']:.6f}；"
             f"C_dt vs S10 C 行 max|diff|={cc['C_dt_vs_S10_C_max_abs_diff']:.6f}"
             f"（容差 5e-4）→ identity_pass={cc['identity_pass']}")
    L.append(f"- 注：{cc['note']}\n")

    if smoke:
        L.append("## 冒烟（子步1）")
        L.append(f"- gates={json.dumps(smoke.get('gates', {}), ensure_ascii=False)}，"
                 f"gates_all_pass={smoke.get('gates_all_pass')}，smoke_sec={smoke.get('smoke_sec')}")
        L.append(f"- seed0 clean 两口径：C_dt={smoke['rows']['seed0_clean_C_dt_meanpool']} / "
                 f"C_no_dt={smoke['rows']['seed0_clean_C_no_dt_meanpool']} → "
                 f"切换差 {smoke['rows']['seed0_clean_switch_pp']}pp\n")

    L.append("## 计时与复现")
    L.append(f"- {json.dumps(timing, ensure_ascii=False)}")
    L.append("- 复现：G:/ai/python11/python.exe experiments/nm_negative_probes.py all"
             "（smoke→main→spectrum→判据→CSV/MD/progress；main 逐行原子落盘可断点续跑；"
             "report 子命令从 progress 确定性重建 CSV/MD 不重算）")
    L.append("")
    L.append("## 限制")
    L.append("- C_no_dt 的 gap-bit 通道必须填值（C 模型 core_in=129）：冻结口径填真实掩码"
             "（只去 dt，隔离推理期因素）；诊断口径 C_no_dt_nogap 填 0（再去 gap 信息，"
             "OOD 于 C 训练分布，仅拆解用）")
    L.append("- 次探针为常数 dt 扫描（统一 elapsed time），非 D2 变 dt 序列；测的是 cell 对 dt "
             "量级的敏感性，用于检验帧化量化候选机制，不直接复现 B 臂推理分布")
    L.append("- 本阶段纯推理：未动任何 ckpt、未训练；GPU 与 S11（已收口）无并行争抢")
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ================= 主流程 =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "main", "spectrum", "all", "report", "status"])
    args = ap.parse_args()
    progress = load_progress()

    if args.cmd == "status":
        print(json.dumps({"rows": len(progress.get("rows", [])),
                          "smoke_pass": (progress.get("smoke") or {}).get("gates_all_pass"),
                          "spectrum": bool(progress.get("spectrum")),
                          "criteria": bool(progress.get("criteria")),
                          "status": progress.get("status")}, ensure_ascii=False))
        return 0

    t0 = time.time()
    x_te, y_te = load_test()
    _t_data = time.time() - t0
    print(f"[np] test={x_te.shape} device={N.DEVICE} "
          f"(data_prep {_t_data:.1f}s)", flush=True)

    if args.cmd == "smoke":
        models = load_models([0])
        rep = run_smoke(models, x_te, y_te)
        progress["smoke"] = rep
        progress["status"] = "smoke_pass" if rep["gates_all_pass"] else "smoke_fail"
        save_progress(progress)
        return 0 if rep["gates_all_pass"] else 2

    if args.cmd == "main":
        models = load_models(N.SEEDS)
        rows, el = run_main(models, x_te, y_te, progress, t0)
        need = len(N.SEEDS) * len(MAIN_CONDS) * N.N_PATTERNS * (len(MAIN_PROBES) + len(DIAG_PROBES))
        print(f"[main] rows={len(rows)}/{need} elapsed={el:.1f}s")
        return 0 if len(rows) >= need else 3

    if args.cmd == "spectrum":
        models = load_models([0])
        spec = {}
        for cond in ["block5-30", "uniform30", "clean"]:
            spec[cond] = dt_response_spectrum(models[0]["ab"], x_te, y_te, cond, 0, 0)
            print(f"[spectrum] {cond}: dt*={spec[cond]['dt_star']} "
                  f"R_max={spec[cond]['R_max']:.6f} flat={spec[cond]['flat']}")
        payload = {"schema": "nm-dt-response-spectrum/v1", "stage": STAGE,
                   "substage": SUBSTAGE, "model": "nmnist_seed0_ab.pt",
                   "arm_caliber": "B (manual-loop, constant timespans dt scan)",
                   "dt_scan": DT_SCAN, "saturation_rule":
                       f"dt* = min dt in {DT_SCAN[1:]} with R(dt) >= {DT_SAT_FRAC}*R_max; "
                       f"none -> '>20'; flat if R_max < {FLAT_EPS}",
                   "conditions": spec}
        tmp = SPECTRUM_OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, SPECTRUM_OUT)
        return 0

    if args.cmd == "all":
        # 子步1 冒烟（不过→partial+断点）
        if not progress.get("smoke"):
            models0 = load_models([0])
            rep = run_smoke(models0, x_te, y_te)
            progress["smoke"] = rep
            progress["status"] = "smoke_pass" if rep["gates_all_pass"] else "smoke_fail"
            save_progress(progress)
            if not rep["gates_all_pass"]:
                print("[all] smoke FAILED → partial + breakpoint (不硬撑)")
                return 2
        # 子步2 主探针
        models = load_models(N.SEEDS)
        rows, el = run_main(models, x_te, y_te, progress, t0)
        need = len(N.SEEDS) * len(MAIN_CONDS) * N.N_PATTERNS * (len(MAIN_PROBES) + len(DIAG_PROBES))
        if len(rows) < need:
            print(f"[all] main incomplete {len(rows)}/{need} → checkpointed")
            progress["status"] = "checkpointed"
            save_progress(progress)
            return 3
        # 子步3 次探针
        if not progress.get("spectrum"):
            spec = {}
            for cond in ["block5-30", "uniform30", "clean"]:
                spec[cond] = dt_response_spectrum(models[0]["ab"], x_te, y_te, cond, 0, 0)
            progress["spectrum"] = spec
            save_progress(progress)
            payload = {"schema": "nm-dt-response-spectrum/v1", "stage": STAGE,
                       "substage": SUBSTAGE, "model": "nmnist_seed0_ab.pt",
                       "arm_caliber": "B (manual-loop, constant timespans dt scan)",
                       "dt_scan": DT_SCAN, "saturation_rule":
                           f"dt* = min dt in {DT_SCAN[1:]} with R(dt) >= {DT_SAT_FRAC}*R_max; "
                           f"none -> '>20'; flat if R_max < {FLAT_EPS}",
                       "conditions": spec}
            tmp = SPECTRUM_OUT + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, SPECTRUM_OUT)
        # 子步4 判据+落表
        crit = compute_criteria(rows, progress["spectrum"])
        progress["criteria"] = crit
        timing = {"schema": "nm-negative-probes-timing/v1",
                  "data_prep_sec": round(_t_data, 1),
                  "smoke_sec": (progress.get("smoke") or {}).get("smoke_sec"),
                  "main_eval_sec": round(el, 1),
                  "total_sec": round(time.time() - t0, 1),
                  "hard_cap_sec": HARD_CAP_SEC,
                  "note": "纯推理探针；数据只载 test；无训练无 ckpt 写入；"
                          "data_prep_sec 单独计量（勿与总耗时混记，S10 同款键名 bug 不复犯）"}
        progress["timing"] = timing
        progress["status"] = "done"
        save_progress(progress)
        write_csv(rows)
        write_md(rows, progress["spectrum"], crit, timing, progress.get("smoke"))
        print("[all] criteria: " + json.dumps(crit, ensure_ascii=False))
        return 0

    if args.cmd == "report":
        rows = progress.get("rows", [])
        spec = progress.get("spectrum")
        if not rows or not spec:
            print("[report] incomplete: rows/spectrum missing; nothing written")
            return 0
        crit = compute_criteria(rows, spec)
        progress["criteria"] = crit
        save_progress(progress)
        timing = progress.get("timing", {"note": "report 模式未重算"})
        write_csv(rows)
        write_md(rows, spec, crit, timing, progress.get("smoke"))
        print("[report] CSV/MD regenerated from progress (no recomputation)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
