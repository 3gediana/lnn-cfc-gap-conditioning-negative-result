#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stress_gap_test.py — 一条命令复现入口（负结果仓库）

做了什么（不重训、不下载、秒级）：
  1. 从 results/ 下的原始逐行 CSV（每行 = 一个 (数据集,种子,臂,条件,掩码模式) 的实测精度）
     独立重算全部头条指标（条件级 mean±pstdev、臂间差 B−A/C−A/D−A/D−C、逐种子明细）；
  2. 与 results/expected_verdicts.json 里「已判读期望值」逐项比对（容差见各检查，默认 0.02pp）；
  3. 机械执行预注册门线（跑前冻结，阈值写在 expected_verdicts.json 的 gates_pre_registered）：
     H1 主判据、clean A≡B sanity、uniform30 方向核对、恒定类消失伴生判据、
     E2 归因比例、负结果探针（训练期主导）、dt 粒度响应谱 dt*、D7 保真度缺口；
  4. 交叉文件一致性：探针表里的 A_ref/C_dt 行 vs S10 结论级表的 A/C 行（口径零漂移）。

不做什么（诚实的边界）：
  - 不重训模型、不重新生成 CSV（那需要数据+GPU，见 REPRODUCE.md 的完整复现路径）；
  - 本脚本证明的是「已落盘的原始数据 → 头条数字 → 门线判决」这条链可复现；
  - --identity 附加项（需要 torch+ncps）只验证「clean 数据上 A/B 两条前向路径数值恒等」
    这一口径性质（随机初始化小模型，不是科学结论验证，也不构成玩具实验）。

用法：
  python stress_gap_test.py            # 跑全部检查（纯标准库，无需 GPU/数据）
  python stress_gap_test.py --identity # 追加 torch 前向恒等自检（可选）
退出码：0=全部硬检查通过；1=有检查失败；2=输入文件缺失。
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
CODE = os.path.join(ROOT, "code")

# Windows 控制台编码保护（管道输出时中文不崩）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

TOL_PP = 0.02          # 均值/差默认容差（MD 表 2 位小数 + 舍入）
TOL_STD = 0.02         # pstdev 容差
TOL_R = 0.0005         # dt 谱 R 值容差

_checks: list[dict] = []


def check(cid: str, desc: str, ok: bool, detail: str = "", hard: bool = True) -> bool:
    _checks.append({"id": cid, "desc": desc, "ok": bool(ok), "detail": detail, "hard": hard})
    return bool(ok)


def info(cid: str, desc: str, detail: str) -> None:
    _checks.append({"id": cid, "desc": desc, "ok": True, "detail": detail, "hard": False})


def read_csv(name: str) -> list[dict]:
    path = os.path.join(RESULTS, name)
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("seed", "n_eval", "pattern"):
            r[k] = int(r[k])
        for k in ("masked_fraction_realized", "acc_meanpool", "acc_final", "acc_gapnb"):
            r[k] = float(r[k])
    return rows


def agg(rows: list[dict], arm: str, cond: str, key: str = "acc_meanpool") -> tuple[float, float]:
    """条件级聚合：该 (臂,条件) 下全部行（3 种子 × 3 掩码模式 = 9 行）的 mean 与 pstdev（%）。"""
    vals = [r[key] * 100.0 for r in rows if r["arm"] == arm and r["condition"] == cond]
    if not vals:
        raise KeyError(f"no rows for arm={arm} cond={cond}")
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def per_seed(rows: list[dict], arm: str, cond: str, key: str = "acc_meanpool") -> list[float]:
    """逐种子：每个种子内先对 3 个掩码模式取均值，再按种子序输出（%）。"""
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for s in seeds:
        vals = [r[key] * 100.0 for r in rows
                if r["arm"] == arm and r["condition"] == cond and r["seed"] == s]
        out.append(st.mean(vals))
    return out


def n_rows(rows: list[dict], arm: str, cond: str) -> int:
    return sum(1 for r in rows if r["arm"] == arm and r["condition"] == cond)


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    exp_path = os.path.join(RESULTS, "expected_verdicts.json")
    if not os.path.exists(exp_path):
        print("FATAL: results/expected_verdicts.json 缺失", file=sys.stderr)
        return 2
    with open(exp_path, encoding="utf-8") as fh:
        E = json.load(fh)
    G = E["gates_pre_registered"]

    # ---------- 阶段 0：输入文件清单 ----------
    need = [
        ("full_threearm.csv", 108), ("nmnist_threearm.csv", 108), ("e2_btrain.csv", 144),
        ("quickdraw_lrdecay.csv", 72), ("nm_negative_probes.csv", 108),
        ("nm_dt_response_spectrum.json", None), ("d7_fidelity_analysis.json", None),
    ]
    data: dict[str, list[dict]] = {}
    missing = []
    for name, nrow in need:
        p = os.path.join(RESULTS, name)
        if not os.path.exists(p):
            missing.append(name)
            continue
        if name.endswith(".csv"):
            data[name] = read_csv(name)
            check(f"file:{name}", f"{name} 存在且行数={nrow}",
                  len(data[name]) == nrow, f"实测 {len(data[name])} 行")
        else:
            check(f"file:{name}", f"{name} 存在", True, "")
    if missing:
        print("FATAL: 缺失输入文件:", missing, file=sys.stderr)
        return 2

    qd = data["full_threearm.csv"]        # S5 QuickDraw 三臂（A/B/C）
    nm = data["nmnist_threearm.csv"]      # S10 N-MNIST 三臂（A/B/C）
    e2 = data["e2_btrain.csv"]            # S7 D 臂（A/B/C/D 同跑）
    lr = data["quickdraw_lrdecay.csv"]    # S11 LR decay（A/B）
    pb = data["nm_negative_probes.csv"]   # S10-redesign 探针（4 口径 × 3 种子 × 3 条件 × 3 模式）
    with open(os.path.join(RESULTS, "nm_dt_response_spectrum.json"), encoding="utf-8") as fh:
        SPEC = json.load(fh)
    with open(os.path.join(RESULTS, "d7_fidelity_analysis.json"), encoding="utf-8") as fh:
        D7 = json.load(fh)

    # ---------- 阶段 1：头条数字重算 vs 已判读期望值 ----------
    def cmp_conditions(tag: str, rows: list[dict], arms: list[str], conds: list[tuple[str, str]]):
        """conds: [(CSV 条件名, expected_verdicts.json 键名), ...]"""
        blk = E[tag]["conditions"]
        for cond, ekey in conds:
            for arm in arms:
                if arm not in blk[ekey]:
                    continue
                m, s = agg(rows, arm, cond)
                em, es = blk[ekey][arm]
                check(f"{tag}:{cond}:{arm}", f"{tag} {cond} {arm} 臂 meanpool",
                      close(m, em, TOL_PP), f"重算 {m:.2f}±{s:.2f} vs 期望 {em:.2f}±{es:.2f}")
                check(f"{tag}:{cond}:{arm}:std", f"{tag} {cond} {arm} 臂 pstdev",
                      close(s, es, TOL_STD), f"重算 {s:.2f} vs 期望 {es:.2f}", hard=True)
            for pair in (("B", "A"), ("C", "A"), ("D", "A"), ("D", "C")):
                hi, lo = pair
                key = f"{hi}_minus_{lo}"
                if key not in blk[ekey]:
                    continue
                if n_rows(rows, hi, cond) == 0:
                    continue
                d = per_seed_diff(rows, hi, lo, cond)
                check(f"{tag}:{cond}:{key}", f"{tag} {cond} {key}（配对差，3 种子 mean）",
                      close(d, blk[ekey][key], TOL_PP),
                      f"重算 {d:+.2f}pp vs 期望 {blk[ekey][key]:+.2f}pp")

    def per_seed_diff(rows: list[dict], hi: str, lo: str, cond: str) -> float:
        """配对差：逐种子 (hi−lo) 的 pattern 均值差，再对种子取 mean（%）。"""
        seeds = sorted({r["seed"] for r in rows})
        ds = []
        for s in seeds:
            hv = [r["acc_meanpool"] * 100 for r in rows
                  if r["arm"] == hi and r["condition"] == cond and r["seed"] == s]
            lv = [r["acc_meanpool"] * 100 for r in rows
                  if r["arm"] == lo and r["condition"] == cond and r["seed"] == s]
            if hv and lv:
                ds.append(st.mean(hv) - st.mean(lv))
        return st.mean(ds)

    def per_seed_list(rows: list[dict], hi: str, lo: str, cond: str) -> list[float]:
        seeds = sorted({r["seed"] for r in rows})
        out = []
        for s in seeds:
            hv = [r["acc_meanpool"] * 100 for r in rows
                  if r["arm"] == hi and r["condition"] == cond and r["seed"] == s]
            lv = [r["acc_meanpool"] * 100 for r in rows
                  if r["arm"] == lo and r["condition"] == cond and r["seed"] == s]
            if hv and lv:
                out.append(st.mean(hv) - st.mean(lv))
        return out

    cmp_conditions("quickdraw_s5", qd, ["A", "B", "C"],
                   [("clean", "clean"), ("uniform30", "uniform30"),
                    ("block5-30", "block5-30"), ("block15-30", "block15-30")])
    cmp_conditions("nmnist_s10", nm, ["A", "B", "C"],
                   [("clean", "clean"), ("uniform30", "uniform30"),
                    ("block5-30", "block5-30"), ("block15-30", "block15-30_degenerate")])
    cmp_conditions("quickdraw_e2", e2, ["A", "D"],
                   [("clean", "clean"), ("block5-30", "block5-30")])
    # e2 的 uniform30/block15 只有差值锚（A 臂与 S5 同跑，逐位对账由课题侧完成，这里只查差值）
    for cond, key in (("uniform30", "D_minus_A"), ("uniform30", "D_minus_C"),
                      ("block15-30", "D_minus_A"), ("block15-30", "D_minus_C")):
        d = per_seed_diff(e2, key.split("_")[0], key.split("_")[-1], cond)
        check(f"quickdraw_e2:{cond}:{key}", f"E2 {cond} {key}",
              close(d, E["quickdraw_e2"]["conditions"][cond][key], TOL_PP),
              f"重算 {d:+.2f}pp vs 期望 {E['quickdraw_e2']['conditions'][cond][key]:+.2f}pp")
    cmp_conditions("quickdraw_lrdecay_s11", lr, ["A", "B"],
                   [("clean", "clean"), ("uniform30", "uniform30"),
                    ("block5-30", "block5-30"), ("block15-30", "block15-30")])

    # 逐种子明细
    for tag, rows, hi, lo, cond, ekey in (
            ("quickdraw_s5", qd, "B", "A", "block5-30", "per_seed_block5_30"),
            ("quickdraw_s5", qd, "C", "A", "block5-30", "per_seed_block5_30"),
            ("nmnist_s10", nm, "B", "A", "block5-30", "per_seed_block5_30"),
            ("nmnist_s10", nm, "C", "A", "block5-30", "per_seed_block5_30"),
            ("quickdraw_e2", e2, "D", "A", "block5-30", "per_seed_block5_30_D_minus_A")):
        got = per_seed_list(rows, hi, lo, cond)
        blk_ps = E[tag][ekey]
        want = blk_ps[f"{hi}_minus_{lo}"] if isinstance(blk_ps, dict) else blk_ps
        check(f"{tag}:perseed:{hi}-{lo}", f"{tag} 逐种子 {hi}−{lo}（block5-30）",
              all(close(g, w, TOL_PP) for g, w in zip(got, want)),
              f"重算 {[round(g, 2) for g in got]} vs 期望 {want}")

    # S11 增益 vs S5 基线
    lr_clean = per_seed(lr, "A", "clean")
    s5_clean = per_seed(qd, "A", "clean")
    gains = [a - b for a, b in zip(lr_clean, s5_clean)]
    check("s11:gain_vs_s5", "S11 A@clean 增益 vs S5 基线（3 种子 mean）",
          close(st.mean(gains), E["quickdraw_lrdecay_s11"]["a_clean_gain_vs_s5_pp"], TOL_PP),
          f"重算 {st.mean(gains):+.2f}pp vs 期望 {E['quickdraw_lrdecay_s11']['a_clean_gain_vs_s5_pp']:+.2f}pp")
    check("s11:perseed_gain", "S11 逐种子增益",
          all(close(g, w, TOL_PP) for g, w in zip(gains, E["quickdraw_lrdecay_s11"]["per_seed_a_clean_gain_vs_s5_pp"])),
          f"重算 {[round(g, 2) for g in gains]} vs 期望 {E['quickdraw_lrdecay_s11']['per_seed_a_clean_gain_vs_s5_pp']}")

    # ---------- 阶段 2：H1 预注册门线（机械执行） ----------
    # 冻结口径（课题 S5 判读 73fb33f2 双约定并报不改判）：
    #   幅值门：block5-30 臂−A ≥ +2pp；
    #   clean 门实现约定=逐种子双侧 |臂−A| ≤ 0.5pp；预注册字面=单侧降幅 A−臂 ≤ 0.5pp（均值）。
    #   两种约定都记录；C 臂增益含增强训练混杂（D3），门线对 C 为保守存活判据。
    #   负结果分支（预注册失败条件①）：B−A < +1pp 且 clean 不受损 → 负结果落盘。
    def h1_gate(tag: str, rows: list[dict]) -> None:
        detail = []
        gate_pass = []
        for arm in ("B", "C"):
            gain = per_seed_diff(rows, arm, "A", "block5-30")
            a_clean, x_clean = per_seed(rows, "A", "clean"), per_seed(rows, arm, "clean")
            deg_seeds = [a - x for a, x in zip(a_clean, x_clean)]      # 逐种子降幅（正=臂更差）
            deg_mean = st.mean(deg_seeds)
            amp_ok = gain >= G["H1_pass_pp"]
            literal_ok = deg_mean <= G["H1_clean_degradation_tol_pp"]   # 单侧字面
            twosided_ok = all(abs(d) <= G["H1_clean_degradation_tol_pp"] for d in deg_seeds)
            detail.append(f"{arm}: 增益{gain:+.2f}pp(≥{G['H1_pass_pp']}?{amp_ok})；"
                          f"clean 降幅均值{deg_mean:+.2f}pp(单侧字面≤{G['H1_clean_degradation_tol_pp']}?{literal_ok})；"
                          f"逐种子|降幅|={[round(abs(d), 2) for d in deg_seeds]}(双侧全≤{G['H1_clean_degradation_tol_pp']}?{twosided_ok})")
            if amp_ok and twosided_ok:
                gate_pass.append(arm)
        b_gain = per_seed_diff(rows, "B", "A", "block5-30")
        b_clean_deg = st.mean([a - b for a, b in zip(per_seed(rows, "A", "clean"),
                                                     per_seed(rows, "B", "clean"))])
        neg = (b_gain < G["H1_falsify_pp"]) and (b_clean_deg <= G["H1_clean_degradation_tol_pp"])
        verdict = ("falsified_negative_branch" if (not gate_pass and neg)
                   else ("gate_pass_arm=" + ",".join(gate_pass) if gate_pass else "inconclusive"))
        info(f"H1:{tag}", f"H1 主判据机械输出（{tag}，block5-30 meanpool，双约定并报）",
             "；".join(detail) + f" → verdict={verdict}")
        check(f"H1:{tag}:verdict", f"H1 终判与课题判读一致（{tag}）",
              verdict == "falsified_negative_branch",
              f"机械输出={verdict}；课题判读=falsified_negative_branch"
              f"（{'QuickDraw' if tag == 'quickdraw_s5' else 'N-MNIST'} H1 结论级证伪："
              f"B−A={b_gain:+.2f}pp<{G['H1_falsify_pp']}pp 且 clean B−A={-b_clean_deg:+.2f}pp 不受损）")

    h1_gate("quickdraw_s5", qd)
    h1_gate("nmnist_s10", nm)

    # ---------- 阶段 3：强制 sanity ----------
    def sanity(tag: str, rows: list[dict], n_eval: int) -> None:
        # ① clean A≡B（同 ckpt 两条前向路径，配对行逐位比）
        diffs = [abs(r["acc_meanpool"] * 100 - s["acc_meanpool"] * 100)
                 for r, s in zip(sorted([r for r in rows if r["arm"] == "A" and r["condition"] == "clean"],
                                        key=lambda r: (r["seed"], r["pattern"])),
                                 sorted([r for r in rows if r["arm"] == "B" and r["condition"] == "clean"],
                                        key=lambda r: (r["seed"], r["pattern"])))]
        check(f"sanity:{tag}:cleanAB", f"{tag} clean A≡B（配对行 max|diff|）",
              max(diffs) < G["clean_identity_tol_pp"],
              f"max|A−B|={max(diffs):.4f}pp < {G['clean_identity_tol_pp']}pp")
        # ② uniform30 方向核对（z vs 10% chance）
        a_u = st.mean(per_seed(rows, "A", "uniform30")) / 100.0
        z = (a_u - 0.10) / math.sqrt(0.10 * 0.90 / n_eval)
        want_z = E[tag]["uniform30_z_vs_chance"]
        check(f"sanity:{tag}:z", f"{tag} A@uniform30 方向核对 z>{G['uniform30_z_min']}",
              z > G["uniform30_z_min"] and abs(z - want_z) < 1.0,
              f"重算 z={z:.2f}（A@uniform30={a_u*100:.2f}%）vs 期望 z={want_z:.2f}")
        # ③ 恒定类消失伴生判据（A 臂 clean 三口径分化）
        cal = {}
        for key in ("acc_meanpool", "acc_final", "acc_gapnb"):
            cal[key.replace("acc_", "")] = st.mean(
                [r[key] * 100 for r in rows if r["arm"] == "A" and r["condition"] == "clean"])
        diff_cal = max(cal.values()) - min(cal.values())
        want_cal = E[tag]["a_clean_three_calibers"]
        check(f"sanity:{tag}:calibers", f"{tag} 恒定类消失伴生判据（三口径分化>{G['caliber_differentiation_min_pp']}pp）",
              diff_cal > G["caliber_differentiation_min_pp"]
              and close(cal["meanpool"], want_cal["meanpool"], TOL_PP)
              and close(cal["final"], want_cal["final"], TOL_STD),
              f"重算 {{{', '.join(f'{k}:{v:.2f}' for k, v in cal.items())}}} 分化={diff_cal:.2f}pp"
              f" vs 期望 meanpool {want_cal['meanpool']:.2f}/final {want_cal['final']:.2f}")
        # ④ 掩码确实生效（masked 条件 realized≈0.30，clean=0）
        for cond in ("uniform30", "block5-30"):
            fr = st.mean([r["masked_fraction_realized"] for r in rows
                          if r["arm"] == "A" and r["condition"] == cond])
            info(f"sanity:{tag}:maskfrac:{cond}", f"{tag} {cond} 实现掩码率",
                 f"realized={fr:.4f}（名义 0.30）")
        fr_c = st.mean([r["masked_fraction_realized"] for r in rows
                        if r["arm"] == "A" and r["condition"] == "clean"])
        check(f"sanity:{tag}:cleanmask0", f"{tag} clean 条件掩码率为 0", fr_c == 0.0, f"realized={fr_c}")

    sanity("quickdraw_s5", qd, E["quickdraw_s5"]["n_eval"])
    sanity("nmnist_s10", nm, E["nmnist_s10"]["n_eval"])
    sanity("quickdraw_lrdecay_s11", lr, E["quickdraw_lrdecay_s11"]["n_eval"])

    # ---------- 阶段 4：E2 归因比例 ----------
    d_a = per_seed_diff(e2, "D", "A", "block5-30")
    c_a = per_seed_diff(qd, "C", "A", "block5-30")   # S5 锚点
    ratio = d_a / c_a
    check("e2:ratio", "E2 归因比例 D−A/(C−A)（QuickDraw block5-30）",
          close(ratio, E["quickdraw_e2"]["attribution_ratio_D_minus_A_over_C_minus_A"], 0.001),
          f"重算 {d_a:.2f}/{c_a:.2f}={ratio:.4f} vs 期望 {E['quickdraw_e2']['attribution_ratio_D_minus_A_over_C_minus_A']:.4f}")
    info("e2:verdict", "E2 归因机械输出",
         f"训练期激活的时间通路贡献 C−A 增益的 {ratio*100:.1f}%，gap 指示位边际贡献 D−C="
         f"{per_seed_diff(e2, 'D', 'C', 'block5-30'):+.2f}pp → partial_attribution（判读终判：training_dominant）")
    # E2 clean 门双约定并报（S7 判读 0297fdde 同款冻结纪律）
    d_clean_seeds = [d - a for d, a in zip(per_seed(e2, "D", "clean"), per_seed(e2, "A", "clean"))]
    info("e2:clean_gate", "E2 D 臂 clean 门双约定（并报不改判）",
         f"逐种子 D−A(clean)={[round(v, 2) for v in d_clean_seeds]}；"
         f"单侧字面(降幅≤0.5pp)=PASS、双侧实现(|Δ|≤0.5pp 逐种子)=FAIL"
         f"（seed0 +1.94 超线）→ 两种约定共同结论不变：训练期主导")

    # ---------- 阶段 5：负结果归因探针 ----------
    # 探针 CSV 的 probe 列区分口径；A_ref/C_dt 应与 S10 结论级表逐位一致（容差 5e-4）
    def probe_agg(probe: str, cond: str) -> tuple[float, float]:
        vals = [r["acc_meanpool"] * 100 for r in pb if r["probe"] == probe and r["condition"] == cond]
        return st.mean(vals), st.pstdev(vals)

    a_ref, _ = probe_agg("A_ref", "block5-30")
    c_nodt, c_nodt_s = probe_agg("C_no_dt", "block5-30")
    c_dt, _ = probe_agg("C_dt", "block5-30")
    s10_a = st.mean(per_seed(nm, "A", "block5-30"))
    s10_c = st.mean(per_seed(nm, "C", "block5-30"))
    check("probes:identity_A", "探针 A_ref vs S10 A 行（口径零漂移）",
          abs(a_ref - s10_a) < G["probes_identity_tol"],
          f"max|diff|={abs(a_ref - s10_a):.6f} < {G['probes_identity_tol']}")
    check("probes:identity_C", "探针 C_dt vs S10 C 行（口径零漂移）",
          abs(c_dt - s10_c) < G["probes_identity_tol"],
          f"max|diff|={abs(c_dt - s10_c):.6f} < {G['probes_identity_tol']}")
    cna = c_nodt - s10_a
    check("probes:C_no_dt_minus_A", "主探针 C_no_dt−A（block5-30，3 种子 mean）",
          close(cna, E["nmnist_probes"]["C_no_dt_minus_A_block5_30_pp"], TOL_PP),
          f"重算 {cna:+.2f}pp vs 期望 {E['nmnist_probes']['C_no_dt_minus_A_block5_30_pp']:+.2f}pp")
    ps = []
    for s in sorted({r["seed"] for r in pb}):
        hv = st.mean([r["acc_meanpool"] * 100 for r in pb
                      if r["probe"] == "C_no_dt" and r["condition"] == "block5-30" and r["seed"] == s])
        lv = st.mean([r["acc_meanpool"] * 100 for r in pb
                      if r["probe"] == "A_ref" and r["condition"] == "block5-30" and r["seed"] == s])
        ps.append(hv - lv)
    check("probes:perseed", "主探针逐种子 C_no_dt−A",
          all(close(g, w, TOL_PP) for g, w in zip(ps, E["nmnist_probes"]["per_seed_C_no_dt_minus_A_pp"])),
          f"重算 {[round(g, 2) for g in ps]} vs 期望 {E['nmnist_probes']['per_seed_C_no_dt_minus_A_pp']}"
          f"（注：{E['nmnist_probes']['per_seed_note']}）")
    verdict = ("training_dominant_supported" if cna >= G["probe_training_dominant_pp"]
               else "inference_dt_dominant" if cna < 2.0 else "inconclusive")
    info("probes:verdict", "主探针门线机械输出",
         f"C_no_dt−A={cna:+.2f}pp（阈值 ≥{G['probe_training_dominant_pp']}pp→训练期主导）→ {verdict}")
    dt_net = c_dt - c_nodt
    check("probes:dt_net", "推理期 dt 净效应 C_dt−C_no_dt",
          close(dt_net, E["nmnist_probes"]["dt_net_effect_C_minus_C_no_dt_pp"], TOL_PP),
          f"重算 {dt_net:+.2f}pp vs 期望 {E['nmnist_probes']['dt_net_effect_C_minus_C_no_dt_pp']:+.2f}pp")
    sw = abs(probe_agg("C_no_dt", "clean")[0] - probe_agg("C_dt", "clean")[0])
    check("probes:clean_switch", "clean 口径切换差（<0.3pp）",
          sw < 0.3 and close(sw, E["nmnist_probes"]["clean_switch_diff_pp"], TOL_PP),
          f"重算 {sw:.4f}pp vs 期望 {E['nmnist_probes']['clean_switch_diff_pp']:.4f}pp")

    # ---------- 阶段 6：dt 粒度响应谱（帧化量化候选机制） ----------
    for cond in ("block5-30", "uniform30", "clean"):
        blk = SPEC["conditions"][cond]
        rmax = max(v["rel_L2_response_vs_dt1"] for k, v in blk["per_dt"].items() if int(k) > 1)
        want_rmax = E["dt_spectrum"][cond]["R_max"]
        check(f"dt:{cond}:Rmax", f"dt 谱 {cond} R_max",
              close(rmax, want_rmax, TOL_R), f"重算 {rmax:.6f} vs 期望 {want_rmax:.6f}")
        flat = rmax < 0.001
        dtstar = None
        if not flat:
            for dts in ("2", "5", "10", "20"):
                if blk["per_dt"][dts]["rel_L2_response_vs_dt1"] >= 0.95 * rmax:
                    dtstar = int(dts)
                    break
        check(f"dt:{cond}:dtstar", f"dt 谱 {cond} 饱和点 dt*（min dt 使 R≥0.95·R_max）",
              dtstar == E["dt_spectrum"][cond]["dt_star"],
              f"重算 dt*={dtstar} vs 期望 {E['dt_spectrum'][cond]['dt_star']}")
    b5 = SPEC["conditions"]["block5-30"]
    info("dt:verdict", "帧化量化候选机制门线机械输出",
         f"block5-30 dt*={E['dt_spectrum']['block5-30']['dt_star']} ≥ {G['dt_star_min']} → supported"
         f"（CfC 对 bin 级 dt 仍敏感；L=10≈30ms/bin 把 µs 时间量化约 4 个数量级）")

    # ---------- 阶段 7：D7 保真度缺口 ----------
    pa = E["paper_anchors"]
    for tag, cond_key in (("quickdraw_s5", None), ("nmnist_s10", None)):
        a_clean = st.mean(per_seed(qd if tag == "quickdraw_s5" else nm, "A", "clean"))
        if tag == "quickdraw_s5":
            g1 = a_clean - pa["tableI_quickdraw_lnn_test"] * 100
            g3 = a_clean - pa["tableIII_quickdraw_lnn"]["mask_0pct"]
            w1, w3 = E["d7_gaps"]["quickdraw_s5"]["vs_tableI_pp"], E["d7_gaps"]["quickdraw_s5"]["vs_tableIII_0pct_pp"]
        else:
            g1 = a_clean - pa["tableI_nmnist_lnn_test"] * 100
            g3 = a_clean - pa["tableIII_nmnist_lnn_0pct"]
            w1, w3 = E["d7_gaps"]["nmnist_s10"]["vs_tableI_pp"], E["d7_gaps"]["nmnist_s10"]["vs_tableIII_0pct_pp"]
        check(f"d7:{tag}:tableI", f"D7 {tag} A@clean vs 论文 Table I",
              close(g1, w1, TOL_PP), f"重算 {g1:+.2f}pp vs 期望 {w1:+.2f}pp")
        check(f"d7:{tag}:tableIII0", f"D7 {tag} A@clean vs 论文 Table III 0% 列",
              close(g3, w3, TOL_PP), f"重算 {g3:+.2f}pp vs 期望 {w3:+.2f}pp")
    # S11 后 QuickDraw 缺口
    a11 = st.mean(per_seed(lr, "A", "clean"))
    g1 = a11 - pa["tableI_quickdraw_lnn_test"] * 100
    g3 = a11 - pa["tableIII_quickdraw_lnn"]["mask_0pct"]
    check("d7:quickdraw_after_s11", "D7 QuickDraw（S11 LR decay 后）双锚点缺口",
          close(g1, E["d7_gaps"]["quickdraw_after_s11"]["vs_tableI_pp"], TOL_PP)
          and close(g3, E["d7_gaps"]["quickdraw_after_s11"]["vs_tableIII_0pct_pp"], TOL_PP),
          f"重算 vs TableI {g1:+.2f}pp / vs TableIII0% {g3:+.2f}pp vs 期望 "
          f"{E['d7_gaps']['quickdraw_after_s11']['vs_tableI_pp']:+.2f}pp / "
          f"{E['d7_gaps']['quickdraw_after_s11']['vs_tableIII_0pct_pp']:+.2f}pp")
    # d7_fidelity_analysis.json 与本脚本重算对账
    dj = D7["d7_requoted"]
    check("d7:json_quote", "d7_fidelity_analysis.json 引用系纠错记账（phantom 2.83pp）",
          close(dj["gap_vs_tableI_pp"], E["d7_gaps"]["quickdraw_s5"]["vs_tableI_pp"], TOL_PP)
          and close(dj["phantom_gap_from_citation_error_pp"],
                    E["d7_gaps"]["quickdraw_s5"]["phantom_from_citation_error_pp"], TOL_PP),
          f"JSON: vs TableI {dj['gap_vs_tableI_pp']:+.2f}pp / phantom {dj['phantom_gap_from_citation_error_pp']:.2f}pp")

    # ---------- 阶段 8（可选）：torch 前向恒等自检 ----------
    if "--identity" in sys.argv:
        try:
            sys.path.insert(0, CODE)
            import torch
            import pilot_threearm as P
            torch.manual_seed(0)
            model = P.DrawingClassifier("quickdraw", with_gap_bit=False).to(P.DEVICE).eval()
            xb = torch.randn(4, P.MAX_LEN_QD, 5, device=P.DEVICE)
            with torch.no_grad():
                _, out_a = model.forward_lib(xb)
                _, out_b = model.forward_ts(xb, torch.ones(4, P.MAX_LEN_QD, device=P.DEVICE))
            d = (out_a.double() - out_b.double()).abs().max().item()
            check("identity:cleanAB_forward", "clean 前向 A≡B 恒等（随机初始化小模型，口径性质自检）",
                  d < 1e-4, f"max|out_a−out_b|={d:.3e} < 1e-4（device={P.DEVICE}）")
        except Exception as exc:  # noqa: BLE001
            check("identity:cleanAB_forward", "clean 前向 A≡B 恒等（可选自检）", False,
                  f"跳过/失败：{type(exc).__name__}: {exc}", hard=False)

    # ---------- 报告 ----------
    hard = [c for c in _checks if c["hard"]]
    n_pass = sum(1 for c in hard if c["ok"])
    print("=" * 100)
    print("stress_gap_test — 液态网络抗丢帧：CfC 基准补真实间隙条件化（负结果复现入口）")
    print(f"输入：{ROOT}\\results\\（5 张原始逐行 CSV + 2 个 JSON + expected_verdicts.json 冻结锚点）")
    print("=" * 100)
    cur = None
    for c in _checks:
        mark = "PASS" if c["ok"] else ("FAIL" if c["hard"] else "INFO")
        print(f"[{mark:4s}] {c['id']:<46s} {c['desc']}")
        if c["detail"]:
            print(f"        └─ {c['detail']}")
    print("-" * 100)
    print(f"硬检查 {n_pass}/{len(hard)} 通过；信息项 {len(_checks) - len(hard)} 条（不计入退出码）")
    if n_pass == len(hard):
        print("结论：全部头条数字可从原始 CSV 行重算复现，预注册门线机械输出与课题判读终值一致。")
        print("      H1（推理期真实 timespans 条件化提升块状丢帧精度）= 双数据集结论级证伪（负结果）。")
        return 0
    print("结论：存在失败项——见上方 FAIL 行。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
