#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_patch_selftest.py — gapnb 向量化补丁的数值等价 + 提速自验（2026-10-26T02:1x）。

对照实现 = 2026-10-26T01:5x S3 首派单元里的原始逐样本代码
（gap_neighborhood_index + 逐行 out[r, idx].mean(0)），在本脚本内逐字保留。
判据：全部用例 hit 计数逐例相等（整数恒等）+ pooled logits max abs diff < 1e-6。
用法：G:/ai/python11/python.exe experiments/eval_patch_selftest.py
"""
import importlib.util
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pilot_threearm", os.path.join(HERE, "pilot_threearm.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def old_gapnb(out, mask, yb, K=P.GAPNB_K):
    """原始逐样本实现（补丁前逐字保留）：返回 (hit 计数, pooled 矩阵)。"""
    nb_idx = P.gap_neighborhood_index(mask, K)
    hit = 0
    pooled_rows = []
    for r, idx in enumerate(nb_idx):
        pooled = out[r, idx].mean(0) if idx else out[r].mean(0)
        pooled_rows.append(pooled)
        hit += int(pooled.argmax(0).item() == yb[r].item())
    return hit, torch.stack(pooled_rows)


def new_gapnb(out, mask, yb, K=P.GAPNB_K):
    """补丁后向量化实现。"""
    w = P.gap_neighborhood_weights(mask, K)
    pooled = (out * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True)
    return (pooled.argmax(1) == yb).sum().item(), pooled


def main() -> int:
    torch.manual_seed(0)
    dev = P.DEVICE
    worst, cases = 0.0, 0
    for cond in P.CONDITIONS:
        for pattern in range(P.N_PATTERNS):
            for rep in range(4):
                B, L = 256, 100
                mask = P.make_mask(cond, B, L, seed=rep * 7 + pattern, pattern=pattern).to(dev)
                out = torch.randn(B, L, 10, device=dev)
                yb = torch.randint(0, 10, (B,), device=dev)
                h_old, p_old = old_gapnb(out, mask, yb)
                h_new, p_new = new_gapnb(out, mask, yb)
                cases += 1
                assert h_old == h_new, f"hit mismatch {cond} p{pattern} rep{rep}: {h_old} vs {h_new}"
                worst = max(worst, (p_old.double() - p_new.double()).abs().max().item())
    # 提速实测：12 批 ×(256,100,10)，与一次 eval_arm_condition 的 gapnb 工作量同构
    mask = P.make_mask("block5-30", 256, 100, 0, 0).to(dev)
    out = torch.randn(256, 100, 10, device=dev)
    yb = torch.randint(0, 10, (256,), device=dev)
    t0 = time.time()
    for _ in range(12):
        old_gapnb(out, mask, yb)
    t_old = time.time() - t0
    t0 = time.time()
    for _ in range(12):
        new_gapnb(out, mask, yb)
    t_new = time.time() - t0
    print(f"device={dev} cases={cases} hit_counts_all_equal=True "
          f"pooled_max_abs_diff={worst:.3e}")
    print(f"gapnb 12-batch cost: old={t_old:.3f}s new={t_new:.4f}s "
          f"speedup={t_old / max(t_new, 1e-9):.0f}x"
          f"（注意：new 侧 GPU 异步未同步，该倍率非墙钟证据；墙钟提速以串行实测 22.5s/行为准，2026-10-26T03:1x 判读勘误）")
    assert worst < 1e-6, f"pooled diff too large: {worst}"

    # ===== 汇总门控回归（2026-10-26T03:1x researcher 判读新增，事故样本驱动）=====
    # 事故①：clean A≡B 阈值原写 abs(diff)<0.1（分数单位=10pp），宽于预注册 <0.1pp 百倍；
    # 事故②：A@uniform30 方向原写 >=0.60，与 design.md §五预注册「显著高于随机水平」不符。
    # 合成行只驱动 summarize 门控逻辑，不碰任何数值路径。
    def _row(seed, arm, cond, acc, n=3000):
        return {"dataset": "quickdraw", "seed": seed, "arm": arm, "condition": cond,
                "pattern": 0, "n_eval": n, "masked_fraction_realized": 0.3,
                "acc_meanpool": acc, "acc_final": acc, "acc_gapnb": acc}

    # 用例 1：clean A−B=0.5pp（分数 0.005）→ A≡B 门必须不过；uniform30=13% (z≈5.5) → 方向门必须过
    s1 = P.summarize([_row(0, "A", "clean", 0.50), _row(0, "B", "clean", 0.505),
                      _row(0, "A", "uniform30", 0.13), _row(0, "B", "uniform30", 0.13)], "quickdraw")
    assert s1["sanity"]["clean_A_equals_B_lt_0p1pp"] is False, "0.5pp 臂间差必须不过 A≡B<0.1pp 门"
    assert s1["sanity"]["A_uniform30_direction_ok"] is True, "13% vs chance10% (n=3000,z>2) 应过方向门"
    # 用例 2：uniform30=11%（z≈1.8<2）→ 方向门必须不过
    s2 = P.summarize([_row(0, "A", "clean", 0.50), _row(0, "B", "clean", 0.50),
                      _row(0, "A", "uniform30", 0.11), _row(0, "B", "uniform30", 0.11)], "quickdraw")
    assert s2["sanity"]["A_uniform30_direction_ok"] is False, "11% vs chance10% (z<2) 不应过方向门"
    # 用例 3：A−B=0.000 → A≡B 门必须过
    s3 = P.summarize([_row(0, "A", "clean", 0.50), _row(0, "B", "clean", 0.50),
                      _row(0, "A", "uniform30", 0.13), _row(0, "B", "uniform30", 0.13)], "quickdraw")
    assert s3["sanity"]["clean_A_equals_B_lt_0p1pp"] is True, "A−B=0 必须过 A≡B 门"
    print("summarize gate regression: 3/3 PASS（A≡B<0.1pp 严判生效；uniform30 z>2 vs chance 生效）")
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
