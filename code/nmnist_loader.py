#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nmnist_loader.py — S8-nmnist-event-loader：N-MNIST 事件 loader 固化（结论级基准数据前置）。

课题：液态网络抗丢帧没接时间-cfc基准补真实间隙条件化-4a1c4c0a
阶段：S8-nmnist-event-loader

格式裁决（2026-10-31 实测 + 论文仓库 lnn_nmnist.py 双向坐实）：
  每个 .bin = 事件序列，每事件 5 字节；40-bit 位域（与 data/nmnist/README.md 位描述一致）：
      b0 = X（0-33）、b1 = Y（0-33）、b2 = (polarity<<7) | (timestamp  bits 22-16)、
      b3 = timestamp bits 15-8、b4 = timestamp bits 7-0
  即  t = ((b2 & 0x7f) << 16) | (b3 << 8) | b4（µs），p = (b2 >> 7) & 1。
  - 与论文仓库 experiments/_repo/lnn_nmnist.py:37-40 逐字一致（x=raw[0::5]、
    y=raw[1::5]、p=(raw[2::5]&128)>>7、t=((raw[2::5]&127)<<16)|(raw[3::5]<<8)|raw[4::5]）。
  - 10-26 wrap 复验（DECISIONS 10-26 07:6x）看到的「p 字节 ∈ {0..4}∪{128..132}」
    即 b2 = polarity<<7 | (t>>16) 的联合分布（t>>16∈0..4 ⇔ t<327680µs≈328ms），
    不是极性本身的取值——布尔化「合规」判定会吞掉这一口径分歧，故本 loader 显式拆分。
  - 10-26 的「t=u16LE 且非单调」是低位双字节截断伪影：u16 回绕制造大量逆序；
    正确解读下事件按时间存储（见 results/nmnist_loader_check.md 全量统计），
    但本 loader 仍不假设排序（span 取 min/max、分箱与输入序无关）。

帧堆积规则（显式定义；L 与时 Bin 无先验，以论文仓库口径为准并做鲁棒化）：
  L=10（EV-06/design.md 协议锁定，pilot_threearm.py 文档串「N-MNIST=10 bin log1p」）；
  每样本按自身时间跨度 [min t, max t] 等分 L 个物理时 Bin，
  bin = min(L-1, floor((t - t_min) / ((t_max - t_min)/L)))；span=0 时全部落 bin 0；
  计数 frames[bin, p, y, x]（空间索引 (y,x)，与论文一致；x/y≥34 的事件按论文丢弃），
  输出 log1p(counts)，dtype float32，形状 (N, L, 2, 34, 34)。
  与论文 _to_frames 的差异仅一处：论文用 t[0],t[-1] 作跨度端点（假设时序存储），
  本 loader 用 min/max——在全量单调文件上二者恒等（check 报告实测），
  在非单调文件上论文版会出现负 bin_idx 的 Python 负索引 silent bug，本版不受影响。

样本聚合：Train 60000 / Test 10000（10 类，按 (label, 文件名) 排序保证确定性；
论文用 glob 无序，本 loader 改为 sorted 以便种子化子集可复现）。

用法：
  python experiments/nmnist_loader.py build --split all     # 物化张量缓存（G: 盘 data/nmnist/cache/）
  python experiments/nmnist_loader.py verify                # 自检：格式/锚点/与论文逐位一致/缓存一致
  from nmnist_loader import load_split                      # x(N,L,2,34,34) f32 memmap, y(N,) i8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA_NM = os.path.join(PROJECT, "data", "nmnist", "unpacked")
CACHE_DIR = os.path.join(PROJECT, "data", "nmnist", "cache")
MANIFEST = os.path.join(CACHE_DIR, "manifest.json")

# ---- 协议常数 ----
L_BINS = 10           # EV-06 / design.md：N-MNIST = 10 bin log1p
SIZE = 34             # 34x34 传感器
EV_BYTES = 5          # 每事件 5 字节
N_TRAIN = 60000
N_TEST = 10000
N_CLASSES = 10
SPLITS = {"train": ("Train", N_TRAIN), "test": ("Test", N_TEST)}

# 10-26 wrap 独立分布复验锚点（notes/DECISIONS.md 2026-10-26 07:6x 条）
ANCHOR = {
    "relpath": "Train/1/17665.bin",
    "n_events": 652,
    "b2_keys": list(range(0, 5)) + list(range(128, 133)),
    "n_b2_129_132": 299,          # 10-26 原文「129-132 合计 299」
    "frac_b2_129_132": 299 / 652,  # ≈0.4586「≈46% 带 128 高位」
    "x_range": (1, 33),
    "y_range": (0, 33),
}


# ================= 事件解析 =================

def parse_bin(path: str):
    """读一个 .bin → (x, y, p, t) int64 数组。5 字节/事件，字段切分见模块文档。"""
    with open(path, "rb") as fh:
        raw = np.frombuffer(fh.read(), dtype=np.uint8)
    if raw.size % EV_BYTES != 0:
        raise ValueError(f"{path}: {raw.size} bytes not divisible by {EV_BYTES}")
    if raw.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.copy(), z.copy(), z.copy()
    b2 = raw[2::EV_BYTES].astype(np.int64)
    x = raw[0::EV_BYTES].astype(np.int64)
    y = raw[1::EV_BYTES].astype(np.int64)
    p = (b2 >> 7) & 1
    t = ((b2 & 0x7F) << 16) | (raw[3::EV_BYTES].astype(np.int64) << 8) | raw[4::EV_BYTES].astype(np.int64)
    return x, y, p, t


def events_to_frames(x, y, p, t, L: int = L_BINS, size: int = SIZE) -> np.ndarray:
    """事件 → (L,2,size,size) float32 log1p 帧。向量化，与论文 _to_frames 逐位一致（verify 断言）。"""
    frames = np.zeros((L, 2, size, size), dtype=np.float32)
    n = t.shape[0]
    if n == 0:
        return frames
    t_min, t_max = int(t.min()), int(t.max())
    if t_max > t_min:
        t_bin = (t_max - t_min) / L
        bidx = np.floor_divide((t - t_min).astype(np.float64), t_bin).astype(np.int64)
        np.clip(bidx, 0, L - 1, out=bidx)
    else:
        bidx = np.zeros(n, dtype=np.int64)
    keep = (x < size) & (y < size)          # 论文口径：x/y≥34 丢弃
    if not keep.all():
        x, y, p, bidx = x[keep], y[keep], p[keep], bidx[keep]
    flat = (bidx * 2 + p) * (size * size) + y * size + x
    counts = np.bincount(flat, minlength=L * 2 * size * size)
    return np.log1p(counts.reshape(L, 2, size, size).astype(np.float32))


def paper_events_to_frames(x, y, p, t, L: int = L_BINS, size: int = SIZE) -> np.ndarray:
    """论文 experiments/_repo/lln_nmnist.py:43-56 的逐事件循环原样转录（仅作等价性基准）。"""
    frames = np.zeros((L, 2, size, size), dtype=np.float32)
    if len(t) == 0:
        return frames
    t_start, t_end = t[0], t[-1]
    t_bin = (t_end - t_start) / L if t_end > t_start else 1
    for i in range(len(t)):
        bin_idx = int((t[i] - t_start) // t_bin)
        bin_idx = min(bin_idx, L - 1)
        if x[i] < size and y[i] < size:
            frames[bin_idx, p[i], y[i], x[i]] += 1.0
    return np.log1p(frames)


def file_to_frames(path: str, L: int = L_BINS, size: int = SIZE) -> np.ndarray:
    x, y, p, t = parse_bin(path)
    return events_to_frames(x, y, p, t, L=L, size=size)


# ================= 样本枚举与张量物化 =================

def iter_samples(split: str, data_dir: str = DATA_NM):
    """确定性样本枚举：(path, label)，按 (label, 文件名) 排序。"""
    phase, expected = SPLITS[split]
    root = os.path.join(data_dir, phase)
    out = []
    for label in range(N_CLASSES):
        folder = os.path.join(root, str(label))
        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)
        for name in sorted(os.listdir(folder)):
            if name.endswith(".bin"):
                out.append((os.path.join(folder, name), label))
    if len(out) != expected:
        raise ValueError(f"{split}: expected {expected} samples, found {len(out)}")
    return out


def cache_paths(split: str, cache_dir: str = CACHE_DIR):
    return (os.path.join(cache_dir, f"{split}_x.f32.npy"),
            os.path.join(cache_dir, f"{split}_y.i64.npy"))


def build_split(split: str, cache_dir: str = CACHE_DIR, L: int = L_BINS,
                data_dir: str = DATA_NM, log=print) -> dict:
    """物化 (N,L,2,34,34) float32 + (N,) int64 到 cache_dir（memmap 落盘，峰值内存低）。"""
    samples = iter_samples(split, data_dir)
    n = len(samples)
    xp, yp = cache_paths(split, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    x = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float32,
                                  shape=(n, L, 2, SIZE, SIZE))
    y = np.empty(n, dtype=np.int64)
    stats = {"n_samples": n, "n_events": 0, "empty_files": 0, "x_ge_34": 0, "y_ge_34": 0,
             "t_min": None, "t_max": None, "mono_files": 0, "nonmono_files": 0,
             "inv_events": 0, "span0_files": 0}
    t0 = time.time()
    for i, (path, label) in enumerate(samples):
        ex, ey, ep, et = parse_bin(path)
        stats["n_events"] += int(et.shape[0])
        if et.shape[0] == 0:
            stats["empty_files"] += 1
        else:
            stats["x_ge_34"] += int((ex >= SIZE).sum())
            stats["y_ge_34"] += int((ey >= SIZE).sum())
            stats["t_min"] = int(et.min()) if stats["t_min"] is None else min(stats["t_min"], int(et.min()))
            stats["t_max"] = int(et.max()) if stats["t_max"] is None else max(stats["t_max"], int(et.max()))
            inv = int((np.diff(et) < 0).sum())
            stats["inv_events"] += inv
            if inv == 0:
                stats["mono_files"] += 1
            else:
                stats["nonmono_files"] += 1
            if int(et.max()) == int(et.min()):
                stats["span0_files"] += 1
        x[i] = events_to_frames(ex, ey, ep, et, L=L)
        y[i] = label
        if (i + 1) % 5000 == 0:
            x.flush()
            log(f"[build {split}] {i + 1}/{n}  ({time.time() - t0:.0f}s)")
    x.flush()
    del x
    np.save(yp, y)
    return stats


def load_split(split: str, cache_dir: str = CACHE_DIR):
    """读缓存张量：x (N,L,2,34,34) float32（memmap 只读）、y (N,) int64。"""
    xp, yp = cache_paths(split, cache_dir)
    if not (os.path.exists(xp) and os.path.exists(yp)):
        raise FileNotFoundError(f"cache missing for {split}: run `python experiments/nmnist_loader.py build --split {split}`")
    x = np.load(xp, mmap_mode="r")
    y = np.load(yp)
    return x, y


def write_manifest(entries: dict, cache_dir: str = CACHE_DIR) -> None:
    man = {"generated_by": "experiments/nmnist_loader.py build",
           "format": {"L": L_BINS, "size": SIZE, "dtype": "float32", "value": "log1p(event counts)",
                      "layout": "(N, L, 2, 34, 34), channel 0=OFF(p=0) 1=ON(p=1), spatial (y,x)"},
           "rebuild_cmd": "python experiments/nmnist_loader.py build --split all",
           "splits": entries}
    os.makedirs(cache_dir, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=1)


# ================= CLI =================

def main(argv=None):
    ap = argparse.ArgumentParser(description="N-MNIST event loader (S8)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--split", default="all", choices=["all", "train", "test"])
    v = sub.add_parser("verify")
    v.add_argument("--equiv-files", type=int, default=200,
                   help="与论文 _to_frames 逐位比对的文件数（确定性抽样）")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        entries = {}
        splits = ["train", "test"] if args.split == "all" else [args.split]
        for sp in splits:
            st = build_split(sp)
            entries[sp] = st
            print(f"[build {sp}] done: {st}")
        write_manifest(entries)
        print(f"[build] manifest -> {MANIFEST}")
        return 0

    if args.cmd == "verify":
        return run_verify(args.equiv_files)
    return 1


def run_verify(equiv_files: int = 200) -> int:
    """自检：①5 字节整除+样本数 ②锚点复现 10-26 ③向量化 vs 论文循环逐位一致
    ④字段范围 ⑤缓存与重新解析一致。任一失败 exit=1。"""
    ok = True

    # ① 样本计数
    for sp, (_, expected) in SPLITS.items():
        n = len(iter_samples(sp))
        good = n == expected
        ok &= good
        print(f"[verify] samples {sp}: {n} (expect {expected}) {'PASS' if good else 'FAIL'}")

    # ② 锚点：复现 10-26 wrap 复验数字
    ax, ay, ap_, at = parse_bin(os.path.join(DATA_NM, ANCHOR["relpath"]))
    b2 = ((ap_ << 7) | (at >> 16)).astype(np.int64)
    keys = sorted(set(b2.tolist()))
    n_hi = int(((b2 >= 129) & (b2 <= 132)).sum())
    checks = {
        "n_events_652": ax.shape[0] == ANCHOR["n_events"],
        "b2_keys": keys == ANCHOR["b2_keys"],
        "n_b2_129_132": n_hi == ANCHOR["n_b2_129_132"],
        "frac_b2_129_132": abs(n_hi / ax.shape[0] - ANCHOR["frac_b2_129_132"]) < 0.01,
        "x_range": (int(ax.min()), int(ax.max())) == ANCHOR["x_range"],
        "y_range": (int(ay.min()), int(ay.max())) == ANCHOR["y_range"],
    }
    for k, v in checks.items():
        ok &= bool(v)
        print(f"[verify] anchor {k}: {'PASS' if v else 'FAIL'}  (b2 hist={ {int(c): int((b2==c).sum()) for c in keys} })")

    # ③ 向量化 vs 论文循环逐位一致（确定性抽样：每类等距取 equiv_files//10 个）
    samples = iter_samples("train")
    step = max(1, len(samples) // max(1, equiv_files))
    picked = samples[::step][:equiv_files]
    bad = []
    for path, _ in picked:
        x, y, p, t = parse_bin(path)
        a = events_to_frames(x, y, p, t)
        b = paper_events_to_frames(x, y, p, t)
        if not np.array_equal(a, b):
            bad.append(os.path.basename(path))
    ok &= not bad
    print(f"[verify] vectorized==paper loop on {len(picked)} files: {'PASS' if not bad else 'FAIL ' + str(bad[:5])}")

    # ④ 字段范围（抽样）
    rng_bad = []
    for path, _ in picked:
        x, y, p, t = parse_bin(path)
        if x.min() < 0 or x.max() > 255 or y.min() < 0 or y.max() > 255:
            rng_bad.append(path)
        if not set(np.unique(p).tolist()) <= {0, 1}:
            rng_bad.append(path)
    ok &= not rng_bad
    print(f"[verify] field ranges (x,y uint8; p in {{0,1}}): {'PASS' if not rng_bad else 'FAIL'}")

    # ⑤ 缓存一致性（若已物化）
    try:
        xc, yc = load_split("train")
        good = xc.shape == (N_TRAIN, L_BINS, 2, SIZE, SIZE) and yc.shape == (N_TRAIN,)
        idxs = [0, len(samples) // 2, len(samples) - 1]
        for i in idxs:
            path, label = samples[i]
            if not (np.array_equal(np.asarray(xc[i]), file_to_frames(path)) and int(yc[i]) == label):
                good = False
        ok &= good
        print(f"[verify] cache train consistent (shape+3 spot reparse): {'PASS' if good else 'FAIL'}")
    except FileNotFoundError:
        print("[verify] cache train: not built (skip)")
    print(f"[verify] OVERALL {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
