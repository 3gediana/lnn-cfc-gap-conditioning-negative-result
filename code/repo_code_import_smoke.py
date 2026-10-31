"""repo_code_import_smoke.py — 打包代码导入级完整性冒烟（不训练、不读数据、不写文件）。

验证 repo/code/ 下 7 个实验脚本在打包目录布局下可独立导入（路径解析 HERE=code/、PROJECT=repo/
与课题内一致），且关键协议常量/类/函数符号存在。这是「代码原样封包」的证据，不构成科学结论验证。
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(REPO, "code")
sys.path.insert(0, CODE)

MODS = {
    "pilot_threearm": ["DrawingClassifier", "make_mask", "d2_timespans", "CONDITIONS", "SEEDS",
                       "UNITS", "BATCH", "LR", "WD", "CLIP", "EPOCHS", "DROP_RATE", "N_PATTERNS",
                       "BLOCK_LENS", "MAX_LEN_QD", "load_quickdraw", "eval_arm_condition"],
    "full_protocol": ["d2_timespans_vec", "EPOCHS_FULL", "SPLIT_SEED", "FULL_TRAIN_N",
                      "FULL_EVAL_N", "EVAL_TOTAL", "main"],
    "nmnist_loader": ["load_split", "events_to_frames", "main"],
    "nmnist_threearm": ["EPOCHS_NM", "load_model_nm", "train_one_nm", "main"],
    "e2_btrain": ["main"],
    "quickdraw_lrdecay": ["main"],
    "nm_negative_probes": ["main"],
}

fails = []
for name, syms in MODS.items():
    try:
        mod = __import__(name)
    except Exception as exc:  # noqa: BLE001
        fails.append(f"{name}: import failed: {type(exc).__name__}: {exc}")
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        continue
    missing = [s for s in syms if not hasattr(mod, s)]
    if missing:
        fails.append(f"{name}: missing symbols {missing}")
        print(f"[FAIL] {name}: missing {missing}")
    else:
        print(f"[PASS] {name}: import ok, {len(syms)} symbols present")

# 协议常量钉死核查（与课题冻结规格一致）
import pilot_threearm as P  # noqa: E402
import full_protocol as F   # noqa: E402
import nmnist_threearm as N  # noqa: E402

consts = [
    ("SEEDS", P.SEEDS, [0, 1, 2]),
    ("UNITS", P.UNITS, 256),
    ("BATCH", P.BATCH, 128),
    ("LR", P.LR, 1e-3),
    ("WD", P.WD, 0.01),
    ("CLIP", P.CLIP, 1.0),
    ("DROP_RATE", P.DROP_RATE, 0.3),
    ("N_PATTERNS", P.N_PATTERNS, 3),
    ("BLOCK_LENS", P.BLOCK_LENS, {"block5-30": 5, "block15-30": 15}),
    ("EPOCHS_FULL", F.EPOCHS_FULL, 30),
    ("SPLIT_SEED", F.SPLIT_SEED, 12345),
    ("EVAL_TOTAL", F.EVAL_TOTAL, 108),
    ("EPOCHS_NM", N.EPOCHS_NM, 10),
    ("CONDITIONS", P.CONDITIONS, ["clean", "uniform30", "block5-30", "block15-30"]),
]
for name, got, want in consts:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] const {name}: {got}" + ("" if ok else f" != {want}"))
    if not ok:
        fails.append(f"const {name}: {got} != {want}")

print("-" * 70)
if fails:
    print(f"RESULT: FAIL ({len(fails)} 项)")
    sys.exit(1)
print("RESULT: PASS — 7 模块导入 + 14 项协议常量与课题冻结规格逐位一致")
sys.exit(0)
