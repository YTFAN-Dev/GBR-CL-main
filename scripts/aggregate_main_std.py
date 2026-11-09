# -*- coding: utf-8 -*-
"""汇总主表（seq-cifar10 / seq-cifar100 / seq-tinyimg）的 mean ± std。
逐 seed 读取 results/<dset>/<model>/epoch1/buf<M>/<seed_dir>/results.pkl，
ACC = mean(all_accuracy_cls[-1])，FGT = mean(all_forgetting_cls[-1])。
种子限定 5000-5009，排除 abl_* 与 '0' 目录。
"""
import os, re, glob, pickle
import numpy as np

CUR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DSETS = [("seq-cifar10", ["200", "500"]),
         ("seq-cifar100", ["2000", "5000"]),
         ("seq-tinyimg", ["2000", "5000"])]

MODELS = ["er", "er_gbr",
          "derpp", "derpp_gbr",
          "mir", "mir_gbr",
          "gss", "gss_gbr"]


def seed_of(name):
    cands = [int(x) for x in re.findall(r"\d{4}", name) if 5000 <= int(x) <= 5009]
    return cands[-1] if cands else None


def load(pkl):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return float(np.mean(d["all_accuracy_cls"][-1])), float(np.mean(d["all_forgetting_cls"][-1]))


def collect(dset, model, buf):
    base = os.path.join(CUR, dset, model, "epoch1", "buf" + buf)
    out = {}
    for d in glob.glob(os.path.join(base, "*")):
        if not os.path.isdir(d):
            continue
        nm = os.path.basename(d)
        if nm.startswith("abl_") or nm == "0":
            continue
        s = seed_of(nm)
        if s is None:
            continue
        p = os.path.join(d, "results.pkl")
        if not os.path.exists(p):
            continue
        try:
            out[s] = load(p)
        except Exception as e:
            print(f"  [warn] {d}: {e}")
    return out


def fmt(vals):
    if not vals:
        return "-"
    a = np.array(vals, dtype=float)
    if len(a) == 1:
        return f"{a.mean():.2f}"
    return f"{a.mean():.2f} $\\pm$ {a.std(ddof=1):.2f}"


for dset, bufs in DSETS:
    print("=" * 110)
    print(f"{dset}")
    print("=" * 110)
    for buf in bufs:
        print(f"\n--- buffer M={buf} ---")
        print(f"{'Method':<20} {'ACC mean±std (n)':<30} {'FGT mean±std (n)':<30}")
        for m in MODELS:
            r = collect(dset, m, buf)
            seeds = sorted(r)
            accs = [r[s][0] for s in seeds]
            fgts = [r[s][1] for s in seeds]
            print(f"{m:<20} {fmt(accs):<30} {fmt(fgts):<30} n={len(seeds)} seeds={seeds}")
    print()