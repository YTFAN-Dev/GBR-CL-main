# -*- coding: utf-8 -*-
"""汇总消融实验结果（run_ablation.bat 产出）
输出:
  Table VI 组件消融: DERPP/MIR/GSS × {base(已有), no_gb_weight, no_gaussian_sampling, full(已有)}
  Table IV: center_weight / edge_weight 敏感性 (ER-GBR)
  Table V : gaussian_sigma / buffer_k 敏感性 (ER-GBR)
读取 results/seq-cifar100/<model>/epoch1/buf2000/abl_<tag>_s<seed>/results.pkl
"""
import os, glob, pickle
import numpy as np

CUR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "seq-cifar100")

def load(pkl):
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    return (float(np.mean(d['all_accuracy_cls'][-1])),
            float(np.mean(d['all_forgetting_cls'][-1])))

def collect(model, exp_prefix):
    """返回 {seed: (acc, fgt)}"""
    out = {}
    for p in glob.glob(os.path.join(CUR, model, 'epoch1', 'buf2000', exp_prefix + '_s*', 'results.pkl')):
        seed = int(os.path.basename(os.path.dirname(p)).split('_s')[-1])
        out[seed] = load(p)
    return out

def fmt5(res):
    seeds = sorted(res)
    accs = [res[s][0] for s in seeds]
    fgts = [res[s][1] for s in seeds]
    ma = f"{np.mean(accs):.2f}" if accs else '-'
    mf = f"{np.mean(fgts):.2f}" if fgts else '-'
    return (f"n={len(seeds)}  acc: " + " ".join(f"{a:.2f}" for a in accs) + f"  mean={ma}"
            f"  | fgt: " + " ".join(f"{f:.2f}" for f in fgts) + f"  mean={mf}")

print("=" * 100)
print("Table VI 组件消融 (seq-cifar100, buf2000)")
print("=" * 100)
for model, name in [("derpp_gbr", "DERPP-GBR"), ("mir_gbr", "MIR-GBR"), ("gss_gbr", "GSS-GBR")]:
    print(f"\n--- {name} ---")
    for tag, label in [("abl_noweight", "no_gb_weight (仅高斯采样)"), ("abl_nogauss", "no_gaussian_sampling (仅加权)")]:
        print(f"  {label:28s}", fmt5(collect(model, tag)))

print()
print("=" * 100)
print("Table IV 权重敏感性 (ER-GBR, seq-cifar100, buf2000)")
print("=" * 100)
for tag, vals, pname in [("cw", [1.2, 1.5, 1.81, 2.2], "center_weight"),
                         ("ew", [0.1, 0.33, 0.6, 0.9], "edge_weight")]:
    print(f"\n--- {pname} ---")
    for v in vals:
        if v in (1.81, 0.33):
            print(f"  {pname}={v:<5} (默认, 见主表 ER-GBR: acc mean=23.84)")
            continue
        print(f"  {pname}={v:<5}", fmt5(collect("er_gbr", f"abl_{tag}{v}")))

print()
print("=" * 100)
print("Table V 高斯采样敏感性 (ER-GBR, seq-cifar100, buf2000)")
print("=" * 100)
for tag, vals, pname in [("sigma", [0.3, 0.6, 0.91, 1.2], "gaussian_sigma"),
                         ("k", [1, 3, 5, 7], "buffer_k")]:
    print(f"\n--- {pname} ---")
    for v in vals:
        if v in (0.91, 3):
            print(f"  {pname}={v:<5} (默认, 见主表 ER-GBR: acc mean=23.84)")
            continue
        print(f"  {pname}={v:<5}", fmt5(collect("er_gbr", f"abl_{tag}{v}")))
