# -*- coding: utf-8 -*-
"""批量补跑 seq-tinyimg 缺失实验。4路并行，带进度日志与崩溃容错。"""
import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MAIN = os.path.join(ROOT, "utils", "main.py")
LOG = os.path.join(ROOT, "results", "_batch_tinyimg.log")
WORKERS = 1

CONFIGS = []
for buf in [2000, 5000]:
    for model in ["er", "er_gbr", "derpp", "derpp_gbr",
                  "mir", "mir_gbr", "gss", "gss_gbr"]:
        CONFIGS.append((model, "seq-tinyimg", buf))

SEEDS = [5000, 5001, 5002, 5003, 5004]

def already_done(model, dataset, buf, seed):
    p = os.path.join(ROOT, "results", dataset, model, f"epoch1", f"buf{buf}", f"formal_s{seed}", "results.pkl")
    return os.path.exists(p)

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def run_one(task):
    model, dataset, buf, seed = task
    exp = f"formal_s{seed}"
    cmd = [PY, MAIN,
           "--model", model,
           "--backbone", "resnet18",
           "--dataset", dataset,
           "--lr", "0.03",
           "--batch_size", "32",
           "--minibatch_size", "32",
           "--n_epochs", "1",
           "--buffer_size", str(buf),
           "--seed", str(seed),
           "--gpu_id", "0",
           "--exp", exp]
    if model.startswith("gss"):
        cmd += ["--batch_num", "1"]
    start = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
        dt = time.time() - start
        ok = r.returncode == 0 and os.path.exists(os.path.join(ROOT, "results", dataset, model, "epoch1", f"buf{buf}", exp, "results.pkl"))
        msg = f"[{'OK ' if ok else 'FAIL'}] {model} {dataset} buf{buf} seed{seed} {dt/60:.1f}min"
        log(msg)
        if not ok:
            log("  stderr tail: " + r.stderr[-1500:].replace("\n", " | "))
        return ok
    except Exception as e:
        log(f"[ERROR] {model} {dataset} buf{buf} seed{seed}: {e}")
        return False

def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log(f"===== TINYIMG BATCH START: {len(CONFIGS)} configs x {len(SEEDS)} seeds, {WORKERS} workers =====")
    total = len(CONFIGS) * len(SEEDS)
    tasks = [(m, d, b, s) for (m, d, b) in CONFIGS for s in SEEDS]
    todo = [t for t in tasks if not already_done(*t)]
    log(f"total={total}, already_done={total - len(todo)}")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for ok in ex.map(run_one, todo):
            done += 1
            if not ok:
                failed += 1
            if done % WORKERS == 0 or not ok:
                log(f"  -> progress {done}/{len(todo)} (failed={failed})")
    log(f"===== TINYIMG BATCH END: done={done}, failed={failed}, skipped={total - len(todo)}, total={total} =====")

if __name__ == "__main__":
    main()
