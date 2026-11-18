"""Benchmarks FP16 vs Q4. Saves every trial to results/benchmark_results.csv.
Example:  python benchmark.py --trials 3 --prompts 3 --max-tokens 128"""
import argparse
import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import memory_monitor as mm
import model_loader as ml

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "benchmark_results.csv"
FIELDS = ["timestamp", "config", "prompt_id", "trial", "load_time_s", "model_ram_mb", "ram_mb",
          "gpu_used_mb", "prompt_tokens", "generated_tokens", "prompt_time_s",
          "generation_time_s", "prompt_tokens_per_s", "tokens_per_s", "n_gpu_layers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["FP16", "Q4"], choices=["FP16", "Q4"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prompts", type=int, default=3, help="how many of the 10 prompts to use")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--gpu-layers", type=int, default=0, help="0 = CPU only; 99 = all layers on GPU")
    a = ap.parse_args()

    prompts = json.load(open(ROOT / "prompts.json", encoding="utf-8"))[: a.prompts]
    rows = []
    for cfg in a.configs:
        print(f"\n=== Loading {cfg} ===")
        ram_before = mm.ram_process_mb()
        llm, load_s = ml.load_model(cfg, a.n_ctx, a.gpu_layers)
        model_ram = mm.ram_process_mb() - ram_before
        print(f"Loaded in {load_s:.2f}s. Warm-up run (not recorded)...")
        ml.generate_timed(llm, "Say hello.", max_tokens=8)
        for p in prompts:
            for t in range(1, a.trials + 1):
                r = ml.generate_timed(llm, p["prompt"], a.max_tokens)
                rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"), "config": cfg,
                    "prompt_id": p["id"], "trial": t, "load_time_s": round(load_s, 3),
                    "model_ram_mb": round(model_ram, 1), "ram_mb": round(mm.ram_process_mb(), 1),
                    "gpu_used_mb": mm.gpu_used_mb(), "prompt_tokens": r["prompt_tokens"],
                    "generated_tokens": r["generated_tokens"],
                    "prompt_time_s": round(r["prompt_time_s"], 4),
                    "generation_time_s": round(r["generation_time_s"], 4),
                    "prompt_tokens_per_s": round(r["prompt_tokens_per_s"], 2),
                    "tokens_per_s": round(r["tokens_per_s"], 2), "n_gpu_layers": a.gpu_layers})
                print(f"{cfg} prompt {p['id']} trial {t}: {r['tokens_per_s']:.1f} tok/s")
        del llm
        gc.collect()

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {OUT}")
    for cfg in a.configs:
        v = [r["tokens_per_s"] for r in rows if r["config"] == cfg]
        if v:
            print(f"{cfg}: mean generation speed = {sum(v) / len(v):.2f} tokens/s")


if __name__ == "__main__":
    main()
