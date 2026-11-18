"""Runs the 10 fixed prompts on both configs and saves answers to results/quality_results.csv.
Example:  python quality_test.py --max-tokens 300"""
import argparse
import csv
import gc
import json
from pathlib import Path

import model_loader as ml

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "quality_results.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["FP16", "Q4"], choices=["FP16", "Q4"])
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--gpu-layers", type=int, default=0)
    a = ap.parse_args()

    prompts = json.load(open(ROOT / "prompts.json", encoding="utf-8"))
    rows = []
    for cfg in a.configs:
        print(f"\n=== {cfg} ===")
        llm, _ = ml.load_model(cfg, a.n_ctx, a.gpu_layers)
        for p in prompts:
            r = ml.generate_timed(llm, p["prompt"], a.max_tokens, temperature=0.0, seed=42)
            print(f"Prompt {p['id']} ({p['topic']}): {r['generated_tokens']} tokens")
            rows.append({"prompt_id": p["id"], "topic": p["topic"], "prompt": p["prompt"],
                         "config": cfg, "answer": r["text"],
                         "generated_tokens": r["generated_tokens"],
                         "total_time_s": round(r["total_time_s"], 2)})
        del llm
        gc.collect()

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {OUT}. Now open the dashboard (Quality tab) and score the answers 1-5.")


if __name__ == "__main__":
    main()
