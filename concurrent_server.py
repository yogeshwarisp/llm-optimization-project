"""20-user serving demo. Three modes, each clearly labelled in the output CSV:
  plan      : CALCULATED memory budget (no model is run)
  simulate  : SIMULATED queue using a speed you supply (no model is run)
  test      : ACTUAL load test against a running llama.cpp server (OpenAI-compatible API)
Results are appended to results/concurrency_results.csv"""
import argparse
import concurrent.futures as cf
import csv
import heapq
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "concurrency_results.csv"
FIELDS = ["timestamp", "mode", "label", "users", "slots", "ok", "failed", "wall_s", "total_tokens",
          "throughput_tok_s", "lat_mean_s", "lat_p50_s", "lat_p95_s"]


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def save(row):
    OUT.parent.mkdir(exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_plan(a):
    kv_tok = 2 * a.layers * a.kv_heads * a.head_dim * a.kv_bytes
    kv_gb = kv_tok * a.ctx * a.users / 1024 ** 3
    total = a.weights_gb + kv_gb + a.overhead_gb
    print("*** CALCULATED ESTIMATE - NOT MEASURED ***")
    print(f"KV cache per token : {kv_tok / 1024:.1f} KiB")
    print(f"KV cache total     : {kv_gb:.2f} GB ({a.users} users x {a.ctx} tokens)")
    print(f"Weights (assumed)  : {a.weights_gb:.2f} GB   Overhead (assumed): {a.overhead_gb:.2f} GB")
    print(f"Total              : {total:.2f} GB of {a.gpu_gb} GB -> "
          f"{'fits on paper' if total <= a.gpu_gb else 'DOES NOT FIT'}")
    print(f"\nServer command (llama.cpp, total context = users x ctx = {a.users * a.ctx}):")
    print(f"llama-server -m models\\MODEL.gguf -c {a.users * a.ctx} -np {a.users} -ngl 99 --port 8080")


def cmd_simulate(a):
    service = a.tokens / a.tps
    free = [0.0] * a.slots  # time when each slot becomes free; all users arrive at t=0
    heapq.heapify(free)
    lats = []
    for _ in range(a.users):
        start = heapq.heappop(free)
        end = start + service
        heapq.heappush(free, end)
        lats.append(end)
    wall = max(lats)
    print("*** SIMULATION - NOT A REAL TEST ***")
    print(f"Service time per request = {service:.1f}s; slots = {a.slots}; users = {a.users}")
    print(f"Mean latency {sum(lats) / len(lats):.1f}s, p95 {pct(lats, .95):.1f}s, wall {wall:.1f}s")
    save({"timestamp": datetime.now().isoformat(timespec="seconds"), "mode": "SIMULATED",
          "label": a.label, "users": a.users, "slots": a.slots, "ok": a.users, "failed": 0,
          "wall_s": round(wall, 2), "total_tokens": a.users * a.tokens,
          "throughput_tok_s": round(a.users * a.tokens / wall, 2),
          "lat_mean_s": round(sum(lats) / len(lats), 2), "lat_p50_s": round(pct(lats, .5), 2),
          "lat_p95_s": round(pct(lats, .95), 2)})


def one_request(url, prompt, max_tokens):
    import requests
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json={"messages": [{"role": "user", "content": prompt}],
                                     "max_tokens": max_tokens, "temperature": 0}, timeout=600)
        r.raise_for_status()
        toks = r.json().get("usage", {}).get("completion_tokens", 0)
        return True, time.perf_counter() - t0, toks
    except Exception as e:
        return False, time.perf_counter() - t0, str(e)


def cmd_test(a):
    prompts = [p["prompt"] for p in json.load(open(ROOT / "prompts.json", encoding="utf-8"))]
    url = f"http://{a.host}:{a.port}/v1/chat/completions"
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=a.users) as ex:
        futs = [ex.submit(one_request, url, prompts[i % len(prompts)], a.max_tokens)
                for i in range(a.users)]
        res = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    ok = [r for r in res if r[0]]
    lats = [r[1] for r in ok]
    toks = sum(r[2] for r in ok)
    for r in res:
        if not r[0]:
            print("Failed:", r[2])
    print("*** ACTUAL TEST - hardware/model:", a.label, "***")
    print(f"{len(ok)}/{a.users} ok, wall {wall:.1f}s, {toks} tokens, {toks / wall:.1f} tok/s total")
    if lats:
        print(f"Latency mean {sum(lats) / len(lats):.1f}s, p50 {pct(lats, .5):.1f}s, p95 {pct(lats, .95):.1f}s")
    save({"timestamp": datetime.now().isoformat(timespec="seconds"), "mode": "ACTUAL",
          "label": a.label, "users": a.users, "slots": a.slots, "ok": len(ok),
          "failed": a.users - len(ok), "wall_s": round(wall, 2), "total_tokens": toks,
          "throughput_tok_s": round(toks / wall, 2) if wall else 0,
          "lat_mean_s": round(sum(lats) / len(lats), 2) if lats else 0,
          "lat_p50_s": round(pct(lats, .5), 2), "lat_p95_s": round(pct(lats, .95), 2)})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--gpu-gb", type=float, default=24)
    p.add_argument("--weights-gb", type=float, default=4.7, help="approx. 7B Q4_K_M file size; check yours")
    p.add_argument("--overhead-gb", type=float, default=1.5, help="assumed runtime/compute buffers")
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--kv-bytes", type=int, default=2, help="2 = FP16 KV cache")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--users", type=int, default=20)
    p.set_defaults(fn=cmd_plan)
    s = sub.add_parser("simulate")
    s.add_argument("--tps", type=float, required=True, help="per-request tokens/s (take from your benchmark)")
    s.add_argument("--tokens", type=int, default=200)
    s.add_argument("--users", type=int, default=20)
    s.add_argument("--slots", type=int, default=4)
    s.add_argument("--label", default="simulation")
    s.set_defaults(fn=cmd_simulate)
    t = sub.add_parser("test")
    t.add_argument("--host", default="127.0.0.1")
    t.add_argument("--port", type=int, default=8080)
    t.add_argument("--users", type=int, default=20)
    t.add_argument("--slots", type=int, default=0, help="the -np value you started the server with")
    t.add_argument("--max-tokens", type=int, default=128)
    t.add_argument("--label", required=True, help='e.g. "Colab T4, Qwen2.5-1.5B Q4"')
    t.set_defaults(fn=cmd_test)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
