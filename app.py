"""Streamlit dashboard.  Run:  streamlit run app.py"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
RES, CH = ROOT / "results", ROOT / "charts"
st.set_page_config(page_title="LLM Optimisation: FP16 vs Q4", page_icon="🧠", layout="wide")

PRESETS = {"Qwen2.5-7B (28 layers, 4 KV heads, dim 128)": (28, 4, 128),
           "Qwen2.5-1.5B (28 layers, 2 KV heads, dim 128)": (28, 2, 128),
           "Custom": (28, 4, 128)}
CRITERIA = ["correctness", "relevance", "completeness", "clarity", "coding_correctness"]


def load_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 0 else None


def kv_gb(layers, kv_heads, head_dim, ctx, users, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * bytes_per * ctx * users / 1024 ** 3


def bar(summary, col, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    vals = summary[col].fillna(0)
    ax.bar(summary.index, vals, color=["#4f46e5", "#10b981"][: len(vals)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    CH.mkdir(exist_ok=True)
    fig.savefig(CH / fname, dpi=150)
    st.pyplot(fig)
    plt.close(fig)


st.title("🧠 Running a Small Open-Source LLM Locally: FP16 vs 4-bit")
tabs = st.tabs(["Overview", "Run", "Performance", "Quality", "KV cache", "Quantisation", "20-user plan"])

with tabs[0]:
    st.markdown("""
**Goal:** compare two configurations of the *same* open-source model (Qwen2.5-1.5B-Instruct, GGUF, run with llama.cpp):
**FP16** (16-bit weights) and **Q4_K_M** (4-bit quantised).

**Rules of this project:** ₹0 budget, free tools only, no paid APIs. No result on any tab is pre-filled:
every number comes from `results/*.csv`, which are created only when you run the scripts on your machine.
""")
    st.info("Steps: 1) `python download_models.py`  2) Run tab → benchmark  3) Run tab → quality test  4) score answers in the Quality tab.")

with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    cfgs = c1.multiselect("Configurations", ["FP16", "Q4"], default=["FP16", "Q4"])
    trials = c2.number_input("Trials", 1, 10, 3)
    nprompts = c3.number_input("Prompts (benchmark)", 1, 10, 3)
    maxtok = c4.number_input("Max new tokens", 16, 512, 128)
    gl = st.number_input("GPU layers (0 = CPU only, 99 = full GPU)", 0, 99, 0)
    st.caption("Long runs are easier to watch in the terminal; this button blocks until finished.")
    b1, b2 = st.columns(2)
    if b1.button("▶ Run benchmark") and cfgs:
        with st.spinner("Benchmarking..."):
            r = subprocess.run([sys.executable, "benchmark.py", "--configs", *cfgs, "--trials", str(trials),
                                "--prompts", str(nprompts), "--max-tokens", str(maxtok),
                                "--gpu-layers", str(gl)], cwd=ROOT, capture_output=True, text=True)
        st.code((r.stdout or "") + (r.stderr or ""))
    if b2.button("▶ Run 10-prompt quality test") and cfgs:
        with st.spinner("Generating answers..."):
            r = subprocess.run([sys.executable, "quality_test.py", "--configs", *cfgs,
                                "--gpu-layers", str(gl)], cwd=ROOT, capture_output=True, text=True)
        st.code((r.stdout or "") + (r.stderr or ""))

with tabs[2]:
    df = load_csv("benchmark_results.csv")
    if df is None:
        st.warning("No benchmark results yet. Run the benchmark first.")
    else:
        g = df.groupby("config")
        summary = g[["load_time_s", "model_ram_mb", "ram_mb", "gpu_used_mb", "prompt_tokens_per_s",
                     "tokens_per_s", "prompt_time_s", "generation_time_s", "generated_tokens"]].mean()
        st.subheader("Averages over all trials")
        st.dataframe(summary.round(2), use_container_width=True)
        a, b, c = st.columns(3)
        with a:
            bar(summary, "tokens_per_s", "Generation speed", "tokens/s", "tokens_per_second.png")
        with b:
            bar(summary, "prompt_tokens_per_s", "Prompt processing speed", "tokens/s", "prompt_speed.png")
        with c:
            bar(summary, "ram_mb", "Process RAM", "MB", "ram_usage.png")
        d, e, _ = st.columns(3)
        with d:
            bar(summary, "load_time_s", "Model load time", "seconds", "load_time.png")
        with e:
            if summary["gpu_used_mb"].notna().any():
                bar(summary, "gpu_used_mb", "GPU VRAM (whole GPU)", "MB", "vram.png")
            else:
                st.info("No NVIDIA GPU data recorded.")
        st.caption("Prompt speed = prompt tokens / time-to-first-token. Generation speed = (tokens-1) / "
                   "(last token time - first token time). Token counts are approximate.")
        with st.expander("Raw trials"):
            st.dataframe(df, use_container_width=True)

with tabs[3]:
    q = load_csv("quality_results.csv")
    if q is None:
        st.warning("No quality results yet. Run the 10-prompt quality test first.")
    else:
        topics = q.drop_duplicates("prompt_id").set_index("prompt_id")["topic"]
        pid = st.selectbox("Prompt", topics.index, format_func=lambda i: f"{i}. {topics[i]}")
        sub = q[q.prompt_id == pid]
        st.markdown(f"**Prompt:** {sub.iloc[0]['prompt']}")
        cols = st.columns(len(sub))
        scores = []
        for col, (_, row) in zip(cols, sub.iterrows()):
            with col:
                st.subheader(row["config"])
                st.caption(f"{row['generated_tokens']} tokens, {row['total_time_s']}s")
                st.write(row["answer"])
                s = {"prompt_id": pid, "config": row["config"]}
                for cname in CRITERIA:
                    s[cname] = st.slider(f"{cname.replace('_', ' ')} (0 = n/a)", 0, 5, 0,
                                         key=f"{pid}-{row['config']}-{cname}")
                scores.append(s)
        if st.button("💾 Save my scores for this prompt"):
            new = pd.DataFrame(scores)
            old = load_csv("quality_scores.csv")
            if old is not None:
                old = old[old.prompt_id != pid]
                new = pd.concat([old, new])
            new.to_csv(RES / "quality_scores.csv", index=False)
            st.success("Saved to results/quality_scores.csv")
        sc = load_csv("quality_scores.csv")
        if sc is not None:
            st.subheader("Your scores so far (mean, 0 = not scored)")
            st.dataframe(sc.replace(0, pd.NA).groupby("config")[CRITERIA].mean().round(2))
            st.caption("Scores are your own manual judgement; with 10 prompts they are observations, not proof.")

with tabs[4]:
    st.markdown("""
**Key and Value tensors:** for every token, each attention layer computes a *Key* (what the token offers for matching)
and a *Value* (the information it contributes). **KV cache** stores these for all earlier tokens so each new token only
computes its own K and V instead of recomputing the whole history. It saves compute but **costs memory**, growing with
context length and with every concurrent user.
""")
    p = st.selectbox("Model preset (verify numbers in the model's config.json)", list(PRESETS))
    L, H, D = PRESETS[p]
    c1, c2, c3 = st.columns(3)
    L = c1.number_input("Layers", 1, 200, L)
    H = c2.number_input("KV heads", 1, 64, H)
    D = c3.number_input("Head dim", 16, 512, D)
    ctx = st.slider("Context per user (tokens)", 256, 32768, 4096, 256)
    users = st.slider("Concurrent users", 1, 64, 20)
    bytes_per = st.selectbox("KV cache precision", [("FP16", 2), ("8-bit", 1)], format_func=lambda x: x[0])[1]
    per_tok = 2 * L * H * D * bytes_per
    m1, m2 = st.columns(2)
    m1.metric("KV cache per token", f"{per_tok / 1024:.1f} KiB")
    m2.metric(f"Total for {users} users", f"{kv_gb(L, H, D, ctx, users, bytes_per):.2f} GiB")
    st.caption("Formula: 2 (K and V) × layers × KV heads × head dim × bytes × tokens × users. Calculated, not measured.")

with tabs[5]:
    st.markdown("""
**FP16:** each weight is a 16-bit floating-point number (2 bytes). **4-bit quantisation** stores weights in about
4 bits using small groups with a shared scale, so the model is much smaller. Rounding loses some precision, so answers
*may* change, but whether quality visibly drops must be checked on your own results.
Savings are **not exactly 75%**: scales/metadata take space, some tensors stay at higher precision, and the runtime
needs extra buffers (KV cache, compute).
""")
    params = st.number_input("Parameters (billions)", 0.1, 100.0, 1.5)
    q4bits = st.number_input("Effective bits per weight for Q4_K_M (approx. 4.5-5)", 4.0, 6.0, 4.85)
    fp16_gb = params * 1e9 * 2 / 1024 ** 3
    q4_gb = params * 1e9 * q4bits / 8 / 1024 ** 3
    a, b, c = st.columns(3)
    a.metric("FP16 weights", f"{fp16_gb:.2f} GiB")
    b.metric("Q4 weights (est.)", f"{q4_gb:.2f} GiB")
    c.metric("Saving", f"{(1 - q4_gb / fp16_gb) * 100:.0f}%", help="Less than 75% because of metadata")
    st.caption("Estimates. Compare with the real file sizes in models/.")

with tabs[6]:
    st.warning("Calculated plan and simulations are NOT test results. Only rows marked ACTUAL come from a real server.")
    c1, c2, c3, c4 = st.columns(4)
    gpu = c1.number_input("GPU memory (GB)", 4.0, 80.0, 24.0)
    wts = c2.number_input("7B Q4 weights (GB, approx.)", 1.0, 40.0, 4.7)
    cx = c3.number_input("Context per user", 512, 16384, 4096, 512)
    us = c4.number_input("Users", 1, 100, 20)
    kv = kv_gb(28, 4, 128, cx, us)
    over = 1.5
    tot = wts + kv + over
    st.table(pd.DataFrame({"Item": ["Weights", "KV cache", "Overhead (assumed)", "Total", "GPU"],
                           "GiB": [wts, round(kv, 2), over, round(tot, 2), gpu]}))
    st.write("Result on paper: " + ("**fits**" if tot <= gpu else "**does not fit**"))
    st.code(f"llama-server -m models\\your-7b-q4.gguf -c {us * cx} -np {us} -ngl 99 --port 8080\n"
            "python concurrent_server.py test --users 20 --label \"your GPU + model\"", language="bash")
    st.markdown("**Bottlenecks:** VRAM (KV cache), memory bandwidth during decoding, queue wait when slots < users, "
                "long prompts blocking short ones.")
    cr = load_csv("concurrency_results.csv")
    if cr is None:
        st.info("No concurrency runs recorded yet. Status: NOT TESTED.")
    else:
        st.dataframe(cr, use_container_width=True)
