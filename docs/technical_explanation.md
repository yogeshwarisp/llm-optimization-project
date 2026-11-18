# Technical Explanation

## 1. KV cache
- **Key and Value tensors:** In each attention layer every token is turned into a query (Q), key (K) and value (V) vector. A token's Q is compared with the Ks of all earlier tokens to decide how much of each V to mix in.
- **What the KV cache stores:** the K and V vectors of every previous token, for every layer.
- **Why it saves computation:** without it, producing each new token would recompute K and V for the whole history. With it, only the new token's K and V are computed and appended.
- **Memory cost:** `2 × layers × KV heads × head dim × bytes × tokens`. Example from architecture figures (verify in config.json): a 7B Qwen2.5-style model with 28 layers, 4 KV heads and head dim 128 uses 2×28×4×128×2 bytes = 57,344 bytes ≈ 56 KiB per token at FP16.
- **Context and users:** memory grows linearly with context length and with the number of concurrent users. 20 users × 4096 tokens ≈ 81,920 tokens ≈ 4.4 GiB (calculated, not measured).

## 2. Quantisation
- **FP16:** 16-bit floating point; 2 bytes per weight. A 1.5B-parameter model ≈ 3 GB.
- **4-bit quantisation:** weights are stored in about 4 bits, in small blocks that share a scale factor. Q4_K_M mixes some higher-precision parts, so the effective size is about 4.5-5 bits per weight.
- **Memory saving:** weights shrink by roughly 3-3.5×, not exactly 4× (75%), because block scales/metadata, higher-precision tensors (such as embeddings) and runtime buffers (KV cache, compute buffers) are added.
- **Quality:** rounding weights adds small errors, so outputs may differ or become slightly worse. How much must be observed from the 10-prompt comparison; it is not guaranteed either way.

## 3. Prompt processing vs generation speed
- **Prompt processing** (prefill): the whole prompt is processed in parallel; measured as time to first token.
- **Generation** (decode): tokens come one at a time, limited mostly by memory bandwidth; measured from first to last token.
- They are reported separately because they scale differently with prompt length and hardware.

## 4. 20-user serving architecture
Clients → request queue → llama.cpp server with 20 slots (`-np 20`) → continuous batching merges the active requests' next-token steps into one GPU pass → responses streamed back. A fixed context per slot (e.g. 4096) lets the KV cache be reserved up front. Bottlenecks: VRAM for KV cache, memory bandwidth, slot count vs users (queue wait), long prompts delaying others.
**Status of evidence:** the 24 GB / 7B / 20-user scenario is a calculation unless you run it on such a GPU.
