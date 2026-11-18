"""Loads the FP16 or Q4 GGUF model with llama.cpp and runs one timed generation."""
import time
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
# Keyword that must appear in the .gguf file name for each configuration
KEYWORDS = {"FP16": "fp16", "Q4": "q4_k_m"}


def find_model(config):
    key = KEYWORDS[config]
    matches = [p for p in MODELS_DIR.glob("*.gguf") if key in p.name.lower()]
    if not matches:
        raise FileNotFoundError(
            f"No '{key}' .gguf file in {MODELS_DIR}. Run:  python download_models.py")
    return matches[0]


def load_model(config, n_ctx=2048, n_gpu_layers=0, n_threads=None):
    """Returns (llm, load_time_seconds)."""
    from llama_cpp import Llama  # imported here so other scripts work without it
    path = find_model(config)
    t0 = time.perf_counter()
    llm = Llama(model_path=str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                n_threads=n_threads, verbose=False)
    return llm, time.perf_counter() - t0


def generate_timed(llm, prompt, max_tokens=128, temperature=0.0, seed=42):
    """Streams one answer and separates the two phases:
    - prompt processing time  = start -> first generated token (time to first token)
    - generation time         = first token -> last token
    Token count = streamed pieces (about 1 token each; approximate)."""
    llm.reset()
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))  # excludes chat-template tokens
    t0 = time.perf_counter()
    t_first = None
    pieces = []
    for chunk in llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature, seed=seed, stream=True):
        piece = chunk["choices"][0]["delta"].get("content")
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
            pieces.append(piece)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    n = len(pieces)
    prompt_time = t_first - t0
    gen_time = t_end - t_first
    return {
        "text": "".join(pieces),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": n,
        "prompt_time_s": prompt_time,
        "generation_time_s": gen_time,
        "total_time_s": t_end - t0,
        "prompt_tokens_per_s": prompt_tokens / prompt_time if prompt_time > 0 else 0.0,
        "tokens_per_s": (n - 1) / gen_time if n > 1 and gen_time > 0 else 0.0,
    }
