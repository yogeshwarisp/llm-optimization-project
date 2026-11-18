"""Downloads the FP16 and Q4_K_M GGUF files of Qwen2.5-1.5B-Instruct (free, no API key)."""
import sys
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"  # check the licence on this page before use
WANT = {"FP16": "fp16", "Q4": "q4_k_m"}


def main():
    files = list_repo_files(REPO)
    print("Files in repo:", [f for f in files if f.endswith(".gguf")])
    for name, key in WANT.items():
        hits = [f for f in files if f.endswith(".gguf") and key in f.lower() and "-of-" not in f]
        if not hits:
            print(f"[!] No single-file '{key}' GGUF found. Open https://huggingface.co/{REPO}/tree/main "
                  "and download one manually into models/.")
            continue
        print(f"Downloading {name}: {hits[0]}")
        hf_hub_download(repo_id=REPO, filename=hits[0], local_dir="models")
    print("Done. Check the models/ folder.")


if __name__ == "__main__":
    sys.exit(main())
