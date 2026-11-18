# Running a Small Open-Source LLM Locally & Optimisation (FP16 vs Q4)

Free (₹0), runs from VS Code on Windows 10/11. Model: Qwen2.5-1.5B-Instruct (GGUF) via llama.cpp.
**No results are pre-filled.** All numbers come from scripts you run.

## Files
| File | Purpose |
|---|---|
| `download_models.py` | Downloads FP16 + Q4_K_M GGUF files into `models/` |
| `model_loader.py` | Loads a model, runs one timed generation |
| `memory_monitor.py` | RAM (psutil) and VRAM (nvidia-smi) |
| `benchmark.py` | Multi-trial speed/memory benchmark → `results/benchmark_results.csv` |
| `quality_test.py` | 10 fixed prompts × 2 configs → `results/quality_results.csv` |
| `app.py` | Streamlit dashboard |
| `concurrent_server.py` | 20-user plan / simulation / real load test |
| `docs/` | Technical explanation, report template, screenshot checklist |

## Setup (VS Code terminal, PowerShell)
Open the folder: **File → Open Folder → llm-optimization-project**, then **Terminal → New Terminal**.

```powershell
python --version                      # must be 3.10 or newer
python -m venv venv                   # 1. create virtual environment
venv\Scripts\Activate.ps1             # 2. activate (you will see (venv) in the prompt)
python -m pip install --upgrade pip
pip install -r requirements.txt       # 3. install dependencies
pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python download_models.py             # 4. download both models (~4 GB total)
python benchmark.py --trials 3 --prompts 3      # 5. benchmark
python quality_test.py                           # 6. 10-prompt quality test
streamlit run app.py                             # 7. dashboard (opens in browser)
python concurrent_server.py plan                 # 8. 20-user memory plan (calculation)
```
Start small: `python benchmark.py --trials 1 --prompts 1 --max-tokens 32` to check everything works.

## 20-user test
1. Download a llama.cpp Windows release (free) from github.com/ggml-org/llama.cpp/releases.
2. Start the server (numbers for 20 users × 4096 tokens): `llama-server -m models\MODEL.gguf -c 81920 -np 20 -ngl 99 --port 8080`
3. In a second terminal: `python concurrent_server.py test --users 20 --label "my GPU, my model"`
4. No big GPU? `python concurrent_server.py simulate --tps <your measured tokens/s> --slots 4`. This is a SIMULATION and is labelled so.
vLLM is Linux-oriented (use WSL2 or Colab) and works best with AWQ/GPTQ weights rather than GGUF, so llama.cpp server is the default here.

## Optional: free Google Colab GPU
New notebook → Runtime → Change runtime type → T4 GPU. Upload the project folder, then:
```
!pip install -r requirements.txt
!pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
!python download_models.py
!python benchmark.py --gpu-layers 99
```
A free T4 has 16 GB, so it cannot honestly claim a 24 GB test. Label such results accordingly.

## Common errors
| Error | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again |
| `pip` tries to compile llama-cpp-python / needs a compiler | Use the `--prefer-binary --extra-index-url` command above; or install Visual Studio Build Tools (C++); or use Python 3.11/3.12 |
| `FileNotFoundError: No 'fp16' .gguf` | Run `python download_models.py`; check the file names in `models/` |
| Out of memory / very slow FP16 | Close other apps; run `--configs Q4`; use the 0.5B model; or use Colab |
| `streamlit` not recognised | Activate the venv first |
| Download fails | Check internet; or download manually from the Hugging Face model page into `models/` |

## Limitations
Small sample (10 prompts), temperature 0, token counts approximate, RAM of memory-mapped files can under-report, VRAM is whole-GPU, scores are manual.
Check the model licence on its Hugging Face page and cite it in your report.
