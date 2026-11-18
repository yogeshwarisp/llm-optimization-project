# Running a Small Open-Source LLM Locally and Optimising it with Quantisation

**Name:** Yogeshwari Sandip Pawar  **Date:** [ ]  **Institution/Course:** [ ]

> Sections marked [RESULT] must be filled only from your own experiments.

## 1. Introduction
[Why local LLMs and optimisation matter: cost, privacy, hardware limits.]

## 2. Objectives
Compare FP16 and 4-bit (Q4_K_M) versions of the same model on speed, memory and answer quality; explain KV cache and quantisation; plan serving 20 concurrent users on one 24 GB GPU.

## 3. Tools and Technologies
Python, VS Code, llama.cpp / llama-cpp-python, GGUF, Hugging Face, Streamlit, Pandas, Matplotlib, psutil, nvidia-smi. All free and open-source.

## 4. System Requirements and Test Machine
| Item | Value |
|---|---|
| CPU | [fill] |
| RAM | [fill] |
| GPU | [fill / none] |
| OS / Python | [fill] |

## 5. Model Details
Qwen2.5-1.5B-Instruct (GGUF). Licence: [check and cite from the model card]. FP16 file: [name, size]. Q4_K_M file: [name, size].

## 6. Experimental Methodology
Same model, runtime, prompts and settings (temperature 0, seed 42). Warm-up run discarded. [N] trials × [N] prompts. Prompt processing = time to first token; generation speed = tokens after the first / generation time.

## 7. FP16 vs Q4 Results [RESULT]
| Metric | FP16 | Q4_K_M |
|---|---|---|
| Load time (s) | [ ] | [ ] |
| Process RAM (MB) | [ ] | [ ] |
| VRAM (MB) | [ ] | [ ] |
| Prompt speed (tok/s) | [ ] | [ ] |
| Generation speed (tok/s) | [ ] | [ ] |

## 8. 10-Prompt Quality Comparison [RESULT]
| # | Topic | FP16 score (0-5) | Q4 score (0-5) | Observation |
|---|---|---|---|---|
| 1 | Artificial Intelligence | [ ] | [ ] | [ ] |
| 2 | Machine Learning | [ ] | [ ] | [ ] |
| 3 | Python programming | [ ] | [ ] | [ ] |
| 4 | Cloud computing | [ ] | [ ] | [ ] |
| 5 | Database concepts | [ ] | [ ] | [ ] |
| 6 | Email writing | [ ] | [ ] | [ ] |
| 7 | RAM vs ROM | [ ] | [ ] | [ ] |
| 8 | Neural networks | [ ] | [ ] | [ ] |
| 9 | Python coding | [ ] | [ ] | [ ] |
| 10 | Cybersecurity | [ ] | [ ] | [ ] |

## 9. KV Cache and 10. Quantisation
See `docs/technical_explanation.md` (rewrite in your own words).

## 11. 20-User Serving Architecture
Design from `docs/technical_explanation.md`. Evidence type: [calculated / simulated / actually tested on ___]. Table of memory budget: [from `concurrent_server.py plan`].

## 12. Limitations
Small prompt set, manual scoring, approximate token counts, single machine, 1.5B model not equal to 7B behaviour, 20-user result not verified unless actually tested.

## 13. Conclusion
[Write only what your data shows.]

## References
Model card, llama.cpp repository, Hugging Face, [add yours].
