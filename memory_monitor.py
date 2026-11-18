"""RAM (psutil) and GPU VRAM (nvidia-smi) readers. GPU functions return None if no NVIDIA GPU."""
import os
import shutil
import subprocess
import psutil


def ram_process_mb():
    """RAM used by this Python process (RSS), in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def ram_system_used_mb():
    """RAM used by the whole system, in MB."""
    return psutil.virtual_memory().used / 1024 ** 2


def gpu_used_mb():
    """Total VRAM in use on GPU 0 (whole GPU, not only this process), or None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


if __name__ == "__main__":
    print("Process RAM (MB):", round(ram_process_mb(), 1))
    print("System RAM used (MB):", round(ram_system_used_mb(), 1))
    print("GPU VRAM used (MB):", gpu_used_mb())
