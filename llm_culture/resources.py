"""Lightweight runtime resource reporting (CPU / memory / GPU)."""
import os


def log_resources(label=""):
    """Print a one-shot snapshot of CPU / memory (and GPU if available) usage.

    Dependency-free by default (Python stdlib). It opportunistically uses two
    optional upgrades if present, but never requires them:
      * `psutil`      -> live process RSS + system RAM used/available/percent
      * `nvidia-smi`  -> per-GPU VRAM used/total + utilization (NVIDIA/Linux/Colab)

    On Apple Silicon (macOS, Unified Memory Architecture) there is no separate
    VRAM: the Metal GPU shares the one system RAM pool, so the reported process
    RSS / system-RAM figures already account for the model's "GPU" memory. That
    is why there is no nvidia-smi-style GPU line on a Mac — it would be redundant.

    :param label: short tag describing when this snapshot was taken
        (e.g. "after model load", "after simulation").
    """
    import platform
    import shutil
    import subprocess
    import sys

    tag = f" [{label}]" if label else ""
    lines = []

    # ---- CPU ----
    n_cpu = os.cpu_count()
    cpu_line = f"CPU: {n_cpu} logical core(s)"
    if hasattr(os, "getloadavg"):
        try:
            load1, load5, load15 = os.getloadavg()
            cpu_line += f" | load avg (1/5/15m): {load1:.2f} / {load5:.2f} / {load15:.2f}"
        except OSError:
            pass
    lines.append(cpu_line)

    # ---- Memory ----
    proc_rss_gb = None
    sys_total_gb = sys_avail_gb = sys_pct = None
    try:
        import psutil  # optional upgrade

        proc_rss_gb = psutil.Process().memory_info().rss / 1e9
        vm = psutil.virtual_memory()
        sys_total_gb, sys_avail_gb, sys_pct = vm.total / 1e9, vm.available / 1e9, vm.percent
    except Exception:
        # stdlib fallback: peak RSS (resource) + total RAM (sysconf)
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss units differ: bytes on macOS, kilobytes on Linux.
            proc_rss_gb = (ru / 1e9) if sys.platform == "darwin" else (ru * 1024 / 1e9)
        except Exception:
            pass
        try:
            sys_total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except (ValueError, OSError, AttributeError):
            pass

    if proc_rss_gb is not None:
        mem_line = f"Process RAM: {proc_rss_gb:.2f} GB"
        if sys_total_gb is not None:
            mem_line += f" of {sys_total_gb:.1f} GB total"
        if sys_avail_gb is not None:
            mem_line += f" ({sys_avail_gb:.1f} GB free, {sys_pct:.0f}% used system-wide)"
        lines.append(mem_line)
    elif sys_total_gb is not None:
        lines.append(f"System RAM: {sys_total_gb:.1f} GB total")

    # ---- GPU (NVIDIA only, via nvidia-smi) ----
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                for row in out.stdout.strip().splitlines():
                    idx, used, total, util = (c.strip() for c in row.split(","))
                    lines.append(f"GPU {idx}: {used}/{total} MiB VRAM used, {util}% util")
        except Exception:
            pass
    elif platform.system() == "Darwin":
        lines.append("GPU: Apple Metal (unified memory — shares the system RAM above)")

    print(f"\n·· resources{tag} " + "·" * 40)
    for line in lines:
        print(f"   {line}")
    print("·" * 54)
