#!/usr/bin/env python
"""Minimal end-to-end smoke test.

Runs the `base` experiment preset through `run_experiment.py` (simulation + analysis)
into a throwaway output folder and checks it produced the expected artifacts:
one simulation file (`output0.json`) and the 7 default analysis plots.

Usage:
    uv run python scripts/smoke_test.py

Exit code 0 = pass, 1 = fail. Downloads the small (135M) GGUF model on first run.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "_smoke"
STRAY = ["outputs", "multirun", ".hydra"]
EXPECTED_PLOTS = 7


def _cleanup() -> None:
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    for name in STRAY:
        shutil.rmtree(REPO_ROOT / name, ignore_errors=True)


def main() -> int:
    _cleanup()
    cmd = [
        sys.executable,
        "run_experiment.py",
        "experiment=base",
        f"output={OUT.relative_to(REPO_ROOT)}",
    ]
    print(f"[smoke] running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(f"[smoke] FAIL — run_experiment exited {proc.returncode}")
        return 1

    sim = OUT / "output0.json"
    plots = sorted(OUT.glob("*.png"))
    ok = True
    if not sim.exists():
        print(f"[smoke] FAIL — missing simulation output: {sim}")
        ok = False
    if len(plots) != EXPECTED_PLOTS:
        print(f"[smoke] FAIL — expected {EXPECTED_PLOTS} plots, found {len(plots)}")
        ok = False

    if ok:
        print(f"[smoke] OK — {sim.name} + {len(plots)} plots in {OUT}")
    _cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
