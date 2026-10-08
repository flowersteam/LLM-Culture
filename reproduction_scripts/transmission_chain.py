"""Reproduction: transmission chain. Config: conf/experiment/repro_transmission_chain.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repro import load_base, run_experiment_set


def build_variants():
    return [(load_base("repro_transmission_chain"), "transmission_chain")]


def main():
    run_experiment_set(build_variants())


if __name__ == "__main__":
    main()
