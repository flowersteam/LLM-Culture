"""Reproduction: network-structure comparison. Config: conf/experiment/repro_network.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repro import load_base, run_experiment_set, Network

STRUCTURES = ("circle", "caveman", "fully_connected")


def build_variants():
    variants = []
    for structure in STRUCTURES:
        exp = load_base("repro_network")
        exp.population.network_structure = Network(structure)
        exp.output = f"results/experiments/network_{structure}"
        variants.append((exp, structure))
    return variants


def main():
    run_experiment_set(build_variants())


if __name__ == "__main__":
    main()
