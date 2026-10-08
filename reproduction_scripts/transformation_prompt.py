"""Reproduction: transformation-prompt comparison. Config: conf/experiment/repro_transformation_prompt.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repro import load_base, run_experiment_set

# Registered prompt_update names (data/parameters/prompt_update.json).
UPDATE_VARIANTS = ("CombineTwo", "MinorChanges", "Repeat", "MaximizeDifference")


def build_variants():
    variants = []
    for name in UPDATE_VARIANTS:
        exp = load_base("repro_transformation_prompt")
        exp.population.agents[0].prompt_update = name
        exp.output = f"results/experiments/transformation_{name}"
        variants.append((exp, name))
    return variants


def main():
    run_experiment_set(build_variants())


if __name__ == "__main__":
    main()
