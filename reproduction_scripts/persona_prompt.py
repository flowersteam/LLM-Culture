"""Reproduction: persona comparison. Config: conf/experiment/repro_persona_prompt.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _repro import load_base, run_experiment_set, AgentConfig


def _single(personality: str, label: str):
    exp = load_base("repro_persona_prompt")
    exp.population.agents[0].personality = personality
    exp.output = f"results/experiments/persona_{label}"
    return exp, label


def _mixed():
    exp = load_base("repro_persona_prompt")
    half = exp.population.n_agents // 2
    exp.population.agents = [
        AgentConfig(count=half, personality="Creative", prompt_init="simple", prompt_update="CombineTwo"),
        AgentConfig(count=half, personality="NotCreative", prompt_init="simple", prompt_update="CombineTwo"),
    ]
    exp.output = "results/experiments/persona_mixed"
    return exp, "mixed"


def build_variants():
    return [
        _single("Creative", "creative"),
        _single("NotCreative", "not_creative"),
        _mixed(),
    ]


def main():
    run_experiment_set(build_variants())


if __name__ == "__main__":
    main()
