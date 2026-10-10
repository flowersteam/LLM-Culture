"""Shared helpers for the config-driven reproduction scripts (no argparse)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved, never shown

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from llm_culture.config import ExperimentConfig, Network, AgentConfig
from llm_culture.analysis.utils import initialize_nltk
from scripts.run_simulation import run_simulation_from_config
from scripts.run_analysis import main_analysis
from scripts.run_comparison_analysis import run_comparison_analysis

ConfigStore.instance().store(name="experiment_schema", node=ExperimentConfig)

FONT_SIZES = {"ticks": 12, "labels": 14, "title": 16}
COMPARISON_SIZES = {"ticks": 16, "labels": 18, "legend": 16, "title": 23, "matrix": 8}


def load_base(preset: str) -> ExperimentConfig:
    """Compose conf/experiment/<preset>.yaml into a typed ExperimentConfig."""
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="config", overrides=[f"experiment={preset}"])
    return OmegaConf.to_object(cfg)


def run_one(exp: ExperimentConfig) -> str:
    """Run simulation + analysis for one config; return its output folder."""
    run_simulation_from_config(exp)
    initialize_nltk()
    main_analysis(exp.output, font_sizes=FONT_SIZES, plot=False, embedding=exp.analysis.embedding)
    return exp.output


def compare(folders, labels):
    run_comparison_analysis(folders, plot=False, scale_y_axis=False, labels=labels, sizes=COMPARISON_SIZES)


def run_experiment_set(variants):
    """Run each (ExperimentConfig, label); compare them if there is more than one."""
    folders, labels = [], []
    for exp, label in variants:
        folders.append(run_one(exp))
        labels.append(label)
    if len(folders) > 1:
        compare(folders, labels)
    print("\n".join(f"{lab}: {fol}" for lab, fol in zip(labels, folders)))
    return folders, labels
