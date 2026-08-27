from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from llm_culture.simulation.utils import run_experiment
from scripts.run_analysis import main_analysis
from scripts.run_comparison_analysis import run_comparison_analysis
#go up one directory to find the scripts folder
import sys
sys.path.append(str(Path(__file__).parent.parent))

FONT_SIZES = {"ticks": 12, "labels": 14, "title": 16}
COMPARISON_SIZES = {
    "ticks": 16, "labels": 18, "legend": 16, "title": 23, "matrix": 8,
}


def run_one(cfg: dict):
    folder = run_experiment(cfg)
    main_analysis(folder, font_sizes=FONT_SIZES, plot=True)
    return folder


def compare(folders, labels):
    return run_comparison_analysis(
        folders,
        plot=True,
        scale_y_axis=False,
        labels=labels,
        sizes=COMPARISON_SIZES,
    )


def parse_common_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--n-agents", type=int, default=None)
    p.add_argument("--n-timesteps", type=int, default=None)
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--llm-backend", choices=["vllm", "llama.cpp"], default=None)
    p.add_argument("--access-url", default=None)
    p.add_argument("--debug", action="store_true")
    return p


def apply_cli(cfg, args):
    if args.n_agents is not None:
        cfg["n_agents"] = args.n_agents
    if args.n_timesteps is not None:
        cfg["n_timesteps"] = args.n_timesteps
    if args.n_seeds is not None:
        cfg["n_seeds"] = args.n_seeds
    if args.model is not None:
        cfg["model"] = args.model
    if args.llm_backend is not None:
        cfg["llm_backend"] = args.llm_backend
    if args.access_url is not None:
        cfg["access_url"] = args.access_url
    if args.debug:
        cfg["debug"] = True
    return cfg

def make_config(args):
    cfg = {
        "n_agents": 50, "n_timesteps": 50, "n_seeds": 1,
        "network_structure": "sequence", "n_cliques": 2,
        "prompt_init": {"name": "kid_init", "prompt": "Imagine that you are telling a story to your kid. What would that story be? Just output the story, nothing else."},
        "prompt_update": {"name": "kid_transform", "prompt": "Here is one or more stories you were told as a kid. It is now your turn to tell a story at your kid. Tell that story. Write only one story. Do not output anything else"},
        "personalities": [{"name": "empty", "prompt": ""}] * 50,
        "output_name": "transmission_chain", "debug": False,
        "llm_backend": "llama.cpp", "model": "unsloth/SmolLM2-135M-Instruct-GGUF", "access_url": None,
        # "llm_backend": "llama.cpp", "model": "TheBloke/Mistral-7B-OpenOrca-GGUF", "access_url": None,

    }
    return apply_cli(cfg, args)

def main():
    args = parse_common_args("Run and analyze a transmission-chain experiment.").parse_args()
    folder = run_one(make_config(args))
    print(f"Results: {folder}")

if __name__ == "__main__":
    main()

