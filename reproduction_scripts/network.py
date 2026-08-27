from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
os.chdir(root)

sys.path.insert(0, str(root))


from llm_culture.simulation.utils import run_experiment
from scripts.run_analysis import main_analysis
from scripts.run_comparison_analysis import run_comparison_analysis

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

def make_base(args):
    cfg = {
        "n_agents": 10, "n_timesteps": 5, "n_seeds": 1,
        "network_structure": "fully_connected", "n_cliques": 2,
        "prompt_init": {"name": "simple", "prompt": "Tell me a story."},
        "prompt_update": {"name": "Combine2", "prompt": "You will receive stories. Pick the two stories you prefer, and create a story that is combination of these two stories. Just output your story, don’t write anything else."},
        "personalities": [{"name": "empty", "prompt": ""}] * 10,
        "output_name": "network", "debug": False,
        "llm_backend": "llama.cpp", "model": "unsloth/SmolLM2-135M-Instruct-GGUF", "access_url": None,
        # "llm_backend": "llama.cpp", "model": "TheBloke/Mistral-7B-OpenOrca-GGUF", "access_url": None,
        }
    return apply_cli(cfg, args)

def main():
    p = parse_common_args("Run and compare network-topology variants exposed by the notebook.")
    p.add_argument("--networks", nargs="+", default=["circle","caveman","fully_connected"],
                   choices=["circle","caveman","fully_connected"])
    args = p.parse_args()
    folders, labels = [], []
    for structure in args.networks:
        cfg = make_base(args); cfg["network_structure"] = structure; cfg["output_name"] = f"network_{structure}"
        folder = run_experiment(cfg); main_analysis(folder, font_sizes=FONT_SIZES, plot=True)
        folders.append(folder); labels.append(structure)
    compare(folders, labels)
    print("\n".join(f"{x}: {y}" for x,y in zip(labels,folders)))

if __name__ == "__main__":
    main()

