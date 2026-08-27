from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

#go up one directory to find the scripts folder
import os
print("Current directory:", os.getcwd())

os.chdir(str(Path(__file__).parent.parent))
print("Changed directory to:", os.getcwd())

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

CREATIVE={"name":"Creative","prompt":"For what follows, pretend that you are a very creative person."}
NOT_CREATIVE={"name":"NotCreative","prompt":"For what follows, pretend that you are not a very creative person."}
EMPTY={"name":"empty","prompt":""}

def make_base(args):
    cfg = {
            "n_agents": 10, "n_timesteps": 10, "n_seeds": 5,
            "network_structure": "fully_connected", "n_cliques": 2,
            "prompt_init": {"name": "simple", "prompt": "Tell me a story."},
            "prompt_update": {"name": "Combine2", "prompt": "You will receive stories. Pick the two stories you prefer, and create a story that is combination of these two stories. Just output your story, don’t write anything else."},
            "personalities": [{"name": "empty", "prompt": ""}] * 10,
            "llm_backend": "llama.cpp", "model": "unsloth/SmolLM2-135M-Instruct-GGUF", "access_url": None,
            # "llm_backend": "llama.cpp", "model": "TheBloke/Mistral-7B-OpenOrca-GGUF", "access_url": None,
        }
    return apply_cli(cfg,args)

def main():
    p=parse_common_args("Run and compare the persona/personalities variants shown in the notebook.")
    p.add_argument("--variants",nargs="+",choices=["creative","not_creative","mixed"],default=["creative","not_creative","mixed"])
    args=p.parse_args()
    folders,labels=[],[]
    for variant in args.variants:
        cfg=make_base(args)
        if variant=="creative": cfg["personalities"]=[copy.deepcopy(CREATIVE)]*cfg["n_agents"]
        elif variant=="not_creative": cfg["personalities"]=[copy.deepcopy(NOT_CREATIVE)]*cfg["n_agents"]
        else:
            half=cfg["n_agents"]//2
            cfg["personalities"]=[copy.deepcopy(CREATIVE)]*half+[copy.deepcopy(NOT_CREATIVE)]*half
        cfg["output_name"]=f"persona_{variant}"
        folder=run_experiment(cfg); main_analysis(folder,font_sizes=FONT_SIZES,plot=True)
        folders.append(folder); labels.append(variant)
    if len(folders)>1: compare(folders,labels)
    print("\n".join(f"{x}: {y}" for x,y in zip(labels,folders)))

if __name__=="__main__":
    main()

