import os
import json
import argparse

from pathlib import Path

import networkx as nx

from llm_culture.simulation.utils import run_simul, build_network_structure, load_named_prompt, load_personalities, log_resources
from llm_culture.simulation.backends import load_llm_backend
from llm_culture.config import ExperimentConfig, Backend, Network, validate_experiment, GenerationConfig
from llm_culture.paths import PROMPT_INIT_JSON, PROMPT_UPDATE_JSON, PERSONALITIES_JSON


def build_parser():
    """Build and return the simulation argument parser.

    Exposed separately from parse_arguments() so other entrypoints (e.g. the
    combined run_experiment.py) can reuse the *exact* same flags instead of
    duplicating them.
    """
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('-na', '--n_agents', type=int, default=2, help='Number of agents.')
    parser.add_argument('-nt', '--n_timesteps', type=int, default=2, help='Number of timesteps.')
    # argument to select the network structure
    parser.add_argument('-ns', '--network_structure', type=str, default='sequence',
                        choices=['sequence', 'fully_connected', 'circle', 'caveman'], help='Network structure.')
    parser.add_argument('-nc', '--n_cliques', type=int, default=2, help='Number of cliques for the Caveman graph')
    # argument to select the prompt_init from the list of prompts
    parser.add_argument('-pi', '--prompt_init', type=str, default='kid',
                        help='Initial prompt.')
    # argument to select the prompt_update from the list of prompts
    parser.add_argument('-pu', '--prompt_update', type=str, default='kid',
                        help='Update prompt.')    
    # select a personality from the list of personalities (no choices)
    parser.add_argument('-pl', '--personality_list', type=str, nargs='+', default=["Empty", "Empty"],
                        help='Personality list (one value per agent, e.g. -pl Empty Empty Empty).')
    # add an option output folder to save the results
    parser.add_argument('-o', '--output', type=str, default='results/default_folder', help='Output folder.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('-url', '--access_url', type=str, default='', help='URL to send the prompt to.')
    parser.add_argument('-s', '--n_seeds', type=int, default=2, help='Number of seeds')
    parser.add_argument('--seed_offset', type=int, default=0, help='Start index for output{i}.json naming when running seeds in parallel.')
    parser.add_argument('--use_vllm', action='store_true', help='Use vllm for local inference instead of a server URL (Linux/GPU only).')
    parser.add_argument('--use_llama_cpp', action='store_true', help='Use llama.cpp for local inference (loads a GGUF model; works on macOS/Linux).')
    parser.add_argument('--model', type=str, default=None, help='Model name/repo id or local path to load for local inference.')
    parser.add_argument('--hf_cache_dir', type=str, default=None, help='Hugging Face cache dir for downloading local models (default: ~/.cache/huggingface).')
    parser.add_argument('--no_instruct', action='store_true', help='Disable instruct mode (use raw completion).')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print each agent\'s generated story text as the simulation runs.')

    return parser


def parse_arguments():
    return build_parser().parse_args()


def args_to_config(args):
    """Map the argparse Namespace onto an ExperimentConfig dataclass.

    This is the single place that translates the standalone CLI's flags (the
    two backend booleans, the `no_instruct` double-negative, the network string)
    into the shared config type.
    """
    if args.use_vllm and args.use_llama_cpp:
        raise ValueError("Choose only one of --use_vllm or --use_llama_cpp.")
    if args.use_vllm:
        backend = Backend.vllm
    elif args.use_llama_cpp:
        backend = Backend.llama_cpp
    else:
        backend = Backend.none

    return ExperimentConfig(
        n_agents=args.n_agents,
        n_timesteps=args.n_timesteps,
        n_seeds=args.n_seeds,
        seed_offset=args.seed_offset,
        network_structure=Network(args.network_structure),
        n_cliques=args.n_cliques,
        prompt_init=args.prompt_init,
        prompt_update=args.prompt_update,
        personality_list=list(args.personality_list),
        generation=GenerationConfig(temperature=args.temperature),
        instruct=not args.no_instruct,
        verbose=args.verbose,
        backend=backend,
        model=args.model,
        access_url=args.access_url,
        hf_cache_dir=args.hf_cache_dir,
        output=args.output,
        debug=args.debug,
    )


def run_simulation_from_config(cfg):
    """Run the simulation from an ExperimentConfig and return the results dict.

    This is the shared core used by both the standalone CLI (via main) and the
    Hydra entrypoint (run_experiment.py), so there is no argparse Namespace on
    the Hydra path.

    :param cfg: an ExperimentConfig instance
    :return: dictionary containing the simulation results
    """
    validate_experiment(cfg)

    output_dict = {}
    n_agents = cfg.n_agents
    n_timesteps = cfg.n_timesteps

    # Select the backend and load a local model if requested
    # (Backend.none -> remote OpenAI-compatible server via cfg.access_url).
    llm_backend, model = load_llm_backend(cfg)
    # Report resource usage once the (potentially large) model is resident in
    # memory, so the footprint of the weights is visible. Skipped for the remote
    # backend, where no model is loaded in this process.
    if model is not None:
        log_resources("after model load")

    # Build the network graph (shared builder; also supports custom structures)
    network_structure, _ = build_network_structure(cfg.network_structure.value, n_agents, cfg.n_cliques)
    output_dict["adjacency_matrix"] = nx.to_numpy_array(network_structure).tolist()

    # Resolve the named prompts / personalities from the parameter files
    prompt_init = load_named_prompt(PROMPT_INIT_JSON, cfg.prompt_init)
    prompt_update = load_named_prompt(PROMPT_UPDATE_JSON, cfg.prompt_update)
    personality_list = load_personalities(PERSONALITIES_JSON, cfg.personality_list)
    output_dict["prompt_init"] = [prompt_init]
    output_dict["prompt_update"] = [prompt_update]
    output_dict["personality_list"] = personality_list

    # Create the output folder if it does not exist
    output_dir = str(cfg.output)
    os.makedirs(os.path.dirname(output_dir + '/'), exist_ok=True)

    backend_desc = llm_backend if llm_backend else f"remote server ({cfg.access_url or 'no url set'})"
    print("\n" + "=" * 64)
    print("SIMULATION")
    print(f"  agents={n_agents}  timesteps={n_timesteps}  seeds={cfg.n_seeds}  network={cfg.network_structure.value}")
    print(f"  backend={backend_desc}" + (f"  model={cfg.model}" if cfg.model else ""))
    print(f"  output folder: {os.path.abspath(output_dir)}")
    print("=" * 64)

    # Run the simulation for each seed
    for i in range(cfg.n_seeds):
        seed_idx = cfg.seed_offset + i
        print(f"Seed {seed_idx}")
        stories = run_simul(
            cfg,
            network_structure,
            prompt_init,
            prompt_update,
            personality_list,
            llm_backend=llm_backend,
            model=model,
        )
        output_dict["stories"] = stories

        out_path = Path(cfg.output, 'output' + str(seed_idx) + '.json')
        with open(out_path, "w") as f:
            json.dump(output_dict, f, indent=4)
        print(f"  seed {seed_idx}: saved {out_path}")

    print(f"Simulation complete — {cfg.n_seeds} seed(s) written to {os.path.abspath(output_dir)}")
    if model is not None:
        log_resources("after simulation")
    return output_dict


def main(args=None):
    """Run the simulation with the given parameters (argparse entrypoint).

    :param args: parsed argparse Namespace, defaults to None (parse from argv)
    :return: dictionary containing the simulation results
    """
    if args is None:
        args = parse_arguments()
    cfg = args_to_config(args)
    return run_simulation_from_config(cfg)


if __name__ == "__main__":
    main()