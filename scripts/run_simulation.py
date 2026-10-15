"""Run only the simulation (no analysis) from a Hydra config.

Like run_experiment.py, this reads the structured `ExperimentConfig` from `conf/`
(no argparse) — it just stops after writing the per-seed output JSON instead of
also producing the analysis plots. Use run_experiment.py for simulation + analysis.

    uv run python scripts/run_simulation.py                      # base preset, sim only
    uv run python scripts/run_simulation.py experiment=base n_seeds=1
    uv run python scripts/run_simulation.py \\
        backend.kind=llama_cpp backend.model=unsloth/SmolLM2-135M-Instruct-GGUF \\
        population.agents.0.count=3 n_timesteps=3
"""
import os
import sys
import json
from pathlib import Path

import networkx as nx
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Ensure the repo root is importable (so `llm_culture` resolves) regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_culture.simulation.utils import (
    run_simul,
    build_network_structure,
    load_named_prompt,
    load_personalities,
)
from llm_culture.resources import log_resources
from llm_culture.simulation.backends import load_llm_backend
from llm_culture.config import ExperimentConfig, validate_experiment
from llm_culture.paths import PROMPT_INIT_JSON, PROMPT_UPDATE_JSON, PERSONALITIES_JSON

cs = ConfigStore.instance()
cs.store(name="experiment_schema", node=ExperimentConfig)


def resolve_agent_specs(population):
    """Expand a PopulationConfig's agent groups into an ordered list of per-agent
    ``(personality_text, prompt_init_text, prompt_update_text)`` triples.

    Registered names are resolved from the parameter files once per group and
    repeated ``count`` times. Agents are laid out group by group onto network
    node indices 0..n-1.
    """
    specs = []
    for group in population.agents:
        personality_text = load_personalities(PERSONALITIES_JSON, [group.personality])[0]
        init_text = load_named_prompt(PROMPT_INIT_JSON, group.prompt_init)
        update_text = load_named_prompt(PROMPT_UPDATE_JSON, group.prompt_update)
        specs.extend([(personality_text, init_text, update_text)] * group.count)
    return specs


def run_simulation_from_config(cfg):
    """Run the simulation from an ExperimentConfig and return the results dict.

    Shared core used by both this script's Hydra entrypoint and run_experiment.py.

    :param cfg: an ExperimentConfig instance
    :return: dictionary containing the simulation results
    """
    validate_experiment(cfg)

    pop = cfg.population
    n_agents = pop.n_agents
    n_timesteps = cfg.n_timesteps

    # Select the backend and load a local model if requested
    # (Backend.none -> remote OpenAI-compatible server via cfg.backend.access_url).
    llm_backend, model = load_llm_backend(cfg)
    # Report resource usage once the (potentially large) model is resident in
    # memory. Skipped for the remote backend, where no model is loaded here.
    if model is not None:
        log_resources("after model load")

    # Build the network graph (shared builder; also supports custom structures)
    network_structure, _ = build_network_structure(
        pop.network_structure.value, n_agents, pop.n_cliques
    )

    # Expand the agent groups into resolved per-agent specs
    agent_specs = resolve_agent_specs(pop)

    output_dict = {}
    output_dict["adjacency_matrix"] = nx.to_numpy_array(network_structure).tolist()
    # Provenance (not consumed by analysis): the per-agent resolved texts.
    output_dict["personality_list"] = [spec[0] for spec in agent_specs]
    output_dict["prompt_init"] = [spec[1] for spec in agent_specs]
    output_dict["prompt_update"] = [spec[2] for spec in agent_specs]

    # Create the output folder if it does not exist
    output_dir = str(cfg.output)
    os.makedirs(os.path.dirname(output_dir + '/'), exist_ok=True)

    backend_desc = llm_backend if llm_backend else f"remote server ({cfg.backend.access_url or 'no url set'})"
    print("\n" + "=" * 64)
    print("SIMULATION")
    print(f"  agents={n_agents}  timesteps={n_timesteps}  seeds={cfg.n_seeds}  network={pop.network_structure.value}")
    print(f"  backend={backend_desc}" + (f"  model={cfg.backend.model}" if cfg.backend.model else ""))
    print(f"  output folder: {os.path.abspath(output_dir)}")
    print("=" * 64)

    # Run the simulation for each seed
    for i in range(cfg.n_seeds):
        seed_idx = cfg.seed_offset + i
        print(f"Seed {seed_idx}")
        stories = run_simul(
            cfg,
            network_structure,
            agent_specs,
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    exp: ExperimentConfig = OmegaConf.to_object(cfg)

    print("\n" + "#" * 64)
    print("# SIMULATION CONFIG")
    print("#" * 64)
    print(OmegaConf.to_yaml(cfg), end="")

    run_simulation_from_config(exp)


if __name__ == "__main__":
    main()
