import os
import json

from pathlib import Path

import networkx as nx

from llm_culture.simulation.utils import run_simul, build_network_structure, load_named_prompt, load_personalities
from llm_culture.simulation.backends import load_llm_backend
from llm_culture.config import (
    ExperimentConfig,
    PopulationConfig,
    AgentConfig,
    BackendConfig,
    Backend,
    GenerationConfig,
)
from llm_culture.paths import PROMPT_INIT_JSON, PROMPT_UPDATE_JSON, PERSONALITIES_JSON

RESULTS_DIR = 'results/experiments'


def run_simulation(
        n_agents,
        n_timesteps,
        n_seeds,
        network_structure_name,
        n_cliques,
        personalities,
        init_prompt,
        update_prompt,
        output_dir,
        server_url,
        use_local_model=False,
        model_source=None,
        hf_cache_dir=None,
        instruct=True,
        temperature=0.8,
        progress_callback=None
    ):
    """Run the simulation with the given parameters
    """
    network_structure, _ = build_network_structure(network_structure_name, n_agents, n_cliques)

    output_dict = {}
    output_dict["adjacency_matrix"] = nx.to_numpy_array(network_structure).tolist()

    prompt_init = load_named_prompt(PROMPT_INIT_JSON, init_prompt)
    prompt_update = load_named_prompt(PROMPT_UPDATE_JSON, update_prompt)
    personality_list = load_personalities(PERSONALITIES_JSON, personalities)
    output_dict["prompt_init"] = [prompt_init]
    output_dict["prompt_update"] = [prompt_update]
    output_dict["personality_list"] = personality_list

    os.makedirs(os.path.dirname(output_dir + '/'), exist_ok=True)

    if use_local_model and not model_source:
        raise ValueError("Please provide a local model path or Hugging Face repo id")

    # One ExperimentConfig for the whole run; the GUI's local backend is llama.cpp.
    # population.agents mirrors the per-agent personas (one group per agent); the
    # network is built separately above, so population.network_structure is left at
    # its default on this path.
    cfg = ExperimentConfig(
        population=PopulationConfig(
            n_cliques=n_cliques,
            agents=[
                AgentConfig(
                    count=1,
                    personality=persona,
                    prompt_init=init_prompt,
                    prompt_update=update_prompt,
                )
                for persona in personalities
            ],
        ),
        backend=BackendConfig(
            kind=Backend.llama_cpp if use_local_model else Backend.none,
            model=model_source,
            access_url=server_url,
            hf_cache_dir=hf_cache_dir,
        ),
        generation=GenerationConfig(temperature=temperature, instruct=instruct),
        n_timesteps=n_timesteps,
        output=output_dir,
        debug=True,
    )

    # Per-agent specs for run_simul: each agent gets its persona text + the shared
    # init/update prompt texts (resolved above).
    agent_specs = [
        (persona_text, prompt_init, prompt_update) for persona_text in personality_list
    ]

    # Load the model once (reused across seeds); remote server -> (False, None).
    llm_backend, model = load_llm_backend(cfg)

    for seed in range(n_seeds):
        print(f"\nSeed {seed}")

        def _seed_progress(current_generation, total_generations):
            if progress_callback is None:
                return

            completed_generations = (seed * n_timesteps) + current_generation
            progress_callback(
                completed_generations,
                n_seeds * n_timesteps,
                seed + 1,
                n_seeds,
                current_generation,
                total_generations,
            )

        stories = run_simul(
            cfg,
            network_structure,
            agent_specs,
            llm_backend=llm_backend,
            model=model,
            progress_callback=_seed_progress,
        )
        
        output_dict["stories"] = stories

        if output_dir:
            with open(Path(output_dir, f'output{seed}.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            raise ValueError("Please provide an output directory for your experiment")
    

