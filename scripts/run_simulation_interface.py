import os
import json

from pathlib import Path

import networkx as nx

from llm_culture.simulation.utils import resolve_model_path, run_simul

RESULTS_DIR = 'results/experiments'


def _create_network_structure(
        network_structure_name, 
        n_agents, 
        n_cliques
    ):
    """Create a network structure based on the given parameters

    :param network_structure_name: name
    :param n_agents: n_agents
    :param n_cliques: n_cliques
    :return: network_structure
    """
    if network_structure_name == 'sequence':
        network_structure = nx.DiGraph()
        for i in range(n_agents - 1):
            network_structure.add_edge(i, i + 1)
    elif network_structure_name == 'circle':
        network_structure = nx.cycle_graph(n_agents)
    elif network_structure_name == 'caveman':
        network_structure = nx.connected_caveman_graph(n_cliques, n_agents // n_cliques)
    elif network_structure_name == 'fully_connected':
        network_structure = nx.complete_graph(n_agents)
    return network_structure


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
    
    json_prompt_init = 'data/parameters/prompt_init.json'
    json_prompt_update = 'data/parameters/prompt_update.json'
    json_personnalities = 'data/parameters/personalities.json'
    
    sequence = True if network_structure_name == 'sequence' else False
    network_structure = _create_network_structure(network_structure_name, n_agents, n_cliques)

    output_dict = {}
    output_dict["adjacency_matrix"] = nx.to_numpy_array(network_structure).tolist()

    # Write the prompts and their description in the output dictionary
    with open(json_prompt_init, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == init_prompt:
                prompt_init = d['prompt']
    output_dict["prompt_init"] = [prompt_init]
    
    with open(json_prompt_update, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == update_prompt:
                prompt_update = d['prompt']
    
    output_dict["prompt_update"] = [prompt_update]

    personality_list = []
    print("\nAgents personalities:")
    with open(json_personnalities, 'r') as file:
        data = json.load(file)
        for perso in personalities:
            print(perso)
            for d in data:
                if d['name'] == perso:
                    personality_list.append(d['prompt'])
    output_dict["personality_list"] = personality_list

    os.makedirs(os.path.dirname(output_dir + '/'), exist_ok=True)

    llm_backend = False
    model = None
    if use_local_model:
        if not model_source:
            raise ValueError("Please provide a local model path or Hugging Face repo id")

        resolved_model_path = resolve_model_path(
            model_source,
            hf_cache_dir or os.path.expanduser("~/.cache/huggingface"),
        )

        resolved_model_path = Path(resolved_model_path)
        if resolved_model_path.is_dir():
            gguf_files = sorted(resolved_model_path.rglob("*.gguf"))
            if len(gguf_files) == 0:
                raise FileNotFoundError(
                    f"No .gguf model file found in downloaded snapshot: {resolved_model_path}"
                )

            selected_gguf = next(
                (model_file for model_file in gguf_files if "q4_k_m" in model_file.name.lower()),
                gguf_files[0],
            )

            if len(gguf_files) > 1:
                print(
                    f"Multiple .gguf model files found in {resolved_model_path}; using {selected_gguf.name}",
                    flush=True,
                )

            resolved_model_path = selected_gguf

        from llama_cpp import Llama

        model = Llama(
            model_path=str(resolved_model_path),
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        llm_backend = "llama.cpp"
    
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
            server_url,
            n_timesteps,
            network_structure,
            prompt_init,
            prompt_update,
            personality_list,
            n_agents,
            sequence=sequence,
            output_folder=output_dir,
            debug=True,
            instruct=instruct,
            llm_backend=llm_backend,
            model=model,
            temperature=temperature,
            progress_callback=_seed_progress,
        )
        
        output_dict["stories"] = stories

        if output_dir:
            with open(Path(output_dir, f'output{seed}.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            raise ValueError("Please provide an output directory for your experiment")
    

