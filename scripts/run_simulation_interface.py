# TODO : Refactor the file to decompose huge functions into smaller ones
# TODO : Combine this file with run_simulation 

import os
import json
from pathlib import Path

import networkx as nx

from llm_culture.simulation.utils import run_simul

RESULTS_DIR = 'Results/experiments'


def _create_network_structure(network_structure_name, n_agents, n_cliques):
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
        server_url
        ):
    
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
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed}")
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
            debug=True
            )
        
        output_dict["stories"] = stories

        if output_dir:
            with open(Path(output_dir, f'output{seed}.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            raise ValueError("Please provide an output directory for your experiment")
    

