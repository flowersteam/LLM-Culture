# TODO : Combine this file with run_simulation 

import os
import json
import argparse
from pathlib import Path

import networkx as nx

from llm_culture.simulation.utils import run_simul


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('-na', '--n_agents', type=int, default=2, help='Number of agents.')
    parser.add_argument('-nt', '--n_timesteps', type=int, default=2, help='Number of timesteps.')
    # add an optional argument that will select a preset of parameters from parameters_sets in data
    # argument to select the network structure
    parser.add_argument('-ns', '--network_structure', type=str, default='sequence',
                        choices=['sequence','fully_connected' 'circle', 'caveman'], help='Network structure.')
    parser.add_argument('-nc', '--n_cliques', type=int, default=2, help='Number of cliques for the Caveman graph')
    # argument to select the prompt_init from the list of prompts
    parser.add_argument('-pi', '--prompt_init', type=str, default='kid',
                        help='Initial prompt.')
    # argument to select the prompt_update from the list of prompts
    parser.add_argument('-pu', '--prompt_update', type=str, default='kid',
                        help='Update prompt.')    
    # select a personality from the list of personalities (no choices)
    parser.add_argument('-pl', '--personality_list', type=str, default= ["Empty", "Empty"],
                        help='Personality list.')
    # add an option output folder to save the results
    parser.add_argument('-o', '--output', type=str, default='results/default_folder', help='Output folder.')
    # create optional argument for the output file name to save in the output folder
    parser.add_argument('-of', '--output_file', type=str, default='output.json', help='Output file name.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('-url', '--access_url', type=str, default='', help='URL to send the prompt to.')
    parser.add_argument('-s', '--n_seeds', type=int, default=2, help='Number of seeds')

    return parser.parse_args()


def prepare_simu(args):
    pass


def main(args=None):

    json_prompt_init = 'llm_culture/data/parameters/prompt_init.json'
    json_prompt_update = 'llm_culture/data/parameters/prompt_update.json'
    json_structure = 'llm_culture/data/parameters/network_structure.json'
    json_personnalities = 'llm_culture/data/parameters/personnalities.json'


    if args is None:
        args = parse_arguments()

    output_dict = {}
    debug = args.debug

    sequence = False

    # If we use a preset, we can use the parameters_sets in data

    # Use the arguments
    n_agents = args.n_agents
    n_timesteps = args.n_timesteps

    network_structure = None
    if args.network_structure == 'sequence':
        network_structure = nx.DiGraph()
        for i in range(n_agents - 1):
            network_structure.add_edge(i, i + 1)
        sequence = True
    elif args.network_structure == 'circle':
        network_structure = nx.cycle_graph(n_agents)
    elif args.network_structure == 'caveman':
        network_structure = nx.connected_caveman_graph(int(args.n_cliques), n_agents // int(args.n_cliques))

    elif args.network_structure == 'fully_connected':
                network_structure = nx.complete_graph(n_agents)

    # save adjacency matrix to output_dict
    output_dict["adjacency_matrix"] = nx.to_numpy_array(network_structure).tolist()

    # prompt_init = prompts.prompt_init_dict[args.prompt_init]
    with open(json_prompt_init, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == args.prompt_init:
                prompt_init = d['prompt']

    # prompt_update = prompts.prompt_update_dict[args.prompt_update]
    with open(json_prompt_update, 'r') as file:
        data = json.load(file)
        for d in data:
            if d['name'] == args.prompt_update:
                prompt_update = d['prompt']



        # personality_dict = getattr(prompts, args.personality_dict)
        # personality_list = prompts.personality_dict_of_lists[args.personality_list]

        personality_list = []
        with open(json_personnalities, 'r') as file:
                    data = json.load(file)
                    for perso in args.personality_list:
                        print(perso)
                        for d in data:
                            if d['name'] == perso:
                                personality_list.append(d['prompt'])


        output_dict["prompt_init"] = [prompt_init]
        output_dict["prompt_update"] = [prompt_update]
        output_dict["personality_list"] = personality_list

    os.makedirs(os.path.dirname(str(args.output) + '/'), exist_ok=True)
    t = input(args.output)
    for i in range(args.n_seeds):
        print(f"Seed {i}")
        stories = run_simul(args.access_url, n_timesteps, network_structure, prompt_init,
                            prompt_update, personality_list, n_agents,
                            sequence=sequence, output_folder=args.output,
                            debug=debug)
        output_dict["stories"] = stories

        # Save the output to a file
        if args.output:
            with open(Path(args.output, 'output'+str(i)+'.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            with open(Path("results/", 'output'+str(i)+'.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
            return output_dict
        
    # get_all_figures(stories, folder_name)


if __name__ == "__main__":
    main()