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
    # create optional argument for the output file name to save in the output folder
    parser.add_argument('-of', '--output_file', type=str, default='output.json', help='Output file name.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('-url', '--access_url', type=str, default='', help='URL to send the prompt to.')
    parser.add_argument('-s', '--n_seeds', type=int, default=2, help='Number of seeds')
    parser.add_argument('--seed_offset', type=int, default=0, help='Start index for output{i}.json naming when running seeds in parallel.')
    parser.add_argument('--use_vllm', action='store_true', help='Use vllm for local inference instead of a server URL.')
    parser.add_argument('--model', type=str, default=None, help='Model name or path to load with vllm.')
    parser.add_argument('--no_instruct', action='store_true', help='Disable instruct mode (use raw completion).')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature.')

    return parser.parse_args()


def main(args=None):
    """Run the simulation with the given parameters

    :param args: simulation parameters, defaults to None
    :return: dictionary containing the simulation results
    """
    json_prompt_init = 'llm_culture/data/parameters/prompt_init.json'
    json_prompt_update = 'llm_culture/data/parameters/prompt_update.json'
    json_personnalities = 'llm_culture/data/parameters/personalities.json'

    #import Path
    from pathlib import Path
    repo_root = Path(__file__).parent.parent
    json_prompt_init = repo_root / json_prompt_init
    json_prompt_update = repo_root / json_prompt_update
    json_personnalities = repo_root / json_personnalities
    

    if args is None:
        args = parse_arguments()

    # initialize the output dictionary for results
    output_dict = {}
    debug = args.debug
    sequence = False

    # Load vllm model if requested
    vllm_model = None
    if args.use_vllm:
        from vllm import LLM
        assert args.model is not None, "--model must be provided when using --use_vllm"
        vllm_model = LLM(model=args.model)

    # Use the arguments
    n_agents = args.n_agents
    n_timesteps = args.n_timesteps

    # handle the network structure
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

    # Create the output folder if it does not exist
    os.makedirs(os.path.dirname(str(args.output) + '/'), exist_ok=True)
    print(args.output)

    # Run the simulation for each seed
    for i in range(args.n_seeds):
        seed_idx = args.seed_offset + i
        print(f"Seed {seed_idx}")
        stories = run_simul(
             args.access_url, 
             n_timesteps, 
             network_structure, 
             prompt_init,
            prompt_update, 
            personality_list, 
            n_agents,
            sequence=sequence, 
            output_folder=args.output,
            debug=debug,
            instruct=not args.no_instruct,
            use_vllm=args.use_vllm,
            model=vllm_model,
            temperature=args.temperature,
        )
        output_dict["stories"] = stories

        # Save the output to a file
        if args.output:
            with open(Path(args.output, 'output'+str(seed_idx)+'.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
        else:
            with open(Path("results/", 'output'+str(seed_idx)+'.json'), "w") as f:
                json.dump(output_dict, f, indent=4)
            return output_dict
        

if __name__ == "__main__":
    main()