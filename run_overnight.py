import argparse
import run_simulation
from pathlib import Path
from tqdm import trange


## VARIABLES
n_agents_var = [2, 5, 10]


prompt_init_var = ['War of the Ghosts inspiration', 'Darwinism inspiration', 'War of the Ghosts rephrase', 'Darwinism rephrase']
prompt_update_var = ['inspiration_multi', 'inspiration_multi', 'rephrase_multi', 'rephrase_multi']
prompts_var = zip(prompt_init_var, prompt_update_var)



## CONSTANTS
n_timesteps = 50   
network_structure_type = 'fully_connected'
n_seeds = 4
n_cliques = 1

# enter the correct url here
access_url = 'https://ve-breathing-msgid-look.trycloudflare.com'


## SIMULATION LOOP
for n_agents in n_agents_var:
    for (prompt_init, prompt_update) in prompts_var:
        print(n_agents, prompt_init, prompt_update)
            
        n_edges = (n_agents ** 2) // 2
        personality_list = ['Empty'] * n_agents


        ## MODIFY MODEL NAME HERE
        model = 'mistral-7b-instruct-v0.2.Q5_K_M.gguf'


        output_folder_path = f"{prompt_init}_{n_agents}Agents_50gen_{model}"


        args = argparse.Namespace()
        args.n_agents = n_agents
        args.n_timesteps = n_timesteps
        args.prompt_init = prompt_init
        args.prompt_update = prompt_update
        args.format_prompt = "Empty"


        args.start_flag = None
        args.end_flag = None
        args.network_structure = network_structure_type
        args.n_seeds = n_seeds
        args.n_edges = n_edges

        args.personality_list = personality_list
        args.output = Path(output_folder_path).absolute()
        args.debug = False
        args.preset = None  # Add the preset attribute
        args.access_url = access_url
        args.n_cliques = n_cliques

        run_simulation.main(args)
