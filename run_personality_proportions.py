import argparse
import run_simulation
from pathlib import Path
from tqdm import trange

n_agents = 10
n_timesteps = 10
prompt_init = 'story teller'
prompt_update = 'Combine 2 Favorites'
network_structure_type = 'fully_connected'
n_seeds = 5
#personality_list = ['The story should be about a hero', 'The story should be about a villain', 'The story should be about a princess', 'The story should be about a dragon', 'The story should be about a knight', 'The story should be about a wizard', 'The story should be about a witch', 'The story should be about a king', 'The story should be about a queen', 'The story should be about a castle']
access_url = 'https://toolkit-had-census-off.trycloudflare.com'
n_cliques = 2

for i in trange(n_agents + 1):
    
    personality_list = ['Very creative'] * i + ['Not Creative'] * (n_agents - i)
    output_folder_path = f"{network_structure_type}_{n_agents}_agents_{n_timesteps}_timesteps_prop_creative_{i}"



    args = argparse.Namespace()
    args.n_agents = n_agents
    args.n_timesteps = n_timesteps
    args.prompt_init = prompt_init
    args.prompt_update = prompt_update
    args.network_structure = network_structure_type
    args.n_seeds = n_seeds

    args.personality_list = personality_list
    args.output = Path(output_folder_path).absolute()
    args.debug = False
    args.preset = None  # Add the preset attribute
    args.access_url = access_url
    args.n_cliques = n_cliques

    run_simulation.main(args)