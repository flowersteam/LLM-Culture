import argparse
import run_simulation
from pathlib import Path
from tqdm import trange

n_agents = 10
n_timesteps = 10
prompt_init = 'permutation_Lorem_Ipsum'
prompt_update = 'permutations'
network_structure_type = 'fully_connected'
n_seeds = 1
#personality_list = ['The story should be about a hero', 'The story should be about a villain', 'The story should be about a princess', 'The story should be about a dragon', 'The story should be about a knight', 'The story should be about a wizard', 'The story should be about a witch', 'The story should be about a king', 'The story should be about a queen', 'The story should be about a castle']
access_url = 'https://av-jobs-confirmation-restoration.trycloudflare.com'
n_cliques = 2

# for i in trange(n_agents // 2 + 1):
    
#     personality_list = ['Romance'] * 2 * i + ['Sci-Fi'] * (n_agents - 2 * i)
#     output_folder_path = f"permut_{network_structure_type}_prop_{2 * i}romance_{(n_agents - 2 * i)}scifi"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "permutations"
#     args.end_flag = "##STORYEND##"
#     args.start_flag = "##STORY##"
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)


# for i in trange(n_agents // 2 + 1):
    
#     personality_list = ['NotSciFi'] * 2 * i + ['Sci-Fi'] * (n_agents - 2 * i)
#     output_folder_path = f"permut_{network_structure_type}_prop_{2 * i}notScifi_{(n_agents - 2 * i)}scifi"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "permutations"
#     args.end_flag = "##STORYEND##"
#     args.start_flag = "##STORY##"
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)



# for i in trange(n_agents // 2 + 1):
    
#     personality_list = ['RomanceNotSciFi'] * 2 * i + ['SciFiNotRomance'] * (n_agents - 2 * i)
#     output_folder_path = f"permut_{network_structure_type}_prop_{2 * i}RomNotScifi_{(n_agents - 2 * i)}ScifiNotRom"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "permutations"
#     args.end_flag = "##STORYEND##"
#     args.start_flag = "##STORY##"
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)

# prompt_update = 'selective permutations'
# n_seeds = 3


# for i in trange(n_agents // 5 + 1):
    
#     personality_list = ['Romance'] * 5 * i + ['Sci-Fi'] * (n_agents - 5 * i)
#     output_folder_path = f"selectivePermut_{network_structure_type}_prop_{5 * i}romance_{(n_agents - 5 * i)}scifi_2"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "permutations"
#     args.end_flag = "##STORYEND##"
#     args.start_flag = "##STORY##"
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)



# prompt_update = 'selective permutations'
# n_seeds = 1


# for i in trange(5):
    
#     personality_list = ['Romance'] * 5 * i + ['Sci-Fi'] * (n_agents - 5 * i)
#     output_folder_path = f"selectivePermut_{network_structure_type}_prop_{5 * i}romance_{(n_agents - 5 * i)}scifi_3"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "permutations"
#     args.end_flag = "##STORYEND##"
#     args.start_flag = "##STORY##"
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)



n_agents = 10
n_timesteps = 10
prompt_init = 'story teller'
prompt_update = 'Combine 2 Favorites'
network_structure_type = 'fully_connected'
n_seeds = 1
access_url = 'https://remedy-manage-italian-impacts.trycloudflare.com'
n_cliques = 2



# for i in trange(n_agents // 2):
    
#     personality_list = ['Sci-Fi'] * 2 * i + ['Empty'] * (n_agents - 2 * i)
#     output_folder_path = f"combineTwo_{network_structure_type}_prop_{2 * i}romance_1"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "Empty"
#     args.end_flag = None
#     args.start_flag = None
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)




# for i in trange(n_agents // 2):
    
#     personality_list = ['Sci-Fi'] * 2 * i + ['Empty'] * (n_agents - 2 * i)
#     output_folder_path = f"combineTwo_{network_structure_type}_prop_{2 * i}romance_2"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "Empty"
#     args.end_flag = None
#     args.start_flag = None
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)





# personality_list = ['Sci-Fi'] * n_agents
# output_folder_path = f"combineTwo_{network_structure_type}_prop_10SciFi_1"



# args = argparse.Namespace()
# args.n_agents = n_agents
# args.n_timesteps = n_timesteps
# args.prompt_init = prompt_init
# args.prompt_update = prompt_update
# args.format_prompt = "Empty"
# args.end_flag = None
# args.start_flag = None
# args.network_structure = network_structure_type
# args.n_seeds = n_seeds

# args.personality_list = personality_list
# args.output = Path(output_folder_path).absolute()
# args.debug = False
# args.preset = None  # Add the preset attribute
# args.access_url = access_url
# args.n_cliques = n_cliques

# run_simulation.main(args)






# n_agents = 10
# n_timesteps = 10
# prompt_init = 'permutation_Lorem_Ipsum'
# prompt_update = 'permutations'
# network_structure_type = 'random'
# n_edges = (n_agents ** 2) // 2
# n_seeds = 1
# #personality_list = ['The story should be about a hero', 'The story should be about a villain', 'The story should be about a princess', 'The story should be about a dragon', 'The story should be about a knight', 'The story should be about a wizard', 'The story should be about a witch', 'The story should be about a king', 'The story should be about a queen', 'The story should be about a castle']
# access_url = 'https://furthermore-sg-nu-craps.trycloudflare.com'
# n_cliques = 2



# for i in trange(n_agents // 2):
    
#     personality_list = ['Sci-Fi'] * 2 * i + ['Romance'] * (n_agents - 2 * i)
#     output_folder_path = f"permutat_{network_structure_type}_{n_edges}edges_{2 * i}SciFi_{n_agents - 2 * i}Romance"



#     args = argparse.Namespace()
#     args.n_agents = n_agents
#     args.n_timesteps = n_timesteps
#     args.prompt_init = prompt_init
#     args.prompt_update = prompt_update
#     args.format_prompt = "Empty"
#     args.end_flag = None
#     args.start_flag = None
#     args.network_structure = network_structure_type
#     args.n_seeds = n_seeds
#     args.n_edges = n_edges

#     args.personality_list = personality_list
#     args.output = Path(output_folder_path).absolute()
#     args.debug = False
#     args.preset = None  # Add the preset attribute
#     args.access_url = access_url
#     args.n_cliques = n_cliques

#     run_simulation.main(args)





n_agents = 50
n_timesteps = 50
prompt_init = 'War of the Ghosts inspiration'
prompt_update = 'inspiration'
network_structure_type = 'sequence'
n_seeds = 4
n_edges = (n_agents ** 2) // 2
#personality_list = ['The story should be about a hero', 'The story should be about a villain', 'The story should be about a princess', 'The story should be about a dragon', 'The story should be about a knight', 'The story should be about a wizard', 'The story should be about a witch', 'The story should be about a king', 'The story should be about a queen', 'The story should be about a castle']
access_url = 'https://cattle-chile-facts-example.trycloudflare.com'





personality_list = ['Empty'] * n_agents
output_folder_path = f"GhostWarInspirationChain50_Mixtral_8x7b_Instruct_formatted3"



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



n_agents = 50
n_timesteps = 50
prompt_init = 'Darwinism inspiration'
prompt_update = 'inspiration'
network_structure_type = 'sequence'
n_seeds = 4
n_edges = (n_agents ** 2) // 2
#personality_list = ['The story should be about a hero', 'The story should be about a villain', 'The story should be about a princess', 'The story should be about a dragon', 'The story should be about a knight', 'The story should be about a wizard', 'The story should be about a witch', 'The story should be about a king', 'The story should be about a queen', 'The story should be about a castle']
access_url = 'https://cattle-chile-facts-example.trycloudflare.com'





personality_list = ['Empty'] * n_agents
output_folder_path = "DarwinismInspirationChain50_Mixtral_8x7b_Instruct_formatted3"



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
