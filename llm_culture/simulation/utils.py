import os
from pathlib import Path

from llm_culture.simulation.agent import Agent
import networkx as nx
import json
PARAMS_DIR = Path("data", "parameters")
PROMPT_INIT_JSON = PARAMS_DIR / "prompt_init.json"
PROMPT_UPDATE_JSON = PARAMS_DIR / "prompt_update.json"
PERSONALITIES_JSON = PARAMS_DIR / "personalities.json"


def init_agents(
        cfg,
        network_structure,
        prompt_init,
        prompt_update,
        personality_list,
        llm_backend=False,
        model=None,
        sampling_params=None,
    ):
    """Initialize the agents from an ExperimentConfig and the runtime objects.

    :param cfg: ExperimentConfig (provides n_agents, access_url, debug, instruct, temperature)
    :param network_structure: the built networkx graph (directed => sequence mode)
    :param prompt_init: resolved initial prompt text
    :param prompt_update: resolved update prompt text
    :param personality_list: resolved personality texts (one per agent)
    :param llm_backend: loaded backend tag ("vllm"/"llama.cpp") or False
    :param model: loaded model instance or None
    :param sampling_params: optional sampling params
    :return: list of agents
    """
    # A sequence chain is the only directed topology; derive it from the graph so
    # this works for every structure (including custom ones).
    sequence = network_structure.is_directed()

    agent_list = []
    wait = 0

    for agent_id in range(cfg.n_agents):
        personality = personality_list[agent_id]
        agent = Agent(
            cfg,
            agent_id,
            prompt_init,
            prompt_update,
            personality,
            wait=wait,
            sequence=sequence,
            llm_backend=llm_backend,
            model=model,
            sampling_params=sampling_params,
        )
        agent_list.append(agent)
        if sequence:
            wait += 1

    return agent_list


def run_simul(
        cfg,
        network_structure,
        prompt_init,
        prompt_update,
        personality_list,
        llm_backend=False,
        model=None,
        sampling_params=None,
        progress_callback=None,
    ):
    """Run the simulation.

    :param cfg: ExperimentConfig providing the scalar simulation params
        (n_agents, n_timesteps, access_url, debug, instruct, temperature,
        verbose, output)
    :param network_structure: the built networkx graph
    :param prompt_init: resolved initial prompt text
    :param prompt_update: resolved update prompt text
    :param personality_list: resolved personality texts (one per agent)
    :param llm_backend: loaded backend tag ("vllm"/"llama.cpp") or False
    :param model: loaded model instance or None
    :param sampling_params: optional sampling params
    :param progress_callback: optional callback(current_step, total_steps)
    :return: stories_history
    """
    # storage for the stories
    stories_history = []

    # initialize the agents
    agent_list = init_agents(
        cfg,
        network_structure,
        prompt_init,
        prompt_update,
        personality_list,
        llm_backend=llm_backend,
        model=model,
        sampling_params=sampling_params,
    )

    for agent in agent_list:
        agent.update_neighbours(network_structure, agent_list)

    # set the path to store the state history
    if cfg.output is None:
        state_history_path = 'results/state_history.json'
    else:
        state_history_path = f'{cfg.output}/state_history.json'

    # run the simulation
    for t in range(cfg.n_timesteps):
        new_stories = update_step(agent_list, t, state_history_path, verbose=cfg.verbose)
        print(f'\nTimestep: {t}')
        print(f'Number of new_stories: {len(new_stories)}')
        stories_history.append(new_stories)
        if progress_callback is not None:
            progress_callback(t + 1, cfg.n_timesteps)

    return stories_history


def update_step(
        agent_list, 
        timestep, 
        state_history_path,
        verbose=False
    ):
    """Update the agents

    :param agent_list: list of agents
    :param timestep: timestep
    :param state_history_path: path to store the state history
    :param verbose: if True, print each agent's generated story text, defaults to False
    :return: new_stories
    """
    # update the prompt of the agents
    new_stories = []

    for agent in agent_list:
        agent.update_prompt()

    for agent in agent_list:
        print(f'Agent: {agent.agent_id}')
        story = agent.get_updated_story()
        if story is not None:
            new_stories.append(story)
            if verbose:
                indented = "\n".join("    " + line for line in str(story).splitlines())
                print(f'  [agent {agent.agent_id}] story:')
                print(indented)

    return new_stories




def register_entry(json_path, name, prompt):
    '''Add {name, prompt} to a parameter JSON file if not already present (matching the framework's schema).'''
    with open(json_path, "r") as f:
        data = json.load(f)
    if not any(d["name"] == name for d in data):
        data.append({"name": name, "prompt": prompt})
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Registered new entry '{name}' in {json_path.name}")
    return name


# def build_network_structure(structure, n_agents, n_cliques=2):
#     '''Build the networkx graph for a given topology name (mirrors scripts/run_simulation.py).'''
#     sequence = False
#     if structure == "sequence":
#         g = nx.DiGraph()
#         for i in range(n_agents - 1):
#             g.add_edge(i, i + 1)
#         sequence = True
#     elif structure == "circle":
#         g = nx.cycle_graph(n_agents)
#     elif structure == "caveman":
#         g = nx.connected_caveman_graph(int(n_cliques), n_agents // int(n_cliques))
#     elif structure == "fully_connected":
#         g = nx.complete_graph(n_agents)
#     else:
#         raise ValueError(f"Unknown network_structure: {structure!r}")
#     return g, sequence

# More flexible version of build_network_structure that allows for custom topologies:
def register_custom_network_structure(name, graph, replace = True):
    '''Register a custom network structure in the framework's own parameter files.'''
    with open(PARAMS_DIR / "network_structures.json", "r") as f:
        data = json.load(f)



    if not any(d["name"] == name for d in data):
        data.append({"name": name, "n_agents": graph.number_of_nodes(), "adjacency": nx.to_dict_of_lists(graph)})
        with open(PARAMS_DIR / "network_structures.json", "w") as f:
            json.dump(data, f, indent=4)
    else:
        if replace:
            for d in data:
                if d["name"] == name:
                    d["n_agents"] = graph.number_of_nodes()
                    d["adjacency"] = nx.to_dict_of_lists(graph)
            with open(PARAMS_DIR / "network_structures.json", "w") as f:
                json.dump(data, f, indent=4)
        
    return name

def load_named_prompt(json_path, name):
    """Return the 'prompt' text for the entry named `name` in a parameter JSON file.

    :param json_path: path to a parameter file (list of {name, prompt} dicts)
    :param name: the registered entry name to look up
    :return: the prompt string
    :raises KeyError: if no entry with that name exists
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    for d in data:
        if d['name'] == name:
            return d['prompt']
    raise KeyError(f"No entry named {name!r} in {json_path}")


def load_personalities(json_path, names):
    """Return the prompt texts for each personality name, in order.

    :param json_path: path to the personalities parameter file
    :param names: iterable of registered personality names (one per agent)
    :return: list of prompt strings, same length/order as `names`
    :raises KeyError: if any name is missing
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    by_name = {d['name']: d['prompt'] for d in data}
    resolved = []
    for name in names:
        if name not in by_name:
            raise KeyError(f"No personality named {name!r} in {json_path}")
        resolved.append(by_name[name])
    return resolved


def build_network_structure(structure, n_agents, n_cliques=2):
    '''Build the networkx graph for a given topology name (mirrors scripts/run_simulation.py).'''
    sequence = False
    if structure == "sequence":
        g = nx.DiGraph()
        for i in range(n_agents - 1):
            g.add_edge(i, i + 1)
        sequence = True
    elif structure == "circle":
        g = nx.cycle_graph(n_agents)
    elif structure == "caveman":
        g = nx.connected_caveman_graph(int(n_cliques), n_agents // int(n_cliques))
    elif structure == "fully_connected":
        g = nx.complete_graph(n_agents)
    else:
        # Check if the structure is a registered custom network structure
        with open(PARAMS_DIR / "network_structures.json", "r") as f:
            custom_structures = json.load(f)
        custom_structure = next((d for d in custom_structures if d["name"] == structure), None)
        if custom_structure is not None:

            adjacency = {
                int(k): v
                for k, v in custom_structure["adjacency"].items()
            }
            g = nx.from_dict_of_lists(adjacency)

            #show graph

        else:
            raise ValueError(f"Unknown network_structure: {structure!r}")
    return g, sequence

def run_experiment(config, repo_dir=None, hf_cache_dir=None):
    '''Run a full LLM-Culture simulation (all seeds) for the given CONFIG dict.

    :param config: a CONFIG-shaped dict (see section 4)
    :return: path to the results folder (str)
    '''
    # 1. Register prompts / personalities into the framework's own parameter files
    register_entry(PROMPT_INIT_JSON, config["prompt_init"]["name"], config["prompt_init"]["prompt"])
    register_entry(PROMPT_UPDATE_JSON, config["prompt_update"]["name"], config["prompt_update"]["prompt"])
    for p in config["personalities"]:
        register_entry(PERSONALITIES_JSON, p["name"], p["prompt"])

    # 2. Build the network
    print(f"Building network structure '{config['network_structure']}' with {config['n_agents']} agents...")
    network_structure, sequence = build_network_structure(
        config["network_structure"], config["n_agents"], config.get("n_cliques", 2)
    )
    adjacency_matrix = nx.to_numpy_array(network_structure).tolist()

    prompt_init_text = config["prompt_init"]["prompt"]
    prompt_update_text = config["prompt_update"]["prompt"]
    personality_texts = [p["prompt"] for p in config["personalities"]]

    # 3. Prepare output folder: results/experiments/<output_name>/
    output_folder = Path("results", "experiments", config["output_name"])
    output_folder.mkdir(parents=True, exist_ok=True)

    model = None
    if config.get("llm_backend") == "vllm":
        import vllm
        model = vllm.LLM(model=config.get("model"), n_gpus=-1)
    elif config.get("llm_backend") == "llama.cpp":
        resolved_model_path = resolve_model_path(
            config.get("model"),
            hf_cache_dir or os.path.expanduser("~/.cache/huggingface"),
        )
        from llama_cpp import Llama
        # Start the llama.cpp server if not already running
        model = Llama(
            model_path=str(resolved_model_path),
            n_ctx=4096,
            n_gpu_layers=-1,  
            verbose=False,
        )

    # 4. Run the simulation for each seed using the framework's own run_simul()
    for seed in range(config["n_seeds"]):
        print(f"\n=== Seed {seed} ===")
        from llm_culture.config import ExperimentConfig

        cfg = ExperimentConfig(
            n_agents=config["n_agents"],
            n_timesteps=config["n_timesteps"],
            access_url=config["access_url"],
            debug=config.get("debug", False),
            temperature=config.get("temperature", 0.8),
            output=str(output_folder),
        )
        stories = run_simul(
            cfg,
            network_structure,
            prompt_init_text,
            prompt_update_text,
            personality_texts,
            llm_backend=config.get("llm_backend", False),
            model=model,
            sampling_params=config.get("sampling_params", None),
        )
        output_dict = {
            "adjacency_matrix": adjacency_matrix,
            "prompt_init": [prompt_init_text],
            "prompt_update": [prompt_update_text],
            "personality_list": personality_texts,
            "stories": stories,
        }
        with open(output_folder / f"output{seed}.json", "w") as f:
            json.dump(output_dict, f, indent=4)

    print(f"\nSaved results to {output_folder}")
    return str(output_folder)


def resolve_model_path(model, hf_cache_dir):
    """
    :param model: a local path to model weights, OR a Hugging Face repo id to download.
    :param hf_cache_dir: directory to download into / read the cache from (the HF_CACHE setting).
    :return: a local filesystem path usable as run_simul(..., model=<this>).
    """
    if os.path.exists(model):
        print(f"Using local model at {model}")
        return model

    models_dir_path = Path("models") / model
    if os.path.exists(models_dir_path):
        print(f"Using local model at {models_dir_path}")
        return models_dir_path
 
    from huggingface_hub import snapshot_download
    print(f"Model '{model}' not found locally — downloading into {hf_cache_dir} (cached after first run)...",
          flush=True)
    local_path = snapshot_download(repo_id=model, cache_dir=hf_cache_dir)
    print(f"Model available at {local_path}")

    local_path = Path(local_path)
    if local_path.is_dir():
        gguf_files = sorted(local_path.rglob("*.gguf"))
        if len(gguf_files) == 0:
            raise FileNotFoundError(
                f"No .gguf model file found in downloaded snapshot: {local_path}"
            )

        selected_gguf = next(
            (model_file for model_file in gguf_files if "q4_k_m" in model_file.name.lower()),
            gguf_files[0],
        )

        if len(gguf_files) > 1:
            print(
                f"Multiple .gguf model files found in {local_path}; using {selected_gguf.name}",
                flush=True,
            )

        return selected_gguf

    return local_path
 