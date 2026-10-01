import os
import json
import textwrap
from pathlib import Path

import networkx as nx
from huggingface_hub import snapshot_download

from llm_culture.simulation.agent import Agent
from llm_culture.config import ExperimentConfig, GenerationConfig
from llm_culture.paths import PARAMS_DIR, PROMPT_INIT_JSON, PROMPT_UPDATE_JSON, PERSONALITIES_JSON


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

    # run the simulation
    for t in range(cfg.n_timesteps):
        # Header BEFORE the agents act, so the log reads top-to-bottom in order.
        print(f"\n┌─ Timestep {t + 1}/{cfg.n_timesteps} " + "─" * 46)
        new_stories = update_step(agent_list, verbose=cfg.verbose)
        plural = "story" if len(new_stories) == 1 else "stories"
        print(f"└─ {len(new_stories)} new {plural} this timestep " + "─" * 34)
        stories_history.append(new_stories)
        if progress_callback is not None:
            progress_callback(t + 1, cfg.n_timesteps)

    return stories_history


def _print_story_block(text, width=88):
    """Pretty-print a story as an indented, word-wrapped box so long generations
    stay readable in the terminal (paragraph breaks preserved)."""
    indent = "   │  "
    print("   ┌─ story " + "─" * (width - 9))
    for paragraph in text.splitlines():
        if paragraph.strip() == "":
            print("   │")
            continue
        for line in textwrap.wrap(paragraph, width=width):
            print(f"{indent}{line}")
    print("   └" + "─" * (width + 1))


def update_step(
        agent_list,
        verbose=False
    ):
    """Update the agents

    :param agent_list: list of agents
    :param verbose: if True, print each agent's generated story text, defaults to False
    :return: new_stories
    """
    # update the prompt of the agents
    new_stories = []

    for agent in agent_list:
        agent.update_prompt()

    for agent in agent_list:
        story = agent.get_updated_story()
        if story is None:
            # Agent produced nothing this step. In a sequence / transmission-chain
            # network this is EXPECTED: only the agent whose turn it is generates;
            # the others are still "waiting" down the chain.
            print(f"   Agent {agent.agent_id}: · skipped (waiting its turn)")
            continue

        new_stories.append(story)
        text = str(story).strip()
        print(
            f"   Agent {agent.agent_id}: ✓ generated "
            f"({len(text.split())} words, {len(text)} chars)"
        )
        if verbose:
            _print_story_block(text)

    return new_stories




def log_resources(label=""):
    """Print a one-shot snapshot of CPU / memory (and GPU if available) usage.

    Dependency-free by default (Python stdlib). It opportunistically uses two
    optional upgrades if present, but never requires them:
      * `psutil`      -> live process RSS + system RAM used/available/percent
      * `nvidia-smi`  -> per-GPU VRAM used/total + utilization (NVIDIA/Linux/Colab)

    On Apple Silicon (macOS, Unified Memory Architecture) there is no separate
    VRAM: the Metal GPU shares the one system RAM pool, so the reported process
    RSS / system-RAM figures already account for the model's "GPU" memory. That
    is why there is no nvidia-smi-style GPU line on a Mac — it would be redundant.

    :param label: short tag describing when this snapshot was taken
        (e.g. "after model load", "after simulation").
    """
    import platform
    import shutil
    import subprocess
    import sys

    tag = f" [{label}]" if label else ""
    lines = []

    # ---- CPU ----
    n_cpu = os.cpu_count()
    cpu_line = f"CPU: {n_cpu} logical core(s)"
    if hasattr(os, "getloadavg"):
        try:
            load1, load5, load15 = os.getloadavg()
            cpu_line += f" | load avg (1/5/15m): {load1:.2f} / {load5:.2f} / {load15:.2f}"
        except OSError:
            pass
    lines.append(cpu_line)

    # ---- Memory ----
    proc_rss_gb = None
    sys_total_gb = sys_avail_gb = sys_pct = None
    try:
        import psutil  # optional upgrade

        proc_rss_gb = psutil.Process().memory_info().rss / 1e9
        vm = psutil.virtual_memory()
        sys_total_gb, sys_avail_gb, sys_pct = vm.total / 1e9, vm.available / 1e9, vm.percent
    except Exception:
        # stdlib fallback: peak RSS (resource) + total RAM (sysconf)
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss units differ: bytes on macOS, kilobytes on Linux.
            proc_rss_gb = (ru / 1e9) if sys.platform == "darwin" else (ru * 1024 / 1e9)
        except Exception:
            pass
        try:
            sys_total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except (ValueError, OSError, AttributeError):
            pass

    if proc_rss_gb is not None:
        mem_line = f"Process RAM: {proc_rss_gb:.2f} GB"
        if sys_total_gb is not None:
            mem_line += f" of {sys_total_gb:.1f} GB total"
        if sys_avail_gb is not None:
            mem_line += f" ({sys_avail_gb:.1f} GB free, {sys_pct:.0f}% used system-wide)"
        lines.append(mem_line)
    elif sys_total_gb is not None:
        lines.append(f"System RAM: {sys_total_gb:.1f} GB total")

    # ---- GPU (NVIDIA only, via nvidia-smi) ----
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                for row in out.stdout.strip().splitlines():
                    idx, used, total, util = (c.strip() for c in row.split(","))
                    lines.append(f"GPU {idx}: {used}/{total} MiB VRAM used, {util}% util")
        except Exception:
            pass
    elif platform.system() == "Darwin":
        lines.append("GPU: Apple Metal (unified memory — shares the system RAM above)")

    print(f"\n·· resources{tag} " + "·" * 40)
    for line in lines:
        print(f"   {line}")
    print("·" * 54)


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
        # Deferred import: vllm is an optional ("serving") extra, absent in base installs.
        import vllm
        model = vllm.LLM(model=config.get("model"), n_gpus=-1)
    elif config.get("llm_backend") == "llama.cpp":
        resolved_model_path = resolve_model_path(
            config.get("model"),
            hf_cache_dir or os.path.expanduser("~/.cache/huggingface"),
        )
        # Deferred import: llama_cpp is an optional ("serving") extra, absent in base installs.
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
        cfg = ExperimentConfig(
            n_agents=config["n_agents"],
            n_timesteps=config["n_timesteps"],
            access_url=config["access_url"],
            debug=config.get("debug", False),
            generation=GenerationConfig(temperature=config.get("temperature", 0.8)),
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


def resolve_model_path(model, hf_cache_dir, gguf_filename=None):
    """
    :param model: a local path to model weights, OR a Hugging Face repo id to download.
    :param hf_cache_dir: directory to download into / read the cache from (the HF_CACHE setting).
    :param gguf_filename: optional substring used to pick a specific .gguf quant file
        from a multi-file snapshot (e.g. "q3_k_m"); falls back to the q4_k_m / first file.
    :return: a local filesystem path usable as run_simul(..., model=<this>).
    """
    if os.path.exists(model):
        print(f"Using local model at {model}")
        return model

    models_dir_path = Path("models") / model
    if os.path.exists(models_dir_path):
        print(f"Using local model at {models_dir_path}")
        return models_dir_path
 
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

        selected_gguf = None
        if gguf_filename:
            selected_gguf = next(
                (f for f in gguf_files if gguf_filename.lower() in f.name.lower()),
                None,
            )
            if selected_gguf is None:
                raise FileNotFoundError(
                    f"No .gguf file matching {gguf_filename!r} in {local_path}; "
                    f"available: {[f.name for f in gguf_files]}"
                )

        if selected_gguf is None:
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
 