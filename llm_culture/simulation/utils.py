import os
import json
import textwrap
from pathlib import Path

import networkx as nx
from huggingface_hub import snapshot_download

from llm_culture.simulation.agent import Agent
from llm_culture.simulation.server_answer import get_answers_batch
from llm_culture.config import (
    ExperimentConfig,
    PopulationConfig,
    AgentConfig,
    BackendConfig,
    Backend,
    Network,
    GenerationConfig,
)
from llm_culture.paths import PARAMS_DIR, PROMPT_INIT_JSON, PROMPT_UPDATE_JSON, PERSONALITIES_JSON


def init_agents(
        cfg,
        network_structure,
        agent_specs,
        llm_backend=False,
        model=None,
        sampling_params=None,
    ):
    """Initialize the agents from an ExperimentConfig and the runtime objects.

    :param cfg: ExperimentConfig (provides the shared scalars via Agent:
        backend.access_url, debug, generation)
    :param network_structure: the built networkx graph (directed => sequence mode)
    :param agent_specs: ordered list of per-agent
        ``(personality_text, prompt_init_text, prompt_update_text)`` triples,
        one per agent (length == population.n_agents)
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

    for agent_id, (personality, prompt_init, prompt_update) in enumerate(agent_specs):
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
        agent_specs,
        llm_backend=False,
        model=None,
        sampling_params=None,
        progress_callback=None,
    ):
    """Run the simulation.

    :param cfg: ExperimentConfig providing the scalar simulation params
        (n_timesteps, verbose) and the shared Agent scalars
    :param network_structure: the built networkx graph
    :param agent_specs: ordered list of per-agent
        ``(personality_text, prompt_init_text, prompt_update_text)`` triples
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
        agent_specs,
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
        new_stories = update_step(
            agent_list,
            verbose=cfg.verbose,
            batch=cfg.generation.batch,
            max_concurrent_requests=cfg.backend.max_concurrent_requests,
        )
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
        verbose=False,
        batch=True,
        max_concurrent_requests=8,
    ):
    """Advance every agent one step: build all prompts, then generate, then store.

    Prompts are built from the previous step's stories before any generation, so the
    eligible agents are independent; ``batch`` only changes how they're run (together
    vs one-by-one), not the result.

    :param agent_list: list of agents
    :param verbose: if True, print each agent's generated story text
    :param batch: generate the step's eligible agents together (vs one-by-one)
    :param max_concurrent_requests: cap for the remote backend's concurrent requests
    :return: new_stories (in agent order)
    """
    # Phase A — every agent (re)builds its prompt from neighbours' previous stories.
    # A waiting agent (e.g. not yet reached in a sequence chain) gets prompt = None.
    for agent in agent_list:
        agent.update_prompt()

    eligible = [agent for agent in agent_list if agent.prompt is not None]
    prompts = [agent.prompt for agent in eligible]

    # Phase B — generate. All eligible agents share one backend/model/params (built
    # from a single config), so read those from a representative agent.
    if prompts:
        ref = eligible[0]
        kwargs = dict(
            access_url=ref.access_url,
            debug=ref.debug,
            instruct=ref.instruct,
            llm_backend=ref.llm_backend,
            model=ref.model,
            sampling_params=ref.sampling_params,
        )
        if batch:
            results = get_answers_batch(
                prompts, ref.generation,
                max_concurrent_requests=max_concurrent_requests, **kwargs,
            )
        else:
            # One at a time, through the same path (single-element batches).
            results = [
                get_answers_batch([p], ref.generation,
                                  max_concurrent_requests=1, **kwargs)[0]
                for p in prompts
            ]
        for agent, text in zip(eligible, results):
            agent.set_story(text)

    # Waiting agents produce nothing; then EVERY agent's wait ticks down exactly
    # once (the same bookkeeping the old per-agent update_story did).
    for agent in agent_list:
        if agent.prompt is None:
            agent.set_story(None)
        agent.decrease_wait()

    # Collect + log in agent order (output format unchanged).
    new_stories = []
    for agent in agent_list:
        story = agent.get_story()
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

    # Build the experiment config once (shared across all seeds) and load the model
    # through the SAME shared helper every other entrypoint uses. This replaces the
    # old hand-rolled loader and fixes its `vllm.LLM(..., n_gpus=-1)` call (`n_gpus`
    # is not a vLLM kwarg) and the undefined `hf_cache_dir` reference in it.
    backend_tags = {"vllm": Backend.vllm, "llama.cpp": Backend.llama_cpp}
    cfg = ExperimentConfig(
        population=PopulationConfig(
            network_structure=Network(config["network_structure"]),
            n_cliques=config.get("n_cliques", 2),
            agents=[AgentConfig(count=1) for _ in range(config["n_agents"])],
        ),
        backend=BackendConfig(
            kind=backend_tags.get(config.get("llm_backend"), Backend.none),
            model=config.get("model"),
            access_url=config["access_url"] or "",
            hf_cache_dir=config.get("hf_cache_dir"),
        ),
        generation=GenerationConfig(temperature=config.get("temperature", 0.8)),
        n_timesteps=config["n_timesteps"],
        output=str(output_folder),
        debug=config.get("debug", False),
    )
    # Deferred import avoids a circular import (backends imports resolve_model_path
    # from this module). Returns the ("vllm" / "llama.cpp" / False) tag run_simul wants.
    from llm_culture.simulation.backends import load_llm_backend
    llm_backend, model = load_llm_backend(cfg)

    # 4. Run the simulation for each seed using the framework's own run_simul().
    #    One agent per personality entry; each gets the shared init/update prompts.
    agent_specs = [
        (persona_text, prompt_init_text, prompt_update_text)
        for persona_text in personality_texts
    ]
    for seed in range(config["n_seeds"]):
        print(f"\n=== Seed {seed} ===")
        stories = run_simul(
            cfg,
            network_structure,
            agent_specs,
            llm_backend=llm_backend,
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
    # A GGUF repo (e.g. TheBloke/*-GGUF) holds ONE model re-encoded at ~15 different
    # quantization levels, each a SEPARATE .gguf file — there is no fp16 base here.
    # Downloading the whole repo pulls every quant (tens of GB). Restrict the fetch
    # to just the quant we want via allow_patterns; `gguf_filename` selects it
    # (default: q4_k_m). fnmatch is case-sensitive, so cover upper/lower spellings
    # (Q4_K_M vs q4_k_m).
    quant_token = gguf_filename or "q4_k_m"
    allow_patterns = sorted({
        f"*{quant_token}*.gguf",
        f"*{quant_token.lower()}*.gguf",
        f"*{quant_token.upper()}*.gguf",
    })
    print(f"  (fetching only files matching {allow_patterns})", flush=True)
    local_path = snapshot_download(
        repo_id=model, cache_dir=hf_cache_dir, allow_patterns=allow_patterns
    )
    print(f"Model available at {local_path}")

    local_path = Path(local_path)
    if local_path.is_dir():
        gguf_files = sorted(local_path.rglob("*.gguf"))
        if len(gguf_files) == 0:
            try:
                from huggingface_hub import HfApi
                available = sorted(
                    f for f in HfApi().list_repo_files(model)
                    if f.lower().endswith(".gguf")
                )
            except Exception:
                available = []
            hint = (
                f"Available .gguf files in {model}: {available}. "
                "Set llama_cpp.gguf_filename to a tag that appears in one of them."
                if available else
                "Could not list the repo's files; check the repo id and "
                "llama_cpp.gguf_filename."
            )
            raise FileNotFoundError(
                f"No .gguf file matched quant {quant_token!r} after a filtered "
                f"download of {model} (allow_patterns={allow_patterns}). " + hint
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
 