"""Structured configuration for experiments (Hydra-friendly dataclasses).

These dataclasses are the single source of truth for an experiment's parameters.
Hydra registers `ExperimentConfig` as a schema (see run_experiment.py), so a YAML
config is validated against it and converted straight into a typed instance via
`OmegaConf.to_object`. Both entrypoints (run_experiment.py = simulation + analysis,
scripts/run_simulation.py = simulation only) read the same config type.

Layout: related knobs are grouped into small sub-configs
(`population`, `backend`, `generation`, `analysis`); cross-cutting scalars
(`n_timesteps`, `n_seeds`, `seed_offset`, `output`, `verbose`, `debug`) stay at
the top level. The population is described as agent *groups* (AgentConfig), not a
per-agent list, so `population.n_agents` is derived rather than a field to sync.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Backend(Enum):
    """Which LLM backend to run against."""
    none = "none"            # remote OpenAI-compatible server via access_url
    vllm = "vllm"            # in-process vLLM (Linux/GPU)
    llama_cpp = "llama_cpp"  # in-process llama.cpp (CPU/Metal, works on macOS)


class Network(Enum):
    """Agent network topology."""
    sequence = "sequence"
    fully_connected = "fully_connected"
    circle = "circle"
    caveman = "caveman"


@dataclass
class LlamaCppConfig:
    """llama.cpp (llama-cpp-python) model-loading options.

    Each non-None field is forwarded as a keyword to ``llama_cpp.Llama(...)``;
    None means "use llama.cpp's own default". The single most important knob for
    fitting a large model on a *small* GPU is ``n_gpu_layers`` (offload only N of
    the model's transformer layers to the GPU and run the rest on CPU).
    """
    n_gpu_layers: int = -1            # -1 = all layers on GPU; lower to fit a small GPU; 0 = CPU-only
    n_ctx: int = 4096                 # context window in tokens; bigger => more KV-cache memory
    n_batch: Optional[int] = None     # prompt (prefill) batch size
    n_threads: Optional[int] = None   # CPU threads for generation (defaults to physical cores)
    n_threads_batch: Optional[int] = None  # CPU threads for prompt batching
    use_mmap: Optional[bool] = None   # memory-map weights from disk (default True)
    use_mlock: Optional[bool] = None  # lock weights in RAM so they are never swapped out
    main_gpu: Optional[int] = None    # index of the GPU to use for single-GPU operations
    flash_attn: Optional[bool] = None # enable flash attention (reduces KV-cache memory) if built with it
    gguf_filename: Optional[str] = None  # substring to pick a specific quant file, e.g. "q3_k_m"
    verbose: bool = False


@dataclass
class VllmConfig:
    """vLLM engine options.

    Each non-None field is forwarded as a keyword to ``vllm.LLM(...)``; None means
    "use vLLM's own default". For a small GPU the levers that matter most are
    ``gpu_memory_utilization``, ``max_model_len`` (caps the KV cache),
    ``quantization``/``dtype``, and ``cpu_offload_gb``.
    """
    gpu_memory_utilization: Optional[float] = None  # fraction of VRAM vLLM may use (default 0.9)
    max_model_len: Optional[int] = None    # cap context length -> smaller KV cache
    tensor_parallel_size: Optional[int] = None  # shard the model across N GPUs
    dtype: Optional[str] = None            # "auto" | "float16" | "bfloat16" | "float32"
    quantization: Optional[str] = None     # "awq" | "gptq" | "fp8" | ... (loads a quantized checkpoint)
    kv_cache_dtype: Optional[str] = None   # "auto" | "fp8" -> shrink the KV cache
    cpu_offload_gb: Optional[float] = None # offload this many GB of weights to CPU RAM
    swap_space: Optional[int] = None       # CPU swap space (GiB) per GPU for KV-cache spill
    enforce_eager: Optional[bool] = None   # disable CUDA graphs: less VRAM, slower
    max_num_seqs: Optional[int] = None     # max sequences batched concurrently


@dataclass
class BackendConfig:
    """Which LLM backend to run against, and how to load the model.

    Only the sub-config matching ``kind`` is used:
      * ``kind=none``      -> talk to a remote OpenAI-compatible server at ``access_url``
                             (no model is loaded in-process; ``llama_cpp`` / ``vllm`` ignored).
      * ``kind=llama_cpp`` -> load ``model`` with llama.cpp, tuned by ``llama_cpp.*``.
      * ``kind=vllm``      -> load ``model`` with vLLM, tuned by ``vllm.*``.
    """
    kind: Backend = Backend.none      # none (remote server) | llama_cpp | vllm
    model: Optional[str] = None       # HF repo id or local path (required for llama_cpp/vllm)
    access_url: str = ""              # remote server URL (used when kind == none)
    hf_cache_dir: Optional[str] = None  # HF download cache (default ~/.cache/huggingface)
    # backend-specific tuning (only the one matching `kind` is read)
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)


@dataclass
class GenerationConfig:
    """How each story is generated: the sampling knobs plus the request style."""
    temperature: float = 0.8
    max_tokens: int = 512
    top_p: float = 0.95
    instruct: bool = True   # True -> chat/instruct API; False -> raw text completion (base models)


@dataclass
class EmbeddingConfig:
    """How stories are vectorized for the story-similarity matrix. "tfidf" (default)
    is reproducible + dependency-light; "huggingface" needs the [embeddings] extra.
    Switching method/model uses a separate analysis cache."""
    method: str = "tfidf"                                  # "tfidf" | "huggingface"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # HF model id (huggingface only)
    batch_size: int = 32                                   # encode batch size (huggingface only)
    device: Optional[str] = None                           # torch device, or None -> library picks


@dataclass
class AnalysisConfig:
    """Post-simulation analysis options."""
    run: bool = True            # False -> simulate only, skip plots
    plot: bool = False          # also open figures interactively (saved either way)
    recompute_cache: bool = True
    font_sizes: Dict[str, int] = field(
        default_factory=lambda: {"ticks": 12, "labels": 14, "title": 16}
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class AnalysisRunConfig:
    """Config for analysing an existing results folder (scripts/run_analysis.py)."""
    folder: str = "results/default_folder"  # results folder to analyze (full or repo-relative)
    plot: bool = False                       # also open figures interactively (saved either way)
    recompute_cache: bool = True             # ignore any existing cache and recompute
    cache_file: Optional[str] = None         # override cache filename (None -> derived from embedding)
    font_sizes: Dict[str, int] = field(
        default_factory=lambda: {"ticks": 12, "labels": 14, "title": 16}
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class ComparisonConfig:
    """Config for comparing several results folders (scripts/run_comparison_analysis.py)."""
    folders: List[str] = field(default_factory=list)  # entries joined under `root`
    root: str = "results/experiments"                 # prefix for each entry ("" -> none)
    labels: Optional[List[str]] = None                # legend labels (None -> folder basenames)
    plot: bool = False                                 # also open figures interactively
    scale_y_axis: bool = False                         # shared y-axis scale across folders
    sizes: Dict[str, int] = field(
        default_factory=lambda: {"ticks": 16, "labels": 18, "legend": 16, "title": 23, "matrix": 8}
    )
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class AgentConfig:
    """One agent *type*: how many agents share it, plus the persona and prompts
    that define how they write / rewrite stories.

    A population (see PopulationConfig) is an ordered list of these groups. The
    agents are laid out group by group onto network node indices 0..n-1, so e.g.
    ``[AgentConfig(count=3, personality="Fantasy"), AgentConfig(count=2,
    personality="SciFi")]`` places 3 Fantasy agents (nodes 0-2) then 2 SciFi
    agents (nodes 3-4). The common homogeneous case is a single group whose
    ``count`` is the population size.

    ``personality`` / ``prompt_init`` / ``prompt_update`` are registered *names*
    looked up in llm_culture/data/parameters/{personalities,prompt_init,
    prompt_update}.json.
    """
    count: int = 2                 # how many agents of this type
    personality: str = "Empty"     # persona name (personalities.json)
    prompt_init: str = "kid"       # seed-story prompt, used on an agent's first (cold) step
    prompt_update: str = "kid"     # transformation instruction, used once it has neighbour stories


@dataclass
class PopulationConfig:
    """The agents and how they are wired together.

    ``n_agents`` is **derived** (``sum`` of the group counts), so there is no
    separate agent-count field to keep in sync with a per-agent list.
    """
    network_structure: Network = Network.sequence
    n_cliques: int = 2  # only used when network_structure == caveman
    agents: List[AgentConfig] = field(default_factory=lambda: [AgentConfig()])

    @property
    def n_agents(self) -> int:
        """Total number of agents = sum of each group's count."""
        return sum(group.count for group in self.agents)


@dataclass
class ExperimentConfig:
    # ---- Who runs + how they're wired + the task prompts ----
    population: PopulationConfig = field(default_factory=PopulationConfig)
    # ---- Where the model runs ----
    backend: BackendConfig = field(default_factory=BackendConfig)
    # ---- How text is generated ----
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # ---- Post-run analysis ----
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # ---- Schedule / repetition (cross-cutting scalars, kept flat) ----
    n_timesteps: int = 2    # number of generations. For a `sequence` network this MUST
                            # equal population.n_agents (one generation per agent); see
                            # validate_experiment.
    n_seeds: int = 2        # independent repeats of the whole run
    seed_offset: int = 0    # start index for output{i}.json naming (parallel seed runs)

    # ---- Output / logging ----
    output: str = "results/default_folder"
    verbose: bool = False   # print each agent's generated story text
    debug: bool = False


def validate_experiment(cfg: ExperimentConfig) -> None:
    """Validate cross-field constraints the schema can't express.

    Raises ValueError with a clear, actionable message when the config is
    inconsistent.
    """
    backend = cfg.backend
    if backend.kind in (Backend.vllm, Backend.llama_cpp) and not backend.model:
        raise ValueError(
            f"backend.kind={backend.kind.value} requires `backend.model` to be set "
            "(a Hugging Face repo id or a local model path)."
        )
    if backend.kind == Backend.none and not backend.access_url:
        raise ValueError(
            "backend.kind=none means 'send requests to a remote OpenAI-compatible "
            "server', but `backend.access_url` is empty — there is nothing to call. "
            "The bare default config has no LLM wired up on purpose; pick one:\n"
            "  • local smoke test (recommended first run):\n"
            "      uv run python run_experiment.py experiment=base\n"
            "  • a local model via llama.cpp:\n"
            "      uv run python run_experiment.py backend.kind=llama_cpp "
            "backend.model=unsloth/SmolLM2-135M-Instruct-GGUF\n"
            "  • a remote OpenAI-compatible server:\n"
            "      uv run python run_experiment.py backend.access_url=http://localhost:8000"
        )

    pop = cfg.population
    if pop.n_agents < 1:
        raise ValueError(
            "population has no agents — the agent group counts sum to "
            f"{pop.n_agents}. Add at least one AgentConfig with count >= 1."
        )
    if pop.network_structure == Network.caveman and pop.n_cliques <= 0:
        raise ValueError("network_structure=caveman requires n_cliques > 0.")

    # A `sequence` network is a transmission chain: agent i generates exactly once,
    # at timestep i, so the number of generations is fixed by the agent count.
    if pop.network_structure == Network.sequence and cfg.n_timesteps != pop.n_agents:
        raise ValueError(
            "network_structure=sequence is a transmission chain: it runs exactly one "
            f"generation per agent, so n_timesteps must equal the number of agents "
            f"({pop.n_agents}). Got n_timesteps={cfg.n_timesteps}. "
            f"Set n_timesteps={pop.n_agents} (or change the agent counts), or use a "
            "non-sequence network (fully_connected / circle / caveman) where the two "
            "are independent."
        )
