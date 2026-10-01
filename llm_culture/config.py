"""Structured configuration for experiments (Hydra-friendly dataclasses).

These dataclasses are the single source of truth for an experiment's parameters.
Hydra registers `ExperimentConfig` as a schema (see run_experiment.py), so a YAML
config is validated against it and converted straight into a typed instance via
`OmegaConf.to_object`. The standalone argparse CLI in scripts/run_simulation.py
also builds an `ExperimentConfig`, so both entrypoints share one config type.

Flat layout for now (every field at the top level); grouping into nested
sub-configs can come later.
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
class GenerationConfig:
    """LLM text-generation sampling options (shared by every backend)."""
    temperature: float = 0.8
    max_tokens: int = 512
    top_p: float = 0.95


@dataclass
class AnalysisConfig:
    """Post-simulation analysis options."""
    run: bool = True            # False -> simulate only, skip plots
    plot: bool = False          # also open figures interactively (saved either way)
    recompute_cache: bool = True
    font_sizes: Dict[str, int] = field(
        default_factory=lambda: {"ticks": 12, "labels": 14, "title": 16}
    )


@dataclass
class ExperimentConfig:
    # ---- Simulation ----
    n_agents: int = 2
    n_timesteps: int = 2
    n_seeds: int = 2
    seed_offset: int = 0
    network_structure: Network = Network.sequence
    n_cliques: int = 2  # only used when network_structure == caveman
    prompt_init: str = "kid"
    prompt_update: str = "kid"
    personality_list: List[str] = field(default_factory=lambda: ["Empty", "Empty"])
    instruct: bool = True   # False -> raw completion mode
    verbose: bool = False   # print each agent's generated story text

    # ---- LLM backend ----
    backend: Backend = Backend.none
    model: Optional[str] = None       # HF repo id or local path (required for vllm/llama_cpp)
    access_url: str = ""              # server URL (used when backend == none)
    hf_cache_dir: Optional[str] = None
    # backend-specific tuning (only the sub-config matching `backend` is used)
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)

    # ---- Generation (sampling) ----
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    # ---- Output ----
    output: str = "results/default_folder"
    debug: bool = False

    # ---- Analysis ----
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def validate_experiment(cfg: ExperimentConfig) -> None:
    """Validate cross-field constraints the schema can't express.

    Raises ValueError with a clear message when the config is inconsistent.
    """
    if cfg.backend in (Backend.vllm, Backend.llama_cpp) and not cfg.model:
        raise ValueError(
            f"backend={cfg.backend.value} requires `model` to be set "
            "(a Hugging Face repo id or a local model path)."
        )
    if cfg.backend == Backend.none and not cfg.access_url:
        raise ValueError(
            "backend=none means 'send requests to a remote OpenAI-compatible "
            "server', but `access_url` is empty — there is nothing to call. "
            "The bare default config has no LLM wired up on purpose; pick one:\n"
            "  • local smoke test (recommended first run):\n"
            "      uv run python run_experiment.py experiment=base\n"
            "  • a local model via llama.cpp:\n"
            "      uv run python run_experiment.py backend=llama_cpp "
            "model=unsloth/SmolLM2-135M-Instruct-GGUF\n"
            "  • a remote OpenAI-compatible server:\n"
            "      uv run python run_experiment.py access_url=http://localhost:8000"
        )
    if len(cfg.personality_list) != cfg.n_agents:
        raise ValueError(
            f"personality_list has {len(cfg.personality_list)} entrie(s) but "
            f"n_agents={cfg.n_agents}; provide exactly one persona per agent."
        )
    if cfg.network_structure == Network.caveman and cfg.n_cliques <= 0:
        raise ValueError("network_structure=caveman requires n_cliques > 0.")
