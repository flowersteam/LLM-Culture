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
from typing import List, Optional


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
    temperature: float = 0.8
    instruct: bool = True   # False -> raw completion mode
    verbose: bool = False   # print each agent's generated story text

    # ---- LLM backend ----
    backend: Backend = Backend.none
    model: Optional[str] = None       # HF repo id or local path (required for vllm/llama_cpp)
    access_url: str = ""              # server URL (used when backend == none)
    hf_cache_dir: Optional[str] = None

    # ---- Output ----
    output: str = "results/default_folder"
    debug: bool = False

    # ---- Analysis ----
    run_analysis: bool = True   # False -> simulate only, skip plots
    plot: bool = False          # also open figures interactively (saved either way)
    recompute_cache: bool = True
    ticks_font_size: int = 12
    labels_font_size: int = 14
    title_font_size: int = 16


def validate_experiment(cfg: ExperimentConfig) -> None:
    """Validate cross-field constraints the schema can't express.

    Raises ValueError with a clear message when the config is inconsistent.
    """
    if cfg.backend in (Backend.vllm, Backend.llama_cpp) and not cfg.model:
        raise ValueError(
            f"backend={cfg.backend.value} requires `model` to be set "
            "(a Hugging Face repo id or a local model path)."
        )
    if len(cfg.personality_list) != cfg.n_agents:
        raise ValueError(
            f"personality_list has {len(cfg.personality_list)} entrie(s) but "
            f"n_agents={cfg.n_agents}; provide exactly one persona per agent."
        )
    if cfg.network_structure == Network.caveman and cfg.n_cliques <= 0:
        raise ValueError("network_structure=caveman requires n_cliques > 0.")
