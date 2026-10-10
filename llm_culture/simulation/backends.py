"""LLM backend selection + model loading.

Shared helper so every entrypoint (the Hydra runner, the standalone CLI, and the
web interface) loads models the same way. The in-process backends (vLLM,
llama.cpp) are imported lazily so that running against a remote server pulls in
neither heavy dependency.

Backend-specific tuning lives in the nested sub-configs on ExperimentConfig
(cfg.llama_cpp / cfg.vllm): every field that is not None is forwarded as a
keyword argument to the backend constructor, so you can tune memory/offload
(e.g. llama_cpp.n_gpu_layers, vllm.gpu_memory_utilization) without touching code.
"""
import os
from dataclasses import asdict

from llm_culture.config import Backend
from llm_culture.simulation.utils import resolve_model_path


def _forward_kwargs(sub_config, drop=()):
    """Turn a backend sub-config dataclass into kwargs, dropping None (= use the
    backend's own default) and any explicitly excluded keys."""
    return {
        key: value
        for key, value in asdict(sub_config).items()
        if value is not None and key not in drop
    }


def load_llm_backend(cfg):
    """Load the model for the backend requested in `cfg`.

    :param cfg: an ExperimentConfig (uses cfg.backend.kind / .model / .hf_cache_dir
        and the matching cfg.backend.llama_cpp / cfg.backend.vllm sub-config)
    :return: (llm_backend_tag, model) where the tag is the string "vllm" /
        "llama.cpp" that get_answer expects, or False for the remote-server path
        (in which case model is None).
    """
    backend = cfg.backend

    if backend.kind == Backend.vllm:
        # Deferred import: vllm is an optional ("serving") extra, absent in base installs.
        from vllm import LLM

        vllm_kwargs = _forward_kwargs(backend.vllm)
        return "vllm", LLM(model=backend.model, **vllm_kwargs)

    if backend.kind == Backend.llama_cpp:
        # Deferred import: llama_cpp is an optional ("serving") extra, absent in base installs.
        from llama_cpp import Llama

        resolved_model_path = resolve_model_path(
            backend.model,
            backend.hf_cache_dir or os.path.expanduser("~/.cache/huggingface"),
            gguf_filename=backend.llama_cpp.gguf_filename,
        )
        # gguf_filename is only used to pick the file above, not a Llama() kwarg.
        llama_kwargs = _forward_kwargs(backend.llama_cpp, drop=("gguf_filename",))
        model = Llama(model_path=str(resolved_model_path), **llama_kwargs)
        return "llama.cpp", model

    # Backend.none -> remote OpenAI-compatible server via cfg.backend.access_url.
    return False, None
