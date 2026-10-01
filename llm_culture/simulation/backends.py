"""LLM backend selection + model loading.

Shared helper so the CLI/Hydra runner (and, later, the web interface) load models
the same way. The in-process backends (vLLM, llama.cpp) are imported lazily so
that running against a remote server pulls in neither heavy dependency.
"""
import os


def load_llm_backend(cfg):
    """Load the model for the backend requested in `cfg`.

    :param cfg: an ExperimentConfig (uses cfg.backend, cfg.model, cfg.hf_cache_dir)
    :return: (llm_backend_tag, model) where the tag is the string "vllm" /
        "llama.cpp" that get_answer expects, or False for the remote-server path
        (in which case model is None).
    """
    from llm_culture.config import Backend

    if cfg.backend == Backend.vllm:
        from vllm import LLM

        return "vllm", LLM(model=cfg.model)

    if cfg.backend == Backend.llama_cpp:
        # Imported here to avoid a hard dependency on utils at module import time.
        from llm_culture.simulation.utils import resolve_model_path
        from llama_cpp import Llama

        resolved_model_path = resolve_model_path(
            cfg.model,
            cfg.hf_cache_dir or os.path.expanduser("~/.cache/huggingface"),
        )
        model = Llama(
            model_path=str(resolved_model_path),
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        return "llama.cpp", model

    # Backend.none -> remote OpenAI-compatible server via cfg.access_url.
    return False, None
