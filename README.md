# LLM-Culture

A framework for studying the **cultural evolution of text in populations of LLMs**.

Agents are organized into a network; each agent rewrites its neighbours' stories
according to a personality and a transformation prompt. You simulate how texts
evolve across generations, then analyze the results with built-in metrics and plots.

![introduction_figure](/static/introduction_figure.png)

## Installation

```bash
git clone git@github.com:flowersteam/LLM-Culture.git
cd LLM-Culture/
```

This project uses [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync                                            # base install
uv sync --extra serving                            # + local inference backends
uv sync --extra embeddings                         # + HuggingFace embeddings for analysis
```

Prefix commands with `uv run` (e.g. `uv run python run_experiment.py ...`).

### Choosing a backend

| `backend.kind` | Where it runs | Notes |
| --- | --- | --- |
| `none` | — | talk to a remote OpenAI-compatible server via `backend.access_url` |
| `llama_cpp` | CPU / Apple Metal (**macOS + Linux**) | loads a local GGUF model |
| `vllm` | **Linux / GPU only** | skipped automatically on macOS |

For `llama_cpp` / `vllm`, `backend.model` is a local path or a Hugging Face repo id
(downloaded to `~/.cache/huggingface` on first use and cached afterwards). For a
multi-file GGUF repo the loader picks the `q4_k_m` file by default, or set
`backend.llama_cpp.gguf_filename`.

### Generation settings (apply to every backend)

The sampling knobs under `generation.*` are backend-agnostic — the **same** values
are sent whether you run `vllm`, `llama_cpp`, or a remote server, so you tune
generation once and switch backends freely:

| Field | Meaning |
| --- | --- |
| `generation.temperature` | randomness (0 = deterministic, higher = more diverse) |
| `generation.top_p` | nucleus sampling cutoff (keep the top-`p` probability mass) |
| `generation.max_tokens` | max tokens generated per story |
| `generation.instruct` | `true` → chat/instruct API (chat template applied); `false` → raw completion for base models |

```bash
uv run python run_experiment.py \
  generation.temperature=0.9 generation.top_p=0.95 generation.max_tokens=256
```

**Remote servers (`backend.kind=none`).** Point `backend.access_url` at any
OpenAI-compatible endpoint (a hosted API, or a vLLM / llama.cpp server you started
yourself). Two optional environment variables cover authenticated or model-validating
servers:

- `OPENAI_API_KEY` — auth token (defaults to `EMPTY`, which local servers accept;
  set it for a genuine hosted API). Never hardcode it — it is read from the environment.
- `OPENAI_MODEL` — the model name to request, for servers that validate it (local
  servers ignore it and serve whatever they loaded).

`backend.max_concurrent_requests` caps how many requests are sent in parallel when a
timestep is batched (remote only — the server's own batching does the real overlap).

```bash
export OPENAI_API_KEY=sk-...                     # only for a hosted/authenticated server
export OPENAI_MODEL=gpt-4o-mini                  # only if the server validates the name
uv run python run_experiment.py \
  backend.kind=none backend.access_url=https://my-server:8000 \
  backend.max_concurrent_requests=8 \
  generation.temperature=0.9 generation.max_tokens=256
```

# TODO --> also explain how to run exps w a remote open ai model ... give several examples

### Tuning generation & inference for your hardware

The sampling knobs (previous section) decide *what* text you get; the knobs below
decide *whether the model fits and how fast it runs*. They are backend-specific. The
two hardware presets (`mac_mx`, `linux_gpu`) are worked examples — read them alongside
this table and copy the one that matches your machine.

**The one mental model:** a run must hold **weights + KV cache** in memory. Weights are
fixed by the model and its quantization; the **KV cache grows with context length and
with how many sequences run at once**. Almost all tuning is "make it fit" (shrink the KV
cache / offload weights) or "go faster" (batch more sequences together).

**llama.cpp (`backend.kind=llama_cpp`) — CPU / Apple Metal.** Loads a quantized GGUF.

| Knob | What it does | How to pick it |
| --- | --- | --- |
| `llama_cpp.n_gpu_layers` | how many transformer layers run on the GPU | **Apple Silicon: `-1`** (all layers; unified memory shares one RAM pool). Discrete GPU: raise until just before VRAM OOM, rest stay on CPU. |
| `llama_cpp.n_ctx` | context window (tokens) | drives KV-cache size — the main memory lever. 4096 is a good default; drop to 2048 on an 8 GB machine. |
| `llama_cpp.gguf_filename` | which quant to load from a multi-quant repo | `q4_k_m` is the size/quality sweet spot; `q5_k_m`/`q6_k` for more quality, `q3_k_m` to save RAM. |
| `llama_cpp.flash_attn` | smaller KV cache (if built with it) | leave `true`; harmless if unsupported. |
| `llama_cpp.n_batch` | prompt (prefill) batch size | raise for faster prompt ingestion if you have headroom. |

*Note:* in-process llama.cpp generates a timestep's agents **sequentially** (the binding
is single-sequence). For real batching on this backend, run the **llama.cpp server** with
`--parallel N --cont-batching` and point at it via `backend.kind=none` (next).

**vLLM (`backend.kind=vllm`) — Linux / NVIDIA GPU.** Loads a full-precision (or quantized)
HF checkpoint and batches the whole timestep natively.

| Knob | What it does | How to pick it |
| --- | --- | --- |
| `vllm.gpu_memory_utilization` | fraction of VRAM vLLM may claim | 0.90 default; lower if other processes share the GPU, raise toward 0.95 for headroom. |
| `vllm.max_model_len` | caps context length → caps KV cache | **the usual OOM lever** — lower it first when you hit CUDA OOM. |
| `vllm.quantization` + an AWQ/GPTQ repo | load 4-bit weights | fits a 7-8B model on a ~12 GB card (fp16 needs ~16 GB of weights alone). |
| `vllm.cpu_offload_gb` | spill weights to CPU RAM | alternative to quantization when a bit short on VRAM (slower). |
| `vllm.tensor_parallel_size` | shard across N GPUs | set to your GPU count for big models. |

**Remote server (`backend.kind=none`).** The server does the batching; you only control
how many requests you send at once via `backend.max_concurrent_requests` (see above).

**Batching across agents (every backend).** `generation.batch` (default `true`) generates
a timestep's independent agents together. It's a near-free speedup on vLLM / a llama.cpp
server / a remote endpoint; a no-op (sequential) for in-process llama.cpp. Set it to
`false` only to debug or to reproduce strictly one-at-a-time behaviour.

## Usage

### Run an experiment (recommended): `run_experiment.py`

The Hydra-driven runner does **simulation + analysis** in one command. Defaults live
in the `ExperimentConfig` dataclass (`llm_culture/config.py`); override any field with
`dotted.key=value`:

```bash
uv run python run_experiment.py \
  backend.kind=llama_cpp backend.model=unsloth/SmolLM2-135M-Instruct-GGUF \
  n_timesteps=2 n_seeds=1 output=results/my_test
```

### Config structure

Related knobs are grouped into small sub-configs; a few cross-cutting scalars stay
at the top level:

| Group / field | Meaning |
| --- | --- |
| `population.network_structure` | `sequence` / `fully_connected` / `circle` / `caveman` |
| `population.agents` | list of agent **groups** — each `{count, personality, prompt_init, prompt_update}` (names registered in `data/parameters/`). `population.n_agents` is the sum of the counts |
| `population.n_cliques` | number of cliques (caveman only) |
| `backend.kind`, `backend.model` | backend selector + model id/path |
| `backend.access_url` | remote server URL (when `kind=none`) |
| `backend.llama_cpp.*` / `backend.vllm.*` | backend tuning (e.g. `backend.llama_cpp.n_gpu_layers`) |
| `generation.temperature` / `.max_tokens` / `.top_p` | sampling knobs |
| `generation.instruct` | chat/instruct API vs raw completion |
| `n_timesteps`, `n_seeds`, `seed_offset` | generations, seeds, seed naming offset |
| `output`, `verbose`, `debug` | results folder, print stories, debug |
| `analysis.run`, `analysis.plot` | skip analysis (simulate only) / also open figures |
| `analysis.embedding.method` | how stories are vectorized for the similarity matrix: `tfidf` (default) or `huggingface` |
| `analysis.embedding.model` | HuggingFace model id when `analysis.embedding.method=huggingface` |

**Agents are described as groups, not a list.** A homogeneous population is a single
group whose `count` is the size; a mixed population lists several types:

```yaml
population:
  network_structure: fully_connected
  agents:
    - { count: 3, personality: Creative }
    - { count: 3, personality: NotCreative }
```

**Transmission chains.** For `network_structure: sequence` the model generates one
story per agent down the chain, so the number of generations equals the agent count —
`n_timesteps` must equal `population.n_agents` (you get a clear error otherwise). Other
networks are populations where every agent regenerates each step, so `n_timesteps` is
independent.

The config is a *structured config*: unknown fields, wrong types, or invalid enum
values are rejected with a clear error before anything runs.

**Presets.** Reusable setups live in `conf/experiment/` — the `base` preset is the
default, so a bare run is a small local smoke test. Copy `base.yaml` to make your own
and select it with `experiment=<name>` (no leading `+`). Ready-made hardware presets:
`mac_mx.yaml` runs a real ~7B model locally on any Apple Silicon Mac (M1/M2/M3/M4,
llama.cpp + Metal), `linux_gpu.yaml` runs one with vLLM on a Linux/NVIDIA box, and
`big_model_small_gpu.yaml` shows partial GPU-offload tuning. `mixed_persona.yaml` is a
heterogeneous population example.

```bash
uv run python run_experiment.py                                   # bare run = base preset
uv run python run_experiment.py n_seeds=2                         # base + override on top
uv run python run_experiment.py experiment=mac_mx                 # ~7B locally on Apple Silicon
uv run python run_experiment.py experiment=linux_gpu              # ~7B with vLLM on a Linux GPU
```

**Inference is its own config group.** The backend/model setup (and all its
hardware-tuning comments) lives in `conf/inference/` — one file per setup, named for
the model + backend (`smollm2_135m_instruct_llama_cpp`,
`mistral_7b_instruct_v0_2_gguf_metal`, `mistral_7b_instruct_v0_2_vllm_gpu`,
`mistral_7b_instruct_v0_2_gguf_partial_offload`). Each experiment preset pulls one in
through its `defaults` list, so you can mix any population with any backend without
editing files:

```bash
# mac_mx population, but the vLLM backend instead of Metal:
uv run python run_experiment.py experiment=mac_mx inference=mistral_7b_instruct_v0_2_vllm_gpu
# base population, but a real 7B on Apple Silicon instead of the tiny smoke model:
uv run python run_experiment.py experiment=base inference=mistral_7b_instruct_v0_2_gguf_metal
```

**Sweeps** (`-m` multirun):

```bash
uv run python run_experiment.py -m generation.temperature=0.7,0.9,1.1
```

### Simulation only: `run_simulation.py`

A sibling Hydra entrypoint that runs the simulation and stops before analysis (same
config, same overrides):

```bash
uv run python scripts/run_simulation.py experiment=base n_seeds=1
uv run python scripts/run_analysis.py folder=results/base_experiment   # analyse later
```

### Web interface

```bash
uv run python web_interface.py
```

Launches a local Flask app to configure/launch runs, register prompts, and browse
plots. Choose `Remote server` mode (paste an OpenAI-compatible URL) or
`Local model` mode (local path or HF repo id).

### Notebook

A Colab notebook is available [here](https://colab.research.google.com/drive/1bD9x4KGus6s0ifRiC1rbaZUMCtAxJywf?usp=sharing)
(select a GPU runtime).

## Reproducibility

Paper data is under `results/experiments/`. Reproduce single-experiment figures
(these analysis scripts are Hydra entrypoints too — override fields with `key=value`):

```bash
uv run python scripts/run_analysis.py folder="results/experiments/Network Structure/CAVEMAN_10_10_combine5seeds"
```

Comparison figures (`folders` entries are names relative to `results/experiments/`
by default — the `root` field; pass them as a Hydra list):

```bash
uv run python scripts/run_comparison_analysis.py 'folders=[Network Structure/CAVEMAN_10_10_combine5seeds, Network Structure/CIRCLE_10_10_combine5seeds]'
```

The `reproduction_scripts/` directory regenerates the paper experiments
(`transmission_chain.py`, `network.py`, `transformation_prompt.py`,
`persona_prompt.py`). They are config-driven: each builds an `ExperimentConfig`
from a `conf/experiment/repro_*.yaml` preset, runs simulation + analysis, and (for
the multi-variant ones) a comparison. Run one with, e.g.:

```bash
uv run python reproduction_scripts/network.py
```

Edit the matching `conf/experiment/repro_*.yaml` to change the model, sizes, or
seeds (e.g. swap `backend.model` for a paper-scale model). You can also run a
single variant through the main entrypoint:
`uv run python run_experiment.py experiment=repro_network`. Outputs differ from the
paper due to generation stochasticity.

> **TODO (dev):** make the library approachable for researchers who aren't software
> specialists — plain-language docs (quickstart, worked examples, an explanation of
> each config knob) and a friendlier interface (the web GUI and/or a guided CLI) so
> running and analysing experiments needs no Python/Hydra knowledge.

## Implemented analysis

Single-experiment plots (registered in `llm_culture/analysis/plots.py`):

| Plot | Description |
| --- | --- |
| Similarity Matrix | similarity between all stories in an experiment |
| Between-Generations Similarity Matrix | each generation vs every other |
| Word Chains | evolution of key words across generations |
| Similarity Graph | generations as a graph weighted by similarity |
| Similarity with First Generation | similarity to the initial generation over time |
| Within-Generation Similarity | how similar stories are inside each generation |
| Successive-Generation Similarity | similarity between consecutive generations |

![analysis_plots](/static/experiment_analysis_figures.png)

### Similarity: TF-IDF or HuggingFace embeddings

The story-similarity metrics are built from a cosine-similarity matrix over the
stories. By default stories are vectorized with **TF-IDF** (word overlap) — fast,
dependency-free, and what the paper used. For **semantic** similarity you can
instead use a HuggingFace sentence-embedding model:

```bash
uv sync --extra embeddings    # one-time: installs sentence-transformers
uv run python run_experiment.py experiment=base \
  analysis.embedding.method=huggingface \
  analysis.embedding.model=sentence-transformers/all-MiniLM-L6-v2
```

`all-MiniLM-L6-v2` is a good, small default (384-dim, fast on CPU). The two methods
write separate analysis caches, so switching back and forth never mixes results.
Everything downstream (plots, between-generation similarities, graphs) is identical
regardless of method — only how each story becomes a vector changes.

## Extending the framework

**Parameters** (named prompts, personalities, network structures) live in
`llm_culture/data/parameters/`.

- *Add a network structure*: register it in
  `llm_culture/data/parameters/network_structures.json` (adjacency-list format), or use
  `register_custom_network_structure` in `llm_culture/simulation/utils.py`. To expose it
  in the GUI, add it to the dropdown in `templates/simulation.html`.
- *Add a plot/metric*: implement the function and register it in
  `llm_culture/analysis/plots.py` (`PLOT_REGISTRY` / `DEFAULT_PLOT_NAMES`) or
  `comparison_plots.py` for comparison plots.
