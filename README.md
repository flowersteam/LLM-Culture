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
and select it with `experiment=<name>` (no leading `+`). See `big_model_small_gpu.yaml`
for backend memory/offload tuning and `mixed_persona.yaml` for a heterogeneous population:

```bash
uv run python run_experiment.py                                   # bare run = base preset
uv run python run_experiment.py n_seeds=2                         # base + override on top
uv run python run_experiment.py experiment=big_model_small_gpu    # switch preset
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
