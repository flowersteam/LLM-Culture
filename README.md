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
```

Prefix commands with `uv run` (e.g. `uv run python run_experiment.py ...`).

### Choosing a backend

| Backend | Where it runs | Notes |
| --- | --- | --- |
| `none` | — | talk to a remote OpenAI-compatible server via `access_url` |
| `llama_cpp` | CPU / Apple Metal (**macOS + Linux**) | loads a local GGUF model |
| `vllm` | **Linux / GPU only** | skipped automatically on macOS |

For `llama_cpp` / `vllm`, `model` is a local path or a Hugging Face repo id (downloaded
to `~/.cache/huggingface` on first use and cached afterwards). For a multi-file GGUF
repo the loader picks the `q4_k_m` file by default, or set `llama_cpp.gguf_filename`.

## Usage

### Run an experiment (recommended): `run_experiment.py`

The Hydra-driven runner does **simulation + analysis** in one command. Defaults live
in the `ExperimentConfig` dataclass (`llm_culture/config.py`); override any field with
`key=value`:

```bash
uv run python run_experiment.py \
  backend=llama_cpp model=unsloth/SmolLM2-135M-Instruct-GGUF \
  n_agents=2 n_timesteps=2 n_seeds=1 output=results/my_test
```

Common fields:

| Field | Meaning |
| --- | --- |
| `n_agents`, `n_timesteps`, `n_seeds` | population size, generations, seeds |
| `network_structure` | `sequence` / `fully_connected` / `circle` / `caveman` |
| `prompt_init`, `prompt_update`, `personality_list` | registered prompt / persona names |
| `backend`, `model` | backend selector + model id/path |
| `generation.temperature`, `generation.max_tokens`, `generation.top_p` | sampling knobs |
| `llama_cpp.*` / `vllm.*` | backend tuning (e.g. `llama_cpp.n_gpu_layers`, `vllm.gpu_memory_utilization`) |
| `instruct`, `verbose` | instruct vs raw completion, print stories as they generate |
| `output` | results folder |
| `analysis.run`, `analysis.plot` | skip analysis (simulate only) / also open figures interactively |

The config is a *structured config*: unknown fields, wrong types, or invalid enum
values are rejected with a clear error before anything runs.

**Presets.** Reusable setups live in `conf/experiment/` — copy `base.yaml` and run it
by name. See `big_model_small_gpu.yaml` for how to tune backend memory/offload:

```bash
uv run python run_experiment.py +experiment=base
uv run python run_experiment.py +experiment=base n_timesteps=10   # override on top
```

**Sweeps** (`-m` multirun):

```bash
uv run python run_experiment.py -m n_agents=2,5,10 +experiment=base
```

### Standalone scripts

The simulation and analysis steps also run on their own (these are what the
reproduction scripts use):

```bash
uv run python scripts/run_simulation.py -na 2 -nt 2 -s 1 -o results/my_test \
  --use_llama_cpp --model unsloth/SmolLM2-135M-Instruct-GGUF
uv run python scripts/run_analysis.py --folder results/my_test
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

Paper data is under `results/experiments/`. Reproduce single-experiment figures:

```bash
uv run python scripts/run_analysis.py --folder "results/experiments/Network Structure/CAVEMAN_10_10_combine5seeds"
```

Comparison figures (names relative to `results/experiments/`, separated by `+`):

```bash
uv run python scripts/run_comparison_analysis.py --dirs "Network Structure/CAVEMAN_10_10_combine5seeds+Network Structure/CIRCLE_10_10_combine5seeds"
```

The `reproduction_scripts/` directory regenerates the paper experiments
(`transmission_chain.py`, `network.py`, `transformation_prompt.py`,
`persona_prompt.py`). Outputs differ from the paper due to generation stochasticity.

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
