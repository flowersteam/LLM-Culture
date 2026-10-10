# LLM-Culture

A framework for studying the **cultural evolution of text in populations of LLMs**.

Agents are organized into a network. Each agent reads the stories of its neighbours
and rewrites them according to a *personality* and a *transformation prompt*. Repeating
this over many generations lets you watch stories evolve — drifting, converging, or
splitting into sub-cultures — much like a transmission-chain or iterated-learning
experiment, but with LLMs as the "participants".

With this framework you can:

- build a population of agents and wire them into a network (a chain, a fully-connected
  group, a ring, or loosely-connected cliques);
- give agents personalities and transmission rules (copy, recombine, innovate, …);
- simulate how texts change across generations, with the LLM of your choice (a small
  one on your laptop, a large one on a GPU server, or a hosted API);
- analyze the results with built-in similarity metrics and plots.

![introduction_figure](/static/introduction_figure.png)

New to LLMs? See [docs/llm_for_researchers.md](docs/llm_for_researchers.md) for a
plain-language primer on the LLM concepts used below (what a model is, how to run one,
and what the generation settings do).

## Installation

```bash
git clone git@github.com:flowersteam/LLM-Culture.git
cd LLM-Culture/
```

This project is managed with [uv](https://docs.astral.sh/uv/), a fast Python package
and environment manager (it replaces `pip` + `venv`: it reads `pyproject.toml` and
builds an isolated environment for you).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv yet
uv sync --extra serving                            # install + local model support
```

- `uv sync` — base install (simulation + analysis).
- `uv sync --extra serving` — also installs the local inference backends (needed to run
  a model on your own machine; use this for the quickstart below).
- `uv sync --extra embeddings` — adds HuggingFace sentence embeddings for the analysis.

Run any command by prefixing it with `uv run` (e.g. `uv run python run_experiment.py`).

## Quickstart

Run the built-in demo — a tiny model, so it works on any laptop with no GPU, no API
key, and no cost (it downloads a ~100 MB model once and caches it):

```bash
uv run python run_experiment.py
```

This runs a small simulation and then its analysis, writing results and plots to
`results/base_experiment/`. That's the whole loop: configure → simulate → analyze.

With no arguments it reads the default experiment preset,
[`conf/experiment/base.yaml`](conf/experiment/base.yaml) — a 4-agent transmission chain
run with a tiny local model. The next section shows how to point it at your own preset.

## Design your own experiment

An experiment is described by a small YAML file (a *preset*). Ready-made presets live in
[`conf/experiment/`](conf/experiment/); the default is
[`base.yaml`](conf/experiment/base.yaml). To make your own, copy one, edit a few lines,
and run it by name.

The fields that define your experimental design:

| Field | What it controls |
| --- | --- |
| `population.network_structure` | who talks to whom: `sequence` (a transmission chain), `fully_connected` (a closed group where everyone reads everyone), `circle` (a ring), `caveman` (sub-groups joined by a bridge) |
| `population.agents` | the agents, as a list of **groups** — each `{count, personality, prompt_init, prompt_update}`. One group = a homogeneous population; several groups = a mixed one |
| `population.n_cliques` | number of sub-groups (for `caveman` only) |
| `generation.temperature` | how faithfully agents copy vs. innovate (0 = faithful, higher = more variation each step) |
| `generation.max_tokens` | maximum story length |
| `n_timesteps` | number of generations |
| `n_seeds` | number of independent repeats of the whole run |
| `output` | folder to write results to |

A `personality` and the two prompts are **names** registered in
[`llm_culture/data/parameters/`](llm_culture/data/parameters/):

- `personality` — a content bias, e.g. `Creative` / `NotCreative`, `Fantasy` / `SciFi`
  ([`personalities.json`](llm_culture/data/parameters/personalities.json)).
- `prompt_init` — the task used to seed the very first story
  ([`prompt_init.json`](llm_culture/data/parameters/prompt_init.json)).
- `prompt_update` — the transmission rule applied each generation, e.g. `Repeat` (copy),
  `MinorChanges` (copy with small edits), `CombineTwo` (recombine), `MaximizeDifference`
  (innovate) ([`prompt_update.json`](llm_culture/data/parameters/prompt_update.json)).

Edit those JSON files to add your own personalities or prompts.

### Example: a mixed-population experiment

Create `conf/experiment/my_run.yaml`:

```yaml
# @package _global_
defaults:
  - /inference: smollm2_135m_instruct_llama_cpp   # which model to run (see below)
  - _self_

population:
  network_structure: fully_connected
  agents:
    - { count: 3, personality: Creative }
    - { count: 3, personality: NotCreative }

generation:
  temperature: 0.8

n_timesteps: 5
n_seeds: 3
output: results/my_run
```

Run it:

```bash
uv run python run_experiment.py experiment=my_run
```

> For a `sequence` network (a transmission chain) each agent generates once, so
> `n_timesteps` must equal the number of agents. For every other network the two are
> independent. You get a clear error if they don't match.

## Choosing where the model runs

The experiment needs an LLM to generate the stories. There are three ways to provide
one — pick based on your resources:

| Option | `backend.kind` | When to use it |
| --- | --- | --- |
| **Your own computer** | `llama_cpp` | piloting and small studies — no setup, no cost; smaller/slower models (what the quickstart uses) |
| **A Linux + NVIDIA GPU machine** | `vllm` | real data collection — large models, fast |
| **A hosted / online API** | `none` | best quality with no local hardware; costs money and needs an API key |

Which model runs (and how it's tuned for your hardware) is set in its own group of
files, [`conf/inference/`](conf/inference/). Each experiment preset pulls one in, and
you can swap it without touching the experiment. Ready-made ones:

```bash
uv run python run_experiment.py experiment=mac_mx      # a real ~7B model on an Apple Silicon Mac
uv run python run_experiment.py experiment=linux_gpu   # a ~7B model with vLLM on a Linux GPU
```

If the words *backend*, *GGUF*, *quantization*, or *context window* are unfamiliar, read
[docs/llm_for_researchers.md](docs/llm_for_researchers.md) first — it explains how to
choose and run a model for your machine. The inference config files are also heavily
commented to guide the choice.

## Generation settings

The same sampling knobs apply to every backend:

- `generation.temperature` — randomness of each rewrite. Think of it as copying fidelity
  vs. mutation rate: `0` is faithful/deterministic, higher values add more variation per
  generation.
- `generation.max_tokens` — maximum length of a generated story.

`generation.top_p` and `generation.instruct` can usually stay at their defaults; see the
[primer](docs/llm_for_researchers.md) for what they do.

```bash
uv run python run_experiment.py generation.temperature=1.0 generation.max_tokens=256
```

## Running simulation and analysis separately

`run_experiment.py` does simulation **and** analysis. You can also run them on their own:

```bash
uv run python scripts/run_simulation.py experiment=base       # simulate only
uv run python scripts/run_analysis.py folder=results/base_experiment   # analyze later
```

## Web interface

A small local web app lets you configure and launch runs, register prompts, and browse
plots from the browser:

```bash
uv run python web_interface.py
```

It starts a local Flask server; open the printed URL and follow the pages to run a
simulation and view its analysis.

## Reading the results

Each run produces these plots (registered in `llm_culture/analysis/plots.py`):

| Plot | What it shows |
| --- | --- |
| Similarity Matrix | similarity between all stories in the experiment |
| Between-Generations Similarity Matrix | each generation compared to every other |
| Word Chains | how key words survive and spread across generations |
| Similarity Graph | generations as a graph weighted by similarity |
| Similarity with First Generation | how far stories drift from the original over time |
| Within-Generation Similarity | how similar stories are inside a generation (convergence) |
| Successive-Generation Similarity | how much changes from one generation to the next |

![analysis_plots](/static/experiment_analysis_figures.png)

By default stories are compared with **TF-IDF** (word overlap) — fast, reproducible, and
what the paper used. For **semantic** similarity instead, install the embeddings extra
and switch the method:

```bash
uv sync --extra embeddings
uv run python run_experiment.py analysis.embedding.method=huggingface
```

## Reproducibility

The paper's data is under `results/experiments/`. Regenerate a single experiment's
figures:

```bash
uv run python scripts/run_analysis.py folder="results/experiments/Network Structure/CAVEMAN_10_10_combine5seeds"
```

Compare several experiments (names are relative to `results/experiments/`):

```bash
uv run python scripts/run_comparison_analysis.py 'folders=[Network Structure/CAVEMAN_10_10_combine5seeds, Network Structure/CIRCLE_10_10_combine5seeds]'
```

The [`reproduction_scripts/`](reproduction_scripts/) directory re-runs the paper's
experiments (`transmission_chain.py`, `network.py`, `transformation_prompt.py`,
`persona_prompt.py`). Each is driven by a `conf/experiment/repro_*.yaml` preset — edit
the preset to change the model, sizes, or seeds. Outputs differ from the paper due to
generation randomness.

```bash
uv run python reproduction_scripts/network.py
```

A Google Colab notebook is also available — it runs the whole configure → simulate →
analyze loop in the cloud with no local setup (choose a GPU runtime for a real model):

# TODO --> update w real link

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/flowersteam/LLM-Culture/blob/dev/notebooks/llm_culture_colab.ipynb)

The notebook lives in the repo at [`notebooks/llm_culture_colab.ipynb`](notebooks/llm_culture_colab.ipynb).

## Extending the framework

- **Add a network structure** — register it in
  `llm_culture/data/parameters/network_structures.json` (adjacency-list format), or use
  `register_custom_network_structure` in `llm_culture/simulation/utils.py`.
- **Add a plot or metric** — implement the function and register it in
  `llm_culture/analysis/plots.py` (`PLOT_REGISTRY` / `DEFAULT_PLOT_NAMES`), or in
  `comparison_plots.py` for comparison plots.
