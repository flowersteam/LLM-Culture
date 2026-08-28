# LLM-Culture

Code for the [Cultural evolution in populations of Large Language Models](https://arxiv.org/abs/2403.08882) paper. This repository provides a comprehensive framework for studying the cultural evolution of linguistic content in populations of Large Language Models (LLM).

It allows organizing LLM agents into networks wherein each agent interacts with neighboring agents by exchanging stories. Each agent can be assigned specific personalities and transmission instructions, serving as prompts for generating new stories from their neighbors’ narratives. Once the network structure and agent characteristics are defined, you can simulate the cultural evolution of texts across generations of agents. We also provide built-in metrics and visualizations to analyze the results.


![introduction_figure](/static/introduction_figure.png)


## Installation 

1 - Clone the repository


```bash
git clone git@github.com:flowersteam/LLM-Culture.git
cd LLM-Culture/
```

2 - Install the dependencies 

```bash
python -m venv myvenv
source myvenv/bin/activate

pip install -r requirements.txt
pip install -e .
```

3 - Choose an LLM backend

The framework supports two ways of talking to a model:

- A remote server exposing an OpenAI-compatible URL. This is the simplest option if you already have an inference endpoint.
- A local Hugging Face model. The code accepts either a local path or a Hugging Face repo id, downloads the snapshot on first use, and reuses the cached copy afterwards.

When you use a Hugging Face repo id, the model is downloaded into the Hugging Face cache directory. By default, that is `~/.cache/huggingface`, unless you set a different cache path in the GUI or in the code. If the snapshot contains several `.gguf` files, the loader prefers the one whose filename contains `q4_k_m` and otherwise falls back to the first `.gguf` file it finds.

To select the right model name, use the repository id shown on the Hugging Face model page, for example `mistralai/Mistral-7B-Instruct-v0.2` or `unsloth/SmolLM2-135M-Instruct-GGUF`. If the repository is gated, make sure your Hugging Face credentials are available before launching the run.
    

## Usage 

You can use the framework both from command-line interface or from a web interface :

### 1 - Command Line 

Run a simulation with your desired parameters (see parameters details above): 

```bash
python3 scripts/run_simulation.py --output_file simulation_test
```

<details>
    
  <summary> Show all parameter flags </summary>
    
  - "-na" : Number of agents (int).

  - "-nt" : Number of timesteps (int).

  - "-ns" : Network structure (choices: 'sequence','fully_connected' 'circle', 'caveman').

  - "-nc" : Number of cliques for a caveman network (int).

  - "-pi": Name of the initialization prompt (str). The prompt should be already registered in llm_culture/data/parameters/prompt_init.json.

  - "-pu" : Name of the transformation prompt (str). The prompt should be already registered in llm_culture/data/parameters/prompt_update.json.

  - "-pl" : Personality list (list of str). Each personality should be already registered in llm_culture/data/parameters/personalities.json. The length of the list of personalities should be equal to the number of agents.

  - "-o" : Name of the folder in which to store results (str).

  - "-url": URL to send the prompt to (str).

</details>

The results of the experiment will be stored in a directory called `results/simulation_test/` in this case. You can then analyze the texts produced with this command:

```bash
python3 scripts/run_analysis.py --dir simulation_test
```

To compare the results of several experiments, run:

```bash
python3 scripts/run_comparison_analysis.py --dirs experiment_1+experiment_2+experiment_3
```

It will store the analysis figures in a directory called `results/experiments_comparisons/experiment_1-experiment_2-experiment_3/`.


### 2 - Web Interface

Launch the web user interface with the following command:

```bash
python3 web_interface.py
```

This starts a local Flask app, typically at `http://127.0.0.1:5000`. From there you can launch a simulation, add prompts, analyze results, and browse previous experiment outputs.

How to use the GUI:

1. Open the simulation page and enter an experiment name. The results will be written under `results/experiments/<experiment name>/`.
2. Set the number of agents, generations, and seeds.
3. Choose a network structure. The GUI currently exposes `fully_connected`, `sequence`, `circle`, and `caveman`. For `caveman`, also set the number of cliques.
4. Pick the initial prompt, update prompt, and personality for each agent from the registered lists. If a value is missing, use the built-in "Add Prompt" form to register a new prompt or personality before starting the run.
5. Choose the LLM mode. In `Remote server` mode, paste the OpenAI-compatible server URL. In `Local model or Hugging Face download` mode, provide either a local model path or a Hugging Face repo id, plus an optional cache directory.
6. Start the simulation. The UI shows progress while the job runs and redirects you to the analysis page when it finishes.
7. Open the analysis or comparison pages to generate plots for one experiment or several experiments.

### Notebook 


We provide a notebook allowing to run experiments on Google Colab: [URL](https://colab.research.google.com/drive/1bD9x4KGus6s0ifRiC1rbaZUMCtAxJywf?usp=sharing)

Note: Make sure to select a GPU-based runtime (e.g. GPU T4). 


## Reproducibility:

### Data

The data presented in the paper is provided in the experiments/ folder. 

To reproduce the figures corresponding to a single experiment, run:

```bash
python3 scripts/run_analysis.py --folder "results/experiments/Network Structure/[experiment_name, e.g. CAVEMAN_10_10_combine5seeds]"
```

To reproduce the figures comparing several variants:

```bash
# Pass experiment names relative to `results/experiments/`, separated by '+'
python3 scripts/run_comparison_analysis.py --dirs "Network Structure/CAVEMAN_10_10_combine5seeds+Network Structure/CIRCLE_10_10_combine5seeds+Network Structure/FC_10_10_combine5seeds
```

The comparison figures are saved under `results/experiments_comparisons/` in a folder named after the joined experiment basenames (see existing folders in `results/experiments_comparisons/`).


### Reproduction Scripts

The scripts provided in reproduction_scripts allow to reproduce the experiments presented in the paper. Note that due to the stochasticity of text generation, the outputs will be different from those presented in the paper. 

It saves their outputs in `results/experiments/`, generates the analysis plots, and then writes the comparison figures in `results/experiments_comparisons/`. The other scripts in that directory follow the same structure for persona, prompt, and transmission-chain comparisons.

To run them: 


```bash
python3 transmission_chain.py
```

```bash
python3 network.py
```

```bash
python3 transformation_prompt.py
```

```bash
python3 persona_prompt.py
```



# Implemented Analysis

The analysis pipeline is split into two layers:


<details>
  <summary> Plots details </summary>

  The default single-experiment plots are registered in `llm_culture/analysis/plots.py` and include:

  | Plot Type | Description |
  | --- | --- |
  | **Similarity Matrix** | Compares the similarity between all the stories generated during an experiment. |
  | **Between Generations Similarity Matrix** | Compares each generation with every other generation. |
  | **Word Chains Plot** | Visualizes the evolution of key words in texts through generations. |
  | **Similarity Graph** | Shows generations as a graph weighted by similarity. |
  | **Similarity with the First Generation** | Tracks similarity to the initial generation across time. |
  | **Within-Generation Similarity** | Tracks how similar stories are inside each generation. |
  | **Successive-Generation Similarity** | Tracks similarity between consecutive generations. |

  ![analysis_plots](/static/experiment_analysis_figures.png)
   


## Building on the framework

### Add a Network Structure

The built-in topologies are `sequence`, `circle`, `caveman`, and `fully_connected`. To add a custom structure, register it in `data/parameters/network_structures.json` using the same adjacency-list format used by the loader in `llm_culture/simulation/utils.py`. The helper `register_custom_network_structure` in that module can update the file for you.

If you want the new structure to appear in the GUI, also add it to the network dropdown in `templates/simulation.html`.

### Add Plots or Metrics

Single-experiment plots are registered in `llm_culture/analysis/plots.py` through `DEFAULT_PLOT_NAMES` and `PLOT_REGISTRY`. Comparison plots are registered in `llm_culture/analysis/comparison_plots.py` through `DEFAULT_COMPARISON_PLOT_NAMES` and `COMPARISON_PLOT_REGISTRY`.

To add a new plot, implement the plotting function, add the required metric data in the analysis step if needed, then register the function in the relevant registry. If the plot should be generated by default, add its name to the corresponding default list.


