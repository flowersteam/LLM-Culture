# LLM-Culture

This repository provides a comprehensive framework for studying the cultural evolution of linguistic content through the utilization of Large Language Model agents. 

It allows organizing LLM agents into networks wherein each agent interacts with neighboring agents by exchanging stories. Each agent can be assigned specific personalities and transmission instructions, serving as prompts for generating new stories from their neighbors’ narratives. Once the network structure and agent characteristics are defined, you can simulate the cultural evolution of texts across generations of agents. After that, you can easiely analyze the evolution of texts produced and compare the results of different experiment configurations.

## Installation 

1- Get the repository

```bash
git git@github.com:jeremyperez2/LLM-Culture.git
cd LLM-Culture/
```
2- Install the dependencies 

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## Usage

Run a simulation with your desired parameters (number of agents, generations, network structure ...) see a complete list in [run_simulation.py](run_simulation.py)): 

```bash
python3 run_simulation.py --output_file simulation_test
```

The results of the experiment will be stored in a directory called ```Results/simulation_test/```. You can then analyze the texts produced with this command : 

```bash
python3 run_analysis.py --dir simulation_test
```

To compare the results of several experiments, you can can run this command (with the experiment names separated by '+' symbols) : 

```bash
python3 run_comparison_analysis --dirs experiment_1+experiment_2+experiment_3
```

It will store the analysis figures in a directory called ```Results/Comparisons/experiment_1-experiment_2-experiment_3/```