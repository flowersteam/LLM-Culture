# LLM-Culture

### TODO : Add a quick description of the Repo, what can be done with it ... 

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

Run a simulation with your desired parameters (see a complete list in [run_simulation.py](run_simulation.py)): 
### TODO : Change --output_dir 

```bash
python3 run_simulation.py --output_file simulation_test
```

The results of the experiment will be stored in a directory called ```Results/simulation_test/```. You can then analyze the texts produced with this command : 

```bash
python3 run_analysis.py --dir simulation_test
```

To compare the results of several experiments, you can can run this command : 

```bash
python3 run_comparison_analysis --dirs experiment_1 experiment_2 experiment_3
```

It will store the analysis figures in a directory called ```Results/Comparisons/experiment_1-experiment_2-experiment_3/```

### TODO : Make a tutorial to explain how to do the same thing with the GUI
