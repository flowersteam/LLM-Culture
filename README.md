# LLM-Culture


## Installation 

1- Get the repository

```bash
git clone git@github.com:jeremyperez2/LLM-Culture.git
cd LLM-Culture/
```
2- Install the dependencies 

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

TOOD, check if need to run this from a new install (I had to do it on a new venv) :

```bash
python -m spacy download en_core_web_lg
```

## Usage

Run a simulation : 
TODO : add the good command 

```bash
example command : run_simu.py --exp_name experiment_name
```

The results of the experiment will be stored in a directory called Results/experiment_name 

Analyze the results and save the figures with : 

```bash
python3 visualization/analyze_results.py
```

Or do an analysis step by step with [this kind of jupyter notebook](caveman_10_6.ipynb)

TODO : Make a quick tutorial to show how to do it with the graphical interface 