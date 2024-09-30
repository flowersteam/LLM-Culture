# LLM-Culture

Code for the [Cultural evolution in populations of Large Language Models](https://arxiv.org/abs/2403.08882) paper. This repository provides a comprehensive framework for studying the cultural evolution of linguistic content in populations of Large Language Models (LLM).

It allows organizing LLM agents into networks wherein each agent interacts with neighboring agents by exchanging stories. Each agent can be assigned specific personalities and transmission instructions, serving as prompts for generating new stories from their neighbors’ narratives. Once the network structure and agent characteristics are defined, you can simulate the cultural evolution of texts across generations of agents. We also provide built-in metrics and vizualizations to analyze the results.


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
```

3 - Install a local LLM (Optional)

If you want to run experiments with a LLM running on your own computer, follow the instructions below. Otherwise, you can use the framework with any LLM hosted on a remote API.

<details>
  <summary> Step-by-step guide to using oogabooga to generate an public URL to request an LLM </summary>
    
  - Manually install oogabooga Text generation web UI by following the steps described here: https://github.com/oobabooga/text-generation-webui (section "Setup details and information about installing manually")
  
  - Launch a server: 

  ```bash
  python server.py  --gradio-auth <your_username>:<your_password> --listen --public-api --share
  ```
  3. This will output an OpenAI-compatible API URL: https://xxxx-xxxx-xxxx-xxxx.trycloudflare, and a "gradio.live" URL: "Running on public URL: https://xxxxxxxx.gradio.live"

  4. Paste the OpenAI-compatible URL in the field "Server access URL" of the LLM-Culture GUI.

  5. Open the gradio.live URL in your browser (use the given username and password to connect). 

  6. Go to the model tab and download a model from [huggingface](https://huggingface.co). We used https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF, with File name "mistral-7b-openorca.Q4_K_M.gguf". Select an appropriate Model loader (we used llama.cpp). 

  7. Click on Load to load the model. 

  8. Once the model is loaded, you can go back to the LLM-Culture GUI and run your simulations!
</details>
    

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

The results of the experiment will be stored in a directory called ```results/simulation_test/``` in this case. You can then analyze the texts produced with this command : 

```bash
python3 scripts/run_analysis.py --dir simulation_test
```

To compare the results of several experiments, you can can run this command (with the experiment names separated by '+' symbols) : 

```bash
python3 scripts/run_comparison_analysis --dirs experiment_1+experiment_2+experiment_3
```

It will store the analysis figures in a directory called ```results/Comparisons/experiment_1-experiment_2-experiment_3/```


### 2 - Web Interface

Launch the web user interface with the following command:

```bash
python3 web_interface.py
```

This will create a link the website (e.g *http://127.0.0.1:5000* in your terminal), just click on it to open the interface on a web browser. You can then run a simulation, analyze it and visualize the results from previous simulations ! You can find below the details of the different simulation parameters, as well as how to use an LLM in our framework.


<!-- ![GUI](/static/web_interface.png) -->

<details>
  
  <summary>Display parameters details</summary>
  
  - Simulation name: Give a name to your simulation. This will be the name of the folder when the simulation results are stored.
  
  - Number of agents: use this to specify how many agents you wish to simulate
    
  - Number of timesteps: use this to specify for how many timesteps the simulation should run

  - Number of seeds: use this to specify how many times the whole simulation should be repeated. 
    
  - Network structure: use this to specify the stucture of the social network. You can view the selected structure by clicking on "Display Graph"

  - Initialization prompts: use this to set the prompt given each agent at the first timestep. You can choose among already registered prompt using the drop-down menu, or add a new prompt to this list by clicking on "Add Prompt...". This will open a window where you can enter the name and content of your new prompt.
    
  - Transmission prompts: use this to set the prompt that will be concatenated with the stories of each agent's neighbors after the first timestep. As for the Initialization prompt, you may select an existing prompt or create a new one.
    
  - Personalities: use this to assign a personality to each agent. The personality will be concatenated with the rest of the prompt. If you want all agents to have the same personality, tick the "Same for all agents" box. You can then select a personality from the drop-down menu or create a new one. If you want agents to have different personalities, untick the "Same for all agents" box and select a personality for each agent.  
    
  - Server access URL: URL to which the requests will be sent to get answers from the LLM. In our case, we generated such an URL using oogabooga (https://github.com/oobabooga/text-generation-webui) and we provide a step-by-step guide below.
</details>


## Implemented Analysis

- Our framework generates several plots for each experiment analysis using `run_analysis.py`.
- In addition to these plots, we track the evolution of several metrics. We also provide a comparison of these metrics across several experiments using `run_comparison_analysis.py`.  

<details>
  <summary> Plots details </summary>

  | Plot Type | Description |
  | --- | --- |
  | **Similarity Matrix (a)** | Compares the similarity between all the stories generated during an experiment. |
  | **Similarity Graph (b)** | Nodes represent stories and are arranged based on their similarities. |
  | **Word Chains Plot (c)** | Visualizes the evolution of key words in texts through generations. |

  ![analysis_plots](/static/experiment_analysis_figures.png)
   
</details>

<details>
  <summary> Metrics details </summary>
  
  - Similarity between new generations of stories and the initial one
  - Similarity within generations and with successive ones
  - Positivity across generations
  - Subjectivity across generations
  - Creativity across generations

  ![comparison_analysis_plots](/static/experiment_analysis_comparison_figures.png)
   
</details>

