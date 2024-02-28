# LLM-Culture

This repository provides a comprehensive framework for studying the cultural evolution of linguistic content in populations of Large Language Models. 

It allows organizing LLM agents into networks wherein each agent interacts with neighboring agents by exchanging stories. Each agent can be assigned specific personalities and transmission instructions, serving as prompts for generating new stories from their neighbors’ narratives. Once the network structure and agent characteristics are defined, you can simulate the cultural evolution of texts across generations of agents. We also provide built-in metrics and vizualizations to analyze the results.


![Alt text](/Images/intro_fig.png)


## Installation 

1- Clone the repository

As this is an anonymous repository, you can use this tool to clone it: https://github.com/fedebotu/clone-anonymous-github


```bash
git clone https://github.com/fedebotu/clone-anonymous-github.git && cd clone-anonymous-github
python3 src/download.py --url https://anonymous.4open.science/r/LLM-Culture-75BE
cd LLM-Culture/
```

2- Install the dependencies 

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## Usage (GUI)

Launch the graphical user interface:

```bash
python3 run_interface.py
```

This will open the GUI. You can then select the simulation parameters. 


![Alt text](/Images/supplementary_screenshot.png?raw=true)

<details>
  
  <summary> Display parameters details</summary>
  
  - Number of agents: use this to specify how many agents you wish to simulate
    
  - Number of timesteps: use this to specify for how many timesteps the simulation should run

  - Number of seeds: use this to specify how many times the whole simulation should be repeated. 
    
  - Initialization prompt: use this to set the prompt given to agents at the first timestep. You can choose among already registered prompt using the drop-down menu, or add a new prompt to this list by clicking on "Add Prompt...". This will open a window where you can enter the name and content of your new prompt.
    
  - Transmission prompt: use this to set the prompt that will be concatenated with the stories of each agent's neighbors after the first timestep. As for the Initialization prompt, you may select an existing prompt or create a new one.
    
  - Network structure: use this to specify the stucture of the social network. You can view the selected structure by clicking on "Display Graph"
    
  - Personality: use this to assign a personality to the agents. The personality will be concatenated with the rest of the prompt. If you want all agents to have the same personality, tick the "Same for all agents" box. You can then select a personality from the drop-down menu or create a new one. If you want agents to have different personalities, untick the "Same for all agents" box and select a personality for each agent.
    
  - Simulation name: Give a name to your simulation. This will be the name of the folder when the simulation results are stored.
    
  - Server access URL: URL to which the requests will be sent to get answers from the LLM. In our case, we generated such an URL using oogabooga (https://github.com/oobabooga/text-generation-webui) and we provide a step-by-step guide below.
</details>

<details>
    
  <summary> Step-by-step guide to using oogabooga to generate an public URL to request an LLM </summary>
    
  - Manually install oogabooga Text generation web UI by following the steps described here: https://github.com/oobabooga/text-generation-webui (section "Setup details and information about installing manually")
  
  - Launch a server: 
  ```bash
  python server.py  --gradio-auth username:password --listen --public-api --share
  ```
  3. This will output an OpenAI-compatible API URL: https://xxxx-xxxx-xxxx-xxxx.trycloudflare, and a "gradio.live" URL: "Running on public URL: https://xxxxxxxx.gradio.live"

  4. Paste the OpenAI-compatible URL in the field "Server access URL" of the LLM-Culture GUI.

  5. Open the gradio.live URL in your browser. 

  6. Go to the model tab and download a model from [huggingface](https://huggingface.co). We used https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF, with File name "mistral-7b-openorca.Q4_K_M.gguf". Select an appropriate Model loader (we used llama.cpp). 

  7. Click on Load to load the model. 

  8. Once the model is loaded, you can go back to the LLM-Culture GUI and run your simulations!
</details>
    


## Usage (command line)



Run a simulation with your desired parameters (see parameters details above): 

```bash
python3 run_simulation.py --output_file simulation_test
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

The results of the experiment will be stored in a directory called ```Results/simulation_test/```. You can then analyze the texts produced with this command : 

```bash
python3 run_analysis.py --dir simulation_test
```

To compare the results of several experiments, you can can run this command (with the experiment names separated by '+' symbols) : 

```bash
python3 run_comparison_analysis --dirs experiment_1+experiment_2+experiment_3
```

It will store the analysis figures in a directory called ```Results/Comparisons/experiment_1-experiment_2-experiment_3/```


## Figures : 




![Alt text](/Images/Fig2a.png?raw=true)

![Alt text](/Images/Fig2b.png?raw=true)

![Alt text](/Images/Fig2c.png?raw=true)
