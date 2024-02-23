# LLM-Culture

This repository provides a comprehensive framework for studying the cultural evolution of linguistic content in populations of Large Language Models. 

It allows organizing LLM agents into networks wherein each agent interacts with neighboring agents by exchanging stories. Each agent can be assigned specific personalities and transmission instructions, serving as prompts for generating new stories from their neighbors’ narratives. Once the network structure and agent characteristics are defined, you can simulate the cultural evolution of texts across generations of agents. We also provide built-in metrics and vizualizations to analyze the results.

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

<details>
  <summary> Display parameters details</summary>
  - Number of agents: use this to specify how many agents you wish to simulate
  - Number of timesteps: use this to specify for how many timesteps the simulation should run
  - Initialization prompt: use this to set the prompt given to agents at the first timestep. You can choose among already registered prompt using the drop-down menu, or add a new prompt to this list by clicking on "Add Prompt...". This will open a window where you can enter the name and content of your new prompt. 
  - Transmission prompt: use this to set the prompt that will be concatenated with the stories of each agent's neighbors after the first timestep. As for the Initialization prompt, you may select an existing prompt or create a new one.
  - Network structure: use this to specify the stucture of the social network. You can view the selected structure by clicking on "Display Graph"
  - Personality: use this to assign a personality to the agents. The personality will be concatenated with the rest of the prompt. If you want all agents to have the same personality, tick the "Same for all agents" box. You can then select a personality from the drop-down menu or create a new one. If you want agents to have different personalities, untick the "Same for all agents" box and select a personality for each agent. 
  - Simulation name: Give a name to your simulation. This will be the name of the folder when the simulation results are stored. 
  - Server access URL: URL to which the requests will be sent to get answers from the LLM. In our case, we generated such an URL using oogabooga (https://github.com/oobabooga/text-generation-webui) and we provide a step-by-step guide below.
  <details>
  <summary> Step-by-step guide to using oogabooga to generate an URL</summary>
    1. Manually install oogabooga Text generation web UI by following the steps described here: https://github.com/oobabooga/text-generation-webui (section "Setup details and information about installing manually")

    2. Launch a server: python server.py  --gradio-auth username:password --listen --public-api --share

    3. This will output an OpenAI-compatible API URL: https://xxxx-xxxx-xxxx-xxxx.trycloudflare, and a "gradio.live" URL: "Running on public URL: https://xxxxxxxx.gradio.live"

    4. Paste the OpenAI-compatible URL in the field "Server access URL" of the LLM-Culture GUI.

    5. Open the gradio.live URL in your browser. 

    6. Go to the model tab and download a model from [huggingface](https://huggingface.co). We used https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF, with File name "mistral-7b-openorca.Q4_K_M.gguf". Select an appropriate Model loader (we used llama.cpp). 

    7. Click on Load to load the model. 

    8. Once the model is loaded, you can go back to the LLM-Culture GUI and run your simulations!
    

      
    </details>
</details>

## Usage (command line)



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
