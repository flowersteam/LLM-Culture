# This file contains the main functions to run the simulation. 
#It is called by the run_simulation.py file.
from llm_culture.simulation.agent import Agent


def init_agents(n_agents, network_structure, prompt_init, prompt_update, personality_list, access_url,
                sequence=False, debug=False):
    agent_list = []
    wait = 0

    for a in range(n_agents):
        perso = personality_list[a]
        agent = Agent(a, network_structure, prompt_init, prompt_update, perso, access_url= access_url, wait=wait,
                      debug=debug, sequence = sequence)
        agent_list.append(agent)
        if sequence:
            wait += 1

    return agent_list


def run_simul(access_url, n_timesteps=5, network_structure=None, prompt_init=None, prompt_update=None, personality_list=None,
              n_agents=5, sequence=False, output_folder=None, debug=False):
    #STRORAGE
    stories_history = []

    #INTIALIZE AGENTS
    agent_list = init_agents(n_agents, network_structure, prompt_init, prompt_update, personality_list, access_url,
                             sequence=sequence, debug=debug)
    


    # print the agent id and wait time
    for agent in agent_list:
        agent.update_neighbours(network_structure, agent_list )
        # print(f'Agent: {agent.agent_id}, wait: {agent.wait}')

    #MAIN LOOP
    if output_folder is None:
        state_history_path = 'Results/state_history.json'
    else:
        state_history_path = f'{output_folder}/state_history.json'
    for t in range(n_timesteps):
        new_stories = update_step(agent_list, t, state_history_path)
        print(f'Timestep: {t}')
        print(f'Number of new_stories: {len(new_stories)}')
        stories_history.append(new_stories)

    return stories_history

def update_step(agent_list, timestep, state_history_path):
    #UPDATE LOOP
    new_stories = []

    for agent in agent_list:
        agent.update_prompt()

    for agent in agent_list:
        print(f'Agent: {agent.agent_id}')
        story = agent.get_updated_story()
        if story is not None:
            new_stories.append(story)
        # update the state history of the agent at the current timestep
        # update_state_history(agent, timestep, state_history_path)

    return new_stories
