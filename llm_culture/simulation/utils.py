from llm_culture.simulation.agent import Agent


def init_agents(
        n_agents, 
        network_structure, 
        prompt_init, 
        prompt_update, 
        personality_list, 
        access_url,
        sequence=False, 
        debug=False
    ):
    """Initialize the agents

    :param n_agents: n_agents
    :param network_structure: network_structure
    :param prompt_init: initial prompt
    :param prompt_update: update prompt
    :param personality_list: personality_list
    :param access_url: url to access the server
    :param sequence: sequence, defaults to False
    :param debug: debug flag, defaults to False
    :return: list of agents
    """
    agent_list = []
    wait = 0

    for agent_id in range(n_agents):
        personality = personality_list[agent_id]
        agent = Agent(
            agent_id, 
            network_structure, 
            prompt_init, 
            prompt_update, 
            personality, 
            access_url=access_url, 
            wait=wait,
            debug=debug, 
            sequence=sequence
        )
        agent_list.append(agent)
        if sequence:
            wait += 1

    return agent_list


def run_simul(
        access_url, 
        n_timesteps=5, 
        network_structure=None, 
        prompt_init=None, 
        prompt_update=None, 
        personality_list=None,
        n_agents=5, 
        sequence=False, 
        output_folder=None, 
        debug=False
    ):
    """Run the simulation

    :param access_url: url to access the server
    :param n_timesteps: n_timesteps, defaults to 5
    :param network_structure: network_structure, defaults to None
    :param prompt_init: prompt_init, defaults to None
    :param prompt_update: prompt_update, defaults to None
    :param personality_list: personality_list, defaults to None
    :param n_agents: n_agents, defaults to 5
    :param sequence: sequence, defaults to False
    :param output_folder: output_folder, defaults to None
    :param debug: debug, defaults to False
    :return: stories_history
    """
    # storage for the stories
    stories_history = []

    # initialize the agents
    agent_list = init_agents(
        n_agents, 
        network_structure, 
        prompt_init, 
        prompt_update, 
        personality_list, 
        access_url,
        sequence=sequence, 
        debug=debug
    )

    for agent in agent_list:
        agent.update_neighbours(network_structure, agent_list )

    # set the path to store the state history
    if output_folder is None:
        state_history_path = 'results/state_history.json'
    else:
        state_history_path = f'{output_folder}/state_history.json'

    # run the simulation
    for t in range(n_timesteps):
        new_stories = update_step(agent_list, t, state_history_path)
        print(f'\nTimestep: {t}')
        print(f'Number of new_stories: {len(new_stories)}')
        stories_history.append(new_stories)

    return stories_history


def update_step(
        agent_list, 
        timestep, 
        state_history_path
    ):
    """Update the agents

    :param agent_list: list of agents
    :param timestep: timestep
    :param state_history_path: path to store the state history
    :return: new_stories
    """
    # update the prompt of the agents
    new_stories = []

    for agent in agent_list:
        agent.update_prompt()

    for agent in agent_list:
        print(f'Agent: {agent.agent_id}')
        story = agent.get_updated_story()
        if story is not None:
            new_stories.append(story)

    return new_stories
