from llm_culture.simulation.server_answer import get_answer

# Separator between an agent's personality and its prompt.
PROMPT_SEP = '\n'


class Agent:
    def __init__(
            self,
            cfg,
            agent_id,
            init_prompt,
            prompt_update,
            personality,
            wait=0,
            sequence=False,
            llm_backend=False,
            model=None,
            sampling_params=None,
        ):
        self.agent_id = agent_id
        self.neighbours = []
        self.init_prompt = init_prompt
        self.prompt_update = prompt_update
        self.personality = personality
        self.story = None
        self.prompt = None
        self.wait = wait
        self.sequence = sequence
        # scalar parameters shared across all agents, read from the experiment config
        self.access_url = cfg.backend.access_url
        self.debug = cfg.debug
        self.instruct = cfg.generation.instruct
        self.generation = cfg.generation
        # runtime objects (loaded model / backend tag)
        self.llm_backend = llm_backend
        self.model = model
        self.sampling_params = sampling_params

    def update_neighbours(self, graph, agentList):
        """Update the neighbours of the agent

        :param graph: networkx graph
        :param agentList: list of agents
        """
        # get the neighbours of the agent from the graph (networkx graph)
        if graph.is_directed():
            # if graph is a DiGraph, the neighbours are the predecessors
            self.neighbours = [agentList[i] for i in list(graph.predecessors(self.agent_id))]
        else:
            self.neighbours = [agentList[i] for i in list(graph.neighbors(self.agent_id))]

    def get_neighbours_stories(self):
        """Get the stories of the neighbours

        :return: list of stories
        """
        return [neighbour.get_story() for neighbour in self.neighbours if neighbour.get_story() is not None]

    def update_prompt(self):
        """Update the prompt of the agent based on the stories of the neighbours
        """
        if (self.wait == 0 and self.sequence) or (self.wait <= 0 and not(self.sequence)):
            neighbours_stories = self.get_neighbours_stories()
            if len(neighbours_stories) != 0:
                prompt = self.personality + PROMPT_SEP + self.prompt_update + str([str(i) + ': '+ neighbours_stories[i] for i in range(len(neighbours_stories))])
            else:
                prompt = self.personality + PROMPT_SEP + self.init_prompt
        else:
            prompt = None
        
        self.prompt = prompt

    def update_story(self):
        """Generate this agent's next story from its current prompt.

        Only used for one-off/manual calls: a normal run generates a whole timestep at
        once via `server_answer.get_answers_batch` and assigns results with `set_story`.
        """
        if (self.wait == 0 and self.sequence) or (self.wait <= 0 and not(self.sequence)):
            self.story = get_answer(
                self.access_url,
                self.prompt,
                self.generation,
                debug=self.debug,
                instruct=self.instruct,
                llm_backend=self.llm_backend,
                model=self.model,
                sampling_params=self.sampling_params,
            )
        else:
            self.story = None
        self.decrease_wait()

    def get_story(self):
        """Get the story of the agent
        """
        return self.story

    def set_story(self, text):
        """Set the agent's story directly (used by the batched update path)."""
        self.story = text

    def decrease_wait(self):
        """Decrease the wait time
        """
        self.wait -= 1

    def __str__(self):
        return f'Agent {self.agent_id}'

