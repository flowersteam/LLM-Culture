# This file contains the Agent class which is used to create the agents in the simulation.

from llm_culture.Simulation.server_answer import get_answer


class Agent:
    def __init__(self, agent_id, network_structure, init_prompt, prompt_update, personality, access_url,
                 wait=0, string_sep='\n', debug=False, sequence = False):
        self.agent_id = agent_id
        self.is_new = True
        self.neighbours = []
       # self.update_neighbours(network_structure)
        self.init_prompt = init_prompt
        self.prompt_update = prompt_update
        self.personality = personality
        self.string_sep = '\n'
        self.story = None
        self.prompt = None
        self.wait = wait
        self.debug = debug
        self.go = True
        self.access_url = access_url
        self.sequence = sequence

    def update_neighbours(self, graph, agentList):
        # get the neighbours of the agent from the graph (networkx graph)
        # if graph is a DiGraph, the neighbours are the predecessors
        if graph.is_directed():
            self.neighbours = [agentList[i] for i in list(graph.predecessors(self.agent_id))]
        else:
            self.neighbours = [agentList[i] for i in list(graph.neighbors(self.agent_id))]

    def get_neighbours_stories(self):
        #print(neighbour.get_story() for neighbour in self.neighbours if neighbour.get_story() is not None)
        return [neighbour.get_story() for neighbour in self.neighbours if neighbour.get_story() is not None]

    def update_prompt(self):
        print(f'wait: {self.wait}')
        print(f'sequence:{self.sequence}' )

        if (self.wait == 0 and self.sequence) or (self.wait <= 0 and not(self.sequence)):
            neighbours_stories = self.get_neighbours_stories()
            if len(neighbours_stories) != 0:
                print('here1')
                prompt = self.personality + self.string_sep + self.prompt_update + str([str(i) + ': '+ neighbours_stories[i] for i in range(len(neighbours_stories))])
            else:
                print('here2')
                prompt = self.personality + self.string_sep + self.init_prompt
            

        else:
            print('here3')
            prompt = None
        
        self.prompt = prompt
        # if self.is_new:
        #     if self.wait == 0:
        #         self.go = True
        #         neighbours_stories = self.get_neighbours_stories()
        #         if len(neighbours_stories) != 0:
        #             prompt = self.personality + self.string_sep + self.prompt_update
        #             prompt += self.string_sep.join(neighbours_stories)
        #             print('here1')
        #         else:
        #             prompt = self.personality + self.string_sep + self.init_prompt
        #             print('here2')
        #         self.is_new = False
        #     else:
        #         prompt = None
        #         print('here3')
        # else:
        #     if self.wait == 0:
        #         prompt = self.personality + self.string_sep + self.prompt_update
        #         # concatenate the neighbours stories and then concatenate to the prompt
        #         neighbours_stories = self.get_neighbours_stories()

        #         if len(neighbours_stories) != 0:
        #             self.go = False
        #             prompt += self.string_sep.join(neighbours_stories)
        #             self.is_new = False
        #             print('here4')
        #     else:
        #         prompt = None
        #         print('here5')
        # self.prompt = prompt

    def update_story(self):
        # if self.wait == 0 and self.go:
        if (self.wait == 0 and self.sequence) or (self.wait <= 0 and not(self.sequence)):
            self.story = get_answer(self.access_url, self.prompt, debug=self.debug)
            #print(f"AgentId: {self.agent_id}) /n Prompt: {self.prompt} /n Answer: {self.story}")
            print(f"Prompt: {self.prompt}")
            print(f"neigbhours stories: {self.get_neighbours_stories()}")
        else:
            self.story = None
        self.decrease_wait()


    def get_story(self):
        return self.story

    def decrease_wait(self):
        self.wait -= 1

    def get_updated_story(self):
        self.update_story()
        story = self.get_story()
        return story

    def get_current_state(self):
        state_dict = {
            "agent_id": self.agent_id,
            "is_new": self.is_new,
            "neighbours": [n.agent_id for n in self.neighbours],
            "init_prompt": self.init_prompt,
            "prompt_update": self.prompt_update,
            "personality": self.personality,
            "string_sep": self.string_sep,
            "story": self.story,
            "prompt": self.prompt,
            "wait": self.wait,
            "debug": self.debug,
            "go": self.go
        }
        return state_dict

    def __str__(self):
        return f'Agent {self.agent_id}'

