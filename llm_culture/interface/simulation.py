import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from PIL import Image

import networkx as nx
from matplotlib import pyplot as plt

import run_simulation
from llm_culture.interface.utils import create_combobox_from_json,  append_to_json, add_item_dialog, remove_item, reveal_content, DotSelector, reveal_add_item, browse_folder
from run_analysis import main_analysis
import os


class SimulationFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self.empty_img_path = 'llm_culture/data/temp/empty_image.png'

        # Create an empty white image
        empty_image = Image.new('RGB', (200, 200), 'white')
        empty_image.save(self.empty_img_path)

        # Create the first horizontal frame
        horizontal_frame1 = tk.Frame(self)
        horizontal_frame1.pack(side='top', fill=tk.BOTH, expand=True)


        # Inside the first horizontal frame, create two vertical frames
        self.vertical_frame1 = ParametersFrame(horizontal_frame1)
        self.vertical_frame1.pack(side='left', fill=tk.BOTH, expand=True)

        # Create the second horizontal frame with the empty image
        self.horizontal_frame2 = GraphImagesFrame(self.vertical_frame1, [self.empty_img_path])
        self.horizontal_frame2.grid(row = 4, column= 6)

        vertical_frame2 = GraphImagesFrame(horizontal_frame1, [])
        vertical_frame2.pack(side='left', fill=tk.BOTH, expand=True)

        # # Create the "Create graph" button
        # create_graph_button = tk.Button(self.vertical_frame1.radio_frame, text="Display graph", command=self.create_graph)
        # create_graph_button.grid(column=6, row=0, sticky='ew', padx=10, pady=5)
        
        run_button = tk.Button(self, text="Run", command=self.run_simu)
        run_button.pack(side='bottom')
        # figures_button = tk.Button(self, text="Get Figures", command=self.create_figures)
        # figures_button.pack(side='bottom')

    def create_figures(self):
        folder_path = self.vertical_frame1.output_folder_path.get().strip()
        main_analysis(folder_path)
        notebook = self.parent
        figure_tab_name = notebook.tabs()[-1]
        figure_tab = notebook.nametowidget(figure_tab_name)

        figure_tab.update_images(folder_path)


    def create_graph(self):
        n_agents = self.vertical_frame1.n_agents_var.get()
        output_folder_path = self.vertical_frame1.output_folder_path.get().strip()  # Get the output folder path

        network_structure_type = self.vertical_frame1.network_structure.get()
   
        if network_structure_type == 'sequence':
            network_structure = nx.DiGraph()
            for i in range(n_agents - 1):
                network_structure.add_edge(i, i + 1)
            sequence = True
        elif network_structure_type == 'circle':
            network_structure = nx.cycle_graph(n_agents)
        elif network_structure_type == 'caveman':
            network_structure = nx.connected_caveman_graph(int(self.vertical_frame1.n_cliques.get()), n_agents // int(self.vertical_frame1.n_cliques.get()))
        elif network_structure_type == 'fully_connected':
            network_structure = nx.complete_graph(n_agents)
        else:
            raise ValueError("Invalid network structure type")


        # Plot the network structure
        plt.figure(figsize=(2, 2))
        nx.draw(network_structure, with_labels=True, node_size = 50, font_size = 6)
        try:
            plt.savefig(output_folder_path + '/network_structure.png')  # Save the plot as an image file
        except:
            plt.savefig(self.empty_img_path)  # Save the plot as an image file
        plt.clf()
        # Add the image to the GraphImagesFrame
        try:
            self.horizontal_frame2.update_image(0, output_folder_path + '/network_structure.png')
        except:
            self.horizontal_frame2.update_image(0, self.empty_img_path)
        # Call the main function from main_simu with the args namespace
            
            
    def run_simu(self):

        #Update the output folder path
        self.vertical_frame1.output_folder_path.set( os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/Results/' + self.vertical_frame1.title_entry.get())

         # Retrieve the values from the ParametersFrame
        n_agents = self.vertical_frame1.n_agents_var.get()
        n_timesteps = self.vertical_frame1.n_timesteps_var.get()
        prompt_init = self.vertical_frame1.prompt_init.get().strip()
        prompt_update = self.vertical_frame1.prompt_update.get().strip()
        network_structure_type = self.vertical_frame1.network_structure.get()
        personality_unique = self.vertical_frame1.personality.get()
        personality_list = self.vertical_frame1.get_personalities()
        output_folder_path = self.vertical_frame1.output_folder_path.get().strip()  # Get the output folder path
        access_url = self.vertical_frame1.access_url.get().strip()
        n_cliques = self.vertical_frame1.n_cliques.get()
        n_seeds = self.vertical_frame1.n_seeds_var.get()

        # Create an instance of argparse.Namespace and set its attributes
        args = argparse.Namespace()
        args.n_agents = n_agents
        args.n_timesteps = n_timesteps
        args.prompt_init = prompt_init
        args.prompt_update = prompt_update
        args.network_structure = network_structure_type
        args.n_seeds = n_seeds
        if self.vertical_frame1.same_personnalities.get():
            print('here2')

            args.personality_list = [personality_unique for _ in range(n_agents)]
        else:
            print('here')
            print(personality_list)
            args.personality_list = personality_list
        args.output = Path(output_folder_path).absolute()
        args.debug = False
        args.preset = None  # Add the preset attribute
        args.access_url = access_url
        args.n_cliques = n_cliques

        if network_structure_type == 'sequence':
            network_structure = nx.DiGraph()
            for i in range(n_agents - 1):
                network_structure.add_edge(i, i + 1)
            sequence = True
        elif network_structure_type == 'circle':
            network_structure = nx.cycle_graph(n_agents)
        elif network_structure_type == 'caveman':
            network_structure = nx.connected_caveman_graph(int(self.vertical_frame1.n_cliques.get()), n_agents // int(self.vertical_frame1.n_cliques.get()))
        elif network_structure_type == 'fully_connected':
            network_structure = nx.complete_graph(n_agents)
        else:
            raise ValueError("Invalid network structure type")
        run_simulation.main(args)

        self.create_figures()

class ParametersFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        json_file_prompt_init = 'llm_culture/data/parameters/prompt_init.json'
        json_file_prompt_update = 'llm_culture/data/parameters/prompt_update.json'
        json_file_structure = 'llm_culture/data/parameters/network_structure.json'
        json_file_personnalities = 'llm_culture/data/parameters/personnalities.json'

        self.parent = parent




        network_structure_options = {
            'sequence': 'Sequence',
            'circle': 'Circle',
            'caveman': 'Caveman',
            'fully_connected': 'Fully connected'#,
            # 'custom': 'Custom'
        }

        self.n_agents_var = tk.IntVar(value=6)
        self.n_timesteps_var = tk.IntVar(value=5)
        self.n_seeds_var = tk.IntVar(value=1)
        # self.prompt_init = tk.Text(self, height=3, width=20)
        self.prompt_init = ttk.Combobox(self, state= 'readonly', width=10)

        self.prompt_update = ttk.Combobox(self, state= 'readonly', width=10)

        self.network_structure = tk.StringVar()
        self.radio_frame = tk.Frame(self, background=self.cget('bg'))
        self.radio_frame.grid(row=5, column=0, padx=10, pady=10, columnspan = 5, sticky = 'W')
        label = tk.Label(self.radio_frame, text='Network Structure: ')
        label.grid(column=0, row=0, sticky='e', padx=10, pady=5)

        self.custom_structure = ttk.Combobox(self.radio_frame, state= 'readonly')

        for i, [key, value] in enumerate(network_structure_options.items()):
            radio_button = tk.Radiobutton(self.radio_frame, text=value, variable=self.network_structure, value=key,  background=self.cget('bg'), command=lambda : self.master.master.create_graph())
            radio_button.grid(column = i + 1, row = 0, padx=20, pady=10, sticky = 'w' )
            if key == 'caveman':
                label = tk.Label(self.radio_frame, text='Number of cliques: ')
                label.grid(column=i + 1, row=1, sticky='w')
                self.n_cliques = tk.Entry(self.radio_frame, width = 2)
                self.n_cliques.insert(0, "2") 
                self.n_cliques.grid(column = i + 2, row = 1, sticky = 'w')
                label.bind('<FocusOut>', lambda: self.master.master.create_graph() )

            # if key == 'custom':
            #     create_combobox_from_json(json_file_structure, self.custom_structure)
            #     self.custom_structure.grid(column=i + 1, row = 1)
            #     add_button_structure = tk.Button(self.radio_frame, text="Add Structure...", command=lambda: add_item_dialog(self, json_file_structure, self.custom_structure), )
            #     add_button_structure.grid(column=i + 2, row=1, sticky='ew', padx=10, pady=5)
            #     remove_button_structure = tk.Button(self.radio_frame, text="Remove this Structure", command=lambda: remove_item(json_file_structure, self.custom_structure))
            #     remove_button_structure.grid(column=i + 3, row=1, sticky='ew', padx=10, pady=5)





        self.personality = ttk.Combobox(self, state= 'readonly', width=10)
        self.output_folder_path = tk.StringVar()  # for the output folder path

        # Define the parameters with labels and input fields
        agent_entry = self.create_integer_input('Number of agents:', 0, self.n_agents_var)
        
       

        def update_perso_list(event):
            _ , _ , self.personalities = self.create_personnality_list(int(self.n_agents_var.get()), json_file_personnalities)

        agent_entry.bind('<FocusOut>', update_perso_list )

        self.create_integer_input('Number of timesteps:', 1, self.n_timesteps_var)

        self.create_integer_input('Number of seeds:', 2, self.n_seeds_var)
        # self.create_text_input('prompt_init:', 2, self.prompt_init,
        #                        "Tell me a story")  # Default text is "Tell me a story"
        
        
        
        
        self.create_selector('Initialization prompt', 3, None, self.prompt_init)
        create_combobox_from_json(json_file_prompt_init, self.prompt_init)
        # Button to add new item
        add_button_prompt_init = tk.Button(self, text="Add Item...", command=lambda: add_item_dialog(self, json_file_prompt_init, self.prompt_init), )
    #add_button.pack(pady=5)
        add_button_prompt_init.grid(column=2, row=3, sticky='ew', padx=10, pady=5)
        remove_button_prompt_init = tk.Button(self, text="Remove this Item", command=lambda: remove_item(json_file_prompt_init, self.prompt_init))
        remove_button_prompt_init.grid(column=3, row=3, sticky='ew', padx=10, pady=5)
        reveal_button_prompt_init = tk.Button(self, text="Reveal content", command=lambda: reveal_content(json_file_prompt_init, self.prompt_init))
        reveal_button_prompt_init.grid(column=4, row=3, sticky='ew', padx=10, pady=5)

        self.create_selector('Transformation prompt', 4, None, self.prompt_update)
        create_combobox_from_json(json_file_prompt_update, self.prompt_update)
        # Button to add new item
        add_button_prompt_update = tk.Button(self, text="Add Item...", command=lambda: add_item_dialog(self, json_file_prompt_update, self.prompt_update), )
    #add_button.pack(pady=5)
        add_button_prompt_update.grid(column=2, row=4, sticky='ew', padx=10, pady=5)
        remove_button_prompt_update = tk.Button(self, text="Remove this Item", command=lambda: remove_item(json_file_prompt_update, self.prompt_update))
        remove_button_prompt_update.grid(column=3, row=4, sticky='ew', padx=10, pady=5)
        reveal_button_prompt_update = tk.Button(self, text="Reveal content", command=lambda: reveal_content(json_file_prompt_update, self.prompt_update))
        reveal_button_prompt_update.grid(column=4, row=4, sticky='ew', padx=10, pady=5)


        






        # self.create_text_input('prompt_update:', 3, self.prompt_update,
        #                        "Here are stories you heard. Tell me a story.")  # Default text is "Here are stories you heard. Tell me a story."

  
        # self.create_selector('network_structure:', 4, network_structure_options, self.network_structure,
        #                      'sequence')  # Default value is 'sequence'




        self.same_personnalities = tk.BooleanVar()

        def toggle_visibility():
            if not(self.same_personnalities.get()):
                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
            else:
                canvas.pack_forget()
                scrollbar.pack_forget()



        checkbox = ttk.Checkbutton(self, text="Same for all agents", variable=self.same_personnalities, command=toggle_visibility)
        checkbox.grid(row = 6, column= 0)

       

        self.create_selector('personality:', 6, None, self.personality,
                             '', column=1)  
        create_combobox_from_json(json_file_personnalities, self.personality)
        # Button to add new item
        add_button_personnalities = tk.Button(self, text="Add Personnality...", command=lambda: add_item_dialog(self, json_file_personnalities, self.personality), )
    #add_button.pack(pady=5)
        add_button_personnalities.grid(column=3, row=6, sticky='ew', padx=10, pady=5)
        remove_button_personnalities = tk.Button(self, text="Remove this Item", command=lambda: remove_item(json_file_personnalities, self.personality))
        remove_button_personnalities.grid(column=4, row=6, sticky='ew', padx=10, pady=5)
        reveal_button_personnalities = tk.Button(self, text="Reveal content", command=lambda: reveal_content(json_file_personnalities, self.personality))
        reveal_button_personnalities.grid(column=5, row=6, sticky='ew', padx=10, pady=5)

        canvas, scrollbar, self.personalities = self.create_personnality_list(int(self.n_agents_var.get()), json_file_personnalities)
        
        



        #Simulation title textfield
        title_text = tk.Label(self, text = 'Simulation title:')
        title_text.grid(row = 10, column= 0, sticky= 'w')
        self.title_entry = tk.Entry(self, width= 10)
        self.title_entry.grid(row = 10, column= 1, sticky= 'w')
        self.title_entry.bind('<FocusOut>', lambda event: self.output_folder_path.set( os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/Results/' + self.title_entry.get()))
        
        # Create a "Browse" button

        
        # folder_text = tk.Label(self, text = 'Save to:')
        # folder_text.grid(row = 10, column= 0, sticky= 'w')
        # browse_button = tk.Button(self, text="Browse", command=lambda: browse_folder(self.output_folder_path))
        # browse_button.grid(row = 10, column= 0, sticky= 'e')

        saveto_label = tk.Label(self, text = 'Save to:', textvariable= self.output_folder_path)
        saveto_label.grid(row = 11, column= 1, columnspan=4, sticky='w')

        folder_label = tk.Label(self, text = 'Save to:')
        folder_label.grid(row = 11, column= 0, columnspan=4, sticky='w')

        self.access_url = tk.Entry(self, width= 10)

        self.create_text_input('Server Access URL:', 12, self.access_url,
                              "")  
        

    def get_personalities(self):
        print(self.personalities)
        return [perso.get() for perso in self.personalities]
    

    def create_personnality_list(self, n_agents, json_file):
        perso_frame = ttk.Frame(self)
        perso_frame.grid(row = 7, column = 0, columnspan=4, sticky= 'w')
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        # Create a canvas with scrollbar
        canvas = tk.Canvas(perso_frame)
        scrollbar = ttk.Scrollbar(perso_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", on_configure)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Add multiple comboboxes to the scrollable frame
        comboboxes = []
        for i in range(n_agents):
            label = tk.Label(scrollable_frame, text= f'Agent {i}')
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            combobox = ttk.Combobox(scrollable_frame)
            combobox.grid(row=i, column=1, padx=5, pady=5)
            create_combobox_from_json(json_file,combobox)
            comboboxes.append(combobox)

        return canvas, scrollbar, comboboxes

    def create_integer_input(self, label_text, row, variable, default_value=None):
        label = tk.Label(self, text=label_text)
        label.grid(column=0, row=row, sticky='w', padx=10, pady=5)
        entry = tk.Entry(self, textvariable=variable, width = 5)
        entry.grid(column=1, row=row, sticky='w', padx=10, pady=5)
        if default_value is not None:
            variable.set(default_value)
        return entry

    def create_text_input(self, label_text, row, text_area, default_text=''):
        label = tk.Label(self, text=label_text)
        label.grid(column=0, row=row, sticky='w', padx=10, pady=5)
        text_area.grid(column=1, row=row, sticky='e', padx=10, pady=5)

    def create_selector(self, label_text, row, options, selector, default_value='', column = 0):
        label = tk.Label(self, text=label_text)
        label.grid(column=column, row=row, sticky='w', padx=10, pady=5)

        # If options is a dictionary, get the keys
        if isinstance(options, dict):
            options = list(options.keys())

        selector['values'] = options
        selector.grid(column= column + 1 , row=row, sticky='e', padx=10, pady=5)
        if default_value is not None:
            selector.set(default_value)

    


class GraphImagesFrame(tk.Frame):
    def __init__(self, parent, image_paths):
        super().__init__(parent)
        self.labels = []  # List to store the label widgets
        self.images = []  # List to store the PhotoImage references

        # Iterate over the provided image paths to create and place image labels
        for index, path in enumerate(image_paths):
            img = tk.PhotoImage(file=path)
            label = tk.Label(self, image=img)
            label.image = img  # Keep a reference to the image
            label.grid(row=index, column=0, padx=5, pady=5)
            self.labels.append(label)
            self.images.append(img)

    def update_image(self, index, new_image_path):
        """Update the image at the given index with a new image."""
        if index < 0 or index >= len(self.labels):
            raise ValueError("Index out of range")

        # Load the new image
        try:
            new_img = tk.PhotoImage(file=new_image_path)  # Save the plot as an image file
        except:
            new_img = tk.PhotoImage(file='data/temp/network_structure.png')  # Save the plot as an image file
        

        # Update the label
        label = self.labels[index]
        label.configure(image=new_img)
        
        # Keep a reference to the new image
        label.image = new_img
        
        # Replace the old image reference with the new one
        self.images[index] = new_img
