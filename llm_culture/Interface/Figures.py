import argparse
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import json

import networkx as nx
from matplotlib import pyplot as plt

import run_simulation
from PIL import Image, ImageTk

from llm_culture.Interface.utils import create_combobox_from_json,  append_to_json, add_item_dialog, remove_item, reveal_content, DotSelector, reveal_add_item, browse_folder
import os 


class FiguresFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        

        self.labels = []
        self.images = []


        # notebook = ttk.Notebook(self)

        # """Update the image at the given index with a new image."""
        # # if index < 0 or index >= len(self.labels):
        # #     raise ValueError("Index out of range")

        # for index, path in enumerate(image_paths):
        #     new_tab = FigureTab(self, path)
        #     notebook.add(new_tab, text = 'Fig')

        # notebook.pack(expand=True, fill='both')


        


    def update_images(self, folder_path):
        for widget in self.winfo_children():
            widget.destroy()
            
        images_paths = []
        images_names = []
        notebook = ttk.Notebook(self)

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith('.png'):
                path = os.path.join(folder_path, file_name)

                new_tab = FigureTab(self, path)
                notebook.add(new_tab, text = file_name)
              
              
            

        """Update the image at the given index with a new image."""
        # if index < 0 or index >= len(self.labels):
        #     raise ValueError("Index out of range")

        # for index, path in enumerate(images_paths):
        #     new_tab = FigureTab(self, path)
        #     notebook.add(new_tab, text = 'Fig')

        notebook.pack(expand=True, fill='both')


class FigureTab(tk.Frame):
    def __init__(self, parent, image_path):
        super().__init__(parent)
        self.labels = []
        self.images = []


        img = tk.PhotoImage(file=image_path)
        label = tk.Label(self, image=img)
        label.image = img  # Keep a reference to the image
        label.grid(row=0, column=0, padx=5, pady=5)
        self.labels.append(label)
        self.images.append(img)