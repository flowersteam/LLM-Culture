import tkinter as tk
from tkinter import ttk
from llm_culture.interface.simulation import SimulationFrame
from llm_culture.interface.figures import FiguresFrame


class CustomNotebookFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        # Create the notebook
        notebook = ttk.Notebook(self)

        # Create the frames for each tab
        tab_simulation = SimulationFrame(notebook)
        tab_figures = FiguresFrame(notebook)

        # Add tabs to the notebook with a label
        notebook.add(tab_simulation, text='Simulation')
        notebook.add(tab_figures, text='Figures')

        # Pack the notebook into the CustomNotebookFrame
        notebook.pack(expand=True, fill='both')