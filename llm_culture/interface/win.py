import tkinter as tk
from llm_culture.interface.notebook import CustomNotebookFrame

class AppWin(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("LLM-Culture")
        self.geometry("800x800")

        canvas = CustomNotebookFrame(self)
        canvas.pack(expand=True, fill=tk.BOTH)

# Création et affichage de la fenêtre
if __name__ == "__main__":
    app = AppWin()
    app.mainloop()
