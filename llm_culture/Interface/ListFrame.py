import tkinter as tk
from tkinter import ttk

class DataFrame(tk.Frame):
    """Frame affichant les attributs de type string d'un objet."""
    def __init__(self, parent, obj, **kwargs):
        super().__init__(parent, **kwargs)
        self.obj = obj
        self.create_widgets()

    def create_widgets(self):
        """Crée les widgets pour chaque attribut de type string de l'objet."""
        # Utiliser __dict__ pour obtenir les attributs dans leur ordre de définition
        for attr_name, attr_value in self.obj.__dict__.items():
            if isinstance(attr_value, str):
                label = tk.Label(self, text=f"{attr_name}: {attr_value}")
                label.pack(fill=tk.X, anchor='w') 

class ListDisplayFrame(tk.Frame):
    """Frame affichant une liste d'objets avec DataFrame."""
    def __init__(self, parent, objects, title="", on_select=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.objects = objects
        self.on_select = on_select
        self.create_widgets(title)

    def create_widgets(self, title):
        # Efface tous les widgets précédents
        for widget in self.winfo_children():
            widget.destroy()

        # Ajouter un titre si fourni
        if title:
            tk.Label(self, text=title).pack()

        # Afficher chaque objet dans la liste avec DataFrame
        for obj in self.objects:
            frame = DataFrame(self, obj)
            frame.pack(padx=5, pady=5)
            frame.bind("<Button-1>", lambda event, obj=obj: self.on_select(obj) if self.on_select else None)

    def add_object(self, obj):
        self.objects.append(obj)
        self.create_widgets("")

    def remove_object(self, obj):
        self.objects.remove(obj)
        self.create_widgets("")
