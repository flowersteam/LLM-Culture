import tkinter as tk
from tkinter import ttk
import json
from tkinter import filedialog

class DotSelector(tk.Frame):
    def __init__(self, master, options, callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.options = options
        self.callback = callback
        
        self.selected_option = tk.StringVar()
        
        for i, option in enumerate(self.options):
            dot = tk.Label(self, text="\u2022", font=("Arial", 20), padx=10)
            dot.grid(row=0, column= i)
            dot.bind("<Button-1>", lambda event, idx=i: self.select_option(idx))
        
    def select_option(self, idx):
        selected_option = self.options[idx]
        self.selected_option.set(selected_option)
        if self.callback:
            self.callback(selected_option)

def browse_folder(folder_var, initialdir = './Results'):
    folder_path = filedialog.askdirectory(initialdir= initialdir)
    if folder_path:
        folder_var.set(folder_path)
        print("Selected folder:", folder_path)

def reveal_add_item(selection, value = None, item = None):
    if selection.get() == value:
        item.grid()
    else:
        item.grid_remove()

def create_combobox_from_json(json_file, combobox):
    # Load data from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
        #for item in data:
        combobox['values'] = [d['name'] for d in data]

    # Add items from JSON data to the combobox
    



def append_to_json(json_file, item):
    # Load existing data from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Append new item to the data
    data.append(item)

    # Write updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def remove_from_json(json_file, combobox):
    # Load existing data from JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    index = combobox.current()
    # Remove item from the data
    del data[index]

    # Write updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

def add_item_dialog(root, json_file, combobox):
            
            # Create a dialog window
            dialog = tk.Toplevel(root)
            
            # Add Name and Prompt fields
            dialog.geometry("300x150")
            dialog.configure(bg="white")
            tk.Label(dialog, text="Name:", bg="white").pack()
            name_entry = tk.Entry(dialog)
            name_entry.pack(pady=5)
            tk.Label(dialog, text="Prompt:", bg="white").pack()
            prompt_entry = tk.Entry(dialog)
            prompt_entry.pack(pady=5)

            # Function to handle adding new item
            def add_item():
                name = name_entry.get()
                prompt = prompt_entry.get()
                if name and prompt:
                    new_item = {'name': name, 'prompt': prompt}
                    append_to_json(json_file, new_item)
                    create_combobox_from_json(json_file, combobox)
                    dialog.destroy()

            # Add Add Button
            add_button = tk.Button(dialog, text="Add", command=add_item)
            add_button.pack(pady=5)

def remove_item(json_file, combobox):
    selected_item = combobox.get()
    if selected_item:
        remove_from_json(json_file, combobox)
        create_combobox_from_json(json_file, combobox)
        combobox.set('')

def reveal_content(json_file, combobox):
    with open(json_file, 'r') as file:
        data = json.load(file)
        index = combobox.current()
        tk.messagebox.showinfo(title = 'Content', message = data[index]['prompt'] ) 
        
