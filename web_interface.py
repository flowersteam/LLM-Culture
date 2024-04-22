import os
import json

from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import send_from_directory

from scripts.run_analysis import main_analysis
from scripts.run_comparison_analysis import run_comparison_analysis
from scripts.run_simulation_interface import run_simulation

app = Flask(__name__)
RESULTS_DIR = 'Results/experiments'
COMPARISON_DIR = 'Results/experiments_comparisons'


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Run simulation
@app.route('/simulation', methods=['GET', 'POST'])
def simulation():
    if request.method == 'POST':
        # You can either add a new prompt 
        if request.form.get('add_prompt'):
            prompt_type = request.form.get('prompt_type')
            name = request.form.get('prompt_name')
            prompt = request.form.get('prompt')
            _write_prompt_option(prompt_type, name, prompt)
        
        # Or run a simulation
        if request.form.get('run_simulation'):
            # Get the general simulation parameters
            experiment_name = request.form.get('name')
            n_agents = int(request.form.get('n_agents'))
            n_timesteps = int(request.form.get('n_timesteps'))
            n_seeds = int(request.form.get('n_seeds'))
            network_structure = request.form.get('network_structure')
            n_cliques = int(request.form.get('n_cliques'))

            # Get the agents parameters
            personalities = []
            init_prompts = []
            update_prompts = []
            for i in range(n_agents):
                personalities.append(request.form.get(f'personality_{i}'))
                init_prompts.append(request.form.get(f'prompt_init_{i}'))
                update_prompts.append(request.form.get(f'prompt_update_{i}'))

            # At the moment the simulation only takes 1 init and update prompt for all the agents (might change later)
            init_prompt = init_prompts[0]
            update_prompt = update_prompts[0]

            output_dir = f"Results/{experiment_name}"
            server_url = request.form.get('server_url')

            print(f"Launching the analysis")

            run_simulation(
                n_agents=n_agents,
                n_timesteps=n_timesteps,
                n_seeds=n_seeds,
                network_structure_name=network_structure,
                n_cliques=n_cliques,
                personalities=personalities,
                init_prompt=init_prompt,
                update_prompt=update_prompt,
                output_dir=output_dir,
                server_url=server_url
            )

            return "Simulation done"
  
    prompt_options = _get_prompt_options()

    return render_template('simulation.html', prompt_options=prompt_options)


# Run analysis
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    result_dirs = _get_results_dir()
    if request.method == 'POST':
        directory = request.form.get('result_dir')
        ticks_font_size = int(request.form.get('ticks_font_size'))
        labels_font_size = int(request.form.get('labels_font_size'))
        title_font_size = int(request.form.get('title_font_size'))

        analyzed_dir = f"{RESULTS_DIR}/{directory}"
        font_sizes = {'ticks': ticks_font_size,
                      'labels': labels_font_size,
                      'title': title_font_size}

        print(f"\nLaunching analysis on the {analyzed_dir} results")
        main_analysis(analyzed_dir, font_sizes, plot=False)

        # Redirect to the new route that serves the generated plots
        return redirect(url_for('plots', dir_name=directory))

    return render_template('analysis.html', result_dirs=result_dirs)


# Run analysis
@app.route('/comparison_analysis', methods=['GET', 'POST'])
def comparison_analysis():
    result_dirs = _get_results_dir()
    if request.method == 'POST':
        selected_dirs = request.form.getlist('result_dir[]')
        ticks_font_size = int(request.form.get('ticks_font_size'))
        labels_font_size = int(request.form.get('labels_font_size'))
        title_font_size = int(request.form.get('title_font_size'))

        dirs_list = [f"{RESULTS_DIR}/{dir_name}" for dir_name in selected_dirs]
        saving_folder = '-'.join(os.path.basename(folder) for folder in dirs_list)

        # Hard encoded the matrix and legend sizes but can add an option on the interface
        font_sizes = {'ticks': ticks_font_size,
                      'labels': labels_font_size,
                      'title': title_font_size,
                      'legend': 16,
                      'matrix': 8}
        
        print(f"\nLaunching comaprison analysis on the {dirs_list} results")
        run_comparison_analysis(dirs_list, plot=False, scale_y_axis=True, labels=selected_dirs,  sizes=font_sizes)

        # Redirect to the new route that serves the generated plots
        return redirect(url_for('plots', dir_name=saving_folder))

    return render_template('comparison_analysis.html', result_dirs=result_dirs)


# Select analyzed dir
@app.route('/results')
def results():
    result_dirs =  _get_results_dir()
    results_comparison_dirs =  _get_results_comparisons_dirs()
    all_results_dirs = result_dirs + results_comparison_dirs
    return render_template('results.html', result_dirs=all_results_dirs)

@app.route('/results/<path:filename>')
def send_result_file_from_directory(filename):
    if os.path.exists(os.path.join(COMPARISON_DIR, filename)):
        results_dir = COMPARISON_DIR
    else:
        results_dir = RESULTS_DIR
    return send_from_directory(results_dir, filename)


# Observe analyzed dir plots
@app.route('/show_plots', methods=['POST'])
def show_plots():
    result_dir = request.form.get('result_dir')
    return redirect(url_for('plots', dir_name=result_dir))

@app.route('/plots/<dir_name>')
def plots(dir_name):
    if os.path.exists(os.path.join(COMPARISON_DIR, dir_name)):
        comparison = True
        results_dir = COMPARISON_DIR
    else:
        comparison = False
        results_dir = RESULTS_DIR

    dir_path = os.path.join(results_dir, dir_name)
    plot_names = [f.replace('.png', '') for f in os.listdir(dir_path) if f.endswith('.png')]
    # Plot the matrix elements before the other figures if clasical experiment, and at the end if comparison
    plot_names.sort(key=lambda name: _matrix_sort_key(name, comparison))
    plot_paths = _get_plot_paths_dict(dir_name, plot_names, comparison)

    return render_template('plots.html', dir_name=dir_name, plot_names=plot_names, plot_paths=plot_paths)


# Helper functions

def _matrix_sort_key(name, comparison):
    # Sort the matrix at the start of the array if not comparison else at the end 
    matrix_prio, other_figs_prio = (1, 0) if comparison else(0, 1)
    return (matrix_prio, name) if "matrix" in name else (other_figs_prio, name)
        
def _get_plot_paths_dict(dir_name, plot_names, comparison):
    plot_paths = {plot_name: f"{dir_name}/{plot_name}.png" for plot_name in plot_names}
    return plot_paths

def _get_results_dir():
    results_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return results_dirs 

def _get_results_comparisons_dirs():
    comparison_dirs = [d for d in os.listdir(COMPARISON_DIR) if os.path.isdir(os.path.join(COMPARISON_DIR, d))]
    return comparison_dirs

def _get_prompt_options():
    option_files = {
        'initial_prompts': 'data/parameters/prompt_init.json',
        'update_prompts': 'data/parameters/prompt_update.json',
        'personalities': 'data/parameters/personalities.json'
    }

    options = {}
    for option_name, file_path in option_files.items():
        with open(file_path, 'r') as f:
            option_data = json.load(f)
        options[option_name] = [o['name'] for o in option_data]

    return options

def _write_prompt_option(prompt_type, name, prompt):
    file_path = f"data/parameters/{prompt_type}.json"
    with open(file_path, 'r') as f:
        prompts = json.load(f)

    new_prompt = {"name": name, "prompt": prompt}
    print(f"{new_prompt = }")
    prompts.append(new_prompt)

    with open(file_path, 'w') as f:
        json.dump(prompts, f, indent=4)


if __name__ == "__main__":
    app.run(debug=True)
