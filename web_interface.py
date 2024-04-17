import os
import json

from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import send_from_directory

from scripts.run_analysis import main_analysis
from scripts.run_simulation_interface import run_simulation

app = Flask(__name__)
RESULTS_DIR = 'Results/experiments'
COMPARISON_DIR = 'Results/experiments_comparisons'
# COMPARISON_DIR = os.path.join(RESULTS_DIR, 'Comparisons')


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

            # TODO : Can be deleted now 
            params = {
            "Experiment name": experiment_name,
            "Number of agents": n_agents,
            "Number of timesteps": n_timesteps,
            "Number of seeds": n_seeds,
            "Network structure": network_structure,
            "Number of cliques": n_cliques,
            "Personalities": personalities,
            "Init prompts": init_prompt,
            "Update prompts": update_prompt,
            "Output directory": output_dir,
            "Server url": server_url
            }

            param_strings = [f"{key}: {value}" for key, value in params.items()]
            param_list = "\n".join(param_strings)

            run_analysis_message = f"Launching the analysis with the following parameters:\n\n{param_list}\n"
            print(run_analysis_message)

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
        # if directory.startswith('Comparisons'):
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

        print(selected_dirs)
        msg = f"{selected_dirs = }"
        return selected_dirs

        analyzed_dir = f"{RESULTS_DIR}/{directory}"
        font_sizes = {'ticks': ticks_font_size,
                      'labels': labels_font_size,
                      'title': title_font_size}

        print(f"\nLaunching analysis on the {analyzed_dir} results")
        # if directory.startswith('Comparisons'):
        main_analysis(analyzed_dir, font_sizes, plot=False)

        # Redirect to the new route that serves the generated plots
        return redirect(url_for('plots', dir_name=directory))

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
    return send_from_directory(RESULTS_DIR, filename)


# Observe analyzed dir plots
@app.route('/show_plots', methods=['POST'])
def show_plots():
    result_dir = request.form.get('result_dir')
    return redirect(url_for('plots', dir_name=result_dir))

@app.route('/plots/<dir_name>')
def plots(dir_name):
    plot_names = [
        'stories_similarity_matrix0',
        'between_gen_similarity_matrix0',
        'generation_similarities_graph0',
        'similarity_first_gen',
        'successive_similarity',
        'within_gen_similarity',
        'positivity_evolution',
        'creativity_evolution',
        'subjectivity_evolution',
        'wordchains0'
    ]

    plot_paths = _get_plot_paths(dir_name, plot_names)

    print(plot_paths)
    return render_template('plots.html', dir_name=dir_name, plot_names=plot_names, plot_paths=plot_paths)


# Helper functions
def _get_plot_paths(dir_name, plot_names):
    plot_paths = {}
    for plot_name in plot_names:
        file_name = f'{plot_name}.png'
        if os.path.exists(os.path.join(RESULTS_DIR, dir_name, file_name)):
            print(dir_name)
            print(file_name)
            plot_paths[plot_name] = f"{dir_name}/{file_name}"
            print(plot_paths[plot_name])
        else:
            print(f"File '{file_name}' not found in directory '{dir_name}'")
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
