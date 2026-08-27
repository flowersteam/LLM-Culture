import os
import json
import threading
import uuid

from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import send_from_directory

from scripts.run_analysis import main_analysis
from scripts.run_comparison_analysis import run_comparison_analysis
from scripts.run_simulation_interface import run_simulation

app = Flask(__name__)
RESULTS_DIR = 'results/experiments'
COMPARISON_DIR = 'results/experiments_comparisons'
SIMULATION_JOBS = {}
SIMULATION_JOBS_LOCK = threading.Lock()


# Home Page
@app.route('/')
def index():
    """Index page

    :return: index.html
    """
    return render_template('index.html')

# Run simulation
@app.route('/simulation', methods=['GET', 'POST'])
def simulation():
    """Run the simulation

    :return: simulation.html
    """
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

            # TODO : Change this mechanism so that you only have to provide experiment name
            output_dir = f"{RESULTS_DIR}/{experiment_name}"
            server_url = request.form.get('server_url')
            model_mode = request.form.get('model_mode', 'server')
            model_source = request.form.get('model_source')
            hf_cache_dir = request.form.get('hf_cache_dir')
            use_local_model = model_mode == 'local'

            if network_structure == 'sequence':
                assert n_agents == n_timesteps, "Number of agents must be equal to the number of timesteps for the sequence network structure"

            print("\nLaunching the analysis")
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
                server_url=server_url,
                use_local_model=use_local_model,
                model_source=model_source,
                hf_cache_dir=hf_cache_dir,
            )

            # Update the results dir with the new experiment and redirect to comparison
            result_dirs = _get_results_dir()
            return render_template('analysis.html', result_dirs=result_dirs)
  
    prompt_options = _get_prompt_options()

    return render_template('simulation.html', prompt_options=prompt_options)


@app.route('/simulation/run', methods=['POST'])
def simulation_run():
    """Start a simulation in the background and return a job id."""
    job_id, error_response = _launch_simulation_job(request.form)
    if error_response is not None:
        return error_response

    return jsonify({"job_id": job_id})


@app.route('/simulation/status/<job_id>', methods=['GET'])
def simulation_status(job_id):
    """Return the current status of a simulation job."""
    with SIMULATION_JOBS_LOCK:
        job = SIMULATION_JOBS.get(job_id)

    if job is None:
        return jsonify({"status": "missing", "message": "Unknown job id"}), 404

    return jsonify(job)

# Run analysis
@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Analyze the results

    :return: analysis.html
    """
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
    """Analyze the results

    :return: comparison_analysis.html
    """
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
    """Select the results directory

    :return: results.html
    """
    result_dirs =  _get_results_dir()
    results_comparison_dirs =  _get_results_comparisons_dirs()
    all_results_dirs = result_dirs + results_comparison_dirs
    return render_template('results.html', result_dirs=all_results_dirs)

@app.route('/results/<path:filename>')
def send_result_file_from_directory(filename):
    """Send the file from the results directory

    :param filename: file name
    :return: file
    """
    if os.path.exists(os.path.join(COMPARISON_DIR, filename)):
        results_dir = COMPARISON_DIR
    else:
        results_dir = RESULTS_DIR
    return send_from_directory(results_dir, filename)

# Observe analyzed dir plots
@app.route('/show_plots', methods=['POST'])
def show_plots():
    """Show the plots of the selected directory

    :return: plots.html
    """
    result_dir = request.form.get('result_dir')
    return redirect(url_for('plots', dir_name=result_dir))

@app.route('/plots/<dir_name>')
def plots(dir_name):
    """Show the plots of the selected directory

    :param dir_name: directory name
    :return: plots.html
    """
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
    """Sort the matrix at the start of the array if not comparison else at the end

    :param name: name of the plot
    :param comparison: wether it is a comparison or not
    :return: tuple with the priority and the name
    """
    # Sort the matrix at the start of the array if not comparison else at the end 
    matrix_prio, other_figs_prio = (1, 0) if comparison else(0, 1)
    return (matrix_prio, name) if "matrix" in name else (other_figs_prio, name)
        
def _get_plot_paths_dict(dir_name, plot_names, comparison):
    """Get the plot paths dictionary

    :param dir_name: directory name
    :param plot_names: plot names
    :param comparison: wether it is a comparison or not
    :return: plot paths dictionary
    """
    plot_paths = {plot_name: f"{dir_name}/{plot_name}.png" for plot_name in plot_names}
    return plot_paths

def _get_results_dir():
    """Get the results directories

    :return: results directories
    """
    results_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return results_dirs 

def _get_results_comparisons_dirs():
    """Get the comparison results directories

    :return: comparison results directories
    """
    comparison_dirs = [d for d in os.listdir(COMPARISON_DIR) if os.path.isdir(os.path.join(COMPARISON_DIR, d))]
    return comparison_dirs

def _get_prompt_options():
    """Get the prompt options

    :return: prompt options
    """
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


def _launch_simulation_job(form_data):
    """Create a background simulation job from submitted form data."""
    try:
        job_id = uuid.uuid4().hex
        job_state = {
            "status": "queued",
            "message": "Experiment in progress",
            "completed_generations": 0,
            "total_generations": 0,
            "current_seed": 0,
            "total_seeds": 0,
            "current_generation": 0,
            "total_generations_per_seed": 0,
        }

        with SIMULATION_JOBS_LOCK:
            SIMULATION_JOBS[job_id] = job_state

        simulation_args = _parse_simulation_form(form_data)

        total_generations = simulation_args["n_timesteps"] * simulation_args["n_seeds"]
        _update_simulation_job(
            job_id,
            status="running",
            message="Experiment in progress",
            total_generations=total_generations,
            total_seeds=simulation_args["n_seeds"],
            total_generations_per_seed=simulation_args["n_timesteps"],
        )

        def progress_callback(completed_generations, total_generations, current_seed, total_seeds, current_generation, total_generations_per_seed):
            _update_simulation_job(
                job_id,
                status="running",
                message="Experiment in progress",
                completed_generations=completed_generations,
                total_generations=total_generations,
                current_seed=current_seed,
                total_seeds=total_seeds,
                current_generation=current_generation,
                total_generations_per_seed=total_generations_per_seed,
            )

        def worker():
            try:
                _update_simulation_job(job_id, status="running", message="Preparing experiment")
                run_simulation(progress_callback=progress_callback, **simulation_args)
                _update_simulation_job(
                    job_id,
                    status="completed",
                    message="Experiment completed",
                    completed_generations=total_generations,
                )
            except Exception as exc:
                _update_simulation_job(job_id, status="error", message=str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return job_id, None
    except Exception as exc:
        return None, (jsonify({"status": "error", "message": str(exc)}), 400)


def _parse_simulation_form(form_data):
    """Parse the simulation form into arguments for run_simulation."""
    experiment_name = form_data.get('name')
    n_agents = int(form_data.get('n_agents'))
    n_timesteps = int(form_data.get('n_timesteps'))
    n_seeds = int(form_data.get('n_seeds'))
    network_structure = form_data.get('network_structure')
    n_cliques = int(form_data.get('n_cliques'))

    personalities = []
    init_prompts = []
    update_prompts = []
    for i in range(n_agents):
        personalities.append(form_data.get(f'personality_{i}'))
        init_prompts.append(form_data.get(f'prompt_init_{i}'))
        update_prompts.append(form_data.get(f'prompt_update_{i}'))

    init_prompt = init_prompts[0]
    update_prompt = update_prompts[0]
    output_dir = f"{RESULTS_DIR}/{experiment_name}"
    server_url = form_data.get('server_url')
    model_mode = form_data.get('model_mode', 'server')
    model_source = form_data.get('model_source')
    hf_cache_dir = form_data.get('hf_cache_dir')
    use_local_model = model_mode == 'local'

    if network_structure == 'sequence':
        assert n_agents == n_timesteps, "Number of agents must be equal to the number of timesteps for the sequence network structure"

    return {
        "n_agents": n_agents,
        "n_timesteps": n_timesteps,
        "n_seeds": n_seeds,
        "network_structure_name": network_structure,
        "n_cliques": n_cliques,
        "personalities": personalities,
        "init_prompt": init_prompt,
        "update_prompt": update_prompt,
        "output_dir": output_dir,
        "server_url": server_url,
        "use_local_model": use_local_model,
        "model_source": model_source,
        "hf_cache_dir": hf_cache_dir,
    }


def _update_simulation_job(job_id, **updates):
    with SIMULATION_JOBS_LOCK:
        job = SIMULATION_JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)

def _write_prompt_option(prompt_type, name, prompt):
    """Write the prompt option

    :param prompt_type: prompt type
    :param name: prompt name
    :param prompt: prompt content
    """
    file_path = f"data/parameters/{prompt_type}.json"
    with open(file_path, 'r') as f:
        prompts = json.load(f)

    new_prompt = {"name": name, "prompt": prompt}
    print(f"{new_prompt = }")
    prompts.append(new_prompt)

    with open(file_path, 'w') as f:
        json.dump(prompts, f, indent=4)


if __name__ == "__main__":
    app.run(debug=True, port=5002)
