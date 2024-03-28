import os

from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import send_from_directory

from dummy_main_analysis import main_analysis

app = Flask(__name__)
RESULTS_DIR = 'Results'


# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Run simulation
# TODO : Add a directory with the different type of prompts and parse it 
@app.route('/simulation', methods=['GET', 'POST'])
def simulation():
    if request.method == 'POST':
        experiment_name = request.form.get('name')
        n_agents = int(request.form.get('n_agents'))
        n_timesteps = int(request.form.get('n_timesteps'))
        network_structure = request.form.get('network_structure')
        n_cliques = int(request.form.get('n_cliques'))
        n_seeds = int(request.form.get('n_seeds'))
        
        # TODO : Change the logic to have a list of agents (maybe with a system of idx if there are many)
        prompt_init = request.form.get('prompt_init')
        prompt_update = request.form.get('prompt_update')
        personality_list = [p.strip() for p in request.form.get('personality_list').split(',')]

        output_dir = f"Results/{experiment_name}"
        output_file = 'output.json'

        return 'Launching the analysis '
    
    return render_template('simulation.html')


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


# Select analyzed dir
@app.route('/results')
def results():
    result_dirs = result_dirs = _get_results_dir()
    return render_template('results.html', result_dirs=result_dirs)

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
    plot_paths = _get_plot_paths(dir_name)
    return render_template('plots.html', dir_name=dir_name, **plot_paths)

  
# Helper functions
def _get_plot_paths(dir_name):
    return {
        'plot1_path': f'{dir_name}/between_gen_similarity_matrix.png',
        'plot2_path': f'{dir_name}/generation_similarities_graph.png'
    }

def _get_results_dir():
    return [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]


if __name__ == "__main__":
    app.run(debug=True)
