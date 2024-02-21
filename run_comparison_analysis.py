import argparse

from llm_culture.analysis.utils import get_stories, get_plotting_infos, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities, get_creativity_indexes
from llm_culture.analysis.comparison_plots import *


def main_analysis(folders, plot, scale_y_axis):
    saving_folder = '-'.join(os.path.basename(folder) for folder in folders)
    data = {}
        
    for folder in folders:
        # Compute all the metric that will be used for plotting
        stories = get_stories(folder)
        n_gen, n_agents, x_ticks_space = get_plotting_infos(stories)
        flat_stories, keywords, stem_words = preprocess_stories(stories)
        similarity_matrix = get_similarity_matrix(flat_stories)
        between_gen_similarity_matrix = compute_between_gen_similarities(similarity_matrix, n_gen, n_agents)
        polarities, subjectivities = get_polarities_subjectivities(stories)
        creativities = get_creativity_indexes(stories, folder)

        data[folder] = {
            'stories': stories,
            'n_gen': n_gen,
            'n_agents': n_agents,
            'x_ticks_space': x_ticks_space,
            'flat_stories': flat_stories,
            'keywords': keywords,
            'stem_words': stem_words,
            'similarity_matrix': similarity_matrix,
            'between_gen_similarity_matrix': between_gen_similarity_matrix,
            'polarities': polarities,
            'subjectivities': subjectivities,
            'creativities': creativities
        }
   
    # Plot all the desired graphs :

    compare_init_generation_similarity_evolution(data, plot, saving_folder, scale_y_axis)

    compare_within_generation_similarity_evolution(data, plot, saving_folder, scale_y_axis)

    compare_successive_generations_similarities(data, plot, saving_folder, scale_y_axis)

    compare_positivity_evolution(data, plot, saving_folder, scale_y_axis)

    compare_subjectivity_evolution(data, plot, saving_folder, scale_y_axis)

    compare_creativity_evolution(data, data, saving_folder, scale_y_axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Enter the names of the experiments separated by '+'
    parser.add_argument("--dirs", type=str, default="Chain_20+Chain_50")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scale_y_axis", action="store_true")
    args = parser.parse_args()

    analyzed_dirs = args.dirs.split('+')
    dirs_list = [f"Results/{dir_name}" for dir_name in analyzed_dirs]

    print(f"\nLaunching analysis on the {args.dirs} results")
    print(f"plot = {args.plot}")
    main_analysis(dirs_list, args.plot, args.scale_y_axis)