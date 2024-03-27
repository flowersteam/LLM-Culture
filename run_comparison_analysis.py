import argparse

from llm_culture.analysis.utils import get_stories, get_plotting_infos, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities, get_creativity_indexes
from llm_culture.analysis.plots import *
from llm_culture.analysis.comparison_plots import *



def main_analysis(folders, plot, scale_y_axis, labels, sizes):
    saving_folder = '-'.join(os.path.basename(folder) for folder in folders)
    if len(saving_folder) > 250:
        saving_folder = saving_folder[:200] + '_NAME_TOO_LONG'
    data = {}
        
    for i, folder in enumerate(folders):
        # Compute all the metric that will be used for plotting
        all_seeds_stories = get_stories(folder)
        n_gen, n_agents, x_ticks_space = get_plotting_infos(all_seeds_stories[0])
        all_seed_flat_stories, all_seed_keywords, all_seed_stem_words = preprocess_stories(all_seeds_stories)
        all_seed_similarity_matrix = get_similarity_matrix(all_seed_flat_stories)
        all_seed_between_gen_similarity_matrix = compute_between_gen_similarities(all_seed_similarity_matrix, n_gen, n_agents)
        all_seed_polarities, all_seed_subjectivities = get_polarities_subjectivities(all_seeds_stories)
        all_seed_creativities = get_creativity_indexes(all_seeds_stories, folder)
        label = labels[i]
        # Plot the individual similarity matrix in the same folder
        for seed in range(len(all_seeds_stories)):
            plot_similarity_matrix(all_seed_similarity_matrix[seed], label, n_gen, n_agents, plot, sizes, saving_folder, seed = seed)
            #plot_similarity_graph(all_seed_between_gen_similarity_matrix[seed], "Comparisons/"+saving_folder, plot, sizes)

        data[folder] = {
            'all_seed_stories': all_seeds_stories,
            'n_gen': n_gen,
            'n_agents': n_agents,
            'x_ticks_space': x_ticks_space,
            'all_seeds_flat_stories': all_seed_flat_stories,
            'all_seeds_keywords': all_seed_keywords,
            'all_seeds_stem_words': all_seed_stem_words,
            'all_seeds_similarity_matrix': all_seed_similarity_matrix,
            'all_seeds_between_gen_similarity_matrix': all_seed_between_gen_similarity_matrix,
            'all_seeds_positivities': all_seed_polarities,
            'all_seeds_subjectivities': all_seed_subjectivities,
            'all_seeds_creativity_indices': all_seed_creativities,
            'label': label
            }
   
    # Plot all the desired graphs :

    compare_init_generation_similarity_evolution(data, plot, sizes, saving_folder, scale_y_axis)
    compare_within_generation_similarity_evolution(data, plot, sizes, saving_folder, scale_y_axis)
    compare_successive_generations_similarities(data, plot, sizes, saving_folder, scale_y_axis)
    compare_positivity_evolution(data, plot, sizes, saving_folder, scale_y_axis)
    compare_subjectivity_evolution(data, plot, sizes, saving_folder, scale_y_axis)
    compare_creativity_evolution(data, plot, sizes, saving_folder, scale_y_axis)

    plot_all_similarity_graphs(data, plot, sizes, saving_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Enter the names of the experiments separated by '+'
    parser.add_argument("--dirs", type=str, default="FC_10_10_combine_crea_not_crea+FC_10_10_combine_creative_2")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scale_y_axis", action="store_true")
    parser.add_argument("--labels", type=str, default="None")
    parser.add_argument("--ticks_font_size", type=int, default=16)
    parser.add_argument("--labels_font_size", type=int, default=18)
    parser.add_argument("--legend_font_size", type=int, default=16)
    parser.add_argument("--title_font_size", type=int, default=23)
    parser.add_argument("--matrix_size", type=int, default=8)
    args = parser.parse_args()

    analyzed_dirs = args.dirs.split('+')
    dirs_list = [f"Results/{dir_name}" for dir_name in analyzed_dirs]

    labels = args.labels.split('+')
    sizes = {'ticks': args.ticks_font_size,
                  'labels': args.labels_font_size,
                  'legend': args.legend_font_size,
                  'title': args.title_font_size,
                  'matrix': args.matrix_size}

    print(f"\nLaunching analysis on the {args.dirs} results")
    print(f"plot = {args.plot}")
    main_analysis(dirs_list, args.plot, args.scale_y_axis, labels, sizes)