""" Test file used to call the main_analysis function, will be replaced later on    """

from llm_culture.analysis.utils import get_stories, get_plotting_infos, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities, get_creativity_indexes
from llm_culture.analysis.plots import *


def main_analysis(folder, font_sizes = {'ticks': 12, 'labels': 14, 'title': 16}, plot=False):
    # Extract data from stories
    all_seeds_stories = get_stories(folder)

    print(len(all_seeds_stories))
    n_seeds = len(all_seeds_stories)
    n_gen, n_agents, x_ticks_space = get_plotting_infos(all_seeds_stories[0]) #same for all seeds

    all_seeds_flat_stories, all_seeds_keywords, all_seeds_stem_words = preprocess_stories(all_seeds_stories)
    all_seeds_similarity_matrix = get_similarity_matrix(all_seeds_flat_stories)
    all_seeds_between_gen_similarity_matrix = compute_between_gen_similarities(all_seeds_similarity_matrix, n_gen, n_agents)
    all_seeds_polarities, all_seeds_subjectivities = get_polarities_subjectivities(all_seeds_stories)
    all_seeds_creativities = get_creativity_indexes(all_seeds_stories, folder)

    # Plot all the desired graphs :
    for seed in range(n_seeds):
        print(seed)
        similarity_matrix = all_seeds_similarity_matrix[seed]
        between_gen_similarity_matrix = all_seeds_between_gen_similarity_matrix[seed]
        stem_words = all_seeds_stem_words[seed]

        # Individual plots for the current seed 
        plot_similarity_matrix(similarity_matrix, n_gen, n_agents, folder, plot, seed=seed, sizes=font_sizes)
        plot_between_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, seed=seed, sizes=font_sizes)
        plot_word_chains(stem_words, folder, plot, x_ticks_space, seed=seed, sizes=font_sizes)
        plot_similarity_graph(between_gen_similarity_matrix, folder, plot, seed=seed, sizes=font_sizes)

    # Plots comparing all the seeds
    plot_init_generation_similarity_evolution(all_seeds_between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes=font_sizes)
    plot_within_gen_similarities(all_seeds_between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes=font_sizes)
    plot_successive_generations_similarities(all_seeds_between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes=font_sizes)
    plot_positivity_evolution(all_seeds_polarities, folder, plot, x_ticks_space, sizes=font_sizes)
    plot_subjectivity_evolution(all_seeds_subjectivities, folder, plot, x_ticks_space, sizes=font_sizes)
    plot_creativity_evolution(all_seeds_creativities, folder, plot, x_ticks_space, sizes=font_sizes)