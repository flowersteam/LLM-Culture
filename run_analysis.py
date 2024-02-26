import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_culture.analysis.utils import get_stories, get_plotting_infos, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities, get_creativity_indexes
from llm_culture.analysis.plots import *


def main_analysis(folder, font_sizes, plot=False):
    # Extract data from stories
    stories = get_stories(folder)
    n_gen, n_agents, x_ticks_space = get_plotting_infos(stories)

    flat_stories, keywords, stem_words = preprocess_stories(stories)
    similarity_matrix = get_similarity_matrix(flat_stories)
    between_gen_similarity_matrix = compute_between_gen_similarities(similarity_matrix, n_gen, n_agents)
    polarities, subjectivities = get_polarities_subjectivities(stories)
    creativities = get_creativity_indexes(stories, folder)


    # Plot all the desired graphs :

    plot_similarity_matrix(similarity_matrix, n_gen, n_agents, folder, plot, font_sizes)
    plot_between_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, font_sizes)
    plot_similarity_graph(between_gen_similarity_matrix, folder, plot, font_sizes)
    plot_init_generation_similarity_evolution(between_gen_similarity_matrix, folder, plot, x_ticks_space, font_sizes)
    plot_within_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, font_sizes)
    plot_successive_generations_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, font_sizes)
    plot_positivity_evolution(polarities, folder, plot, x_ticks_space, font_sizes)
    plot_subjectivity_evolution(subjectivities, folder, plot, x_ticks_space, font_sizes)
    plot_word_chains(stem_words, folder, plot, x_ticks_space, font_sizes)
    plot_creativity_evolution(creativities, folder, plot, x_ticks_space, font_sizes)
    # 1 = low creativity and 0 = high crzativity with this function


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="Chain_20")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--ticks_font_size", type=int, default=12)
    parser.add_argument("--labels_font_size", type=int, default=14)
    parser.add_argument("--title_font_size", type=int, default=16)
    args = parser.parse_args()

    analyzed_dir = f"Results/{args.dir}"
    font_sizes = {'ticks': args.ticks_font_size,
                  'labels': args.labels_font_size,
                  'title': args.title_font_size}
    
    print(f"\nLaunching analysis on the {analyzed_dir} results")
    print(f"plot = {args.plot}")
    main_analysis(analyzed_dir, font_sizes, args.plot)