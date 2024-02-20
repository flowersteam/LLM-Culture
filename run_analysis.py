import json
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_culture.Analysis.utils import extract_keywords, lemmatize_stemming
from llm_culture.Analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities, compute_creativity_indexes
from llm_culture.Analysis.plots import *


def main_analysis(folder, plot=False):
    #  # Plot the network structure
    # plt.figure(figsize=(1, 1))
    # # TODO : What is network structure
    # nx.draw(network_structure, with_labels=True)
    # plt.savefig(folder + '/network_structure.png')  # Save the plot as an image file
    # plt.clf()

    ## Retrive stories
    json_file  = folder + '/output.json'
    with open(json_file, 'r') as file:
            data = json.load(file)
            stories = data['stories']

    n_gen = len(stories)
    n_agents = len(stories[0])
    x_ticks_space = n_gen // 10 if n_gen >= 20 else 1


    flat_stories = [stories[i][j] for i in range(len(stories)) for j in range(len(stories[0]))]
    all_stories_words = [stories[i][j].split(' ') for i in range(len(stories)) for j in range(len(stories[0]))]
    
        
    keywords = [list(map(extract_keywords, s)) for s in stories]
    stem_words = [[list(map(lemmatize_stemming, keyword)) for keyword in keyword_list] for keyword_list in keywords]
    flat_keywords = list(map(extract_keywords,flat_stories))

    vect = TfidfVectorizer(min_df=1, stop_words="english")     
    tfidf = vect.fit_transform(flat_stories)                                                                                                                                                                                                                       
    similarity_matrix = tfidf * tfidf.T 
    similarity_matrix = similarity_matrix.toarray()

    # Plot all the desired graphs :

    plot_similarity_matrix(similarity_matrix, n_gen, n_agents, folder, plot)

    between_gen_similarity_matrix = compute_between_gen_similarities(similarity_matrix, n_gen, n_agents)

    plot_between_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space)

    plot_similarity_graph(between_gen_similarity_matrix, folder, plot)

    plot_init_generation_similarity_evolution(between_gen_similarity_matrix, folder, plot, x_ticks_space)

    plot_within_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space)

    plot_successive_generations_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space)

    polarities, subjectivities = get_polarities_subjectivities(stories)

    plot_positivity_evolution(polarities, folder, plot, x_ticks_space)

    plot_subjectivity_evolution(subjectivities, folder, plot, x_ticks_space)

    plot_word_chains(stem_words, folder, plot, x_ticks_space)

    print("Computing creativity indexes of stories ...")
    creativities = compute_creativity_indexes(stories)

    # TODO : Change the function because 1 = low creativity and 0 = high 
    plot_creativity_evolution(creativities, folder, plot, x_ticks_space)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="Results/Chain_20")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print(f"\nLaunching analysis on the {args.dir} results")
    print(f"plot = {args.plot}")
    main_analysis(args.dir, args.plot)