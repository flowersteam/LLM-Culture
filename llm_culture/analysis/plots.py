import importlib
import json
import os
import re
from glob import glob

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt 


def plot_similarity_matrix(similarity_matrix, n_gen, n_agents, folder, plot, sizes, save=True, seed = 0):
    plt.figure(figsize=(8, 8))
    plt.imshow(similarity_matrix, vmin=0, vmax=1, cmap='viridis')

    n_texts = similarity_matrix.shape[0]
    if n_texts < 20:
        x_ticks_space = 1
    elif n_texts >= 50:
        x_ticks_space = 10
    else:
        x_ticks_space = 5
    
    plt.xlabel('History idx', fontsize=sizes['labels'])
    plt.ylabel('History idx', fontsize=sizes['labels'])
    plt.title('Stories similarity Matrix', fontsize=sizes['title'])
    
    # Add black lines to delimit generations
    for i in range(int(n_gen)):      
        plt.axvline(x = i * n_agents - 0.5, color = 'black')
        plt.axhline(y = i * n_agents - 0.5, color = 'black')

    plt.xticks(range(0, similarity_matrix.shape[0], x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(range(0, similarity_matrix.shape[0], x_ticks_space), fontsize=sizes['ticks'])

    cbar = plt.colorbar(pad=0.02, shrink=0.83)

    if save:
        #check if folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(folder + '/stories_similarity_matrix'+str(seed)+'.png')
        print("Saved stories_similarity_matrix"+str(seed)+".png")

    if plot:
        plt.show()



def plot_between_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes, save=True, seed = 0):
    plt.figure(figsize=(8, 8))
    plt.xlabel('Generation idx', fontsize=sizes['labels'])
    plt.ylabel('Generation idx', fontsize=sizes['labels'])
    plt.title('Between generations similarity Matrix', fontsize=sizes['title'])
    plt.xticks(range(0, between_gen_similarity_matrix[0].shape[0], x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(range(0, between_gen_similarity_matrix[0].shape[0], x_ticks_space), fontsize=sizes['ticks'])

    plt.imshow(between_gen_similarity_matrix, vmin=0, vmax=1, cmap='viridis')
    cbar = plt.colorbar(pad=0.02, shrink=0.83)
   
    if save:
        plt.savefig(folder + '/between_gen_similarity_matrix' + str(seed)+'.png')
        print("Saved between_gen_similarity_matrix"+str(seed)+".png")
    
    if plot:
        plt.show()


def plot_init_generation_similarity_evolution(all_seeds_between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes, save=True, scale_y_axis = False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with the initial generation', fontsize=sizes['title'])
    plt.xlabel('Generations', fontsize=sizes['labels'])
    plt.ylabel('Similarity with first generation', fontsize=sizes['labels'])

    plt.ylim(0, 1)
    plt.xticks(range(0, all_seeds_between_gen_similarity_matrix[0].shape[0], x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.grid()

    mean_line, = plt.plot(range(1, all_seeds_between_gen_similarity_matrix[0].shape[0]), np.mean(all_seeds_between_gen_similarity_matrix, axis = 0)[0, 1:], label='Similarity with First History')
    plt.fill_between(range(1, all_seeds_between_gen_similarity_matrix[0].shape[0]), 
                     np.mean(all_seeds_between_gen_similarity_matrix, axis = 0)[0, 1:] - np.std(all_seeds_between_gen_similarity_matrix, axis = 0)[0, 1:],
                     np.mean(all_seeds_between_gen_similarity_matrix, axis = 0)[0, 1:] + np.std(all_seeds_between_gen_similarity_matrix, axis = 0)[0, 1:],
                     label='Similarity with First History', alpha=0.2)
    
    color = mean_line.get_color()
    for i in range(len(all_seeds_between_gen_similarity_matrix)):
        plt.plot(range(1, all_seeds_between_gen_similarity_matrix[0].shape[0]), all_seeds_between_gen_similarity_matrix[i][0, 1:], alpha=0.2, color = color)
    if save:
        plt.savefig(folder + '/similarity_first_gen.png')
        print("Saved similarity_first_gen.png")
    
    if plot:
        plt.show()
    

def plot_within_gen_similarities(all_seeds_between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes,save=True, scale_y_axis = False):
    plt.figure(figsize=(10, 6))
    plt.title('Within generations texts similarities', fontsize=sizes['title'])
    plt.xlabel('Generations', fontsize=sizes['labels'])
    plt.ylabel('Within-generation similarity', fontsize=sizes['labels'])
    plt.xticks(range(0, all_seeds_between_gen_similarity_matrix[0].shape[0], x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
    plt.ylim(0 , 1.1)
    plt.grid()

    mean_line, = plt.plot( np.diag(np.mean(all_seeds_between_gen_similarity_matrix, axis = 0)))
    
    plt.fill_between(range(all_seeds_between_gen_similarity_matrix[0].shape[0]), 
                     np.diag(np.mean(all_seeds_between_gen_similarity_matrix, axis = 0)) - np.diag(np.std(all_seeds_between_gen_similarity_matrix, axis = 0)),
                     np.diag(np.mean(all_seeds_between_gen_similarity_matrix, axis = 0)) + np.diag(np.std(all_seeds_between_gen_similarity_matrix, axis = 0)),
                     alpha=0.2)
    
    color = mean_line.get_color()
    for i in range(len(all_seeds_between_gen_similarity_matrix)):
        plt.plot(np.diag(all_seeds_between_gen_similarity_matrix[i]), alpha=0.2, color = color)

    if save:
        plt.savefig(folder + '/within_gen_similarity.png')
        print("Saved within_gen_similarity.png")
    if plot:
        plt.show()


def plot_successive_generations_similarities(all_seeds_between_gen_similarity_matrix, folder, plot, x_ticks_space, sizes, save=True, scale_y_axis = False):
    plt.figure(figsize=(10, 6))
    plt.title('Successive generations similarities', fontsize=sizes['title'])
    plt.xlabel('Generations', fontsize=sizes['labels'])
    plt.ylabel('Similarity between successive generations', fontsize=sizes['labels'])
    plt.xticks(range(0, all_seeds_between_gen_similarity_matrix[0].shape[0], x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.grid()
    n_seeds = len(all_seeds_between_gen_similarity_matrix)
    all_seeds_successive_sim = [[all_seeds_between_gen_similarity_matrix[seed][i, i+1] for i in range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1)] for seed in range(n_seeds)]
    
    print(len(all_seeds_successive_sim))

    mean_line, = plt.plot(np.mean(all_seeds_successive_sim, axis = 0))

    plt.fill_between(range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1), 
                     np.mean(all_seeds_successive_sim, axis = 0) - np.std(all_seeds_successive_sim, axis = 0),
                     np.mean(all_seeds_successive_sim, axis = 0) + np.std(all_seeds_successive_sim, axis = 0),
                     alpha=0.2)
    color = mean_line.get_color()
    for i in range(n_seeds):
        plt.plot(all_seeds_successive_sim[i], alpha=0.2, color = color)
    if save:
        plt.savefig(folder + '/successive_similarity.png')
        print("Saved successive_similarity.png")
    if plot:
        plt.show()


def plot_positivity_evolution(all_seeds_positivities, folder, plot, x_ticks_space, sizes, save=True, scale_y_axis = False):
    plt.figure(figsize=(10, 6))
    # 1 = positive, -1 = negative on the y axis
    all_seeds_gen_positivities = []
    for polarities in all_seeds_positivities:
        gen_positivities = []
        for p in polarities:
            gen_positivities.append(np.mean(p))
        all_seeds_gen_positivities.append(gen_positivities)

    plt.title("Evolution of positivity across generations", fontsize=sizes['title'])
    plt.ylabel("Positivity", fontsize=sizes['labels'])
    plt.ylabel("Generation", fontsize=sizes['labels'])

    plt.xticks(range(0, len(gen_positivities), x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(np.linspace(-1, 1, 11), fontsize=sizes['ticks'])
    plt.grid()
    plt.ylim(-1, 1)


    plt.grid()

    mean_line, = plt.plot(np.mean(all_seeds_gen_positivities, axis = 0))
    color = mean_line.get_color()

    plt.fill_between(range(len(gen_positivities)), 
                     np.mean(all_seeds_gen_positivities, axis = 0) - np.std(all_seeds_gen_positivities, axis = 0),
                     np.mean(all_seeds_gen_positivities, axis = 0) + np.std(all_seeds_gen_positivities, axis = 0),
                     alpha=0.2)
    for i in range(len(all_seeds_gen_positivities)):
        plt.plot(all_seeds_gen_positivities[i], alpha=0.2, color = color)

    if save:
        plt.savefig(folder + '/positivity_evolution.png')
        print("Saved positivity_evolution.png")

    if plot:
        plt.show()



def plot_subjectivity_evolution(all_seeds_subjectivities, folder, plot, x_ticks_space, sizes, save=True, scale_y_axis = False):
    """Subjectivity is the output that lies within [0,1] and refers to personal opinions and judgments"""
    plt.figure(figsize=(10, 6))
    all_see_gen_subjectivities = []
    for subjectivities in all_seeds_subjectivities:
        gen_subjectivities = []
        for gen_subjectivity in subjectivities:
            gen_subjectivities.append(np.mean(gen_subjectivity))
        all_see_gen_subjectivities.append(gen_subjectivities)

    plt.title("Evolution of subjectivity across generations", fontsize=sizes['title'])
    plt.ylabel("Positivity", fontsize=sizes['labels'])
    plt.ylabel("Generation", fontsize=sizes['labels'])

    plt.xticks(range(0, len(gen_subjectivities), x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
    plt.grid()

    mean_line, = plt.plot(np.mean(all_see_gen_subjectivities, axis = 0))
    color = mean_line.get_color()
    plt.fill_between(range(len(gen_subjectivities)), 
                     np.mean(all_see_gen_subjectivities, axis = 0) - np.std(all_see_gen_subjectivities, axis = 0),
                     np.mean(all_see_gen_subjectivities, axis = 0) + np.std(all_see_gen_subjectivities, axis = 0),
                     alpha=0.2)
    
    for i in range(len(all_see_gen_subjectivities)):
        plt.plot(all_see_gen_subjectivities[i], alpha=0.2, color = color)

    if save:
        plt.savefig(folder + '/subjectivity_evolution.png')
        print("Saved subjectivity_evolution.png")

    if plot:
        plt.show()

    

# def plot_creativity_evolution(all_seeds_creativity_indices, folder, plot, x_ticks_space, sizes, save=True, scale_y_axis = False):
#     """1 = high creativity and 0 = low creativity"""
#     plt.figure(figsize=(10, 6))
#     all_seeds_gen_creativities = []
#     for creativity_indices in all_seeds_creativity_indices:
#         gen_creativities = [1 - np.mean(gen_creativity) for gen_creativity in creativity_indices]
#         all_seeds_gen_creativities.append(gen_creativities)
#
#     plt.figure(figsize=(10, 6))
#     
#     plt.title("Creativity index evolution", fontsize=sizes['title'])
#     plt.xlabel("Generation", fontsize=sizes['labels'])
#     plt.ylabel("Creativity Index", fontsize=sizes['labels'])
#     plt.xticks(range(0, len(creativity_indices), x_ticks_space), fontsize=sizes['ticks'])
#     plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
#     plt.ylim(0, 1)
#
#     plt.grid()
#
#     mean_line, = plt.plot(np.mean(all_seeds_gen_creativities, axis = 0))
#     color = mean_line.get_color()
#     plt.fill_between(range(len(creativity_indices)),
#                     np.mean(all_seeds_gen_creativities, axis = 0) - np.std(all_seeds_gen_creativities, axis = 0),
#                     np.mean(all_seeds_gen_creativities, axis = 0) + np.std(all_seeds_gen_creativities, axis = 0),
#                     alpha=0.2)
#     for i in range(len(all_seeds_gen_creativities)):
#         plt.plot(all_seeds_gen_creativities[i], alpha=0.2, color = color)
#
#     if save:
#         plt.savefig(folder + '/creativity_evolution.png')
#         print("Saved creativity_evolution.png")
#
#     if plot:
#         plt.show()


def plot_similarity_graph(between_gen_similarity_matrix, folder, plot, sizes, save=True, seed = 0):
    plt.figure(figsize=(8, 8))
    n_generations = between_gen_similarity_matrix.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n_generations))
    connected_edges_idx = []

    for i in range(n_generations):
        for j in range(n_generations):
            if i != j:
                similarity = between_gen_similarity_matrix[i, j]
                G.add_edge(i, j, weight=similarity)
                if i == j - 1:
                     # We keep the indexes of the succesive generations edges to plot them at the end 
                     connected_edges_idx.append((len(G.edges)) - 1)

    np.random.seed(0)
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=range(n_generations), cmap='cool')
    nx.draw_networkx_labels(G, pos)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # Use the previously computed indexes of successive generations to either:
    # - Only plot them
    # hide_plotted_weights = [val if idx in connected_edges_idx else 0 for idx, val in enumerate(weights)] 
    # - Increase their width compared to others
    incr_plotted_weights = [3 if idx in connected_edges_idx else val for idx, val in enumerate(weights)] 
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=incr_plotted_weights, edge_color='gray')

    plt.title('Evolution of similarities between generations', fontsize=sizes['labels'])
    plt.axis('off')

    if save:
        plt.savefig(folder + '/generation_similarities_graph'+str(seed)+'.png')
        print("Saved generation_similarities_graph"+str(seed)+".png")
    
    if plot:
        plt.show()

    return G


def plot_word_chains(word_lists, folder, plot, ticks_space, sizes, save=True, seed = 0):
    flatten_list_of_lists = [item for sublist in word_lists for item in sublist]
    known_words = {}

    for i, word_list in enumerate(flatten_list_of_lists):
        for j, word in enumerate(word_list):
            if word in known_words:
                known_words[word].append(i)

            else:
                known_words[word] = [i]

    # Scale the fig size with the number of words (not optimal but works atm)
    x_fig_size = len(known_words) // 10
    plt.figure(figsize=(x_fig_size, 6))
    plt.title("Evolution of words presence in texts across generations", fontsize=sizes['title'])
    plt.ylabel("Presence across generations", fontsize=sizes['labels'])
    plt.xlabel("Words", fontsize=sizes['labels'])
    ax = plt.gca()

    for i, key in enumerate(known_words.keys()):
        values = known_words[key]
        for j in values:

            size = values.count(j)

            color = 'black'
                
            ax.plot([i, i],[j, j + 1], '-', color = color)
            ax.plot(i,j, 'o', markersize = size, color = color)

    xtick_labels = list(known_words.keys())

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Add half of the words on top and half on the bottom of x axis
    upper_xtick_labels = [''] * len(xtick_labels)
    upper_xtick_labels[::2] = xtick_labels[::2]
    lower_xtick_labels = [''] * len(xtick_labels)
    lower_xtick_labels[1::2] = xtick_labels[1::2]

    ax.set_xticks(range(0, len(xtick_labels), 2))
    ax.set_xticklabels(upper_xtick_labels[0::2], rotation=90, ha='center')
    ax2.set_xticks(range(1, len(xtick_labels), 2))
    ax2.set_xticklabels(lower_xtick_labels[1::2], rotation=90, ha='center')

    plt.yticks(range(0, len(word_lists), ticks_space))
    plt.tight_layout()

    # See if we wanna keep the grids
    ax.grid()
    ax2.grid()

    if save:
        plt.savefig(folder + '/wordchains'+str(seed)+'.png')
        print("Saved wordchains"+str(seed)+".png")
    
    if plot:
        plt.show()
        

# def plot_embedding(folder, plot, sizes, save=True, model_name='sentence-transformers/all-mpnet-base-v2', random_state=42, points=None):

#     def _seed_from_filename(path):
#         filename = os.path.basename(path)
#         match = re.match(r'output(\d+)\.json$', filename)
#         if match:
#             return int(match.group(1))
#         if filename == 'output.json':
#             return 0
#         return filename

#     if points is None:
#         try:
#             sentence_transformers = importlib.import_module('sentence_transformers')
#         except ImportError as exc:
#             raise ImportError(
#                 "plot_embedding requires sentence-transformers. Install it with: pip install sentence-transformers"
#             ) from exc

#         try:
#             umap = importlib.import_module('umap')
#         except ImportError as exc:
#             raise ImportError(
#                 "plot_embedding requires umap-learn. Install it with: pip install umap-learn"
#             ) from exc

#         output_files = glob(os.path.join(folder, 'output*.json'))
#         if len(output_files) == 0:
#             raise FileNotFoundError(f"No output*.json files found in {folder}")

#         output_files = sorted(output_files, key=_seed_from_filename)

#         all_texts = []
#         points = []

#         for output_file in output_files:
#             seed = _seed_from_filename(output_file)
#             with open(output_file, 'r') as f:
#                 content = json.load(f)

#             stories = content.get('stories', [])
#             for gen_idx, generation_stories in enumerate(stories):
#                 for story_idx, story in enumerate(generation_stories):
#                     if not isinstance(story, str):
#                         continue
#                     all_texts.append(story)
#                     points.append(
#                         {
#                             'seed': seed,
#                             'gen': gen_idx,
#                             'story_idx': story_idx,
#                         }
#                     )

#         if len(all_texts) == 0:
#             raise ValueError('No valid stories found in output files.')

#         model = sentence_transformers.SentenceTransformer(model_name)
#         embeddings = model.encode(all_texts, show_progress_bar=False)

#         reducer = umap.UMAP(n_components=2, random_state=random_state)
#         reduced = reducer.fit_transform(embeddings)

#         for i in range(len(points)):
#             points[i]['x'] = reduced[i, 0]
#             points[i]['y'] = reduced[i, 1]

#     points_by_seed_and_gen = {}
#     seeds = sorted({point['seed'] for point in points})
#     for point in points:
#         points_by_seed_and_gen.setdefault(point['seed'], {}).setdefault(point['gen'], []).append(point)

#     cmap = plt.cm.get_cmap('tab20', max(len(seeds), 1))
#     seed_to_color = {seed: cmap(i) for i, seed in enumerate(seeds)}

#     plt.figure(figsize=(10, 8))

#     for seed in seeds:
#         gen_map = points_by_seed_and_gen[seed]
#         sorted_gens = sorted(gen_map.keys())

#         for gen_idx in sorted_gens[:-1]:
#             current_points = gen_map[gen_idx]
#             next_points = gen_map.get(gen_idx + 1, [])
#             for p1 in current_points:
#                 for p2 in next_points:
#                     plt.plot(
#                         [p1['x'], p2['x']],
#                         [p1['y'], p2['y']],
#                         color=seed_to_color[seed],
#                         alpha=0.15,
#                         linewidth=0.6,
#                     )

#     for seed in seeds:
#         seed_points = [point for point in points if point['seed'] == seed]
#         plt.scatter(
#             [point['x'] for point in seed_points],
#             [point['y'] for point in seed_points],
#             color=seed_to_color[seed],
#             alpha=0.85,
#             s=20,
#             label=f'Seed {seed}',
#         )

#     plt.title('UMAP projection of stories embeddings', fontsize=sizes['title'])
#     plt.xlabel('UMAP 1', fontsize=sizes['labels'])
#     plt.ylabel('UMAP 2', fontsize=sizes['labels'])
#     plt.xticks(fontsize=sizes['ticks'])
#     plt.yticks(fontsize=sizes['ticks'])
#     plt.legend()
#     plt.tight_layout()

#     if save:
#         plt.savefig(folder + '/embedding_umap.png')
#         print('Saved embedding_umap.png')

#     if plot:
#         plt.show()
#     else:
#         plt.close()

#     return points


def display_graph(network_structure):
    pos = nx.spring_layout(network_structure)  # positions for all nodes

    nx.draw_networkx_nodes(network_structure, pos, node_size=700)
    nx.draw_networkx_edges(
        network_structure, 
        pos,
        arrowstyle='->',
        arrowsize=30  # Increase the arrowsize value
    )
    
    nx.draw_networkx_labels(network_structure, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()


DEFAULT_PLOT_NAMES = [
    "plot_similarity_matrix",
    "plot_between_gen_similarities",
    "plot_word_chains",
    "plot_similarity_graph",
    "plot_init_generation_similarity_evolution",
    "plot_within_gen_similarities",
    "plot_successive_generations_similarities",
]


PLOT_REGISTRY = {
    "plot_similarity_matrix": {
        "function": plot_similarity_matrix,
        "group": "seed",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "similarity_matrix": analysis_data["all_seeds_similarity_matrix"][seed],
            "n_gen": analysis_data["n_gen"],
            "n_agents": analysis_data["n_agents"],
            "folder": folder,
            "plot": plot,
            "sizes": sizes,
            "seed": seed,
        },
    },
    "plot_between_gen_similarities": {
        "function": plot_between_gen_similarities,
        "group": "seed",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "between_gen_similarity_matrix": analysis_data["all_seeds_between_gen_similarity_matrix"][seed],
            "folder": folder,
            "plot": plot,
            "x_ticks_space": analysis_data["x_ticks_space"],
            "sizes": sizes,
            "seed": seed,
        },
    },
    "plot_word_chains": {
        "function": plot_word_chains,
        "group": "seed",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "word_lists": analysis_data["all_seeds_stem_words"][seed],
            "folder": folder,
            "plot": plot,
            "ticks_space": analysis_data["x_ticks_space"],
            "sizes": sizes,
            "seed": seed,
        },
    },
    "plot_similarity_graph": {
        "function": plot_similarity_graph,
        "group": "seed",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "between_gen_similarity_matrix": analysis_data["all_seeds_between_gen_similarity_matrix"][seed],
            "folder": folder,
            "plot": plot,
            "sizes": sizes,
            "seed": seed,
        },
    },
    "plot_init_generation_similarity_evolution": {
        "function": plot_init_generation_similarity_evolution,
        "group": "summary",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "all_seeds_between_gen_similarity_matrix": analysis_data["all_seeds_between_gen_similarity_matrix"],
            "folder": folder,
            "plot": plot,
            "x_ticks_space": analysis_data["x_ticks_space"],
            "sizes": sizes,
        },
    },
    "plot_within_gen_similarities": {
        "function": plot_within_gen_similarities,
        "group": "summary",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "all_seeds_between_gen_similarity_matrix": analysis_data["all_seeds_between_gen_similarity_matrix"],
            "folder": folder,
            "plot": plot,
            "x_ticks_space": analysis_data["x_ticks_space"],
            "sizes": sizes,
        },
    },
    "plot_successive_generations_similarities": {
        "function": plot_successive_generations_similarities,
        "group": "summary",
        "build_kwargs": lambda analysis_data, folder, plot, sizes, seed=None: {
            "all_seeds_between_gen_similarity_matrix": analysis_data["all_seeds_between_gen_similarity_matrix"],
            "folder": folder,
            "plot": plot,
            "x_ticks_space": analysis_data["x_ticks_space"],
            "sizes": sizes,
        },
    },
}


def normalize_plot_names(plot_names=None):
    if plot_names is None:
        return list(DEFAULT_PLOT_NAMES)

    normalized_names = []
    for plot_name in plot_names:
        if isinstance(plot_name, dict):
            normalized_names.append(plot_name["function"])
        else:
            normalized_names.append(plot_name)
    return normalized_names


def run_configured_plots(analysis_data, folder, plot=False, sizes=None, plot_names=None):
    sizes = sizes or {"ticks": 12, "labels": 14, "title": 16}
    for plot_name in normalize_plot_names(plot_names):
        plot_config = PLOT_REGISTRY.get(plot_name)
        if plot_config is None:
            raise KeyError(f"Unknown plot: {plot_name!r}")

        if plot_config["group"] == "seed":
            for seed in range(analysis_data["n_seeds"]):
                kwargs = plot_config["build_kwargs"](analysis_data, folder, plot, sizes, seed=seed)
                plot_config["function"](**kwargs)
        else:
            kwargs = plot_config["build_kwargs"](analysis_data, folder, plot, sizes)
            plot_config["function"](**kwargs)
