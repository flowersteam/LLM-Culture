import json
import os 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from llm_culture.analysis.plots import plot_similarity_graph
from llm_culture.analysis.utils import get_SBERT_similarity, get_similarity_matrix, preprocess_stories 
import networkx as nx

PAD = 20
LABEL_PAD = 10
MATRIX_SIZE = 10

def compare_init_generation_similarity_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with the initial generation', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Similarity with first generation', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        num_points = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        value = np.mean(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0)[0, 1:]
        std = np.std(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0)[0, 1:]
        label = data[folder]['label']
        plt.plot(range(1, num_points), value, label=label)
        plt.fill_between(range(1, num_points), value - std, value + std, alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/similarity_first_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    
def compare_within_generation_similarity_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(top=0.9)
    plt.title('Evolution of similarity within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Similarity within generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-0.1, 1.1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        value = np.diag(np.mean(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0))
        print("------within------", data[folder]['all_seeds_between_gen_similarity_matrix'] )
        print("------within mean------", value )
        label = data[folder]['label']
        std = np.diag(np.std(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0))


        plt.plot(value, label=label)
        plt.fill_between(range(0, len(value)), value - std, value + std, alpha=0.3)
    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/similarity_within_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_successive_generations_similarities(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with previous generation', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Similarity with previous generation', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-0.1, 1.1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        all_seeds_between_gen_similarity_matrix = data[folder]['all_seeds_between_gen_similarity_matrix']
        successive_sim = [[all_seeds_between_gen_similarity_matrix[seed][i, i+1] for i in range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1)] for seed in range(len(all_seeds_between_gen_similarity_matrix))]
        value = np.mean(successive_sim, axis = 0)
        label = data[folder]['label']
        std = np.std(successive_sim, axis = 0)

        plt.plot(range(1, len(value) + 1), value, label=label)
        plt.fill_between(range(1, len(value) + 1), value - std, value + std, alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/similarity_successive_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_positivity_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of positivity within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Positivity value', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-1, 1)
        plt.yticks(np.linspace(-1, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        all_seeds_positivities = data[folder]['all_seeds_positivities']

        all_seeds_gen_positivities = []
        for polarities in all_seeds_positivities:
            gen_positivities = []
            for p in polarities:
                gen_positivities.append(np.mean(p))
            all_seeds_gen_positivities.append(gen_positivities)

        label = data[folder]['label']


        plt.plot(np.mean(all_seeds_gen_positivities, axis=0), label=label)
        plt.fill_between(range(0, len(all_seeds_gen_positivities[0])), np.mean(all_seeds_gen_positivities, axis=0) - np.std(all_seeds_gen_positivities, axis=0), np.mean(all_seeds_gen_positivities, axis=0) + np.std(all_seeds_gen_positivities, axis=0), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/positivity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
        

def compare_subjectivity_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of subjectivity within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Subjectivity value', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        all_seeds_subjectivities = data[folder]['all_seeds_subjectivities']
        all_see_gen_subjectivities = []
        for subjectivities in all_seeds_subjectivities:
            gen_subjectivities = []
            for gen_subjectivity in subjectivities:
                gen_subjectivities.append(np.mean(gen_subjectivity))
            all_see_gen_subjectivities.append(gen_subjectivities)
        
        label = data[folder]['label']
        plt.plot(np.mean(all_see_gen_subjectivities, axis=0), label=label)
        plt.fill_between(range(0, len(all_see_gen_subjectivities[0])), np.mean(all_see_gen_subjectivities, axis=0) - np.std(all_see_gen_subjectivities, axis=0), np.mean(all_see_gen_subjectivities, axis=0) + np.std(all_see_gen_subjectivities, axis=0), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/subjectivity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_creativity_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of creativity within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Creativity index', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        all_seeds_creativity_indices = data[folder]['all_seeds_creativity_indices']
        all_seeds_gen_creativities = []
        for creativity_indices in all_seeds_creativity_indices:
            gen_creativities = [1 - np.mean(gen_creativity) for gen_creativity in creativity_indices]
            all_seeds_gen_creativities.append(gen_creativities)

        label = data[folder]['label']
        print(len(all_seeds_gen_creativities))
        print(np.std(all_seeds_gen_creativities, axis=0))
        plt.plot(np.mean(all_seeds_gen_creativities, axis=0), label=label)
        plt.fill_between(range(0, len(all_seeds_gen_creativities[0])), np.mean(all_seeds_gen_creativities, axis=0) - np.std(all_seeds_gen_creativities, axis=0), np.mean(all_seeds_gen_creativities, axis=0) + np.std(all_seeds_gen_creativities, axis=0), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/creativity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()



def plot_similarity_matrix(similarity_matrix, label, n_gen, n_agents, plot, sizes, saving_folder=None, seed = 0):
    plt.figure(figsize=(sizes['matrix'], sizes['matrix']))
    plt.imshow(similarity_matrix, vmin=0, vmax=1, cmap='viridis')

    n_texts = similarity_matrix.shape[0]
    if n_texts < 20:
        x_ticks_space = 1
    elif n_texts >= 50:
        x_ticks_space = 10
    else:
        x_ticks_space = 5
    
    plt.xlabel('History idx', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('History idx', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.title(f'Stories similarity matrix for {label}', fontsize=sizes['title'], pad=PAD)
    
    # Add black lines to delimit generations
    for i in range(n_gen):      
        plt.axvline(x = i * n_agents - 0.5, color = 'black')
        plt.axhline(y = i * n_agents - 0.5, color = 'black')

    plt.xticks(range(0, similarity_matrix.shape[0], x_ticks_space), fontsize=sizes['ticks'])
    plt.yticks(range(0, similarity_matrix.shape[0], x_ticks_space), fontsize=sizes['ticks'])

    cbar = plt.colorbar(pad=0.02, shrink=0.84)

    if saving_folder:
        saving_name = f'/stories_similarity_matrix_{label}_{seed}.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")

    if plot:
        plt.show()


def plot_all_similarity_graphs(data, plot, fontsizes, saving_folder=None, save = True, initial_story = 'In the quaint town of Willowbrook, a peculiar old bookstore stood at the corner of Elm Street. It was rumored to hold books of ancient magic. One rainy afternoon, Sarah stumbled upon it. Drawn by an unseen force, she entered. The books whispered secrets, promising power. Sarah hesitated, but curiosity won. She chose a tome bound in cracked leather. As she flipped through its pages, a surge of energy enveloped her. Outside, the rain ceased, and the sun bathed Willowbrook in golden light. Sarah smiled, realizing she held the key to changing her world forever.'):
    plt.clf()

#     all_folder_stories = []

#     for folder in data:
#         all_folder_stories.append(data[folder]['all_seed_stories'])

#     plt.figure(figsize=(8, 8))
#     n_folders = len(data)
#     n_seeds_per_folder = [len(data[folder]['all_seed_stories']) for folder in data]
#     total_seeds = sum(n_seeds_per_folder)
#     n_generations = len(all_folder_stories[0][0])





#     # max_similarity = np.max(between_gen_similarity_matrix)
#     # min_similarity = np.min(between_gen_similarity_matrix)

#     G = nx.Graph()
#     G.add_nodes_from(range(n_generations))
#     connected_edges_idx = []

#     np.random.seed(0)
#     print("Drawing similarity graph...")
#     #positions = np.random.rand(n_generations, 2)
#     for f1 in range(n_folders):
#         print("f1", f1)
#         for s1 in range(n_seeds_per_folder[f1]):
#             print("s1", s1)
#             for f2 in range(n_folders):
#                 print("f2", f2)
#                 for s2 in range(n_seeds_per_folder[f2]):
#                     print("s2", s2)
#                     for i in range(n_generations):
#                         print("i", i)
#                         for j in range(n_generations):
#                             print("j", j)
#                             if not(i == j and f1 == f2 and s1 == s2):
#                                 # similarity = (between_gen_similarity_matrix[i, j] - min_similarity) / (max_similarity - min_similarity)
#                                 # similarity = between_gen_similarity_matrix[i, j]
#                                 similarity = get_SBERT_similarity(all_folder_stories[f1][s1][i], all_folder_stories[f2][s2][j])
#                                 G.add_edge(i, j, weight=similarity)
#                                 if f1 == f2 and s1 == s2 and i == j - 1:
#                                     # We keep the indexes of the succesive generations edges to plot them at the end 
#                                     connected_edges_idx.append((len(G.edges)) - 1)

#     #np.random.seed(0)
#     pos = nx.spring_layout(G, iterations = 10000)

#     nx.draw_networkx_nodes(G, pos, node_size=300, node_color=range(n_generations), cmap='cool')
#     nx.draw_networkx_labels(G, pos)

#     edges = G.edges()
#     weights = [G[u][v]['weight'] for u, v in edges]
#     # Use the previously computed indexes of successive generations to either:
#     # - Only plot them
#     # hide_plotted_weights = [val if idx in connected_edges_idx else 0 for idx, val in enumerate(weights)] 
#     # - Increase their width compared to others
#     incr_plotted_weights = [3 if idx in connected_edges_idx else 0 for idx, val in enumerate(weights)] 
#     nx.draw_networkx_edges(G, pos, edgelist=edges, width=incr_plotted_weights, edge_color='gray', alpha=0.5, 
#                            arrowstyle='->',
#                            arrowsize=30)

#     plt.title('Evolution of similarities between generations', fontsize=sizes['labels'])
#     plt.axis('off')

#     if save:
#         plt.savefig(folder + '/generation_similarities_graph'+str(seed)+'.png')
#         print("Saved generation_similarities_graph"+str(seed)+".png")
    
#     if plot:
#         plt.show()

#     return G
        
    plt.figure(figsize=(16, 16))

    all_folder_stories = [data[folder]['all_seed_stories'] for folder in data]
    print(np.array(all_folder_stories).shape)
    n_folders = len(data)
    n_seeds_per_folder = [len(data[folder]['all_seed_stories']) for folder in data]
    total_seeds = sum(n_seeds_per_folder)
    n_generations = len(all_folder_stories[0][0])

    connected_edges_idx = [] 

    G = nx.Graph()
    G.add_nodes_from(range(n_generations*total_seeds))

    print("Drawing similarity graph...")
    try:
        with open(f'Results/Comparisons/{saving_folder}/connected_edges_idx.json', 'r') as f:
            connected_edges_idx = json.load(f)
        print("Connected edges loaded")
        with open('Results/Comparisons/{saving_folder}/graph.json', 'r') as f:
            graph_data = json.load(f)
            G = nx.node_link_graph(graph_data)
        print("Graph loaded")
    except:

        for f1 in trange(n_folders):
            for s1 in range(n_seeds_per_folder[f1]):
                for f2 in trange(n_folders):  # Avoid redundant calculations
                    for s2 in range(n_seeds_per_folder[f2]):
                        stories1 = np.concatenate((np.array([initial_story]), np.array(all_folder_stories[f1][s1]).flatten()))
                        stories2 = np.concatenate((np.array([initial_story]), np.array(all_folder_stories[f2][s2]).flatten()))
                        similarities = get_SBERT_similarity(stories1, stories2)
                        for i in range(n_generations+1):
                            for j in range(n_generations+1):  
                                similarity = similarities[i][j]
                                if not(i == j and f1 == f2 and s1 == s2):
                                    node1 = f1 * n_seeds_per_folder[f1] * n_generations + s1 * n_generations + i
                                    node2 = f2 * n_seeds_per_folder[f2] * n_generations + s2 * n_generations + j
                                    G.add_edge(node1, node2, weight=similarity)
                                    if f1 == f2 and s1 == s2 and i == j - 1:
                                        connected_edges_idx.append((node1, node2))

                                
        # Save connected_edges_idx to JSON
        with open(f'Results/Comparisons/{saving_folder}/connected_edges_idx.json', 'w') as f:
            json.dump(connected_edges_idx, f)
        print("Connected edges saved")



        # Save the graph to JSON
        graph_data = nx.node_link_data(G)
        with open(f'Results/Comparisons/{saving_folder}/graph.json', 'w') as f:
            json.dump(graph_data, f)
        print("Graph saved")

  

    pos = nx.spring_layout(G, iterations=10000, dim = 3)

    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges() if (u,v) in connected_edges_idx ])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    color_list = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'gray']
    colors = list(np.array([[color_list[i]] * n_generations for i in range(total_seeds)]).flatten())
    sizes = np.array(list(range(n_generations)) * total_seeds).flatten()
    sizes = ([500] + [100] * n_generations) * total_seeds
    scatter = ax.scatter(*node_xyz.T, s=100, ec="w", c=colors)

    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    ax.add_artist(legend1)

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", alpha=0.5)


    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


    _format_axes(ax)
    fig.tight_layout()
    plt.show()

    pos = nx.spring_layout(G, iterations=10000)

    color_list = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'black', 'gray']
    colors = list(np.array([[color_list[i]] * n_generations for i in range(total_seeds)]).flatten())
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=colors, cmap='cool')
    labels = {n:n%n_generations for n in range(n_generations*total_seeds)}
    #nx.draw_networkx_labels(G, pos, labels=labels)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    incr_plotted_weights = [3 if (u, v) in connected_edges_idx else 0 for u, v in edges]  # Simplified edge weight calculation
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=incr_plotted_weights, edge_color='gray', alpha=0.5,
                           arrowstyle='->',
                           arrowsize=30)

    plt.title('Evolution of similarities between generations', fontsize=fontsizes['labels'])
    plt.axis('off')

    if saving_folder:
        saving_name = f'/similarity_graph.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")

    
    plt.show()





    return G



    all_folder_stories = []
    for folder in data:
        all_folder_stories += data[folder]['all_seed_stories']
    
    all_folder_flat_stories, all_folder_keywords, all_folder_stem_words = preprocess_stories(all_folder_stories)

    all_folder_similarity_matrix = get_similarity_matrix(all_folder_flat_stories)

    plot_similarity_graph(all_folder_similarity_matrix, saving_folder, plot, sizes)

