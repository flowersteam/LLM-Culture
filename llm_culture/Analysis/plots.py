import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 


def plot_similarity_matrix(similarity_matrix, n_gen, n_agents, folder, plot, save=True):
    plt.figure(figsize=(8, 8))
    plt.imshow(similarity_matrix, vmin=0, vmax=1, cmap='viridis')

    n_texts = similarity_matrix.shape[0]
    if n_texts < 20:
        x_ticks_space = 1
    elif n_texts >= 50:
        x_ticks_space = 10
    else:
        x_ticks_space = 5
        
    plt.title('Stories similarity Matrix')
    for i in range(n_gen):      
        plt.axvline(x = i * n_agents - 0.5, color = 'black')
        plt.axhline(y = i * n_agents - 0.5, color = 'black')
    plt.xticks(range(0, similarity_matrix.shape[0], x_ticks_space))
    plt.yticks(range(0, similarity_matrix.shape[0], x_ticks_space))

    cbar = plt.colorbar(pad=0.02, shrink=0.83)

    if save:
        plt.savefig(folder + '/stories_similarity_matrix.png')
        print("Saved stories_similarity_matrix.png")

    if plot:
        plt.show()


def plot_between_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, save=True):
    plt.figure(figsize=(8, 8))
    plt.title('Between generations similarity Matrix')
    plt.xticks(range(0, between_gen_similarity_matrix.shape[0], x_ticks_space))
    plt.yticks(range(0, between_gen_similarity_matrix.shape[0], x_ticks_space))

    plt.imshow(between_gen_similarity_matrix, vmin=0, vmax=1, cmap='viridis')
    cbar = plt.colorbar(pad=0.02, shrink=0.83)
   
    if save:
        plt.savefig(folder + '/between_gen_similarity_matrix.png')
        print("Saved between_gen_similarity_matrix.png")
    
    if plot:
        plt.show()


def plot_init_generation_similarity_evolution(between_gen_similarity_matrix, folder, plot, x_ticks_space, save=True):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with the initial generation')
    plt.xlabel('Generations')
    plt.ylabel('Similarity with first generation')

    plt.ylim(0, 1)
    plt.xticks(range(0, between_gen_similarity_matrix.shape[0], x_ticks_space))
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()

    plt.plot(range(1, between_gen_similarity_matrix.shape[0]), between_gen_similarity_matrix[0, 1:], label='Similarity with First History')

    if save:
        plt.savefig(folder + '/similarity_first_gen.png')
        print("Saved similarity_first_gen.png")
    
    if plot:
        plt.show()
    

def plot_within_gen_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, save=True):
    plt.figure(figsize=(10, 6))
    plt.title('Within generations texts similarities')
    plt.xlabel('Generations')
    plt.ylabel('Within-generation similarity')
    plt.xticks(range(0, between_gen_similarity_matrix.shape[0], x_ticks_space))
    plt.ylim(0 , 1.1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    plt.plot(np.diag(between_gen_similarity_matrix))

    if save:
        plt.savefig(folder + '/within_gen_similarity.png')
        print("Saved within_gen_similarity.png")

    if plot:
        plt.show()


def plot_successive_generations_similarities(between_gen_similarity_matrix, folder, plot, x_ticks_space, save=True):
    plt.figure(figsize=(10, 6))
    successive_sim = [between_gen_similarity_matrix[i, i+1] for i in range(between_gen_similarity_matrix.shape[0] - 1)]
    plt.title('Successive generations similarities')
    plt.xlabel('Generations')
    plt.ylabel('Similarity between successive generations')
    plt.xticks(range(0, between_gen_similarity_matrix.shape[0], x_ticks_space))
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    plt.ylim(0, 1)
    plt.plot(successive_sim)

    if save:
        plt.savefig(folder + '/successive_similarity.png')
        print("Saved successive_similarity.png")

    if plot:
        plt.show()


def plot_positivity_evolution(polarities, folder, plot, x_ticks_space, save=True):
    plt.figure(figsize=(10, 6))
    # 1 = positive, -1 = negative on the y axis
    gen_positivities = []
    for gen_polarities in polarities:
        gen_positivities.append(np.mean(gen_polarities))

    plt.title("Evolution of positivity across generations")
    plt.ylabel("Positivity")
    plt.ylabel("Generation")

    plt.xticks(range(0, len(gen_positivities), x_ticks_space))
    plt.yticks(np.linspace(-1, 1, 11))
    plt.grid()
    plt.ylim(-1, 1)

    plt.plot(gen_positivities)

    if save:
        plt.savefig(folder + '/positivity_evolution.png')
        print("Saved positivity_evolution.png")

    if plot:
        plt.show()


def plot_subjectivity_evolution(subjectivities, folder, plot, x_ticks_space, save=True):
    """Subjectivity is the output that lies within [0,1] and refers to personal opinions and judgments"""
    plt.figure(figsize=(10, 6))
    gen_subjectivities = []
    for gen_subjectivity in subjectivities:
        gen_subjectivities.append(np.mean(gen_subjectivity))

    plt.title("Evolution of subjectivity across generations")
    plt.ylabel("Positivity")
    plt.ylabel("Generation")

    plt.xticks(range(0, len(gen_subjectivities), x_ticks_space))
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid()
    plt.ylim(0, 1)

    plt.plot(gen_subjectivities)

    if save:
        plt.savefig(folder + '/positivity_evolution.png')
        print("Saved subjectivity_evolution.png")

    if plot:
        plt.show()

    
def plot_creativity_evolution(creativity_indices, folder, plot, x_ticks_space, save=True):
    """!!! Warning here 1 = low creativity and 0 = high creativity !!!"""
    
    plt.figure(figsize=(10, 6))
    gen_creativities = [np.mean(gen_creativity) for gen_creativity in creativity_indices]
    plt.figure(figsize=(10, 6))
    
    plt.title("Creativity Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Creativity Index")
    plt.xticks(range(0, len(creativity_indices), x_ticks_space))
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 1)
    plt.grid()
    plt.plot(gen_creativities)

    if save:
        plt.savefig(folder + '/creativity_evolution.png')
        print("Saved creativity_evolution.png")

    if plot:
        plt.show()



def plot_similarity_graph(between_gen_similarity_matrix, folder, plot, save=True):
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

    plt.title('Evolution of similarities between generations')
    plt.axis('off')

    if save:
        plt.savefig(folder + '/generation_similarities_graph.png')
        print("Saved generation_similarities_graph.png")
    
    if plot:
        plt.show()

    return G


def plot_word_chains(word_lists, folder, plot, x_ticks_space, save=True):

    flatten_list_of_lists = [np.array(l).flatten() for l in word_lists]
    known_words = {}

    for i, word_list in enumerate(flatten_list_of_lists):
        for j, word in enumerate(word_list):
            if word in known_words:
                known_words[word].append(i)

            else:
                known_words[word] = [i]

    # Scale the fig size with the number of words (not optimal but works atm)
    # TODO : Maybe scale the y_axis too ? 
    x_fig_size = len(known_words) // 10
    plt.figure(figsize=(x_fig_size, 6))
    # TODO : Maybe add a legend to explain how the plot works
    plt.ylabel("Generations")
    plt.xlabel("Words")
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

    # Set the lower and upper ticks labels 
    upper_xtick_labels = [''] * len(xtick_labels)
    upper_xtick_labels[::2] = xtick_labels[::2]
    lower_xtick_labels = [''] * len(xtick_labels)
    lower_xtick_labels[1::2] = xtick_labels[1::2]

    # Plot them
    ax.set_xticks(range(0, len(xtick_labels), 2))
    ax.set_xticklabels(upper_xtick_labels[0::2], rotation=90, ha='center')
    ax2.set_xticks(range(1, len(xtick_labels), 2))
    ax2.set_xticklabels(lower_xtick_labels[1::2], rotation=90, ha='center')

    plt.yticks(range(0, len(word_lists), x_ticks_space))
    
    # See if we wanna keep the grids
    ax.grid()
    ax2.grid()

    if save:
        plt.savefig(folder + '/wordchains.png')
        print("Saved wordchains.png")
    
    if plot:
        plt.show()
        