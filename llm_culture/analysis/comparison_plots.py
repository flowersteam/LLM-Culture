import json
import os 
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import trange

from GRUEN.main import get_focus_score, get_grammaticality_score, get_redundancy_score, preprocess_candidates
from llm_culture.analysis.plots import plot_similarity_graph
from llm_culture.analysis.utils import convert_to_json_serializable, get_SBERT_similarity, get_embeddings, get_similarity_matrix, preprocess_stories 
import networkx as nx
from scipy.spatial import ConvexHull
import umap
from sklearn.neighbors import KernelDensity
import diptest


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
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
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

    last_gens_values = []
    last_gens_stds = []

    for folder in data:
        value = np.diag(np.mean(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0))

        last_gens_values.append(value[-1])
       
        label = data[folder]['label']
        std = np.diag(np.std(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0))

        last_gens_stds.append(std[-1])


        plt.plot(value, label=label)
        plt.fill_between(range(0, len(value)), value - std, value + std, alpha=0.3)
    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/similarity_within_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()

    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(top=0.9)
    plt.title('Cultural diversity', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Cultural diversity', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.xticks(np.arange(len(last_gens_values)), [data[folder]['label'] for folder in data], fontsize=sizes['ticks'], rotation=45)
    plt.grid()

    last_gens_stds = np.array(last_gens_stds)

    last_gens_values = np.ones(len(last_gens_values)) - np.array(last_gens_values)

    plt.plot(last_gens_values)
    plt.fill_between(range(0, len(last_gens_values)), last_gens_values - last_gens_stds, last_gens_values + last_gens_stds, alpha=0.3)
    #plt.xticks = [data[folder]['label'] for folder in data]
    #plt.ylabel = 'Diversity'
    if saving_folder:
        saving_name = '/similarity_within_gen_comparison_LAST.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    
    plt.clf()


def compare_change_frequency_magnitude(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    for bandwith in [0.001, 0.01, 0.05, 0.1, 0.2]:

        plt.figure(figsize=(10, 6))
        plt.title('Change magntiude and frequencies', fontsize=sizes['title'], pad=PAD)


        max_num_ticks = 0 

        for folder in data:
            num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
            if num_ticks > max_num_ticks:
                max_num_ticks = num_ticks
                x_ticks_space = data[folder]['x_ticks_space']
        
        if scale_y_axis:
            plt.ylim(0, 1)
            plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
        
        #plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
        plt.grid()

        color_list = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver']

        for i, folder in enumerate(data):
            color = color_list[i]
            all_seeds_between_gen_similarity_matrix = data[folder]['all_seeds_between_gen_similarity_matrix']
            successive_sim = [[all_seeds_between_gen_similarity_matrix[seed][i, i+1] for i in range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1)] for seed in range(len(all_seeds_between_gen_similarity_matrix))]
            
            for seed in range(len(all_seeds_between_gen_similarity_matrix)):
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwith).fit(np.array(successive_sim[seed]).reshape(-1, 1))
                x = np.linspace(0, 1, 1000)
                log_dens = kde.score_samples(x.reshape(-1, 1))

                local_maxes = []
                for i in range(1, len(log_dens) - 1):
                    if log_dens[i] > log_dens[i-1] and log_dens[i] > log_dens[i+1]:
                        local_maxes.append((x[i], np.exp(log_dens[i])))
                
                plt.scatter([x[0] for x in local_maxes], [x[1] for x in local_maxes], color=color)

                # max, argmax = np.max(np.exp(log_dens)), np.argmax(np.exp(log_dens))

                # plt.scatter(argmax/1000, max, color=color)
        
        handles = [plt.Line2D([0], [0], marker='o', color=color_list[i], markerfacecolor=color, markersize=10, label=data[folder]['label']) for i, folder in enumerate(data)]
        
        plt.legend(handles=handles, loc='upper right')

        plt.xlabel('Change magnitude', fontsize=sizes['labels'], labelpad=LABEL_PAD)
        plt.ylabel('Frequency', fontsize=sizes['labels'], labelpad=LABEL_PAD)






        
        if saving_folder:
            saving_name = f'/change_freq_magni_bandwith_{bandwith}.png'
            os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
            plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
            print(f"Saved {saving_name}")
        
        if plot:
            plt.show()




def compare_dips(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Distributions of distances between successive generations', fontsize=sizes['title'], pad=PAD)


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

    for i, folder in enumerate(data):
        all_seeds_between_gen_similarity_matrix = data[folder]['all_seeds_between_gen_similarity_matrix']
        successive_sim = [[all_seeds_between_gen_similarity_matrix[seed][i, i+1] for i in range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1)] for seed in range(len(all_seeds_between_gen_similarity_matrix))]

        dips = []
        for seed in range(len(all_seeds_between_gen_similarity_matrix)):
            
            dip, pval = diptest.diptest(np.array(successive_sim[seed]))
            dips.append(dip)
        
        plt.errorbar([i], [np.mean(dips)], yerr=[np.std(dips)], fmt='o', label=data[folder]['label'])
        plt.xticks([i], [data[folder]['label']])
        plt.grid()






    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/dips.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    
    #plt.show()


    
    

def compare_similarity_distribution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Distributions of distances between successive generations', fontsize=sizes['title'], pad=PAD)


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

    for i, folder in enumerate(data):
        all_seeds_between_gen_similarity_matrix = data[folder]['all_seeds_between_gen_similarity_matrix']
        successive_sim = [[all_seeds_between_gen_similarity_matrix[seed][i, i+1] for i in range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1)] for seed in range(len(all_seeds_between_gen_similarity_matrix))]
        flat_successive_sim = np.array(successive_sim).flatten()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(np.array(flat_successive_sim).reshape(-1, 1))
        x = np.linspace(0, 1, 1000)
        log_dens = kde.score_samples(x.reshape(-1, 1))
        plt.plot(x, np.exp(log_dens), label=data[folder]['label'])
        plt.hist(flat_successive_sim, bins=20, density=True, alpha=0.3)
        saving_name = f'/similarity_distributions_{data[folder]['label']}.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
        
        #plt.show()
        plt.clf()
        dips = []
        for seed in range(len(all_seeds_between_gen_similarity_matrix)):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(np.array(successive_sim[seed]).reshape(-1, 1))
            x = np.linspace(0, 1, 1000)
            log_dens = kde.score_samples(x.reshape(-1, 1))
            plt.plot(x, np.exp(log_dens), label=data[folder]['label'])
            plt.hist(successive_sim[seed], bins=20, density=True, alpha=0.3)
            saving_name = f'/similarity_distributions_{data[folder]['label']}{seed}.png'
            os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
            plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
            #plt.show()
            plt.clf()

            dip, pval = diptest.diptest(flat_successive_sim)
            print(f'res{saving_name} : {dip}, {pval}')
            dips.append(dip)
        
        plt.clf()
        plt.errorbar([i], [np.mean(dips)], yerr=[np.std(dips)], fmt='o', label=data[folder]['label'])
        plt.xticks([i], [data[folder]['label']])
        plt.grid()






    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/similarity_distributions.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    
    #plt.show()


    
    


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
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", transparent=True)
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


def plot_all_similarity_graphs(data, plot, fontsizes, saving_folder=None, save = True, initial_story = ''):
    
    
    plt.close('all')


    all_folder_stories = [data[folder]['all_seed_stories'] for folder in data]
    #print(np.array(all_folder_stories).shape)
    n_folders = len(data)
    n_seeds_per_folder = [len(data[folder]['all_seed_stories']) for folder in data]
    total_seeds = sum(n_seeds_per_folder)
    n_generations = len(all_folder_stories[0][0])
    n_stories_per_generation = len(all_folder_stories[0][0][0])

    labels = [data[folder]['label'] for folder in data]

    connected_edges_idx = [] 

    G = nx.Graph()
    G.add_nodes_from(range(n_generations*total_seeds))

    print("Computing similarity graph...")
    try:
        with open(f'Results/Comparisons/{saving_folder}/connected_edges_idx.json', 'r') as f:
            connected_edges_idx = json.load(f)
        print("Connected edges loaded")
        with open(f'Results/Comparisons/{saving_folder}/graph.json', 'r') as f:
            graph_data = json.load(f)
            G = nx.node_link_graph(graph_data)
        print("Graph loaded")
    except:
       

        for f1 in trange(n_folders):
            for s1 in range(n_seeds_per_folder[f1]):
                for f2 in trange(n_folders):  # Avoid redundant calculations
                    for s2 in range(n_seeds_per_folder[f2]):
                        # stories1 = np.concatenate((np.array([initial_story]), np.array(all_folder_stories[f1][s1]).flatten()))
                        stories1 = np.array(all_folder_stories[f1][s1]).flatten()
                        # stories2 = np.concatenate((np.array([initial_story]), np.array(all_folder_stories[f2][s2]).flatten()))
                        stories2 = np.array(all_folder_stories[f2][s2]).flatten()
                        similarities = get_SBERT_similarity(stories1, stories2)
                        for i in range(n_generations):
                            for j in range(n_generations):  
                                similarity = similarities[i][j]
                                if not(i == j and f1 == f2 and s1 == s2):
                                    node1 = f1 * n_seeds_per_folder[f1] * (n_generations) + s1 * (n_generations) + i
                                    node2 = f2 * n_seeds_per_folder[f2] * (n_generations) + s2 * (n_generations) + j
                                    G.add_edge(node1, node2, weight=similarity)
                                    if f1 == f2 and s1 == s2 and i == j - 1:
                                        connected_edges_idx.append((node1, node2))


                                
        # Save connected_edges_idx to JSON
        with open(f'Results/Comparisons/{saving_folder}/connected_edges_idx.json', 'w') as f:
            json.dump(connected_edges_idx, f)
        print("Connected edges saved")



        # Save the graph to JSON
        graph_data = nx.readwrite.json_graph.node_link_data(G)
        graph_data = convert_to_json_serializable(graph_data)
        with open(f'Results/Comparisons/{saving_folder}/graph.json', 'w') as f:
            #dump = json.dumps(graph_data)
            json.dump(graph_data, f)
        print("Graph saved")

  
    # print("Drawing 3D graph...")
    # pos = nx.spring_layout(G, iterations=1000, dim = 3)

    # node_xyz = np.array([pos[v] for v in sorted(G)])
    # edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges() if (u,v) in connected_edges_idx ])

    # # Create the 3D figure
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")

    # # Plot the nodes - alpha is scaled by "depth" automatically
    # color_list = ['blue', 'red', 'green', 'paleturquoise', 'purple', 'orange', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver']
    # colors = list(np.array([[color_list[i]] * n_seeds_per_folder[i] * n_generations * n_stories_per_generation for i in range(n_folders)]).flatten())

    # # colors = list(np.array([[['black'] + [color_list[i]] * n_generations] for i in range(total_seeds)]).flatten())
    # sizes = np.array(list(range(n_generations)) * total_seeds).flatten()
    # # sizes = ([500] + [100] * n_generations) * total_seeds
    # sizes = ([100] * n_generations) * total_seeds
    # scatter = ax.scatter(*node_xyz.T, s=100, ec="w", c=colors)

    # handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color) for label, color in zip(labels, color_list)]

    # plt.legend(handles=handles, loc='upper right')


    # # Plot the edges
    # for vizedge in edge_xyz:
    #     ax.plot(*vizedge.T, color="tab:gray", alpha=0.5)


    # def _format_axes(ax):
    #     """Visualization options for the 3D axes."""
    #     # Turn gridlines off
    #     ax.grid(False)
    #     # Suppress tick labels
    #     for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
    #         dim.set_ticks([])
    #     # Set axes labels
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_zlabel("z")


    # _format_axes(ax)
    # fig.tight_layout()
    # plt.show()
    # plt.clf()








    # print("Drawing 2D graph...")

    # pos = nx.spring_layout(G, iterations=1000, dim=2)

    # color_list = ['blue', 'red', 'green', 'paleturquoise', 'purple', 'orange', 'pink', 'brown', 'black', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver']
    # colors = list(np.array([[color_list[i]] * n_seeds_per_folder[i] * n_generations * n_stories_per_generation for i in range(n_folders)]).flatten())


    # shape_list = ['^', 'o', 's', 'x', '+', 'D', 'v', 'p', 'P', 'X']
    # # shapes = list(np.array([[shape_list[i]] * n_generations for i in range(total_seeds)]).flatten())

    # edge_colors = list(np.array([[color_list[i]] * (n_generations - 1) for i in range(total_seeds)]).flatten())
    # #sizes = np.array(list([100] + [50] * (n_generations - 2) + [100]) * total_seeds).flatten()
    # sizes = list(np.linspace(10, 100, n_generations)) * total_seeds
    # #alphas = list(np.linspace(0.8, 1, n_generations)) * total_seeds
    # #shapes = list(np.array( list(['^'] + ['o'] * ( n_generations - 2) + ['s']) * total_seeds).flatten())
    # shapes = ['o'] * total_seeds * n_generations
    # for node, shape, size, color in zip(G.nodes, shapes, sizes, colors):
    #     nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_shape=shape, node_color=color, alpha=0.8)

    # edges = G.edges()
    # incr_plotted_weights = [3 if (u, v) in connected_edges_idx else 0 for u, v in edges]  # Simplified edge weight calculation
    # nx.draw_networkx_edges(G, pos, edgelist=connected_edges_idx, width=1, edge_color=edge_colors, alpha=0.1)

    # plt.title('Evolution of similarities between generations', fontsize=fontsizes['labels'])
    # plt.axis('off')

    # # Creating legend handles using group labels
    # handles = [plt.Line2D([0], [0], marker='o', label=label, color = color, markersize=10, markerfacecolor=color) for shape, label, color in zip(shape_list, labels, color_list)]

    # plt.legend(handles=handles, loc='upper right')

    # if saving_folder:
    #     saving_name = f'/similarity_graph2D.png'
    #     os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
    #     plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
    #     print(f"Saved {saving_name}")

        


    # plt.show()

    # for i in range(total_seeds - 2):

    #     highlighted_nodes_pos = np.array([pos[node] for node in range(i * n_generations, (i + 1) * n_generations)])
    #     hull = ConvexHull(highlighted_nodes_pos)
    #     for simplex in hull.simplices:
    #         plt.plot(highlighted_nodes_pos[simplex, 0], highlighted_nodes_pos[simplex, 1], color = color_list[i])

    #     plt.fill(highlighted_nodes_pos[hull.vertices,0], highlighted_nodes_pos[hull.vertices,1], color =color_list[i], alpha=0.1)




    
    # plt.show()





    # #return G


    matrix = np.array([])

    try:
        with open(f'Results/Comparisons/{saving_folder}/embeddings.npy', 'rb') as f:
            matrix = np.load(f)
        print("Embeddings loaded")
    except:

        print("Computing embeddings...")

        for folder in data:
            for stories in data[folder]['all_seed_stories']:

                print(np.array(stories).flatten().shape)


            
                embeddings = get_embeddings(np.array(stories).flatten())
                matrix = np.concatenate((matrix, embeddings), axis=0) if matrix.size else embeddings
        
        # Save the embeddings to a file
        with open(f'Results/Comparisons/{saving_folder}/embeddings.npy', 'wb') as f:
            np.save(f, matrix)
        print("Embeddings saved")

        print("Embeddings shape:", matrix.shape)

    viz_methods = {'TSNE-perplex5-lr200': TSNE(n_components=2, perplexity= 5, random_state=42, init='random', learning_rate=200),
                   'TSNE-perplex20-lr200': TSNE(n_components=2, perplexity= 20, random_state=42, init='random', learning_rate=200),
                   'TSNE-perplex5-lr50': TSNE(n_components=2, perplexity= 20, random_state=42, init='random', learning_rate=200),
                   'TSNE-perplex20-lr50': TSNE(n_components=2, perplexity= 20, random_state=42, init='random', learning_rate=200),
                   'UMAP-100nb': umap.UMAP(n_neighbors=100),
                   'UMAP-200nb': umap.UMAP(n_neighbors=200),
                   'UMAP-500nb': umap.UMAP(n_neighbors=500),
                   'UMAP-500nb': umap.UMAP(n_neighbors=500),
                   'PCA': PCA(n_components=2)

    }

    #viz_methods = {'PCA': PCA(n_components=2)}



    for method in viz_methods.items():
        name, fit = method
        print(fit, name)
        
        print("Computing t-SNE...")

        

        # Create a t-SNE model and transform the data
        if initial_story != '':
            initial_story_embedding = get_embeddings([initial_story])

            matrix = np.concatenate((initial_story_embedding, matrix), axis=0)

        vis_dims = fit.fit_transform(np.array(matrix))
        vis_dims.shape




        x = [x for x,y in vis_dims]
        y = [y for x,y in vis_dims]


        color_list = ['blue', 'red', 'green', 'purple', 'orange', 'black', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver', 'lime', 'teal', 'indigo', 'maroon', 'olive', 'navy', 'fuchsia', 'aqua', 'silver']
        colors = list(np.array([[color_list[i]] * n_seeds_per_folder[i] * n_generations * n_stories_per_generation for i in range(n_folders)]).flatten())
         
        shape_list = ['o', '^', 's', 'x', '+', 'D', 'v', 'p', 'P', 'X']

        shapes = list(np.array([[[shape_list[i] ] * n_generations * n_stories_per_generation  for i in range(n_seeds_per_folder[j])] for j in range(n_folders)]).flatten())

        sizes = np.array(list([10] *  n_stories_per_generation * (n_generations - 1) + [200] * n_stories_per_generation )   * total_seeds).flatten()

        fig, ax  = plt.subplots(1, 1, figsize=(20, 20))

        for xi, yi, color, shape, size in zip(x, y, colors, shapes, sizes):
            plt.scatter(xi, yi, c=color, s=size, marker=shape, alpha=0.3)
        #plt.scatter(x[1:], y[1:], c=colors,alpha=0.3)
        handles = [plt.Line2D([0], [0], marker='o', label=label, color = color, markersize=10, markerfacecolor=color) for label, color in zip(labels, color_list)]

        plt.scatter(x[0], y[0], c='black', s=100, marker='x', label='Initial story')

        plt.legend(handles=handles, loc='upper right')

        plt.title(name)


        if saving_folder:
            saving_name = f'/{name}graph_noedges_2D.png'
            os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
            plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
            print(f"Saved {saving_name}")

            


        if plot:
            plt.show()

        # ##Only last generation
        # # matrix = np.reshape(matrix[1:], (total_seeds, n_generations, n_stories_per_generation, -1))

        # vis_dims = fit.transform(np.array(matrix[1:]).reshape(total_seeds * n_generations * n_stories_per_generation, -1))

        # x = [x for x,y in vis_dims]
        # y = [y for x,y in vis_dims]


        # # x = np.reshape(x[1:], (total_seeds, n_generations, n_stories_per_generation))
        # # y = np.reshape(y[1:], (total_seeds, n_generations, n_stories_per_generation))

        # vis_dims = fit.transform(np.array([initial_story_embedding]).reshape(1, -1))

        # fig, ax  = plt.subplots(1, 1, figsize=(20, 20))

        # for i in range(total_seeds):
        #     for k in range(n_stories_per_generation):
        #         xi = x[i * n_stories_per_generation + k]
        #         yi = y[i * n_stories_per_generation + k]
        #         ax.scatter(xi, yi, color = color_list[i//2], marker = shape_list[i%2], s=100, )  

            
            
        # plt.title(name)
        # if saving_folder:
        #     saving_name = f'/{name}graph_noedges_2D_last_gen.png'
        #     os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        #     plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        #     print(f"Saved {saving_name}")
        # plt.show()




        # ## 3D Layered graph

        # x = np.reshape(x, (total_seeds, n_generations, n_stories_per_generation))
        # y = np.reshape(y, (total_seeds, n_generations, n_stories_per_generation))


        # # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection="3d")

        # # x = np.reshape(x[1:], (total_seeds, n_generations, n_stories_per_generation))
        # # y = np.reshape(y[1:], (total_seeds, n_generations, n_stories_per_generation))

        # # for i in range(total_seeds):
        # #     for j in range(n_generations):
        # #         for k in range(n_stories_per_generation):
        # #             xi = x[i, j, k]
        # #             yi = y[i, j, k]

                    
                    
        # #             ax.scatter([xi], [yi], [j], color = color_list[i])
        
        # # plt.show()

       

        # # vis_dims0 = fit.transform(np.array([initial_story_embedding]).reshape(1, -1))

        # # x0 = [x for x,y in vis_dims0]
        # # y0 = [y for x,y in vis_dims0]

        # # for i in range(total_seeds):
        # #     fig = plt.figure()
        # #     ax = fig.add_subplot(111, projection="3d")
        
        # #     for k in range(n_stories_per_generation):
        # #         xi = np.concatenate((x0, x[i, :, k].flatten()))
        # #         yi = np.concatenate((y0, y[i, :, k].flatten()))
        # #         print(yi)
        # #         zi = np.arange(n_generations+1)

                
                

        # #         ax.plot(xi, yi, zi, color = color_list[i], alpha=0.5)
        
        # #     plt.show()

        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(111, projection="3d")
        # ax.set_zlim((0, n_generations))


        # def rotate(gen):
        #     ax.clear()
        #     for i in range(total_seeds):
        
        #         for k in range(n_stories_per_generation):
        #             # xi = np.concatenate((x0, x[i, :gen, k].flatten()))
        #             # yi = np.concatenate((y0, y[i, :gen, k].flatten()))
        #             xi = x[i, :gen, k].flatten()
        #             yi = y[i, :gen, k].flatten()
        #             zi = np.arange(gen+1)

                    
                    

        #             ax.plot(xi, yi, zi, color = color_list[i], alpha=0.5)

       
        # # Uncomment the code block below
        # # for gen in range(n_generations):
        # #     rotate(gen)

        # print("Making animation")
        # rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(1, n_generations), interval=1000)
        # if saving_folder:
        #         saving_name = f'/{name}_animation.gif'
        #         os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        #         plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        #         rot_animation.save(f"Results/Comparisons/{saving_folder}/{saving_name}", dpi=80, writer='imagemagick')
        #         print(f"Saved {saving_name}")
                






        print("Drawing Embedding graph...")

        fig = plt.figure(figsize=(20, 20))

        pos = vis_dims

        
        shape_list = ['o', '^', 's', 'x', '+', 'D', 'v', 'p', 'P', 'X']
        # shapes = list(np.array([[shape_list[i]] * n_generations for i in range(total_seeds)]).flatten())

        # edge_colors = list(np.array([[color_list[i]] * (n_generations - 1) for i in range(total_seeds)]).flatten())
        edge_colors = list(np.array([[color_list[i]] * n_seeds_per_folder[i] * (n_generations-1) * n_stories_per_generation for i in range(n_folders)]).flatten())

        sizes = np.array(list([200] + [10] * (n_generations - 2) + [200]) * total_seeds).flatten()
        #sizes = list(np.linspace(10, 100, n_generations)) * total_seeds
        #alphas = list(np.linspace(0.8, 1, n_generations)) * total_seeds
        shapes = list(np.array( list(['^'] + ['o'] * ( n_generations - 2) + ['s']) * total_seeds).flatten())
        #shapes = ['o'] * total_seeds * n_generations
        for node, shape, size, color in zip(G.nodes, shapes, sizes, colors):
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size, node_shape=shape, node_color=color, alpha=0.3)

        edges = G.edges()
        incr_plotted_weights = [3 if (u, v) in connected_edges_idx else 0 for u, v in edges]  
        nx.draw_networkx_edges(G, pos, edgelist=connected_edges_idx, width=1, edge_color=edge_colors, alpha=0.5)

        plt.title('Evolution of similarities between generations', fontsize=fontsizes['labels'])
        plt.axis('off')

        # Creating legend handles using group labels
        handles = [plt.Line2D([0], [0], marker='o', label=label, color = color, markersize=10, markerfacecolor=color) for label, color in zip(labels, color_list)]

        plt.legend(handles=handles, loc='upper right')

        if saving_folder:
            saving_name = f'/{name}_embedding_graph2D.png'
            os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
            plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
            print(f"Saved {saving_name}")

            



        if plot:
            plt.show()
        
        

        #return G



    all_folder_stories = []
    for folder in data:
        all_folder_stories += data[folder]['all_seed_stories']
    
    all_folder_flat_stories, all_folder_keywords, all_folder_stem_words = preprocess_stories(all_folder_stories)

    all_folder_similarity_matrix = np.array(get_similarity_matrix(all_folder_flat_stories))

    plot_similarity_graph(all_folder_similarity_matrix, saving_folder, plot, sizes)



def plot_between_treatment_matrix(data,  plot, fontsizes, saving_folder=None, save = True, initial_story = 'But I must explain to you how all this mistaken idea of denouncing of a pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but occasionally circumstances occur in which toil and pain can procure him some great pleasure.'):


    print("Computing similarity matrix...")
    all_folder_stories = [data[folder]['all_seed_stories'] for folder in data]
    labels = [data[folder]['label'] for folder in data]

    try:
        with open(f'Results/Comparisons/{saving_folder}/similarity_matrix.npy', 'rb') as f:
            sim_matrix = np.load(f)
        print("Similarity matrix loaded")
    except:

        sim_matrix = np.zeros((len(all_folder_stories) * len(all_folder_stories[0]), len(all_folder_stories)* len(all_folder_stories[0])))

        for f1 in trange(len(all_folder_stories)):
            for f2 in trange(len(all_folder_stories)):
                for s1 in range(len(all_folder_stories[f1])):
                    for s2 in range(len(all_folder_stories[f2])):

                        stories1 = np.array(all_folder_stories[f1][s1]).flatten()
                        stories2 = np.array(all_folder_stories[f2][s2]).flatten()
                        similarities = get_SBERT_similarity(stories1, stories2)
                        sim_matrix[f1 * len(all_folder_stories[f1]) + s1 , f2 * len(all_folder_stories[f2]) + s2] = np.mean(similarities)
        
        # Save the similarity matrix to a file
        with open(f'Results/Comparisons/{saving_folder}/similarity_matrix.npy', 'wb') as f:
            np.save(f, sim_matrix)
        print("Similarity matrix saved")
        

    labels = [label + ' seed' + str(i) for label in labels for i in range(len(all_folder_stories[0]))]

    print(labels)


    plt.figure(figsize=(20, 20))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title('Similarity between treatments', fontsize=fontsizes['title'], pad=PAD)
    plt.xticks(range(len(all_folder_stories) * len(all_folder_stories[0])), labels, fontsize=fontsizes['ticks'], rotation=90)
    plt.yticks(range(len(all_folder_stories) * len(all_folder_stories[0])), labels, fontsize=fontsizes['ticks'])
    plt.colorbar(pad=0.02, shrink=0.84)
    plt.tight_layout()

    if saving_folder:
        saving_name = '/treatment_similarity_matrix.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", bbox_inches='tight')
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()

    plt.close('all')


    # mean_similarities_pure_0 = [np.mean(sim_matrix[0][i:i+2] ) for i in range(0, len(sim_matrix[0]), 2)]
    # mean_similarities_pure_1 = [np.mean(sim_matrix[-1][i:i+2] ) for i in range(0, len(sim_matrix[1]), 2)]    
    # std_similarities_pure_0 = [np.std(sim_matrix[0][i:i+2] ) for i in range(0, len(sim_matrix[0]), 2)]
    # std_similarities_pure_1 = [np.std(sim_matrix[-1][i:i+2] ) for i in range(0, len(sim_matrix[1]), 2)]
    # plt.plot(mean_similarities_pure_0, label = 'Similarity with the pure SciFi')
    # plt.fill_between(range(len(mean_similarities_pure_0)), mean_similarities_pure_0 - std_similarities_pure_0, mean_similarities_pure_0 + std_similarities_pure_0, alpha=0.3)

    # plt.plot(mean_similarities_pure_1, label = 'Similarity with the pure Romance')
    # plt.fill_between(range(len(mean_similarities_pure_1)), mean_similarities_pure_1 - std_similarities_pure_0, mean_similarities_pure_1 + std_similarities_pure_0, alpha=0.3)
    # plt.legend(fontsize=fontsizes['legend'])
    # plt.xticks(range(len(sim_matrix[0])), labels, fontsize=fontsizes['ticks'], rotation=90)

    # if saving_folder:
    #     saving_name = '/similarity_with_pure.png'
    #     os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
    #     plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
    #     print(f"Saved {saving_name}")
    # plt.show()
    # if plot:
    #     plt.show()


def plot_between_treatment_matrix_last_gen(data,  plot, fontsizes, saving_folder=None, save = True, initial_story = 'But I must explain to you how all this mistaken idea of denouncing of a pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but occasionally circumstances occur in which toil and pain can procure him some great pleasure.'):


    print("Computing similarity matrix...")
    all_folder_stories = [data[folder]['all_seed_stories'] for folder in data]
    labels = [data[folder]['label'] for folder in data]

    try:
        with open(f'Results/Comparisons/{saving_folder}/similarity_matrix_last_gen.npy', 'rb') as f:
            sim_matrix = np.load(f)
        print("Similarity matrix last gen loaded")
    except:

        sim_matrix = np.zeros((len(all_folder_stories) * len(all_folder_stories[0]), len(all_folder_stories)* len(all_folder_stories[0])))

        for f1 in trange(len(all_folder_stories)):
            for f2 in trange(len(all_folder_stories)):
                for s1 in range(len(all_folder_stories[f1])):
                    for s2 in range(len(all_folder_stories[f2])):

                        stories1 = np.array(all_folder_stories[f1][s1][-1]).flatten()
                        stories2 = np.array(all_folder_stories[f2][s2][-1]).flatten()
                        similarities = get_SBERT_similarity(stories1, stories2)
                        sim_matrix[f1 * len(all_folder_stories[f1]) + s1 , f2 * len(all_folder_stories[f2]) + s2] = np.mean(similarities)
        
        # Save the similarity matrix to a file
        with open(f'Results/Comparisons/{saving_folder}/similarity_matrix_last_gen.npy', 'wb') as f:
            np.save(f, sim_matrix)
        print("Similarity matrix last gen saved")
        

    labels = [label + ' seed' + str(i) for label in labels for i in range(len(all_folder_stories[0]))]

    print(labels)


    plt.figure(figsize=(20, 20))
    plt.imshow(sim_matrix, cmap='viridis')
    plt.title('Similarity between treatments (last gen)', fontsize=fontsizes['title'], pad=PAD)
    plt.xticks(range(len(all_folder_stories) * len(all_folder_stories[0])), labels, fontsize=fontsizes['ticks'], rotation=90)
    plt.yticks(range(len(all_folder_stories) * len(all_folder_stories[0])), labels, fontsize=fontsizes['ticks'])
    plt.colorbar(pad=0.02, shrink=0.84)
    plt.tight_layout()

    if saving_folder:
        saving_name = '/treatment_similarity_matrix_last_gen.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}", bbox_inches='tight')
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()

    plt.close('all')


def plot_convergence_matrix(data,  plot, fontsizes, saving_folder=None, save = True):


    print("Computing convergence matrix...")
    all_folder_stories = [data[folder]['all_seed_stories'] for folder in data]
    labels = [data[folder]['label'] for folder in data]
    intial_stories = [data[folder]['initial_story'] for folder in data]

    try:
        with open(f'Results/Comparisons/{saving_folder}/convergence_matrix.npy', 'rb') as f:
            sim_matrix = np.load(f)
        print("convergence_matrix loaded")
    except:

        sim_matrix = np.zeros((len(all_folder_stories) * len(all_folder_stories[0]), len(all_folder_stories)* len(all_folder_stories[0])))

        for f1 in trange(len(all_folder_stories)):
            initial_story1 = intial_stories[f1]
            for f2 in trange(len(all_folder_stories)):
                initial_story2 = intial_stories[f2]
                for s1 in range(len(all_folder_stories[f1])):
                    for s2 in range(len(all_folder_stories[f2])):

                        stories1 = np.array(all_folder_stories[f1][s1][-1]).flatten()
                        stories2 = np.array(all_folder_stories[f2][s2][-1]).flatten()
                        similarities = get_SBERT_similarity(stories1, stories2)
                        intial_similarity = get_SBERT_similarity([initial_story1], [initial_story2])

                        convergence = np.mean(similarities) - np.mean(intial_similarity)

                        sim_matrix[f1 * len(all_folder_stories[f1]) + s1 , f2 * len(all_folder_stories[f2]) + s2] = convergence

        
        # Save the similarity matrix to a file
        with open(f'Results/Comparisons/{saving_folder}/convergence_matrix.npy', 'wb') as f:
            np.save(f, sim_matrix)
        print("convergence_matrix saved")
        

    labels = [label + ' seed' + str(i) for label in labels for i in range(len(all_folder_stories[0]))]

    print(labels)




    plt.figure(figsize=(20, 20))
    plt.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            text = plt.text(j, i, round(sim_matrix[i][j], 2),
                        ha="center", va="center", color="black", fontweight='bold')
    plt.title('Convergence between treatments (last gen)', fontsize=fontsizes['title'], pad=PAD)
    plt.xticks(range(len(all_folder_stories) * len(all_folder_stories[0])), labels, fontsize=fontsizes['ticks'], rotation=90)
    plt.yticks(range(len(all_folder_stories) * len(all_folder_stories[0])), labels, fontsize=fontsizes['ticks'])
    plt.colorbar(pad=0.02, shrink=0.84)
    plt.tight_layout()



    if saving_folder:
        saving_name = '/convergence_matrix.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()

    plt.close('all')






def compare_gramm_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of grammaticallity within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Grammaticality value', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
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
        all_seeds_grammaticalities = data[folder]['all_seeds_grammaticality_scores']
        
        all_seeds_gen_grammaticalities = []
        for polarities in all_seeds_grammaticalities:
            gen_positivities = []
            for p in polarities:
                gen_positivities.append(np.mean(p))
            all_seeds_gen_grammaticalities.append(gen_positivities)

        label = data[folder]['label']


        plt.plot(np.mean(all_seeds_gen_grammaticalities, axis=0), label=label)
        plt.fill_between(range(0, len(all_seeds_gen_grammaticalities[0])), np.mean(all_seeds_gen_grammaticalities, axis=0) - np.std(all_seeds_gen_grammaticalities, axis=0), np.mean(all_seeds_gen_grammaticalities, axis=0) + np.std(all_seeds_gen_grammaticalities, axis=0), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/grammaticality_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
        
        
def compare_redundancy_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of redundancy within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Redundancy value', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
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
        all_seeds_redundancy = data[folder]['all_seeds_redundancy_scores']
        
        all_seeds_gen_redundancy = []
        for polarities in all_seeds_redundancy:
            gen_positivities = []
            for p in polarities:
                gen_positivities.append(np.mean(p))
            all_seeds_gen_redundancy.append(gen_positivities)

        label = data[folder]['label']


        plt.plot(np.mean(all_seeds_gen_redundancy, axis=0), label=label)
        plt.fill_between(range(0, len(all_seeds_gen_redundancy[0])), np.mean(all_seeds_gen_redundancy, axis=0) - np.std(all_seeds_gen_redundancy, axis=0), np.mean(all_seeds_gen_redundancy, axis=0) + np.std(all_seeds_gen_redundancy, axis=0), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/redundancy_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_focus_evolution(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of focus within generations', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Focus value', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    
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
        all_seeds_focus = data[folder]['all_seeds_focus_scores']
        
        all_seeds_gen_focus = []
        for polarities in all_seeds_focus:
            gen_positivities = []
            for p in polarities:
                gen_positivities.append(np.mean(p))
            all_seeds_gen_focus.append(gen_positivities)

        label = data[folder]['label']


        plt.plot(np.mean(all_seeds_gen_focus, axis=0), label=label)
        plt.fill_between(range(0, len(all_seeds_gen_focus[0])), np.mean(all_seeds_gen_focus, axis=0) - np.std(all_seeds_gen_focus, axis=0), np.mean(all_seeds_gen_focus, axis=0) + np.std(all_seeds_gen_focus, axis=0), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/focus_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
        

def compare_langkit_score(data, plot, sizes, saving_folder, scale_y_axis):
    print("Comparing langkit scores...")
    langkit_scores = ['all_seeds_flesch_reading_ease', 'all_seeds_automated_readability_index', 'all_seeds_aggregate_reading_level', 'all_seeds_syllable_count', 'all_seeds_lexicon_count', 'all_seeds_sentence_count', 'all_seeds_character_count', 'all_seeds_letter_count', 'all_seeds_polysyllable_count', 'all_seeds_monosyllable_count', 'all_seeds_difficult_words', 'all_seeds_difficult_words_ratio', 'all_seeds_polysyllable_ratio', 'all_seeds_monosyllable_ratio']
    
    for langkit_score in langkit_scores:
        plt.figure(figsize=(10, 6))
        plt.title(f'{langkit_score} score evolution', fontsize=sizes['title'], pad=PAD)
        plt.xlabel('Generations', fontsize=sizes['labels'], labelpad=LABEL_PAD)
        plt.ylabel(f'{langkit_score} score', fontsize=sizes['labels'], labelpad=LABEL_PAD)
        
        max_num_ticks = 0 

        for folder in data:
            num_ticks = data[folder]['all_seeds_between_gen_similarity_matrix'][0].shape[0]
            if num_ticks > max_num_ticks:
                max_num_ticks = num_ticks
                x_ticks_space = data[folder]['x_ticks_space']

        if scale_y_axis:
            plt.ylim(0, 100)
            plt.yticks(np.linspace(0, 100, 11), fontsize=sizes['ticks'])

        plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
        plt.grid()

        for folder in data:
            all_seeds_langkit = data[folder][langkit_score]

            #print(all_seeds_langkit)
            
            all_seeds_gen_langkit = []
            for polarities in all_seeds_langkit:
                gen_positivities = []
                for p in polarities:
                    gen_positivities.append(np.mean(p))
                all_seeds_gen_langkit.append(gen_positivities)

            label = data[folder]['label']


            plt.plot(np.mean(all_seeds_gen_langkit, axis=0), label=label)
            plt.fill_between(range(0, len(all_seeds_gen_langkit[0])), np.mean(all_seeds_gen_langkit, axis=0) - np.std(all_seeds_gen_langkit, axis=0), np.mean(all_seeds_gen_langkit, axis=0) + np.std(all_seeds_gen_langkit, axis=0), alpha=0.3)

        plt.legend(fontsize=sizes['legend'])
        
        if saving_folder:
            saving_name = f'/{langkit_score}_gen_comparison.png'
            os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
            plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
            print(f"Saved {saving_name}")
        
        if plot:
            plt.show()




