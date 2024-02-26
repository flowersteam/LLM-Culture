import os 
import numpy as np
import matplotlib.pyplot as plt 

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
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        num_points = data[folder]['between_gen_similarity_matrix'].shape[0]
        value = data[folder]['between_gen_similarity_matrix'][0, 1:]
        label = data[folder]['label']
        plt.plot(range(1, num_points), value, label=label)

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
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-0.1, 1.1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        value = np.diag(data[folder]['between_gen_similarity_matrix'])
        label = data[folder]['label']

        plt.plot(value, label=label, alpha=0.7)

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
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-0.1, 1.1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])
    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        between_gen_similarity_matrix = data[folder]['between_gen_similarity_matrix']
        successive_sim = [between_gen_similarity_matrix[i, i+1] for i in range(between_gen_similarity_matrix.shape[0] - 1)]
        label = data[folder]['label']

        plt.plot(successive_sim, label=label, alpha=0.7)

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
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-1, 1)
        plt.yticks(np.linspace(-1, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        gen_positivities = []
        for gen_polarities in data[folder]['polarities']:
            gen_positivities.append(np.mean(gen_polarities))
        label = data[folder]['label']

        plt.plot(gen_positivities, label=label, alpha=0.7)

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
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        gen_positivities = []
        for gen_polarities in data[folder]['subjectivities']:
            gen_positivities.append(np.mean(gen_polarities))
        
        label = data[folder]['label']

        plt.plot(gen_positivities, label=label, alpha=0.7)

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
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11), fontsize=sizes['ticks'])

    plt.xticks(range(0, max_num_ticks, x_ticks_space), fontsize=sizes['ticks'])
    plt.grid()

    for folder in data:
        gen_creativities = [np.mean(gen_creativity) for gen_creativity in data[folder]['creativities']]
        label = data[folder]['label']

        plt.plot(gen_creativities, label=label, alpha=0.7)

    plt.legend(fontsize=sizes['legend'])
    
    if saving_folder:
        saving_name = '/creativity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()



def plot_similarity_matrix(similarity_matrix, label, n_gen, n_agents, plot, sizes, saving_folder=None):
    plt.figure(figsize=(MATRIX_SIZE, MATRIX_SIZE))
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
        saving_name = f'/stories_similarity_matrix_{label}.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")

    if plot:
        plt.show()