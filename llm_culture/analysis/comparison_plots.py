import os 
import numpy as np
import matplotlib.pyplot as plt 


def compare_init_generation_similarity_evolution(data, plot, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with the initial generation')
    plt.xlabel('Generations')
    plt.ylabel('Similarity with first generation')
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    plt.xticks(range(0, max_num_ticks, x_ticks_space))
    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11))
    plt.grid()

    for folder in data:
        num_points = data[folder]['between_gen_similarity_matrix'].shape[0]
        value = data[folder]['between_gen_similarity_matrix'][0, 1:]
        plt.plot(range(1, num_points), value, label=folder)

    plt.legend()
    
    if saving_folder:
        saving_name = '/similarity_first_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    
def compare_within_generation_similarity_evolution(data, plot, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity within generations')
    plt.xlabel('Generations')
    plt.ylabel('Similarity within generations')
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-0.1, 1.1)
        plt.yticks(np.linspace(0, 1, 11))

    plt.xticks(range(0, max_num_ticks, x_ticks_space))
    plt.grid()

    for folder in data:
        value = np.diag(data[folder]['between_gen_similarity_matrix'])
        plt.plot(value, label=folder, alpha=0.7)

    plt.legend()
    
    if saving_folder:
        saving_name = '/similarity_within_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_successive_generations_similarities(data, plot, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with previous generation')
    plt.xlabel('Generations')
    plt.ylabel('Similarity with previous generation')
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-0.1, 1.1)
        plt.yticks(np.linspace(0, 1, 11))
    plt.xticks(range(0, max_num_ticks, x_ticks_space))
    plt.grid()

    for folder in data:
        between_gen_similarity_matrix = data[folder]['between_gen_similarity_matrix']
        successive_sim = [between_gen_similarity_matrix[i, i+1] for i in range(between_gen_similarity_matrix.shape[0] - 1)]
        plt.plot(successive_sim, label=folder, alpha=0.7)

    plt.legend()
    
    if saving_folder:
        saving_name = '/similarity_successive_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_positivity_evolution(data, plot, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of positivity within generations')
    plt.xlabel('Generations')
    plt.ylabel('Positivity value')
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(-1, 1)
        plt.yticks(np.linspace(-1, 1, 11))

    plt.xticks(range(0, max_num_ticks, x_ticks_space))
    plt.grid()

    for folder in data:
        gen_positivities = []
        for gen_polarities in data[folder]['polarities']:
            gen_positivities.append(np.mean(gen_polarities))
        plt.plot(gen_positivities, label=folder, alpha=0.7)

    plt.legend()
    
    if saving_folder:
        saving_name = '/positivity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
        

def compare_subjectivity_evolution(data, plot, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of subjectivity within generations')
    plt.xlabel('Generations')
    plt.ylabel('Subjectivity value')
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11))

    plt.xticks(range(0, max_num_ticks, x_ticks_space))
    plt.grid()

    for folder in data:
        gen_positivities = []
        for gen_polarities in data[folder]['subjectivities']:
            gen_positivities.append(np.mean(gen_polarities))
        plt.plot(gen_positivities, label=folder, alpha=0.7)

    plt.legend()
    
    if saving_folder:
        saving_name = '/subjectivity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_creativity_evolution(data, plot, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of creativity within generations')
    plt.xlabel('Generations')
    plt.ylabel('Creativity index')
    
    max_num_ticks = 0 

    for folder in data:
        num_ticks = data[folder]['between_gen_similarity_matrix'].shape[0]
        if num_ticks > max_num_ticks:
            max_num_ticks = num_ticks
            x_ticks_space = data[folder]['x_ticks_space']

    if scale_y_axis:
        plt.ylim(0, 1)
        plt.yticks(np.linspace(0, 1, 11))

    plt.xticks(range(0, max_num_ticks, x_ticks_space))
    plt.grid()

    for folder in data:
        gen_creativities = [np.mean(gen_creativity) for gen_creativity in data[folder]['creativities']]
        plt.plot(gen_creativities, label=folder, alpha=0.7)

    plt.legend()
    
    if saving_folder:
        saving_name = '/creativity_gen_comparison.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()