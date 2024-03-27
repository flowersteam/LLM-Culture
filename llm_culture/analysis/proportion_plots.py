import matplotlib.pyplot as plt
import numpy as np
import os
PAD = 20
LABEL_PAD = 10
MATRIX_SIZE = 10



def compare_init_generation_similarity_proportion(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('Evolution of similarity with the initial generation', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Number of creative agents', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Similarity with first generation', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    values = []
    stds = []
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
        value = np.mean(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0)[0, -1]
        std = np.std(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0)[0, -1]
        label = data[folder]['label']
        print(label)
        values.append(value)
        stds.append(std)


    plt.plot(values, label=label)
    plt.fill_between(range(len(values)), np.array(values) - np.array(stds), np.array(values) + np.array(stds), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])


    if saving_folder:
        saving_name = '/similarity_first_gen_proportion.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    


def compare_within_gen_similarity_proportion(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Number of creative agents', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Within-generation similarity', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    values = []
    stds = []
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
        value = np.diag(np.mean(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0))[-1]
        label = data[folder]['label']
        std = np.diag(np.std(data[folder]['all_seeds_between_gen_similarity_matrix'], axis = 0))[-1]
        values.append(value)
        stds.append(std)



    plt.plot(values, label=label)
    plt.fill_between(range(len(values)), np.array(values) - np.array(stds), np.array(values) + np.array(stds), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])


    if saving_folder:
        saving_name = '/similarity_within_gen_proportion.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()
    

def compare_successive_similarity_proportion(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Number of creative agents', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Similarity with previous generation', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    values = []
    stds = []
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
        all_seeds_between_gen_similarity_matrix = data[folder]['all_seeds_between_gen_similarity_matrix']
        successive_sim = [[all_seeds_between_gen_similarity_matrix[seed][i, i+1] for i in range(all_seeds_between_gen_similarity_matrix[0].shape[0] - 1)] for seed in range(len(all_seeds_between_gen_similarity_matrix))]
        value = np.mean(successive_sim, axis = 0)[-1]
        label = data[folder]['label']
        std = np.std(successive_sim, axis = 0)[-1]
        values.append(value)
        stds.append(std)



    plt.plot(values, label=label)
    plt.fill_between(range(len(values)), np.array(values) - np.array(stds), np.array(values) + np.array(stds), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])


    if saving_folder:
        saving_name = '/successive_similarity_proportion.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_positivity_proportion(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Number of creative agents', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Positivity', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    values = []
    stds = []
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
        all_seeds_positivities = data[folder]['all_seeds_positivities']

        all_seeds_gen_positivities = []
        for polarities in all_seeds_positivities:
            gen_positivities = []
            for p in polarities:
                gen_positivities.append(np.mean(p))
            all_seeds_gen_positivities.append(gen_positivities[-1])

        label = data[folder]['label']


        value = np.mean(all_seeds_gen_positivities, axis=0)
        std = np.std(all_seeds_gen_positivities, axis=0)

        values.append(value)
        stds.append(std)



    plt.plot(values, label=label)
    plt.fill_between(range(len(values)), np.array(values) - np.array(stds), np.array(values) + np.array(stds), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])


    if saving_folder:
        saving_name = '/positivity_proportion.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_subjectivity_proportion(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Number of creative agents', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Subjectivity', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    values = []
    stds = []
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
        all_seeds_subjectivities = data[folder]['all_seeds_subjectivities']
        all_see_gen_subjectivities = []
        for subjectivities in all_seeds_subjectivities:
            gen_subjectivities = []
            for gen_subjectivity in subjectivities:
                gen_subjectivities.append(np.mean(gen_subjectivity))
            all_see_gen_subjectivities.append(gen_subjectivities[-1])
        
        label = data[folder]['label']
        value = np.mean(all_see_gen_subjectivities, axis=0)
        std = np.std(all_see_gen_subjectivities, axis=0)

        values.append(value)
        stds.append(std)



    plt.plot(values, label=label)
    plt.fill_between(range(len(values)),np.array(values) - np.array(stds), np.array(values) + np.array(stds), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])


    if saving_folder:
        saving_name = '/subjectivity_proportion.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()


def compare_creativity_proportion(data, plot, sizes, saving_folder=None, scale_y_axis=False):
    plt.figure(figsize=(10, 6))
    plt.title('', fontsize=sizes['title'], pad=PAD)
    plt.xlabel('Number of creative agents', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    plt.ylabel('Creativity', fontsize=sizes['labels'], labelpad=LABEL_PAD)
    values = []
    stds = []
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
        all_seeds_creativity_indices = data[folder]['all_seeds_creativity_indices']
        all_seeds_gen_creativities = []
        for creativity_indices in all_seeds_creativity_indices:
            gen_creativities = [1 - np.mean(gen_creativity) for gen_creativity in creativity_indices]
            all_seeds_gen_creativities.append(gen_creativities[-1])

        label = data[folder]['label']
        value = np.mean(all_seeds_gen_creativities, axis=0)
        std = np.std(all_seeds_gen_creativities, axis=0)
       
        values.append(value)
        stds.append(std)



    plt.plot(values, label=label)
    plt.fill_between(range(len(values)), np.array(values) - np.array(stds), np.array(values) + np.array(stds), alpha=0.3)

    plt.legend(fontsize=sizes['legend'])


    if saving_folder:
        saving_name = '/creativity_proportion.png'
        os.makedirs(f"Results/Comparisons/{saving_folder}", exist_ok=True)
        plt.savefig(f"Results/Comparisons/{saving_folder}/{saving_name}")
        print(f"Saved {saving_name}")
    
    if plot:
        plt.show()