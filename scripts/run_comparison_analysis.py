"""Compare several existing results folders (Hydra entrypoint).

Config: ComparisonConfig (llm_culture/config.py) via conf/comparison.yaml.

    uv run python scripts/run_comparison_analysis.py \\
        'folders=[Network Structure/CAVEMAN_10_10_combine5seeds, Network Structure/CIRCLE_10_10_combine5seeds]'
"""
import os
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Ensure the repo root is importable (so `llm_culture` resolves) regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_culture.analysis.utils import get_stories, get_plotting_infos, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities
from llm_culture.analysis.comparison_plots import run_configured_comparison_plots
from llm_culture.analysis.embedders import make_embedder
from llm_culture.config import EmbeddingConfig, ComparisonConfig

cs = ConfigStore.instance()
cs.store(name="comparison_schema", node=ComparisonConfig)

RESULTS_DIR = 'results/experiments'
COMPARISON_DIR = 'results/experiments_comparisons'

def run_comparison_analysis(folders, plot, scale_y_axis, labels, sizes, plot_configs=None,
                            embedding=None):
    """Compare the story evolution across several result folders.

    :param folders: list of result folders to compare
    :param plot: also open figures interactively
    :param scale_y_axis: use a shared y-axis scale across folders
    :param labels: per-folder labels for the legends
    :param sizes: plot font sizes
    :param plot_configs: subset of comparison plots (None -> defaults)
    :param embedding: EmbeddingConfig picking how stories are vectorized (None -> TF-IDF default)
    """
    # One embedder for all folders so a HF model is loaded only once.
    embedding = embedding or EmbeddingConfig()
    embedder = make_embedder(embedding.method, embedding.model, embedding.batch_size, embedding.device)
    saving_folder = '-'.join(os.path.basename(folder) for folder in folders)
    data = {}
    
    # Extract, analyze and plot the data for each different seed
    for i, folder in enumerate(folders):
        # Compute all the metric that will be used for plotting
        all_seeds_stories = get_stories(folder)
        n_gen, n_agents, x_ticks_space = get_plotting_infos(all_seeds_stories[0])
        all_seed_flat_stories, all_seed_keywords, all_seed_stem_words = preprocess_stories(all_seeds_stories)
        all_seed_similarity_matrix = get_similarity_matrix(all_seed_flat_stories, embedder)
        all_seed_between_gen_similarity_matrix = compute_between_gen_similarities(all_seed_similarity_matrix, n_gen, n_agents)
        all_seed_polarities, all_seed_subjectivities = get_polarities_subjectivities(all_seeds_stories)
        # all_seed_creativities = get_creativity_indexes(all_seeds_stories, folder)
        label = labels[i]
        data[folder] = {
            'all_seed_stories': all_seeds_stories,
            'n_gen': n_gen,
            'n_agents': n_agents,
            'x_ticks_space': x_ticks_space,
            'all_seeds_flat_stories': all_seed_flat_stories,
            'all_seeds_keywords': all_seed_keywords,
            'all_seeds_stem_words': all_seed_stem_words,
            'all_seeds_similarity_matrix': all_seed_similarity_matrix,
            'all_seeds_between_gen_similarity_matrix': all_seed_between_gen_similarity_matrix,
            # 'all_seeds_positivities': all_seed_polarities,
            # 'all_seeds_subjectivities': all_seed_subjectivities,
            # 'all_seeds_creativity_indices': all_seed_creativities,
            'label': label
            }
   
    # Plot all the desired graphs to compare the different seeds:
    run_configured_comparison_plots(data, plot, sizes, saving_folder, scale_y_axis, plot_names=plot_configs)


@hydra.main(version_base=None, config_path="../conf", config_name="comparison")
def main(cfg: DictConfig) -> None:
    conf: ComparisonConfig = OmegaConf.to_object(cfg)

    print("\n" + "#" * 64)
    print("# COMPARISON CONFIG")
    print("#" * 64)
    print(OmegaConf.to_yaml(cfg), end="")

    if not conf.folders:
        raise ValueError(
            "No folders to compare — set `folders=[...]` (names/paths joined under "
            f"`root`, default '{RESULTS_DIR}'). Example:\n"
            "  uv run python scripts/run_comparison_analysis.py "
            "'folders=[Network Structure/CAVEMAN_10_10_combine5seeds, "
            "Network Structure/CIRCLE_10_10_combine5seeds]'"
        )

    dirs_list = [os.path.join(conf.root, name) if conf.root else name for name in conf.folders]
    labels = conf.labels if conf.labels else [os.path.basename(d) for d in dirs_list]
    if len(labels) != len(dirs_list):
        raise ValueError(
            f"Got {len(labels)} labels for {len(dirs_list)} folders — provide one "
            "label per folder, or omit `labels` to use the folder basenames."
        )

    print(f"\nLaunching comparison analysis on {len(dirs_list)} folder(s) (plot={conf.plot})")
    run_comparison_analysis(
        dirs_list,
        conf.plot,
        conf.scale_y_axis,
        labels,
        conf.sizes,
        embedding=conf.embedding,
    )


if __name__ == "__main__":
    main()