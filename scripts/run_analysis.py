"""Analyse an existing results folder (Hydra entrypoint).

Config: AnalysisRunConfig (llm_culture/config.py) via conf/analysis.yaml.

    uv run python scripts/run_analysis.py folder=results/my_run
"""
import re
import sys
from pathlib import Path

import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Ensure the repo root is importable (so `llm_culture` resolves) regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_culture.analysis.utils import get_stories, get_plotting_infos, initialize_nltk, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities
from llm_culture.analysis.plots import run_configured_plots
from llm_culture.analysis.embedders import make_embedder, TFIDF
from llm_culture.config import EmbeddingConfig, AnalysisRunConfig

cs = ConfigStore.instance()
cs.store(name="analysis_schema", node=AnalysisRunConfig)


def _embedding_cache_name(method, model, base="analysis_cache_df"):
    """Cache filename keyed on the embedding method+model so switching embeddings
    never silently reuses another method's cached similarity matrix. TF-IDF keeps
    the historical name so existing (paper) caches still load."""
    if method == TFIDF:
        return f"{base}.pkl"
    safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{method}_{model}").strip("_")
    return f"{base}__{safe}.pkl"




def _compute_analysis_data(folder, font_sizes, embedding):
    all_seeds_stories = get_stories(folder)

    print(f"Number of stories: {len(all_seeds_stories)}\n")
    n_seeds = len(all_seeds_stories)
    n_gen, n_agents, x_ticks_space = get_plotting_infos(all_seeds_stories[0])

    all_seeds_flat_stories, all_seeds_keywords, all_seeds_stem_words = preprocess_stories(all_seeds_stories)
    # Built lazily (cache-miss path) so a HF model loads once and is reused across seeds.
    embedder = make_embedder(embedding.method, embedding.model, embedding.batch_size, embedding.device)
    all_seeds_similarity_matrix = get_similarity_matrix(all_seeds_flat_stories, embedder)
    all_seeds_between_gen_similarity_matrix = compute_between_gen_similarities(all_seeds_similarity_matrix, n_gen, n_agents)
    all_seeds_polarities, all_seeds_subjectivities = get_polarities_subjectivities(all_seeds_stories)
    # all_seeds_creativities = get_creativity_indexes(all_seeds_stories, folder)
    # embedding_points = plot_embedding(folder, False, sizes=font_sizes, save=False)

    return {
        'n_seeds': n_seeds,
        'n_gen': n_gen,
        'n_agents': n_agents,
        'x_ticks_space': x_ticks_space,
        'all_seeds_stories': all_seeds_stories,
        'all_seeds_flat_stories': all_seeds_flat_stories,
        'all_seeds_keywords': all_seeds_keywords,
        'all_seeds_stem_words': all_seeds_stem_words,
        'all_seeds_similarity_matrix': all_seeds_similarity_matrix,
        'all_seeds_between_gen_similarity_matrix': all_seeds_between_gen_similarity_matrix,
        # 'all_seeds_polarities': all_seeds_polarities,
        # 'all_seeds_subjectivities': all_seeds_subjectivities,
        # 'all_seeds_creativities': all_seeds_creativities,
        # 'embedding_points': embedding_points,
    }


def _load_or_compute_analysis_data(folder, font_sizes, cache_file_name='analysis_cache_df.pkl',
                                   force_recompute_cache=False, embedding=None):
    cache_path = Path(folder) / cache_file_name

    if cache_path.exists() and not force_recompute_cache:
        print(f"Loading cached analysis DataFrame from {cache_path}")
        cache_df = pd.read_pickle(cache_path)
        if len(cache_df) == 0:
            raise ValueError(f"Cache file is empty: {cache_path}")
        return cache_df.iloc[0].to_dict()

    print("No usable cache found, computing analysis data...")
    analysis_data = _compute_analysis_data(folder, font_sizes, embedding or EmbeddingConfig())
    cache_df = pd.DataFrame([analysis_data])
    cache_df.to_pickle(cache_path)
    print(f"Saved analysis cache DataFrame to {cache_path}")
    return analysis_data


def main_analysis(
    folder,
    font_sizes={'ticks': 12, 'labels': 14, 'title': 16},
    plot=False,
    cache_file_name=None,
    force_recompute_cache=False,
    plot_configs=None,
    embedding=None,
):
    """Run the analysis + plots on a results folder.

    :param folder: results folder to analyze
    :param font_sizes: plot font sizes
    :param plot: also open figures interactively (saved either way)
    :param cache_file_name: override the cache file; None -> derived from embedding method+model
    :param force_recompute_cache: ignore any existing cache and recompute
    :param plot_configs: subset of plots to render (None -> defaults)
    :param embedding: EmbeddingConfig picking how stories are vectorized (None -> TF-IDF default)
    """
    if embedding is None:
        embedding = EmbeddingConfig()
    if cache_file_name is None:
        cache_file_name = _embedding_cache_name(embedding.method, embedding.model)
    analysis_data = _load_or_compute_analysis_data(
        folder,
        font_sizes,
        cache_file_name=cache_file_name,
        force_recompute_cache=force_recompute_cache,
        embedding=embedding,
    )

    run_configured_plots(analysis_data, folder, plot=plot, sizes=font_sizes, plot_names=plot_configs)
    print(f"\nAnalysis complete — plots + cache saved to {Path(folder).resolve()}")

def _resolve_folder(folder: str) -> str:
    """Resolve the folder to analyze: use the path as given, falling back to a
    cwd-relative path, and raise a clear error if neither exists."""
    path = Path(folder)
    if not path.exists():
        path = Path.cwd() / folder
    if not path.exists():
        raise FileNotFoundError(f"Analysis folder not found: {folder}")
    return str(path)


@hydra.main(version_base=None, config_path="../conf", config_name="analysis")
def main(cfg: DictConfig) -> None:
    conf: AnalysisRunConfig = OmegaConf.to_object(cfg)

    print("\n" + "#" * 64)
    print("# ANALYSIS CONFIG")
    print("#" * 64)
    print(OmegaConf.to_yaml(cfg), end="")

    initialize_nltk()
    folder = _resolve_folder(conf.folder)
    print(f"\nLaunching analysis on the {folder} results (plot={conf.plot})")
    main_analysis(
        folder,
        conf.font_sizes,
        conf.plot,
        cache_file_name=conf.cache_file,
        force_recompute_cache=conf.recompute_cache,
        embedding=conf.embedding,
    )


if __name__ == "__main__":
    main()
