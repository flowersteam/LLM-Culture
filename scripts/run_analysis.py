import re
import argparse
from pathlib import Path
import pandas as pd

from llm_culture.analysis.utils import get_stories, get_plotting_infos, initialize_nltk, preprocess_stories, get_similarity_matrix
from llm_culture.analysis.utils import compute_between_gen_similarities, get_polarities_subjectivities
from llm_culture.analysis.plots import run_configured_plots
from llm_culture.analysis.embedders import make_embedder, TFIDF
from llm_culture.config import EmbeddingConfig


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="FC_10_10_combine_mixedPop_5seeds")
    parser.add_argument("--folder", type=str, default=None, help="Full or repo-relative folder to analyze (overrides --dir)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--cache_file", type=str, default=None,
                        help="override the cache filename (default: derived from the embedding method/model)")
    parser.add_argument("--force_recompute_cache", action="store_true")
    parser.add_argument("--embedding-method", dest="embedding_method", type=str,
                        default="tfidf", choices=["tfidf", "huggingface"])
    parser.add_argument("--embedding-model", dest="embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", dest="embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding-device", dest="embedding_device", type=str, default=None)
    parser.add_argument("--ticks_font_size", type=int, default=12)
    parser.add_argument("--labels_font_size", type=int, default=14)
    parser.add_argument("--title_font_size", type=int, default=16)
    args = parser.parse_args()

    initialize_nltk()

    if args.folder is not None:
        analyzed_dir = Path(args.folder)
        if not analyzed_dir.exists():
            analyzed_dir = Path.cwd() / args.folder
    else:
        analyzed_dir = Path("results") / args.dir

    if not analyzed_dir.exists():
        raise FileNotFoundError(f"Analysis folder not found: {analyzed_dir}")

    analyzed_dir = str(analyzed_dir)
    
    font_sizes = {
        'ticks': args.ticks_font_size,
        'labels': args.labels_font_size,
        'title': args.title_font_size
    }
    
    print(f"\nLaunching analysis on the {analyzed_dir} results")
    print(f"plot = {args.plot}")
    main_analysis(
        analyzed_dir,
        font_sizes,
        args.plot,
        cache_file_name=args.cache_file,
        force_recompute_cache=args.force_recompute_cache,
        embedding=EmbeddingConfig(
            method=args.embedding_method,
            model=args.embedding_model,
            batch_size=args.embedding_batch_size,
            device=args.embedding_device,
        ),
    )
