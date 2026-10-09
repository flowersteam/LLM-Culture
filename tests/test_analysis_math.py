"""Characterization tests for the analysis (similarity-matrix) math.

These pin the CURRENT numeric output of the analysis pipeline so a refactor,
dependency bump, or new embedder can't silently change the project's scientific
numbers. Golden values recorded 2026-10-09 from the TF-IDF path (the paper default).

The core suite is deterministic and network-free: TfidfVectorizer uses sklearn's
own tokenizer/stopwords (not NLTK), and TextBlob's sentiment analyzer is bundled.
The HuggingFace case is opt-in and skips when the extra/model is unavailable.
"""
import numpy as np
import pytest

from llm_culture.analysis.utils import (
    get_similarity_matrix,
    get_similarity_matrix_single_seed,
    compute_between_gen_similarities,
    compute_between_gen_similarities_single_seed,
    get_polarities_subjectivities_single_seed,
    get_similarity,
)

ATOL = 1e-9

# Fixed fixture: 2 generations x 2 agents, with deliberate word overlap
# (gen0[0] vs gen1[0] share "cat sat ... mat") so the matrix is non-trivial.
STORIES = [
    ["the cat sat on the warm mat", "a dog ran across the green field"],
    ["the cat sat quietly on the mat", "birds fly high over the blue sea"],
]
N_GEN, N_AGENTS = 2, 2
FLAT = [STORIES[i][j] for i in range(N_GEN) for j in range(N_AGENTS)]

# --- recorded golden values (TF-IDF) ---
EXPECTED_SIM = np.array([
    [1.0, 0.0, 0.6509328139962117, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.6509328139962117, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
EXPECTED_BETWEEN = np.array([
    [0.5, 0.16273320349905293],
    [0.16273320349905293, 0.5],
])
EXPECTED_WITHIN = [0.5, 0.5]                       # np.diag(between)
EXPECTED_SUCCESSIVE = [0.16273320349905293]        # between[i, i+1]
EXPECTED_FIRST_GEN = [0.16273320349905293]         # between[0, 1:]
EXPECTED_POLARITIES = [[0.6, -0.2], [0.0, 0.32]]
EXPECTED_SUBJECTIVITIES = [[0.6, 0.3], [0.3333333333333333, 0.5133333333333333]]


# ---------- similarity matrix (TF-IDF) ----------

def test_similarity_matrix_single_seed_golden():
    sim = get_similarity_matrix_single_seed(FLAT)
    assert sim.shape == (4, 4)
    np.testing.assert_allclose(sim, EXPECTED_SIM, atol=ATOL)


def test_similarity_matrix_invariants():
    sim = get_similarity_matrix_single_seed(FLAT)
    np.testing.assert_allclose(sim, sim.T, atol=ATOL)          # symmetric
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=ATOL)   # self-similarity
    assert sim.min() >= -ATOL and sim.max() <= 1.0 + ATOL      # TF-IDF >= 0


def test_similarity_matrix_multi_seed_matches_single():
    mats = get_similarity_matrix([FLAT, FLAT])
    assert len(mats) == 2
    for m in mats:
        np.testing.assert_allclose(m, EXPECTED_SIM, atol=ATOL)


def test_tfidf_equivalence_to_raw_vectorizer():
    """Standing guard for the PR-5 claim: the embedder's cosine matrix reproduces
    the paper's original ``tfidf * tfidf.T`` (recorded diff was ~3e-16)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(min_df=1, stop_words="english").fit_transform(FLAT)
    raw = (X @ X.T).toarray()
    sim = get_similarity_matrix_single_seed(FLAT)
    np.testing.assert_allclose(sim, raw, atol=ATOL)


# ---------- between-generation matrix ----------

def test_between_gen_single_seed_golden():
    sim = get_similarity_matrix_single_seed(FLAT)
    between = compute_between_gen_similarities_single_seed(sim, N_GEN, N_AGENTS)
    assert between.shape == (N_GEN, N_GEN)
    np.testing.assert_allclose(between, EXPECTED_BETWEEN, atol=ATOL)


def test_between_gen_multi_seed():
    sim = get_similarity_matrix_single_seed(FLAT)
    out = compute_between_gen_similarities([sim, sim], N_GEN, N_AGENTS)
    assert len(out) == 2
    for b in out:
        np.testing.assert_allclose(b, EXPECTED_BETWEEN, atol=ATOL)


# ---------- derived metrics (currently computed inline in plots.py) ----------
# These replicate the exact one-liners used by the plot functions so the
# semantics stay pinned even though the code lives next to matplotlib.

def test_derived_within_gen_similarity():
    between = compute_between_gen_similarities_single_seed(
        get_similarity_matrix_single_seed(FLAT), N_GEN, N_AGENTS)
    within = np.diag(between)                                   # plot_within_gen_similarities
    np.testing.assert_allclose(within, EXPECTED_WITHIN, atol=ATOL)


def test_derived_successive_similarity():
    between = compute_between_gen_similarities_single_seed(
        get_similarity_matrix_single_seed(FLAT), N_GEN, N_AGENTS)
    successive = [between[i, i + 1] for i in range(between.shape[0] - 1)]  # plot_successive_generations_similarities
    np.testing.assert_allclose(successive, EXPECTED_SUCCESSIVE, atol=ATOL)


def test_derived_similarity_with_first_gen():
    between = compute_between_gen_similarities_single_seed(
        get_similarity_matrix_single_seed(FLAT), N_GEN, N_AGENTS)
    first = between[0, 1:]                                      # plot_init_generation_similarity_evolution
    np.testing.assert_allclose(first, EXPECTED_FIRST_GEN, atol=ATOL)


# ---------- helpers ----------

def test_get_similarity_cosine_and_zero_guard():
    assert get_similarity(np.array([1.0, 0, 0]), np.array([1.0, 0, 0])) == pytest.approx(1.0)
    assert get_similarity(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(0.0)
    assert get_similarity(np.array([0.0, 0, 0]), np.array([1.0, 0, 0])) == 0.0  # zero-vector guard


def test_sentiment_golden():
    pol, subj = get_polarities_subjectivities_single_seed(STORIES)
    np.testing.assert_allclose(pol, EXPECTED_POLARITIES, atol=ATOL)
    np.testing.assert_allclose(subj, EXPECTED_SUBJECTIVITIES, atol=ATOL)


# ---------- HuggingFace embedder (opt-in) ----------

def test_huggingface_embedder_invariants():
    """Structural invariants only (not golden numbers — weights/version drift).
    Skips when the extra isn't installed or the model isn't cached offline."""
    pytest.importorskip("sentence_transformers")
    from llm_culture.analysis.embedders import make_embedder, HUGGINGFACE

    try:
        embedder = make_embedder(method=HUGGINGFACE)
        sim = get_similarity_matrix_single_seed(FLAT, embedder)
    except Exception as exc:  # model download/load unavailable offline
        pytest.skip(f"HF model unavailable: {exc}")

    assert sim.shape == (4, 4)
    np.testing.assert_allclose(sim, sim.T, atol=1e-5)
    np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)
    assert sim.min() >= -1.0 - 1e-6 and sim.max() <= 1.0 + 1e-6
