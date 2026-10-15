"""Pluggable text embedders for the story-similarity analysis.

Each embedder maps texts -> an (N, D) matrix (sparse for TF-IDF, dense for HF);
callers compare rows with cosine similarity. Downstream code only ever sees the
resulting N x N similarity matrix, so swapping embedders is transparent to it.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

TFIDF = "tfidf"
HUGGINGFACE = "huggingface"
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TfidfEmbedder:
    """TF-IDF bag-of-words vectors (default). Rows are L2-normalized, so cosine
    similarity reproduces the paper's original ``tfidf * tfidf.T``."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(min_df=1, stop_words="english")

    def embed(self, texts):
        return self._vectorizer.fit_transform(list(texts))


class SentenceTransformerEmbedder:
    """Dense sentence embeddings from a HuggingFace sentence-transformers model,
    loaded once and reused. Vectors are L2-normalized."""

    def __init__(self, model_name=DEFAULT_HF_MODEL, batch_size=32, device=None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "The 'huggingface' embedding method needs sentence-transformers: "
                "uv sync --extra embeddings"
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts):
        return self._model.encode(
            list(texts), batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )


def make_embedder(method=TFIDF, model_name=DEFAULT_HF_MODEL, batch_size=32, device=None):
    """Build an embedder from AnalysisConfig's embedding fields.

    :param method: "tfidf" (default, reproducible) or "huggingface"
    :param model_name: HF repo id (huggingface only)
    :param batch_size: encode batch size (huggingface only)
    :param device: torch device override, or None to let the library choose
    """
    if method == TFIDF:
        return TfidfEmbedder()
    if method == HUGGINGFACE:
        return SentenceTransformerEmbedder(model_name, batch_size=batch_size, device=device)
    raise ValueError(f"Unknown embedding_method={method!r}; expected {TFIDF!r} or {HUGGINGFACE!r}.")
