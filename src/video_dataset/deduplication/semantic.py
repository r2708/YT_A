"""Semantic similarity for descriptions: sentence-transformers when installed, TF-IDF cosine otherwise."""

from __future__ import annotations

from video_dataset.utils.logging import get_logger

log = get_logger("dedup.semantic")


class SemanticSimilarity:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.backend = "tfidf"
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name, device=device)
            self.backend = "sentence_transformers"
        except Exception:
            self._model = None

    def similar_pairs(self, texts: list[str], threshold: float = 0.92) -> list[tuple[int, int, float]]:
        if len(texts) < 2:
            return []

        if self._model is not None:
            emb = self._model.encode(texts, normalize_embeddings=True)
            sims = emb @ emb.T
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
            sims = (vec @ vec.T).toarray()
        pairs = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sims[i, j])
                if s >= threshold:
                    pairs.append((i, j, round(s, 4)))
        return pairs
