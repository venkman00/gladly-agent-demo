"""Semantic search over help-center articles with BGE embeddings.

Why this is enough: a few hundred articles fit in memory as a numpy matrix, and
with L2-normalized vectors cosine similarity is a single matrix-vector product.
Reach for a vector DB (OpenSearch, pgvector) when the corpus outgrows RAM or
changes faster than you redeploy.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

# BGE is trained with an instruction prefix on the *query* side only (asymmetric retrieval).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
KB_PATH = Path(__file__).parent.parent / "data" / "kb.json"


class Article(BaseModel):
    id: str
    title: str
    body: str


class Hit(BaseModel):
    article: Article
    score: float


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class BGEEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import: only where needed

        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class KnowledgeBase:
    def __init__(self, articles: list[Article], embedder: Embedder):
        self.articles = articles
        self.embedder = embedder
        # Embed once at startup (Lambda cold start); queries reuse the matrix.
        self.matrix = embedder.encode([f"{a.title}\n{a.body}" for a in articles])

    @classmethod
    def from_file(cls, embedder: Embedder, path: Path = KB_PATH) -> "KnowledgeBase":
        return cls([Article(**a) for a in json.loads(path.read_text())], embedder)

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[Hit]:
        q = self.embedder.encode([BGE_QUERY_PREFIX + query])[0]
        scores = self.matrix @ q  # cosine similarity, since both sides are normalized
        top = np.argsort(-scores)[:k]
        return [Hit(article=self.articles[i], score=round(float(scores[i]), 3)) for i in top if scores[i] >= min_score]
