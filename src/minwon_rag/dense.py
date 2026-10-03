"""dense 검색: 문장 임베딩 + 정규화 내적(코사인) 전수 탐색.

코퍼스가 수만 건 수준이라 근사 색인(ANN) 없이 행렬 곱 한 번으로 전수 탐색한다.
임베딩 모델은 sentence-transformers로 불러오며 Apple Silicon에서는 mps를 쓴다.
문서는 청킹하지 않고 한 건을 벡터 하나로 만든다(모델 최대 길이에서 잘림).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """L2 정규화된 (len(texts), dim) 배열."""
        ...


def pick_device(preferred: str = "") -> str:
    if preferred:
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "", max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=pick_device(device))
        self.model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


class DenseIndex:
    def __init__(self, docids: Sequence[str], vectors: np.ndarray):
        if len(docids) != len(vectors):
            raise ValueError("docids와 벡터 수가 다릅니다")
        self.docids = list(docids)
        self.vectors = vectors.astype(np.float32)

    def top_k(self, query_vec: np.ndarray, k: int) -> list[tuple[str, float]]:
        s = self.vectors @ query_vec.astype(np.float32).ravel()
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(self.docids[i], float(s[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: str | Path, docids: Sequence[str]) -> DenseIndex:
        return cls(docids, np.load(path))


class DenseRetriever:
    def __init__(self, embedder: Embedder, index: DenseIndex):
        self.embedder = embedder
        self.index = index

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.index.top_k(self.embedder.encode([query])[0], k)
