"""하이브리드 결합(RRF)과 크로스인코더 리랭킹.

RRF는 점수 척도가 다른 BM25와 코사인 유사도를 정규화 없이 순위만으로 합친다.
    rrf(d) = Σ_i 1 / (k + rank_i(d)),  rank는 1부터, 목록에 없으면 기여 0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from rag_eval.matrix import Retriever


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, depth: int = 100):
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.depth = depth

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        depth = max(k, self.depth)
        rankings = [[d for d, _ in r.search(query, depth)] for r in self.retrievers]
        return rrf(rankings, self.rrf_k)[:k]


class PairScorer(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    def __init__(self, model_name: str, device: str = "", max_length: int = 512):
        from sentence_transformers import CrossEncoder

        from minwon_rag.dense import pick_device

        self.model = CrossEncoder(model_name, device=pick_device(device), max_length=max_length)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16)]


class RerankRetriever:
    """기반 검색 상위 ``candidates`` 건만 다시 매기고, 그 뒤 순위는 기반 순서를 유지한다."""

    def __init__(
        self,
        base: Retriever,
        scorer: PairScorer,
        doc_texts: Mapping[str, str],
        candidates: int = 50,
    ):
        self.base = base
        self.scorer = scorer
        self.doc_texts = doc_texts
        self.candidates = candidates

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        hits = self.base.search(query, max(k, self.candidates))
        head = [d for d, _ in hits[: self.candidates]]
        scores = self.scorer.score(query, [self.doc_texts[d] for d in head])
        reranked = sorted(zip(head, scores, strict=True), key=lambda x: -x[1])
        # 리랭크 밖 문서는 리랭크 점수보다 항상 낮게 둔다
        floor = min(scores, default=0.0) - 1.0
        tail = [(d, floor - i) for i, (d, _) in enumerate(hits[self.candidates :], start=1)]
        return (reranked + tail)[:k]
