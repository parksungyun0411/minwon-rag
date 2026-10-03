"""설정 파일(configs/matrix.yaml) → 검색기 팩토리.

모델은 설정마다가 아니라 별칭마다 한 번만 적재한다(같은 임베딩 모델을 dense·hybrid가 공유).
테스트에서는 ``embedder_factory`` 와 ``scorer_factory`` 로 가짜 모델을 주입한다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rag_eval.matrix import Retriever

from minwon_rag.config import Settings
from minwon_rag.dense import DenseIndex, DenseRetriever, Embedder, SentenceTransformerEmbedder
from minwon_rag.fusion import CrossEncoderScorer, HybridRetriever, PairScorer, RerankRetriever
from minwon_rag.lexical import BM25Index, BM25Retriever, KiwiTokenizer, Tokenizer


@dataclass
class MatrixConfig:
    configs: list[dict[str, Any]]
    models: dict[str, str]
    ks: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    depth: int = 100
    rrf_k: int = 60
    rerank_candidates: int = 50
    baseline: str | None = None
    answer_config: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> MatrixConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        cfg = cls(**raw)
        for c in cfg.configs:
            for key in ("dense", "rerank"):
                if c.get(key) and c[key] not in cfg.models:
                    raise ValueError(f"{c['name']}: 알 수 없는 모델 별칭 {c[key]}")
            if not c.get("dense") and not c.get("lexical"):
                raise ValueError(f"{c['name']}: dense나 lexical 중 하나는 있어야 합니다")
        return cfg

    def dense_aliases(self) -> list[str]:
        return sorted({c["dense"] for c in self.configs if c.get("dense")})

    def uses_lexical(self) -> bool:
        return any(c.get("lexical") for c in self.configs)


def dense_path(index_dir: Path, alias: str) -> Path:
    return index_dir / f"dense__{alias}.npy"


class Components:
    """모델·색인을 별칭 단위로 한 번만 만든다."""

    def __init__(
        self,
        cfg: MatrixConfig,
        settings: Settings,
        doc_texts: dict[str, str],
        embedder_factory: Callable[[str], Embedder] | None = None,
        scorer_factory: Callable[[str], PairScorer] | None = None,
        tokenizer: Tokenizer | None = None,
    ):
        self.cfg = cfg
        self.settings = settings
        self.doc_texts = doc_texts
        self._embedder_factory = embedder_factory or (
            lambda name: SentenceTransformerEmbedder(name, settings.device)
        )
        self._scorer_factory = scorer_factory or (
            lambda name: CrossEncoderScorer(name, settings.device)
        )
        self._tokenizer = tokenizer
        self._embedders: dict[str, Embedder] = {}
        self._dense: dict[str, DenseRetriever] = {}
        self._scorers: dict[str, PairScorer] = {}
        self._lexical: Retriever | None = None

    @property
    def docids(self) -> list[str]:
        return json.loads((self.settings.index_dir / "docids.json").read_text(encoding="utf-8"))

    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            self._tokenizer = KiwiTokenizer()
        return self._tokenizer

    def embedder(self, alias: str) -> Embedder:
        if alias not in self._embedders:
            self._embedders[alias] = self._embedder_factory(self.cfg.models[alias])
        return self._embedders[alias]

    def dense(self, alias: str) -> DenseRetriever:
        if alias not in self._dense:
            index = DenseIndex.load(dense_path(self.settings.index_dir, alias), self.docids)
            self._dense[alias] = DenseRetriever(self.embedder(alias), index)
        return self._dense[alias]

    def lexical(self) -> Retriever:
        if self._lexical is None:
            if self.settings.lexical_backend == "opensearch":
                from minwon_rag.opensearch_backend import OpenSearchRetriever

                self._lexical = OpenSearchRetriever(self.settings.opensearch_url, "minwon")
            elif self.settings.lexical_backend == "kiwi_bm25":
                bm25 = BM25Index.load(self.settings.index_dir / "bm25.pkl")
                self._lexical = BM25Retriever(bm25, self.tokenizer())
            else:
                raise ValueError(f"알 수 없는 LEXICAL_BACKEND: {self.settings.lexical_backend}")
        return self._lexical

    def scorer(self, alias: str) -> PairScorer:
        if alias not in self._scorers:
            self._scorers[alias] = self._scorer_factory(self.cfg.models[alias])
        return self._scorers[alias]

    def build(self, spec: dict[str, Any]) -> Retriever:
        parts: list[Retriever] = []
        if spec.get("lexical"):
            parts.append(self.lexical())
        if spec.get("dense"):
            parts.append(self.dense(spec["dense"]))
        base = (
            parts[0] if len(parts) == 1 else HybridRetriever(parts, self.cfg.rrf_k, self.cfg.depth)
        )
        if spec.get("rerank"):
            return RerankRetriever(
                base, self.scorer(spec["rerank"]), self.doc_texts, self.cfg.rerank_candidates
            )
        return base

    def factories(self) -> dict[str, Callable[[], Retriever]]:
        return {c["name"]: (lambda c=c: self.build(c)) for c in self.cfg.configs}


def build_indexes(
    cfg: MatrixConfig,
    settings: Settings,
    corpus: list[dict[str, Any]],
    text_of: Callable[[dict[str, Any]], str],
    embedder_factory: Callable[[str], Embedder] | None = None,
    tokenizer: Tokenizer | None = None,
    log: Callable[[str], None] = print,
) -> None:
    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    docids = [r["docid"] for r in corpus]
    texts = [text_of(r) for r in corpus]
    (out / "docids.json").write_text(json.dumps(docids), encoding="utf-8")
    make_embedder = embedder_factory or (
        lambda name: SentenceTransformerEmbedder(name, settings.device)
    )
    for alias in cfg.dense_aliases():
        path = dense_path(out, alias)
        if path.exists():
            log(f"[index] {alias}: 기존 색인 사용 ({path})")
            continue
        log(f"[index] {alias}: {len(texts)}건 임베딩 ({cfg.models[alias]})")
        DenseIndex(docids, make_embedder(cfg.models[alias]).encode(texts)).save(path)
    if cfg.uses_lexical():
        if settings.lexical_backend == "opensearch":
            from minwon_rag.opensearch_backend import build_index

            n = build_index(settings.opensearch_url, "minwon", zip(docids, texts, strict=True))
            log(f"[index] OpenSearch {n}건")
        else:
            tok = tokenizer or KiwiTokenizer()
            BM25Index().fit(docids, [tok(t) for t in texts]).save(out / "bm25.pkl")
            log(f"[index] BM25 {len(docids)}건")
