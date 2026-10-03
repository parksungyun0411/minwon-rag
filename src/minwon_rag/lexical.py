"""어휘 검색: Kiwi 형태소 토큰 + 프로세스 내 Okapi BM25 (기본 백엔드).

BM25 점수 (Lucene 방식 idf):
    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score    = Σ_{t ∈ 질의(중복 제거)} idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))
문서별 항 가중치를 색인 시점에 희소 행렬로 미리 계산해 두고, 질의 시에는 열을 더하기만 한다.
OpenSearch 백엔드는 ``opensearch_backend`` 에 있다(선택).
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

Tokenizer = Callable[[str], list[str]]

# 명사·외국어·한자·숫자·어근·동사/형용사 어간만 남긴다 (조사·어미·기호 제외).
KIWI_KEEP_TAGS = frozenset({"NNG", "NNP", "NR", "SL", "SH", "SN", "XR", "VV", "VA"})


class KiwiTokenizer:
    def __init__(self, keep_tags: frozenset[str] = KIWI_KEEP_TAGS):
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()
        self.keep_tags = keep_tags

    def __call__(self, text: str) -> list[str]:
        out = []
        for tok in self._kiwi.tokenize(text):
            tag = tok.tag.split("-")[0]  # 'VV-R' 같은 불규칙 표지 제거
            if tag in self.keep_tags:
                out.append(tok.form.lower() if tag == "SL" else tok.form)
        return out


def whitespace_tokenizer(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.docids: list[str] = []
        self.weights: sparse.csc_matrix | None = None

    def fit(self, docids: Sequence[str], tokenized: Sequence[Sequence[str]]) -> BM25Index:
        if len(docids) != len(tokenized):
            raise ValueError("docids와 토큰 목록 길이가 다릅니다")
        rows, cols, vals = [], [], []
        for i, toks in enumerate(tokenized):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            for j, c in counts.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n = len(docids)
        tf = sparse.csr_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(n, len(self.vocab))
        )
        dl = np.asarray([len(t) for t in tokenized], dtype=np.float64)
        avgdl = dl.mean() if n else 0.0
        df = np.bincount(np.asarray(cols, dtype=np.int64), minlength=len(self.vocab))
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * dl / avgdl) if avgdl else np.zeros(n)
        tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.weights = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.docids = list(docids)
        return self

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        assert self.weights is not None, "fit() 먼저"
        ids = sorted({self.vocab[t] for t in query_tokens if t in self.vocab})
        if not ids:
            return np.zeros(len(self.docids))
        return np.asarray(self.weights[:, ids].sum(axis=1)).ravel()

    def top_k(self, query_tokens: Sequence[str], k: int) -> list[tuple[str, float]]:
        s = self.scores(query_tokens)
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(self.docids[i], float(s[i])) for i in idx if s[i] > 0]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> BM25Index:
        with open(path, "rb") as f:
            return pickle.load(f)


class BM25Retriever:
    def __init__(self, index: BM25Index, tokenizer: Tokenizer):
        self.index = index
        self.tokenizer = tokenizer

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.index.top_k(self.tokenizer(query), k)


def idf_value(n: int, df: int) -> float:
    """테스트·설명용: 위 BM25의 idf 식."""
    return math.log1p((n - df + 0.5) / (df + 0.5))
