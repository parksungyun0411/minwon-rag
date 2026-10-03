"""BM25·RRF·리랭크 손계산 검증과, 가짜 임베딩으로 색인 → 매트릭스 실행까지 연결 검증."""

import math

import numpy as np
import pytest
from rag_eval import GoldenSet
from rag_eval.matrix import run_matrix

from minwon_rag.config import Settings
from minwon_rag.dense import DenseIndex
from minwon_rag.fusion import HybridRetriever, RerankRetriever, rrf
from minwon_rag.lexical import BM25Index, idf_value, whitespace_tokenizer
from minwon_rag.pipeline import Components, MatrixConfig, build_indexes


def test_bm25_hand_computed():
    # 문서 3개, 평균 길이 (2+3+1)/3 = 2
    docs = [["a", "b"], ["a", "a", "c"], ["c"]]
    idx = BM25Index(k1=1.2, b=0.75).fit(["d0", "d1", "d2"], docs)
    s = idx.scores(["a"])
    idf_a = math.log1p((3 - 2 + 0.5) / (2 + 0.5))  # df(a)=2
    # d0: tf=1, dl=2 → 1·2.2 / (1 + 1.2·(0.25 + 0.75·1))
    assert s[0] == pytest.approx(idf_a * 2.2 / (1 + 1.2))
    # d1: tf=2, dl=3 → 2·2.2 / (2 + 1.2·(0.25 + 0.75·1.5))
    assert s[1] == pytest.approx(idf_a * 4.4 / (2 + 1.2 * 1.375))
    assert s[2] == 0.0
    assert idf_value(3, 2) == pytest.approx(idf_a)
    # 질의 중복 토큰은 한 번만, 없는 토큰은 무시
    assert np.allclose(idx.scores(["a", "a", "zzz"]), s)
    # d1: 4.4/3.65 > d0: 2.2/2.2 → d1이 1위, 점수 0인 d2는 빠짐
    assert idx.top_k(["a"], 5) == [("d1", s[1]), ("d0", s[0])]
    assert [d for d, _ in idx.top_k(["c"], 1)] == ["d2"]


def test_bm25_save_load(tmp_path):
    idx = BM25Index().fit(["x", "y"], [["가", "나"], ["나"]])
    idx.save(tmp_path / "bm.pkl")
    again = BM25Index.load(tmp_path / "bm.pkl")
    assert np.allclose(again.scores(["가"]), idx.scores(["가"]))


def test_rrf_hand_computed():
    fused = dict(rrf([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["d"] == pytest.approx(1 / 62)
    assert rrf([["a", "b", "c"], ["b", "d"]], k=60)[0][0] == "b"


class ListRetriever:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query, k):
        return [(d, 1.0 / (i + 1)) for i, d in enumerate(self.docs[:k])]


class LengthScorer:
    """텍스트 길이를 점수로 쓰는 가짜 크로스인코더."""

    def score(self, query, texts):
        return [float(len(t)) for t in texts]


def test_hybrid_and_rerank():
    h = HybridRetriever([ListRetriever(["a", "b", "c"]), ListRetriever(["b", "d"])], rrf_k=60)
    assert [d for d, _ in h.search("q", 2)] == ["b", "a"]
    texts = {"a": "x", "b": "xxx", "c": "xx", "d": "xxxx"}
    rr = RerankRetriever(ListRetriever(["a", "b", "c", "d"]), LengthScorer(), texts, candidates=3)
    out = rr.search("q", 4)
    # 상위 3건만 길이순 재정렬, d는 뒤에 남는다
    assert [d for d, _ in out] == ["b", "c", "a", "d"]
    assert out[3][1] < out[2][1]


class HashEmbedder:
    """글자 해시 bag-of-chars 가짜 임베딩 (dim 64, L2 정규화)."""

    def __init__(self, name: str = "stub"):
        self.name = name

    def encode(self, texts, batch_size=32):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.replace(" ", ""):
                out[i, hash(ch) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def test_dense_index_top_k():
    vecs = np.eye(3, dtype=np.float32)
    idx = DenseIndex(["a", "b", "c"], vecs)
    assert [d for d, _ in idx.top_k(np.array([0.1, 0.9, 0.0]), 2)] == ["b", "a"]


def test_pipeline_end_to_end(tmp_path):
    cfg = MatrixConfig(
        configs=[
            {"name": "bm25", "lexical": True},
            {"name": "dense", "dense": "stub"},
            {"name": "hybrid-rerank", "dense": "stub", "lexical": True, "rerank": "rr"},
        ],
        models={"stub": "stub-model", "rr": "stub-reranker"},
        ks=[1, 2],
        depth=3,
        rerank_candidates=2,
    )
    settings = Settings(data_dir=tmp_path, lexical_backend="kiwi_bm25")
    corpus = [
        {"docid": "c1", "text": "주민등록 등본 발급 온라인"},
        {"docid": "c2", "text": "자동차 등록 이전 서류"},
        {"docid": "c3", "text": "세금 신고 기한 연장"},
    ]
    build_indexes(
        cfg,
        settings,
        corpus,
        lambda r: r["text"],
        embedder_factory=HashEmbedder,
        tokenizer=whitespace_tokenizer,
        log=lambda _: None,
    )
    assert (tmp_path / "index" / "dense__stub.npy").exists()
    comps = Components(
        cfg,
        settings,
        {r["docid"]: r["text"] for r in corpus},
        embedder_factory=HashEmbedder,
        scorer_factory=lambda _: LengthScorer(),
        tokenizer=whitespace_tokenizer,
    )
    gs = GoldenSet(
        name="real",
        queries={"q1": "등본 발급", "q2": "세금 신고"},
        qrels={"q1": {"c1": 2}, "q2": {"c3": 1}},
    )
    rows = run_matrix(comps.factories(), [gs], ks=cfg.ks, depth=cfg.depth, log=lambda _: None)
    by = {(r["config"], r["qid"]): r for r in rows}
    assert by[("bm25", "q1")]["hit@1"] == 1.0
    assert by[("bm25", "q2")]["hit@1"] == 1.0
    assert len(rows) == 6


def test_matrix_config_validation(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("models: {}\nconfigs:\n  - name: x\n    dense: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="모델 별칭"):
        MatrixConfig.load(p)


def test_repo_matrix_config_loads():
    cfg = MatrixConfig.load("configs/matrix.yaml")
    assert cfg.baseline in {c["name"] for c in cfg.configs}
    assert cfg.answer_config in {c["name"] for c in cfg.configs}
    assert cfg.dense_aliases() == ["bge-m3", "kure"]


def test_kiwi_tokenizer():
    pytest.importorskip("kiwipiepy")
    from minwon_rag.lexical import KiwiTokenizer

    toks = KiwiTokenizer()("주민등록등본을 인터넷으로 발급받으려면")
    assert "인터넷" in toks and "발급" in toks
    assert "을" not in toks and "으로" not in toks


def test_settings_paths():
    s = Settings(data_dir=__import__("pathlib").Path("d"))
    assert str(s.index_dir) == "d/index" and str(s.golden_dir) == "d/golden"
