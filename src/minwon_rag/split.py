"""시점 분할: 기준일 전 민원 = 검색 코퍼스, 기준일 당일부터의 민원 = 평가 질의 후보.

평가 질의는 기준일 당일부터의 민원 중 자동 라벨이 붙는(관련 문서 1건 이상, 상한 이하) 것에서
시드 고정으로 ``eval_size`` 건을 뽑는다. 질의가 코퍼스보다 늦게 등록됐으므로
"과거 사례로 새 민원에 답한다"는 실제 사용 순서를 그대로 따른다.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from rag_eval import GoldenSet

from minwon_rag.labels import LawLabeler, label_stats
from minwon_rag.normalize import query_text


def auto_cutoff(records: Sequence[dict[str, Any]], quantile: float = 0.9) -> str:
    dates = sorted(r["reg_date"] for r in records if r["reg_date"])
    if not dates:
        raise ValueError("신청일이 있는 레코드가 없습니다")
    return dates[min(len(dates) - 1, int(len(dates) * quantile))]


def time_split(
    records: Sequence[dict[str, Any]],
    cutoff: str,
    eval_size: int,
    seed: int = 42,
    max_relevant: int = 100,
) -> tuple[list[dict[str, Any]], GoldenSet, dict[str, Any]]:
    """(코퍼스, 실제 질의 골든셋, 분할 통계)."""
    usable = [r for r in records if r["reg_date"] and (r["question"] or r["title"])]
    corpus = [r for r in usable if r["reg_date"] < cutoff and r["answer"]]
    later = [r for r in usable if r["reg_date"] >= cutoff]
    labeler = LawLabeler(corpus, max_relevant=max_relevant)
    labelled = []
    for r in later:
        rels = labeler.label(r)
        if labeler.eligible(rels):
            labelled.append((r, rels))
    stats: dict[str, Any] = {
        "cutoff": cutoff,
        "records": len(records),
        "corpus": len(corpus),
        "after_cutoff": len(later),
        "after_cutoff_with_laws": sum(1 for r in later if r["laws"]),
        "eligible": len(labelled),
    }
    if len(labelled) < eval_size:
        raise ValueError(
            f"라벨이 붙는 평가 후보가 {len(labelled)}건뿐입니다 (필요 {eval_size}). "
            "SPLIT_CUTOFF를 앞당기거나 EVAL_SIZE를 줄이세요."
        )
    picked = random.Random(seed).sample(labelled, eval_size)
    gs = GoldenSet(
        name="real",
        queries={r["docid"]: query_text(r) for r, _ in picked},
        qrels={r["docid"]: rels for r, rels in picked},
        meta={
            r["docid"]: {"reg_date": r["reg_date"], "agency": r["agency"], "reference": r["answer"]}
            for r, _ in picked
        },
    )
    stats["labels"] = label_stats(gs.qrels)
    return corpus, gs, stats
