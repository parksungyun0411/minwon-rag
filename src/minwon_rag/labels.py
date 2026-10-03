"""관련성 라벨 자동 생성.

평가 질의(기준일 이후 민원)마다 코퍼스(기준일 이전 민원)에서 관련 문서를 규칙으로 고른다.
기관 답변에 붙은 인용 법령(lawList)을 근거로 쓴다.

- grade 2 (강): 인용 법령의 조문까지 같은 항목(fullName)을 하나 이상 공유
- grade 1 (약): 법령명(lwrdNm)을 공유하고 처리기관 코드가 같음
- 처리기관만 같은 경우는 라벨로 쓰지 않는다. 한 기관 아래 민원이 수천 건이라 recall이 의미를 잃는다.

관련 문서가 ``max_relevant`` 를 넘는 질의는 평가에서 뺀다(흔한 법령 하나로 수백 건이 관련이 되는 경우).
라벨은 "같은 법 조항을 근거로 답한 과거 민원"이라는 대리 지표이며, 사람이 판정한 주제 관련성과 같지 않다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any


class LawLabeler:
    def __init__(self, corpus: Iterable[dict[str, Any]], max_relevant: int = 100):
        self.max_relevant = max_relevant
        self.by_article: dict[str, set[str]] = defaultdict(set)
        self.by_law_agency: dict[tuple[str, str], set[str]] = defaultdict(set)
        for rec in corpus:
            for law in rec["laws"]:
                if law["full_name"]:
                    self.by_article[law["full_name"]].add(rec["docid"])
                if law["law_name"] and rec["agency_code"]:
                    self.by_law_agency[(law["law_name"], rec["agency_code"])].add(rec["docid"])

    def label(self, rec: dict[str, Any]) -> dict[str, int]:
        rels: dict[str, int] = {}
        for law in rec["laws"]:
            if law["law_name"] and rec["agency_code"]:
                for d in self.by_law_agency.get((law["law_name"], rec["agency_code"]), ()):
                    rels.setdefault(d, 1)
        for law in rec["laws"]:
            if law["full_name"]:
                for d in self.by_article.get(law["full_name"], ()):
                    rels[d] = 2
        rels.pop(rec["docid"], None)
        return rels

    def eligible(self, rels: dict[str, int]) -> bool:
        return 0 < len(rels) <= self.max_relevant


def label_stats(qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    sizes: Sequence[int] = sorted(len(r) for r in qrels.values())
    if not sizes:
        return {"queries": 0}
    return {
        "queries": len(sizes),
        "relevant_min": sizes[0],
        "relevant_median": sizes[len(sizes) // 2],
        "relevant_max": sizes[-1],
        "with_strong": sum(1 for r in qrels.values() if any(g == 2 for g in r.values())),
    }
