"""답변 초안 생성 (LLM 판정 표본용).

검색된 과거 민원 답변을 근거로 새 민원의 답변 초안을 쓰게 한다. 결과는
rag-eval-harness 의 ``rag-eval judge`` 입력 형식(answers.jsonl)으로 저장한다.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

from rag_eval import GoldenSet
from rag_eval.llm import LLMProvider
from rag_eval.matrix import Retriever

SYSTEM = (
    "당신은 국민신문고 민원 답변 초안을 작성하는 공무원 보조자입니다. "
    "주어진 과거 민원 답변에 근거한 내용만 쓰고, 근거가 부족하면 '추가 확인 필요'라고 씁니다."
)

TEMPLATE = """[새 민원]
{question}

[참고할 과거 민원과 답변]
{contexts}

위 과거 사례에 근거해 새 민원에 대한 답변 초안을 작성하세요. 관련 법령과 담당 기관 안내가 근거에 있으면 포함합니다."""


def draft_answers(
    golden: GoldenSet,
    retriever: Retriever,
    doc_texts: Mapping[str, str],
    provider: LLMProvider,
    n: int = 50,
    top_k: int = 5,
    seed: int = 42,
    max_context_chars: int = 1500,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    qids = golden.judged_qids()
    qids = random.Random(seed).sample(qids, min(n, len(qids)))
    rows = []
    for i, qid in enumerate(qids, start=1):
        question = golden.queries[qid]
        hits = retriever.search(question, top_k)
        contexts = [doc_texts[d][:max_context_chars] for d, _ in hits]
        joined = "\n\n".join(f"({j}) {c}" for j, c in enumerate(contexts, start=1))
        answer = provider.complete(SYSTEM, TEMPLATE.format(question=question, contexts=joined))
        rows.append(
            {
                "qid": qid,
                "question": question,
                "answer": answer,
                "reference": golden.meta.get(qid, {}).get("reference", ""),
                "contexts": contexts,
                "context_docids": [d for d, _ in hits],
            }
        )
        if i % 10 == 0:
            log(f"[answers] {i}/{len(qids)}")
    return rows
