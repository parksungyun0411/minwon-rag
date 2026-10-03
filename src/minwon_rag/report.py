"""데이터 규모 보고서 (분할 통계·라벨 분포). 검색 지표 보고서는 rag-eval-harness 가 만든다."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_eval import GoldenSet


def render_data_report(
    stats: dict[str, Any], goldens: Sequence[GoldenSet], out_dir: str | Path
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        ("정규화 레코드", stats["records"]),
        ("분할 기준일", stats["cutoff"]),
        ("코퍼스 (기준일 이전, 답변 있음)", stats["corpus"]),
        ("기준일 이후 민원", stats["after_cutoff"]),
        ("└ 인용 법령 있음", stats["after_cutoff_with_laws"]),
        ("└ 자동 라벨 가능 (관련 1~상한 건)", stats["eligible"]),
    ]
    lines = ["# 데이터 규모", "", "| 항목 | 값 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "## 골든셋",
        "",
        "| 골든셋 | 질의 | 관련 문서 중앙값 | 관련 문서 최대 | grade 2 포함 질의 |",
    ]
    lines += ["|---|---|---|---|---|"]
    for g in goldens:
        sizes = sorted(len([d for d, v in g.qrels[q].items() if v > 0]) for q in g.judged_qids())
        strong = sum(1 for q in g.judged_qids() if 2 in g.qrels[q].values())
        if sizes:
            lines.append(
                f"| `{g.name}` | {len(sizes)} | {sizes[len(sizes) // 2]} | {sizes[-1]} | {strong} |"
            )
    path = out / "data.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
