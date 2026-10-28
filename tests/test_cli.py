"""CLI: split → synth(가짜 LLM) → 데이터 보고서. 레코드는 테스트용 합성 값이다."""

import json

import rag_eval.llm
from rag_eval import GoldenSet
from rag_eval.types import write_jsonl

from minwon_rag.cli import main
from minwon_rag.report import render_data_report


def _rec(docid, date, agency, law, article):
    return {
        "docid": docid,
        "duty": "tqapttn",
        "title": f"합성 제목 {docid}",
        "question": f"합성 질의 {docid}",
        "answer": f"합성 답변 {docid}",
        "answer_raw": f"합성 답변 {docid}",
        "agency": agency,
        "agency_code": agency,
        "dept": "",
        "reg_date": date,
        "laws": [{"full_name": f"{law} / {article}", "law_name": law}],
        "subjects": [],
    }


class FakeLLM:
    name = "fake"

    def complete(self, system, prompt, max_tokens=1024):
        return "합성 질의 문장"


def test_split_synth_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MINWON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SPLIT_CUTOFF", "2025-01-01")
    monkeypatch.setenv("EVAL_SIZE", "2")
    records = [
        _rec("c1", "2024-01-01", "A", "가법", "제1조"),
        _rec("c2", "2024-02-01", "A", "가법", "제2조"),
        _rec("c3", "2024-03-01", "B", "나법", "제1조"),
        _rec("q1", "2025-02-01", "A", "가법", "제1조"),
        _rec("q2", "2025-03-01", "B", "나법", "제1조"),
    ]
    write_jsonl(tmp_path / "data" / "processed" / "records.jsonl", records)

    main(["split"])
    stats = json.loads((tmp_path / "results" / "split_stats.json").read_text(encoding="utf-8"))
    assert stats["corpus"] == 3 and stats["eligible"] == 2
    real = GoldenSet.load(tmp_path / "data" / "golden" / "real")
    assert real.qrels["q1"] == {"c1": 2, "c2": 1}

    monkeypatch.setattr(rag_eval.llm, "get_provider", lambda: FakeLLM())
    main(["synth", "--n", "2"])
    syn = GoldenSet.load(tmp_path / "data" / "golden" / "synthetic")
    assert len(syn.queries) == 2
    for qid, meta in syn.meta.items():
        src = meta["source_docid"]
        assert syn.qrels[qid][src] == 2  # 출처 문서는 grade 2, 같은 라벨 규칙도 적용
    assert syn.qrels["syn-c1"] == {"c1": 2, "c2": 1}

    path = render_data_report(stats, [real, syn], tmp_path / "reports")
    text = path.read_text(encoding="utf-8")
    assert "`real`" in text and "`synthetic`" in text
    assert "| `real` | 2 | 2 | 2 | 2 | 3 | 1 |" in text  # 관련 판정 3쌍 중 grade 1은 q1-c2 하나
