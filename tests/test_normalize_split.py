"""정규화·라벨·시점 분할. 레코드는 모두 테스트용 합성 값이다."""

import json
from pathlib import Path

import pytest

from minwon_rag.labels import LawLabeler
from minwon_rag.normalize import clean_text, normalize_item, strip_boilerplate
from minwon_rag.split import auto_cutoff, time_split

FIX = Path(__file__).parent / "fixtures"


def test_clean_text_double_escaped():
    raw = "첫 줄&lt;br /&gt;\n&lt;br /&gt;\n둘째&nbsp;줄 &quot;인용&quot; &lt;b&gt;굵게&lt;/b&gt;"
    assert clean_text(raw) == '첫 줄\n\n둘째 줄 "인용" 굵게'
    assert clean_text(None) == ""


def test_strip_boilerplate_keeps_body():
    text = (
        "안녕하십니까? 항상 가상행정에 대한 관심과 협조에 감사드립니다.\n"
        "본문 판단입니다.\n민원인께 항상 건강과 행복이 함께 하시길 기원합니다. 감사합니다."
    )
    assert strip_boilerplate(text) == "본문 판단입니다."
    assert strip_boilerplate("감사합니다.") == "감사합니다."  # 전부 지워지면 원문 유지


def test_normalize_item_fixture():
    item = json.loads((FIX / "item_1001.json").read_text(encoding="utf-8"))["resultData"]
    rec = normalize_item(item)
    assert rec["docid"] == "1001"
    assert rec["reg_date"] == "2025-01-05"
    assert rec["laws"] == [{"full_name": "가상법 / 제1조(목적)", "law_name": "가상법"}]
    assert rec["answer"] == "가상 답변 본문입니다. 온라인으로 신청할 수 있습니다."
    assert rec["subjects"] == []


def _rec(docid, date, agency, laws, answer="답"):
    return {
        "docid": docid,
        "reg_date": date,
        "agency": agency,
        "agency_code": agency,
        "title": f"제목{docid}",
        "question": f"질의{docid}",
        "answer": answer,
        "laws": [{"full_name": f"{n} / {a}" if a else "", "law_name": n} for n, a in laws],
    }


CORPUS = [
    _rec("c1", "2024-01-01", "A", [("가법", "제1조")]),
    _rec("c2", "2024-02-01", "A", [("가법", "제2조")]),
    _rec("c3", "2024-03-01", "B", [("가법", "제1조")]),
    _rec("c4", "2024-04-01", "B", [("나법", "제9조")]),
]


def test_law_labeler_grades():
    lab = LawLabeler(CORPUS)
    q = _rec("q1", "2025-01-01", "A", [("가법", "제1조")])
    # 같은 조문 → 2 (c1, c3), 같은 법·같은 기관 → 1 (c2), 다른 법 → 없음
    assert lab.label(q) == {"c1": 2, "c3": 2, "c2": 1}
    assert lab.label(_rec("q2", "2025-01-01", "C", [("다법", "제1조")])) == {}
    assert LawLabeler(CORPUS, max_relevant=2).eligible(lab.label(q)) is False


def test_time_split():
    later = [
        _rec("q1", "2025-01-01", "A", [("가법", "제1조")]),
        _rec("q2", "2025-01-02", "B", [("나법", "제9조")]),
        _rec("q3", "2025-01-03", "C", [("다법", "제1조")]),  # 라벨 없음 → 제외
    ]
    corpus, gs, stats = time_split(CORPUS + later, "2025-01-01", eval_size=2, seed=0)
    assert [r["docid"] for r in corpus] == ["c1", "c2", "c3", "c4"]
    assert set(gs.queries) == {"q1", "q2"}
    assert gs.qrels["q2"] == {"c4": 2}
    assert gs.meta["q1"]["reference"] == "답"
    assert stats["eligible"] == 2 and stats["after_cutoff"] == 3
    with pytest.raises(ValueError, match="평가 후보"):
        time_split(CORPUS + later, "2025-01-01", eval_size=3)


def test_auto_cutoff():
    recs = [_rec(str(i), f"2024-01-{i:02d}", "A", []) for i in range(1, 11)]
    assert auto_cutoff(recs, 0.9) == "2024-01-10"
