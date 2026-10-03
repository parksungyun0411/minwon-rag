"""원본 항목 → 정규화 레코드.

답변 본문(ansCntnCl)은 참고문서 예시에서 ``&lt;br /&gt;`` 처럼 HTML이 한 번 더 이스케이프된 채로 온다.
이스케이프를 풀고 태그를 줄바꿈·공백으로 바꾼다. 기관 답변 첫머리·끝의 정형 인사말은
검색 신호가 아니라 잡음이므로 규칙 목록에 맞는 문장만 지운다(본문 판단 문장은 건드리지 않는다).
"""

from __future__ import annotations

import html
import re
from typing import Any

_BR = re.compile(r"<\s*br\s*/?\s*>", re.I)
_BLOCK = re.compile(r"</?\s*(p|div|li|tr)[^>]*>", re.I)
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t 　]+")
_NEWLINES = re.compile(r"\n{3,}")

_GREETINGS = [
    re.compile(r"^안녕하(십니까|세요)[.?!]?\s*"),
    re.compile(r"^(항상\s*)?[^.\n]{0,40}(관심|협조)[^.\n]{0,20}감사(드립니다|합니다)[.!]?\s*"),
]
_CLOSINGS = [
    re.compile(r"\s*[^.\n]{0,60}(건강|행복)[^.\n]{0,30}(기원|바랍)[^.\n]*[.!]?\s*$"),
    re.compile(r"\s*감사합니다[.!]?\s*$"),
]


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    prev = None
    s = text
    while prev != s:  # 이중 이스케이프를 끝까지 푼다
        prev, s = s, html.unescape(s)
    s = _BR.sub("\n", s)
    s = _BLOCK.sub("\n", s)
    s = _TAG.sub(" ", s)
    s = _SPACES.sub(" ", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    return _NEWLINES.sub("\n\n", s).strip()


def strip_boilerplate(answer: str) -> str:
    s = answer
    for pat in _GREETINGS:
        s = pat.sub("", s, count=1).lstrip()
    for _ in range(3):
        before = s
        for pat in _CLOSINGS:
            s = pat.sub("", s).rstrip()
        if s == before:
            break
    return s or answer


def _date(raw: Any) -> str:
    digits = re.sub(r"\D", "", str(raw or ""))
    if len(digits) < 8:
        return ""
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    listed = item.get("_list") or {}
    laws = []
    for law in item.get("lawList") or []:
        full = clean_text(law.get("fullName"))
        name = clean_text(law.get("lwrdNm")) or full.split("/")[0].strip()
        if full or name:
            laws.append({"full_name": full, "law_name": name})
    subjects = [
        clean_text(s.get("subjectName")) for s in item.get("subjList") or [] if isinstance(s, dict)
    ]
    answer = clean_text(item.get("ansCntnCl"))
    return {
        "docid": str(item["faqNo"]),
        "duty": item.get("dutySctnNm") or listed.get("dutySctnNm", ""),
        "title": clean_text(item.get("qnaTitl") or listed.get("title")),
        "question": clean_text(item.get("qstnCntnCl")),
        "answer": strip_boilerplate(answer),
        "answer_raw": answer,
        "agency": item.get("ancName") or listed.get("ancName", ""),
        "agency_code": str(item.get("ancCode") or listed.get("ancCode") or ""),
        "dept": item.get("deptName") or "",
        "reg_date": _date(item.get("regDate") or listed.get("regDate")),
        "laws": laws,
        "subjects": [s for s in subjects if s],
    }


def query_text(rec: dict[str, Any]) -> str:
    """시민 질의: 제목 + 질의 본문."""
    return "\n".join(x for x in (rec["title"], rec["question"]) if x)


def doc_text(rec: dict[str, Any]) -> str:
    """검색 대상 문서: 과거 민원의 제목·질의·답변 전체."""
    return "\n".join(x for x in (rec["title"], rec["question"], rec["answer"]) if x)
