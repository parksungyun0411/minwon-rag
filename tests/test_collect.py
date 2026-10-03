"""수집기: 합성 응답(tests/fixtures)을 MockTransport로 돌려준다. 실제 API를 부르지 않는다."""

import json
from pathlib import Path

import httpx
import pytest

from minwon_rag.collect import (
    ApiError,
    MinwonClient,
    collect_items,
    collect_list,
    iter_jsonl,
)
from minwon_rag.config import Settings

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _client(handler, **overrides) -> MinwonClient:
    s = Settings(
        service_key="TEST",
        api_base="https://example.invalid/svc",
        calls_per_second=0,
        page_size=2,
        **overrides,
    )
    http = httpx.Client(base_url=s.api_base, transport=httpx.MockTransport(handler))
    return MinwonClient(s, http=http, sleep=lambda _: None)


def api_handler(request: httpx.Request) -> httpx.Response:
    p = dict(request.url.params)
    assert p["serviceKey"] == "TEST"
    if request.url.path.endswith("/PolicyQnaList"):
        assert p["type"] == "3" and p["recordCountPerPage"] == "2"
        page = {"1": "list_page1.json", "2": "list_page2.json"}.get(p["firstIndex"])
        if page is None:
            return httpx.Response(200, json={"resultCode": "S00", "resultList": []})
        return httpx.Response(200, json=_load(page))
    if request.url.path.endswith("/PolicyQnaItem"):
        assert p["dutySctnNm"] in ("tqapttn", "tqaplcy")
        if p["faqNo"] == "1001":
            return httpx.Response(200, json=_load("item_1001.json"))
        if p["faqNo"] == "1002":
            return httpx.Response(200, json={"resultCode": "E01", "resultMessage": "가상 오류"})
        data = _load("item_1001.json")
        data["resultData"] = {**data["resultData"], "faqNo": p["faqNo"]}
        return httpx.Response(200, json=data)
    return httpx.Response(404)


def test_collect_list_and_items(tmp_path):
    client = _client(api_handler)
    list_path, items_path = tmp_path / "list.jsonl", tmp_path / "items.jsonl"
    assert collect_list(client, list_path, log=lambda _: None) == 3
    # 다시 돌려도 중복 없이 0건 추가
    assert collect_list(client, list_path, log=lambda _: None) == 0
    assert [r["faqNo"] for r in iter_jsonl(list_path)] == ["1001", "1002", "1003"]

    fetched = collect_items(client, list_path, items_path, log=lambda _: None)
    assert fetched == 2  # 1002는 resultCode 오류로 건너뜀
    items = list(iter_jsonl(items_path))
    assert {i["faqNo"] for i in items} == {"1001", "1003"}
    assert items[0]["_list"]["ancName"] == "가상기관A"
    # 이어 받기: 이미 받은 건은 다시 부르지 않음
    assert collect_items(client, list_path, items_path, log=lambda _: None) == 0


def test_budget_stops_and_resumes(tmp_path):
    list_path, items_path = tmp_path / "list.jsonl", tmp_path / "items.jsonl"
    collect_list(_client(api_handler), list_path, log=lambda _: None)
    small = _client(api_handler, daily_call_budget=1)
    assert collect_items(small, list_path, items_path, log=lambda _: None) == 1
    assert collect_items(_client(api_handler), list_path, items_path, log=lambda _: None) == 1


def test_non_json_error_is_reported():
    def handler(request):
        return httpx.Response(200, text=(FIX / "error_xml.txt").read_text(encoding="utf-8"))

    with pytest.raises(ApiError, match="JSON이 아닌 응답"):
        _client(handler).list_page(1, 10)


def test_retry_on_5xx_then_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=_load("list_page2.json"))

    data = _client(handler).list_page(1, 10)
    assert calls["n"] == 3 and data["resultList"][0]["faqNo"] == "1003"


def test_overlapping_pages_stop(tmp_path):
    """firstIndex가 오프셋 의미라 다음 페이지가 거의 겹치면 멈춘다."""

    def handler(request):
        return httpx.Response(200, json={**_load("list_page1.json"), "resultCount": "100"})

    added = collect_list(_client(handler), tmp_path / "l.jsonl", log=lambda _: None)
    assert added == 2


def test_missing_key():
    with pytest.raises(RuntimeError):
        MinwonClient(Settings(service_key=""))
