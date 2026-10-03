"""국민권익위 「민원정책 질의응답 조회서비스」 수집기 (data.go.kr 15074671).

명세 출처: data.go.kr 상세 페이지의 요청·출력 항목 표와 기관 참고문서
「[국권위]openAPI 민원정책 질의응답 조회서비스(0).docx」 (2026-10-03 확인).

- 목록 ``PolicyQnaList``: serviceKey, firstIndex(페이지 번호), recordCountPerPage, type(1 민원·2 정책·3 둘 다),
  keyword, searchType, regFrom, regTo → resultCode, resultMessage, resultCount, resultList[
  faqNo, title, ancName, ancCode, regDate(yyyyMMddHHmmss), dutySctnNm]
- 항목 ``PolicyQnaItem``: serviceKey, faqNo, dutySctnNm → resultData{faqNo, dutySctnNm, qnaTitl,
  qstnCntnCl, ansCntnCl, ancName, deptName, regDate(yyyyMMdd), ancCode, deptCode,
  lawList[{fullName, lwrdNm, lwrdUrl}], subjList}
- 응답 형식은 JSON (참고문서 "교환 데이터 표준: JSON").

미확인 (키 발급 후 첫 호출에서 확인할 것)
- 응답을 감싸는 키 ``resultList``/``resultData``/``lawList`` 는 참고문서의 응답 예시에서만 확인했다.
  필드 표에는 이 래퍼 이름이 없다.
- ``firstIndex`` 는 명세상 "페이지 번호"다. 레코드 오프셋일 가능성에 대비해, 새 레코드가 나오지 않는
  페이지에서 멈추고 경고를 남긴다.
- 게이트웨이 오류(인증키 오류·한도 초과)가 JSON이 아닌 XML로 올 수 있다. JSON 파싱에 실패하면
  본문 앞부분을 담아 ``ApiError`` 를 낸다.
- ``subjList`` 원소 구조는 예시가 null 이라 확인하지 못했다(필드 표의 subjectNo·subjectName·subjectCont로 추정).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

from minwon_rag.config import Settings

LIST_OP = "PolicyQnaList"
ITEM_OP = "PolicyQnaItem"
SUCCESS_CODE = "S00"


class ApiError(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    """하루 호출 예산 소진. 다음 날 같은 명령으로 이어 받는다."""


class MinwonClient:
    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
    ):
        if not settings.service_key:
            raise RuntimeError("DATA_GO_KR_SERVICE_KEY가 비어 있습니다 (.env.example 참고)")
        self.settings = settings
        self.http = http or httpx.Client(base_url=settings.api_base, timeout=30)
        self.sleep = sleep
        self.max_retries = max_retries
        self.calls = 0
        self._min_interval = 1.0 / settings.calls_per_second if settings.calls_per_second > 0 else 0
        self._last = 0.0

    def _get(self, op: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.calls >= self.settings.daily_call_budget:
            raise BudgetExhausted(f"호출 예산 {self.settings.daily_call_budget}건 소진")
        query = {"serviceKey": self.settings.service_key, **params}
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                resp = self.http.get(f"/{op}", params=query)
            except httpx.TransportError as e:
                last_err = e
                self.sleep(2**attempt)
                continue
            if resp.status_code >= 500:
                last_err = ApiError(f"HTTP {resp.status_code}")
                self.sleep(2**attempt)
                continue
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                raise ApiError(
                    f"JSON이 아닌 응답 (HTTP {resp.status_code}): {resp.text[:300]}"
                ) from e
            code = str(data.get("resultCode", ""))
            if code != SUCCESS_CODE:
                raise ApiError(f"resultCode={code} resultMessage={data.get('resultMessage')}")
            return data
        raise ApiError(f"{op} 재시도 {self.max_retries}회 실패: {last_err}")

    def list_page(self, page: int, size: int, qna_type: int = 3, **filters: str) -> dict[str, Any]:
        return self._get(
            LIST_OP,
            {"firstIndex": page, "recordCountPerPage": size, "type": qna_type, **filters},
        )

    def item(self, faq_no: str, duty: str) -> dict[str, Any]:
        data = self._get(ITEM_OP, {"faqNo": faq_no, "dutySctnNm": duty})
        item = data.get("resultData")
        if not isinstance(item, dict):
            raise ApiError(f"resultData 없음: faqNo={faq_no}")
        return item


def _read_ids(path: Path, key: str) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {str(json.loads(line)[key]) for line in f if line.strip()}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def collect_list(
    client: MinwonClient, out_path: Path, log: Callable[[str], None] = print, **filters: str
) -> int:
    """목록 전체를 첫 페이지부터 다시 훑어 새 faqNo만 덧붙인다. 새로 추가된 건수를 반환."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _read_ids(out_path, "faqNo")
    size = client.settings.page_size
    added = 0
    page = 1
    total: int | None = None
    prev_ids: set[str] = set()
    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            data = client.list_page(page, size, **filters)
            if total is None and data.get("resultCount") is not None:
                total = int(data["resultCount"])
                log(f"[collect] 전체 {total}건 (resultCount)")
            rows = data.get("resultList") or []
            if not rows:
                break
            ids = {str(r["faqNo"]) for r in rows}
            # 페이지 번호 의미라면 연속 페이지는 (수집 중 신규 등록분 몇 건을 빼면) 겹치지 않는다.
            if len(ids & prev_ids) > len(ids) // 2:
                log(
                    f"[collect] {page}페이지가 앞 페이지와 절반 이상 겹침 — firstIndex 의미 확인 필요, 중단"
                )
                break
            prev_ids = ids
            new = [r for r in rows if str(r["faqNo"]) not in seen]
            for r in new:
                seen.add(str(r["faqNo"]))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += len(new)
            if total is not None and page * size >= total:
                break
            page += 1
    log(f"[collect] 목록 신규 {added}건, 누적 {len(seen)}건")
    return added


def collect_items(
    client: MinwonClient, list_path: Path, items_path: Path, log: Callable[[str], None] = print
) -> int:
    """목록에 있고 아직 받지 않은 항목만 받는다. 예산이 떨어지면 받은 데까지 저장하고 멈춘다."""
    done = _read_ids(items_path, "faqNo")
    todo = [r for r in iter_jsonl(list_path) if str(r["faqNo"]) not in done]
    log(f"[collect] 항목 대상 {len(todo)}건 (이미 받은 {len(done)}건)")
    fetched = 0
    with open(items_path, "a", encoding="utf-8") as f:
        for r in todo:
            try:
                item = client.item(str(r["faqNo"]), str(r["dutySctnNm"]))
            except BudgetExhausted:
                log(f"[collect] 오늘 예산 소진: {fetched}건 저장. 내일 같은 명령으로 이어 받습니다")
                break
            except ApiError as e:
                log(f"[collect] faqNo={r['faqNo']} 실패: {e}")
                continue
            item["faqNo"] = str(item.get("faqNo") or r["faqNo"])
            item["_list"] = r
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            fetched += 1
            if fetched % 500 == 0:
                log(f"[collect] 항목 {fetched}/{len(todo)}")
    return fetched
