"""선택 백엔드: OpenSearch + nori 형태소 분석기 BM25.

이 모듈은 테스트되지 않았다. 개발 환경에 Docker가 없어 ``docker/`` 의 compose 파일로 띄운
OpenSearch에 실제로 붙여 본 적이 없다. ``LEXICAL_BACKEND=opensearch`` 로 켠다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

INDEX_BODY = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {
            "analyzer": {
                "ko": {"type": "custom", "tokenizer": "nori_tokenizer", "filter": ["lowercase"]}
            }
        },
    },
    "mappings": {"properties": {"text": {"type": "text", "analyzer": "ko"}}},
}


def _client(url: str):
    from opensearchpy import OpenSearch

    return OpenSearch(hosts=[url], timeout=60)


def build_index(url: str, index: str, docs: Iterable[tuple[str, str]]) -> int:
    from opensearchpy import helpers

    client = _client(url)
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    client.indices.create(index=index, body=INDEX_BODY)
    actions = ({"_index": index, "_id": d, "_source": {"text": t}} for d, t in docs)
    ok, _ = helpers.bulk(client, actions, chunk_size=500)
    client.indices.refresh(index=index)
    return ok


class OpenSearchRetriever:
    def __init__(self, url: str, index: str):
        self.client = _client(url)
        self.index = index

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        body: dict[str, Any] = {"size": k, "query": {"match": {"text": query}}, "_source": False}
        res = self.client.search(index=self.index, body=body)
        return [(h["_id"], float(h["_score"])) for h in res["hits"]["hits"]]
