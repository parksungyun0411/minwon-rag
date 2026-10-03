"""환경 변수 설정. 키는 코드·저장소에 두지 않고 ``.env`` (커밋 제외)에서 읽는다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    service_key: str = field(default_factory=lambda: _env("DATA_GO_KR_SERVICE_KEY"))
    api_base: str = field(
        default_factory=lambda: _env(
            "MINWON_API_BASE", "https://apis.data.go.kr/1140100/CivilPolicyQnaService"
        )
    )
    # 개발계정 일 10,000건. 목록·항목 호출을 합친 하루 예산.
    daily_call_budget: int = field(
        default_factory=lambda: int(_env("MINWON_DAILY_CALL_BUDGET", "9500"))
    )
    page_size: int = field(default_factory=lambda: int(_env("MINWON_PAGE_SIZE", "100")))
    # 기관 문서상 최대 30 tps. 여유를 두고 초당 호출 수를 제한한다.
    calls_per_second: float = field(
        default_factory=lambda: float(_env("MINWON_CALLS_PER_SECOND", "5"))
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("MINWON_DATA_DIR", "data")))
    # 비우면 신청일 90 분위수를 기준일로 쓴다.
    split_cutoff: str = field(default_factory=lambda: _env("SPLIT_CUTOFF"))
    eval_size: int = field(default_factory=lambda: int(_env("EVAL_SIZE", "150")))
    seed: int = field(default_factory=lambda: int(_env("SEED", "42")))
    lexical_backend: str = field(default_factory=lambda: _env("LEXICAL_BACKEND", "kiwi_bm25"))
    opensearch_url: str = field(
        default_factory=lambda: _env("OPENSEARCH_URL", "http://localhost:9200")
    )
    device: str = field(default_factory=lambda: _env("MODEL_DEVICE"))  # 비우면 자동(mps/cuda/cpu)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"
