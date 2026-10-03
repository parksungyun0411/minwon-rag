# 순서: collect → index(분할·라벨 포함) → eval → report
# LLM 키가 있으면 index 뒤에 synth(합성 골든셋), eval 뒤에 answers(답변 초안 표본)
RUN = uv run $(if $(wildcard .env),--env-file .env,)
ANSWERS_N ?= 50
SYNTH_N ?= 100

.PHONY: setup test lint collect split synth index eval answers report

setup:
	uv sync --extra models --extra llm

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

collect:  ## 목록·항목 수집 + 정규화. 하루 예산을 넘으면 멈추고, 다음 날 다시 실행하면 이어 받는다
	$(RUN) minwon collect

split:  ## 시점 분할 + 자동 관련성 라벨 → data/golden/real
	$(RUN) minwon split

synth: split  ## LLM 합성 질의 골든셋 → data/golden/synthetic
	$(RUN) minwon synth --n $(SYNTH_N)

index: split  ## dense 임베딩 + BM25 색인
	$(RUN) minwon index

eval:  ## configs/matrix.yaml 의 모든 설정 × 골든셋 → results/per_query.jsonl
	$(RUN) minwon eval

answers:  ## 답변 초안 표본 → results/answers.jsonl (rag-eval judge 입력)
	$(RUN) minwon answers --n $(ANSWERS_N)

report:  ## results/ → reports/*.md, *.png
	$(RUN) minwon report
