# Kaggle T4 실행 기록

2026-10-27에 색인(`minwon index`)과 평가(`minwon eval`)를 Kaggle 무료 GPU에서 돌린 절차와 환경이다.
수집(`make collect`)과 시점 분할(`make split`)은 로컬에서 했고, 보고서(`make report`)도 로컬에서 받은 `results/`로 만들었다.
결과 표는 [README](../README.md#평가)와 [reports/retrieval.md](../reports/retrieval.md).

## 환경

| 항목 | 값 |
|---|---|
| 실행 | Kaggle 스크립트 커널, `kaggle kernels push --accelerator NvidiaTeslaT4`, 비공개 |
| GPU | Tesla T4 15GB × 2 중 한 장, 드라이버 580.178.04 |
| 코드 | 이 저장소 커밋 `1ee2d24`, 패키지는 `uv sync --frozen --extra models`(uv.lock 그대로, torch 2.14.1+cu130) |
| 입력 | 로컬 `make split` 결과(`data/processed/corpus.jsonl` 14,858건, `data/golden/real/` 질의 150건)를 비공개 데이터셋으로 올림 |
| 임베딩 | `nlpai-lab/KURE-v1` (`8b418a5`), `BAAI/bge-m3` (`5617a9f`) |
| 리랭커 | `BAAI/bge-reranker-v2-m3` (`953dc6f`) |

모델 리비전은 커널이 Hugging Face Hub에서 받은 시점의 커밋이다. `configs/matrix.yaml`은 리비전을 고정하지 않는다.

```bash
kaggle datasets create -p <corpus·golden 폴더>      # 비공개
kaggle kernels push -p <scripts/kaggle_t4.py 와 kernel-metadata.json 폴더> --accelerator NvidiaTeslaT4
kaggle kernels status <id>                          # RUNNING → COMPLETE
kaggle kernels output <id> -p out                   # 끝난 뒤에만 받을 수 있다
cp -R out/out/results/. results/ && make report
```

## 소요 시간

| 단계 | 시간 |
|---|---:|
| 환경 준비(clone, uv sync, 모델 정보) | 약 2분 |
| `minwon index` (임베딩 2종 × 14,858건, BM25) | 30분 30초 |
| `minwon eval` (7개 설정 × 150건) | 18분 37초 |

## 지연 시간을 읽을 때

`reports/retrieval.md`의 p95 지연은 이 커널 안에서 잰 질의 한 건의 검색 시간이다. 임베딩과 리랭커는 T4 한 장, BM25는 CPU에서 돌았다.
리랭커 설정의 4.3초는 상위 50건을 질의와 짝지어 크로스인코더에 넣는 시간이며, 하드웨어와 배치 크기에 따라 크게 달라진다.

## 기록에 남은 실수

- 이 실행의 `logs/freeze.txt`는 `.venv`가 아니라 Kaggle 기본 파이썬의 패키지 목록을 찍었다. 실제 실행 환경은 uv.lock이고, `.venv` 파이썬으로 찍은 `logs/models.txt`의 torch 버전(2.14.1+cu130)이 uv.lock(2.14.1)과 같은 것으로 확인했다. `scripts/kaggle_t4.py`는 `.venv`를 찍도록 고쳤다.
- 최대 메모리를 남기려던 `/usr/bin/time -v`는 커널 이미지에 없어 기록되지 않았다.
