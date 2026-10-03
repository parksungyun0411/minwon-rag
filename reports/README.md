# reports/

측정 전 — `make eval` 실행 후 자동 생성

`make report` 가 아래 파일을 이 디렉터리에 씁니다. 손으로 숫자를 적지 않습니다.

| 파일 | 내용 | 원자료 |
|---|---|---|
| `data.md` | 수집 건수, 분할 기준일, 코퍼스·평가 질의 수, 라벨 분포 | `results/split_stats.json`, `data/golden/` |
| `retrieval.md` | 설정별 hit@k·recall@k·MRR·nDCG·mAP, Wilson 구간, 기준 설정 대비 McNemar, p95 지연 | `results/per_query.jsonl` |
| `recall_at_k_<골든셋>.png` | 설정별 recall@k 곡선 | 같음 |
