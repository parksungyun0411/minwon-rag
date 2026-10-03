"""``minwon`` 명령행. Makefile 타깃이 이 하위 명령을 차례로 부른다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval import GoldenSet
from rag_eval.types import read_jsonl, write_jsonl

from minwon_rag.config import Settings
from minwon_rag.normalize import doc_text, normalize_item

CONFIG_PATH = Path("configs/matrix.yaml")
RESULTS = Path("results")
REPORTS = Path("reports")

SYNTH_TEMPLATE = """다음은 행정기관이 국민신문고 민원에 단 공식 답변입니다.
이 답변이 해결해 줄 민원을 시민이 직접 쓴 것처럼 하나 작성하세요.
- 시민이 국민신문고 게시판에 쓰는 구어체로, 제목 없이 본문만 씁니다.
- 답변의 문장·법령 이름을 그대로 옮기지 않습니다.
- 민원 본문만 출력합니다.

[공식 답변]
{text}
"""


def _records(s: Settings) -> list[dict]:
    return list(read_jsonl(s.processed_dir / "records.jsonl"))


def _corpus(s: Settings) -> list[dict]:
    return list(read_jsonl(s.processed_dir / "corpus.jsonl"))


def cmd_collect(s: Settings, args: argparse.Namespace) -> None:
    from minwon_rag.collect import MinwonClient, collect_items, collect_list

    client = MinwonClient(s)
    list_path, items_path = s.raw_dir / "list.jsonl", s.raw_dir / "items.jsonl"
    if not args.items_only:
        collect_list(client, list_path)
    collect_items(client, list_path, items_path)
    print(f"[collect] 호출 {client.calls}건")
    cmd_normalize(s, args)


def cmd_normalize(s: Settings, args: argparse.Namespace) -> None:
    items = s.raw_dir / "items.jsonl"
    n = write_jsonl(
        s.processed_dir / "records.jsonl", (normalize_item(i) for i in read_jsonl(items))
    )
    print(f"[normalize] {n}건 → {s.processed_dir / 'records.jsonl'}")


def cmd_split(s: Settings, args: argparse.Namespace) -> None:
    from minwon_rag.split import auto_cutoff, time_split

    records = _records(s)
    cutoff = s.split_cutoff or auto_cutoff(records)
    corpus, golden, stats = time_split(records, cutoff, s.eval_size, s.seed)
    write_jsonl(s.processed_dir / "corpus.jsonl", corpus)
    golden.save(s.golden_dir / "real")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[split] 기준일 {cutoff}: 코퍼스 {stats['corpus']}건, 평가 {len(golden.queries)}건")


def cmd_synth(s: Settings, args: argparse.Namespace) -> None:
    from rag_eval.llm import get_provider
    from rag_eval.synth import generate_synthetic_golden

    from minwon_rag.labels import LawLabeler

    corpus = _corpus(s)
    labeler = LawLabeler(corpus)
    by_id = {r["docid"]: r for r in corpus}
    # 실제 질의와 같은 라벨 규칙이 적용되도록 인용 법령이 있는 문서에서만 출처를 뽑는다.
    pool = [
        {"docid": r["docid"], "text": r["answer"]}
        for r in corpus
        if r["laws"] and len(labeler.label(r)) < labeler.max_relevant
    ]
    gs = generate_synthetic_golden(
        pool, get_provider(), n=args.n, seed=s.seed, template=SYNTH_TEMPLATE, name="synthetic"
    )
    for qid, meta in gs.meta.items():
        src = meta["source_docid"]
        gs.qrels[qid] = {**labeler.label(by_id[src]), src: 2}
    gs.save(s.golden_dir / "synthetic")
    print(f"[synth] {len(gs.queries)}건 → {s.golden_dir / 'synthetic'}")


def _components(s: Settings):
    from minwon_rag.pipeline import Components, MatrixConfig

    cfg = MatrixConfig.load(CONFIG_PATH)
    texts = {r["docid"]: doc_text(r) for r in _corpus(s)}
    return cfg, Components(cfg, s, texts)


def cmd_index(s: Settings, args: argparse.Namespace) -> None:
    from minwon_rag.pipeline import MatrixConfig, build_indexes

    build_indexes(MatrixConfig.load(CONFIG_PATH), s, _corpus(s), doc_text)


def _goldens(s: Settings) -> list[GoldenSet]:
    sets = [GoldenSet.load(s.golden_dir / "real", name="real")]
    if (s.golden_dir / "synthetic" / "queries.jsonl").exists():
        sets.append(GoldenSet.load(s.golden_dir / "synthetic", name="synthetic"))
    return sets


def cmd_eval(s: Settings, args: argparse.Namespace) -> None:
    from rag_eval.matrix import run_matrix

    cfg, comps = _components(s)
    goldens = _goldens(s)
    print(f"[eval] 골든셋: {', '.join(g.name for g in goldens)}")
    run_matrix(comps.factories(), goldens, ks=cfg.ks, depth=cfg.depth, out_dir=RESULTS)


def cmd_answers(s: Settings, args: argparse.Namespace) -> None:
    from rag_eval.llm import get_provider

    from minwon_rag.answer import draft_answers

    cfg, comps = _components(s)
    name = cfg.answer_config or cfg.configs[-1]["name"]
    spec = next(c for c in cfg.configs if c["name"] == name)
    rows = draft_answers(
        GoldenSet.load(s.golden_dir / "real", name="real"),
        comps.build(spec),
        comps.doc_texts,
        get_provider(),
        n=args.n,
        seed=s.seed,
    )
    write_jsonl(RESULTS / "answers.jsonl", rows)
    print(f"[answers] {name}: {len(rows)}건 → {RESULTS / 'answers.jsonl'}")


def cmd_report(s: Settings, args: argparse.Namespace) -> None:
    from rag_eval.report import render_matrix_report

    from minwon_rag.pipeline import MatrixConfig
    from minwon_rag.report import render_data_report

    cfg = MatrixConfig.load(CONFIG_PATH)
    rows = list(read_jsonl(RESULTS / "per_query.jsonl"))
    order = [c["name"] for c in cfg.configs]
    print(render_matrix_report(rows, REPORTS, baseline=cfg.baseline, config_order=order))
    stats_path = RESULTS / "split_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(render_data_report(stats, _goldens(s), REPORTS))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="minwon")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser(
        "collect", help="목록·항목 수집 후 정규화 (하루 예산 초과 시 다음 날 이어서)"
    )
    c.add_argument("--items-only", action="store_true", help="목록은 건너뛰고 항목만 이어 받기")
    c.set_defaults(func=cmd_collect)
    sub.add_parser("normalize").set_defaults(func=cmd_normalize)
    sub.add_parser("split", help="시점 분할 + 자동 라벨 → 실제 질의 골든셋").set_defaults(
        func=cmd_split
    )
    sy = sub.add_parser("synth", help="LLM 합성 질의 골든셋")
    sy.add_argument("--n", type=int, default=100)
    sy.set_defaults(func=cmd_synth)
    sub.add_parser("index").set_defaults(func=cmd_index)
    sub.add_parser("eval").set_defaults(func=cmd_eval)
    a = sub.add_parser("answers", help="답변 초안 표본 생성 (LLM 판정 입력)")
    a.add_argument("--n", type=int, default=50)
    a.set_defaults(func=cmd_answers)
    sub.add_parser("report").set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    args.func(Settings(), args)


if __name__ == "__main__":
    main()
