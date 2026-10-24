"""Benchmark CLI: quality, replay latency and OCR recovery."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from ..ocr import RapidOcrEngine
from ..pipeline import Pipeline
from ..translate import TranslationCache, TranslationService
from .ocr_check import check_ocr, render_line
from .perf import run_replay
from .quality import score_quality

CORPUS = Path(__file__).parent / "corpus" / "quality.jsonl"
LOCAL_ENGINES = ["nllb", "envit5"]


def load_corpus(path: Path = CORPUS) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build_engine(name: str):
    from ..app import build_local

    return build_local(name)


def translate_all(engine, texts: list[str], batch: int = 8) -> list[str]:
    out: list[str] = []
    for start in range(0, len(texts), batch):
        out.extend(engine.translate(texts[start : start + batch]))
    return out


def quality_rows(engines: dict, corpus: list[dict]) -> list[dict]:
    sources = [row["text"] for row in corpus]
    refs = [row["ref_vi"] for row in corpus]
    rows = []
    for name, engine in engines.items():
        hypotheses = translate_all(engine, sources)
        report = score_quality(list(zip(sources, hypotheses)), refs=refs)
        rows.append({"engine": name, **report})
    return rows


def perf_rows(engines: dict, corpus: list[dict], workdir: Path) -> list[dict]:
    frames = [render_line(row["text"], workdir / f"frame_{i}.png") for i, row in enumerate(corpus[:5])]
    rows = []
    for name, engine in engines.items():
        pipeline = Pipeline(
            None,  # replay calls ocr.read/service.translate directly
            RapidOcrEngine(),
            TranslationService(engine, TranslationCache()),
            lambda event: None,
        )
        result = run_replay(frames, pipeline, repeats=2)
        rows.append({"engine": name, **result})
    return rows


def write_report(path: Path, quality: list[dict], perf: list[dict], ocr: dict | None) -> None:
    lines = ["# Benchmark report", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if quality:
        lines += [
            "## Translation quality",
            "",
            "| engine | chrF++ | numbers | proper nouns | length ratio | empty |",
            "|---|---|---|---|---|---|",
        ]
        for row in quality:
            lines.append(
                f"| {row['engine']} | {row['chrf_pp']} | {row['number_preserved']:.2f} "
                f"| {row['proper_noun_preserved']:.2f} | {row['length_ratio']:.2f} | {row['empty_count']} |"
            )
        lines.append("")
    if perf:
        lines += [
            "## Replay latency (ms)",
            "",
            "| engine | OCR p50 | MT p50 | e2e p50 | e2e p95 | e2e p99 | ocr calls |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in perf:
            lines.append(
                f"| {row['engine']} | {row['ocr']['p50']} | {row['mt']['p50']} "
                f"| {row['e2e']['p50']} | {row['e2e']['p95']} | {row['e2e']['p99']} | {row['ocr_calls']} |"
            )
        lines.append("")
    if ocr:
        lines += [
            "## OCR recovery",
            "",
            f"mean ratio {ocr['mean_ratio']} over {ocr['n']} rendered lines; exact {ocr['exact_rate']}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--perf", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--engine", default="all")
    parser.add_argument("--out", default="bench/report.md")
    args = parser.parse_args()

    run_all = not (args.quality or args.perf or args.ocr)
    names = LOCAL_ENGINES if args.engine == "all" else [args.engine]
    engines = {}
    for name in names:
        if name not in LOCAL_ENGINES:
            print(f"skip {name}: benchmark builds local engines only")
            continue
        engines[name] = build_engine(name)

    corpus = load_corpus()
    quality = quality_rows(engines, corpus) if (run_all or args.quality) else []
    with tempfile.TemporaryDirectory() as tmp:
        perf = perf_rows(engines, corpus, Path(tmp)) if (run_all or args.perf) else []
    ocr = None
    if run_all or args.ocr:
        ocr = check_ocr(RapidOcrEngine(), [row["text"] for row in corpus[:30]])

    write_report(Path(args.out), quality, perf, ocr)
    print(f"report written to {args.out}")
    for row in quality:
        print(row)
    for row in perf:
        print(row)
    if ocr:
        print(ocr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
