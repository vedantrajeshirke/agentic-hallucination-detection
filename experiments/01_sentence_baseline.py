"""Experiment 01: sentence-level baseline verifier on RAGTruth.

Every response sentence is one claim, checked by DeBERTa NLI against the example's evidence
passages. Two subcommands:

  run       score sentences, appending one JSON line per response (resumable: finished
            response ids are skipped on restart, so Colab disconnects lose little)
  evaluate  tune the entailment threshold on a dev scores file, apply it to a report file,
            write metrics JSON + a markdown table

Examples:
  # local smoke run: 10 dev responses per task, base model
  uv run python experiments/01_sentence_baseline.py run --split dev --per-task 10 --model base
  # full test set on Colab
  uv run python experiments/01_sentence_baseline.py run --split test --model large
  uv run python experiments/01_sentence_baseline.py evaluate \
      --tune results/01_sentence_baseline/scores_dev_large_per_passage.jsonl \
      --report results/01_sentence_baseline/scores_test_large_per_passage.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.claims.decompose import split_sentences
from src.data.ragtruth import COMMIT, TASKS, evidence_passages, load, make_dev_split

OUT_DIR = REPO_ROOT / "results" / "01_sentence_baseline"
SEED = 0


def _git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        return out + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def select_examples(split: str, per_task: int | None, seed: int = SEED):
    if split == "test":
        pool = load("test")
    else:
        train, dev = make_dev_split(load("train"), dev_frac=0.1, seed=seed)
        pool = dev if split == "dev" else train
    if per_task is None:
        return pool
    rng = random.Random(seed)
    out = []
    for task in TASKS:
        group = [e for e in pool if e.task_type == task]
        out += rng.sample(group, min(per_task, len(group)))
    return out


def cmd_run(args) -> None:
    from src.verify.nli import NLIModel
    from src.verify.scorer import Verifier

    examples = select_examples(args.split, args.per_task)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}{'' if args.per_task is None else f'_n{args.per_task}'}_{args.model}_{args.mode}"
    scores_path = out_dir / f"scores_{tag}.jsonl"
    done = set()
    if scores_path.exists():
        with scores_path.open() as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [e for e in examples if e.id not in done]
    print(f"{len(examples)} examples, {len(done)} already scored, {len(todo)} to go -> {scores_path}")
    if not todo:
        return

    verifier = Verifier(NLIModel(args.model, batch_size=args.batch_size, fp16=args.fp16), mode=args.mode)
    config = {
        "experiment": "01_sentence_baseline",
        "split": args.split,
        "per_task": args.per_task,
        "seed": SEED,
        "ragtruth_commit": COMMIT,
        "code_commit": _git_commit(),
        "zero_hours_rule": "closed",
        "claims": "sentences (src/claims/decompose.py:split_sentences)",
        **verifier.config(),
        "torch": __import__("torch").__version__,
        "host": platform.node(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / f"config_{tag}.json").write_text(json.dumps(config, indent=2))

    t0 = time.perf_counter()
    with scores_path.open("a") as out:
        for i, e in enumerate(todo, 1):
            claims = split_sentences(e.response)
            passages = evidence_passages(e)
            raw = verifier.raw_scores([c.text for c in claims], passages) if claims else []
            sentences = [
                {
                    "start": c.span[0],
                    "end": c.span[1],
                    "per_passage": [
                        [round(max(w.entailment for w in wins), 5), round(max(w.contradiction for w in wins), 5)]
                        for wins in per_premise
                    ],
                }
                for c, per_premise in zip(claims, raw)
            ]
            out.write(json.dumps({"id": e.id, "task": e.task_type, "sentences": sentences}) + "\n")
            out.flush()
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.perf_counter() - t0)
                print(f"  {i}/{len(todo)} responses, {rate * 60:.1f}/min, eta {(len(todo) - i) / rate / 60:.0f} min", flush=True)


def _read(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _md_table(result: dict, title: str) -> list[str]:
    lines = [f"### {title}", "", "| level | task | n | precision | recall | F1 |", "|---|---|---|---|---|---|"]
    for level in ("sentence_level", "response_level"):
        for task, m in result[level].items():
            lines.append(
                f"| {level.split('_')[0]} | {task} | {m['n']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
            )
    lines += ["", "Gold type -> predicted label (gold-hallucinated sentences):", ""]
    lines += [f"- {k}: {v}" for k, v in result["type_confusion"].items()]
    lines += ["", f"Sentences with conflicting evidence: {result['conflicting_sentences']}", ""]
    return lines


def cmd_evaluate(args) -> None:
    from src.eval.metrics import evaluate, tune_t_entail

    examples = {e.id: e for e in load()}
    tune_recs, report_recs = _read(args.tune), _read(args.report)
    t, dev_f1 = tune_t_entail(tune_recs, examples)
    same = Path(args.tune).resolve() == Path(args.report).resolve()
    results = {
        "tuned_on": args.tune,
        "reported_on": args.report,
        "t_entail": t,
        "tune_f1": dev_f1,
        "warning": "threshold tuned and reported on the same file: NOT a valid result" if same else None,
        "main": evaluate(report_recs, examples, t_entail=t),
        "excluding_implicit_true": evaluate(report_recs, examples, t_entail=t, exclude_implicit_true=True),
    }
    stem = Path(args.report).stem.replace("scores_", "")
    out = Path(args.report).parent / f"summary_{stem}.json"
    out.write_text(json.dumps(results, indent=2))
    md = [f"# Experiment 01: sentence baseline ({stem})", ""]
    if same:
        md += ["> **Warning:** threshold tuned on the same data it is reported on. Smoke check only.", ""]
    md += [f"Entailment threshold {t} (tuned on `{Path(args.tune).name}`, F1 {dev_f1:.3f}).", ""]
    md += _md_table(results["main"], "All labels")
    md += _md_table(results["excluding_implicit_true"], "Excluding implicit_true labels")
    (Path(args.report).parent / f"summary_{stem}.md").write_text("\n".join(md))
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    r.add_argument("--per-task", type=int, default=None, help="sample N responses per task (default: all)")
    r.add_argument("--model", choices=["base", "large"], default="large")
    r.add_argument("--mode", choices=["per_passage", "concat"], default="per_passage")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--fp16", action="store_true", help="half precision (CUDA only)")
    r.add_argument("--out-dir", default=str(OUT_DIR))
    r.set_defaults(func=cmd_run)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--tune", required=True, help="scores file to tune the threshold on (dev)")
    ev.add_argument("--report", required=True, help="scores file to report metrics on")
    ev.set_defaults(func=cmd_evaluate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
