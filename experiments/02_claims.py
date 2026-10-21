"""Experiment 02: claim-level vs sentence-level checking, x whole-passage vs small evidence windows.

Pipeline v2 (fixes from error analysis 01: splitter abbreviations, review wording, alnum-overlap
gold rule). Every config is scored at sentence level against the same RAGTruth labels, so all four
are directly comparable:

                 evidence=passage            evidence=s3 (3-sentence windows)
  claims=sentences  baseline rerun with fixes   evidence fix only
  claims=llm        claim splitting only        both

Subcommands (all resumable; finished response ids are skipped on restart):
  decompose  Llama 3.1 (Ollama) splits each sentence into claims -> claims_<split>.jsonl
  run        DeBERTa NLI on every claim                           -> scores_<split>_<claims>_<evidence>.jsonl
  evaluate   tune t_entail on a dev scores file, report on another -> summary_*.json / .md

  uv run python experiments/02_claims.py decompose --split dev --per-task 100 --workers 4
  uv run python experiments/02_claims.py run --split dev --per-task 100 --claims llm --evidence s3 --fp16
  uv run python experiments/02_claims.py evaluate --tune <dev scores> --report <test scores>
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.claims.decompose import DECOMPOSE_MODEL, decompose_response, split_sentences
from src.data.ragtruth import COMMIT, evidence_passages, select
from src.eval.report import done_ids, read_jsonl, run_metadata, write_summary

OUT_DIR = REPO_ROOT / "results" / "02_claims"
SEED = 0
EVIDENCE = {"passage": None, "s3": 3}


def split_tag(split: str, per_task: int | None) -> str:
    return f"{split}{'' if per_task is None else f'_n{per_task}'}"


def _progress(i: int, n: int, t0: float, every: int = 10) -> None:
    if i % every == 0 or i == n:
        rate = i / (time.perf_counter() - t0)
        print(f"  {i}/{n} responses, {rate * 60:.1f}/min, eta {(n - i) / rate / 60:.0f} min", flush=True)


def cmd_decompose(args) -> None:
    import ollama

    examples = select(args.split, args.per_task, SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"claims_{split_tag(args.split, args.per_task)}.jsonl"
    done = done_ids(path)
    todo = [e for e in examples if e.id not in done]
    print(f"{len(examples)} examples, {len(done)} decomposed, {len(todo)} to go -> {path}")
    if not todo:
        return
    client = ollama.Client()
    digest = next((m.digest for m in client.list().models if m.model == args.model), None)
    (out_dir / f"config_claims_{split_tag(args.split, args.per_task)}.json").write_text(json.dumps({
        "experiment": "02_claims/decompose", "split": args.split, "per_task": args.per_task, "seed": SEED,
        "ragtruth_commit": COMMIT, "llm": args.model, "llm_digest": digest,
        "llm_options": {"temperature": 0, "seed": SEED},
        "workers": args.workers, **run_metadata(),
    }, indent=2))

    lock = threading.Lock()
    # only bad model output falls back; connection errors should stop the run
    BAD_OUTPUT = (ValueError, KeyError, TypeError, ollama.ResponseError)

    def work(e):
        sents = split_sentences(e.response)
        texts = [c.text for c in sents]
        status = "ok"
        try:
            claims = decompose_response(texts, e.question, model=args.model, client=client, seed=SEED)
        except BAD_OUTPUT as exc:  # malformed model output: retry once with another seed, then fall back
            try:
                claims = decompose_response(texts, e.question, model=args.model, client=client, seed=SEED + 1)
                status = "retried"
            except BAD_OUTPUT:
                claims, status = [[t] for t in texts], f"fallback: {type(exc).__name__}"
        return {
            "id": e.id,
            "status": status,
            "sentences": [{"start": c.span[0], "end": c.span[1], "claims": cl} for c, cl in zip(sents, claims)],
        }

    t0 = time.perf_counter()
    with path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, e) for e in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()
            _progress(i, len(todo), t0)
    statuses = [r["status"] for r in read_jsonl(path)]
    print(f"done. status counts: { {s: statuses.count(s) for s in set(statuses)} }")


def cmd_run(args) -> None:
    from src.verify.nli import NLIModel
    from src.verify.scorer import Verifier

    examples = {e.id: e for e in select(args.split, args.per_task, SEED)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stag = split_tag(args.split, args.per_task)
    tag = f"{stag}_{args.claims}_{args.evidence}"
    path = out_dir / f"scores_{tag}.jsonl"

    if args.claims == "llm":
        claims_path = out_dir / f"claims_{stag}.jsonl"
        decomposed = {r["id"]: r for r in read_jsonl(claims_path)}
        missing = [i for i in examples if i not in decomposed]
        if missing:
            sys.exit(f"{len(missing)} responses not decomposed yet; run `decompose` first ({claims_path})")
    done = done_ids(path)
    todo = [e for i, e in examples.items() if i not in done]
    print(f"{len(examples)} examples, {len(done)} scored, {len(todo)} to go -> {path}")
    if not todo:
        return

    verifier = Verifier(
        NLIModel(args.model, batch_size=args.batch_size, fp16=args.fp16),
        evidence_sentences=EVIDENCE[args.evidence],
    )
    (out_dir / f"config_{tag}.json").write_text(json.dumps({
        "experiment": "02_claims/run", "split": args.split, "per_task": args.per_task, "seed": SEED,
        "ragtruth_commit": COMMIT, "zero_hours_rule": "closed", "claims": args.claims,
        "evidence": args.evidence, **verifier.config(), **run_metadata(),
    }, indent=2))

    t0 = time.perf_counter()
    with path.open("a") as out:
        for i, e in enumerate(todo, 1):
            if args.claims == "llm":
                sentences = decomposed[e.id]["sentences"]
            else:
                sentences = [
                    {"start": c.span[0], "end": c.span[1], "claims": [c.text]} for c in split_sentences(e.response)
                ]
            flat = [c for s in sentences for c in s["claims"]]
            raw = iter(verifier.raw_scores(flat, evidence_passages(e)) if flat else [])
            rows = []
            for s in sentences:
                claims = []
                for text in s["claims"]:
                    per_premise = next(raw)
                    claims.append({
                        "text": text,
                        "per_passage": [
                            [round(max(w.entailment for w in wins), 5), round(max(w.contradiction for w in wins), 5)]
                            for wins in per_premise
                        ],
                    })
                rows.append({"start": s["start"], "end": s["end"], "claims": claims})
            out.write(json.dumps({"id": e.id, "task": e.task_type, "sentences": rows}) + "\n")
            out.flush()
            _progress(i, len(todo), t0)


def cmd_evaluate(args) -> None:
    write_summary(args.tune, args.report, "Experiment 02: claims x evidence")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decompose")
    d.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    d.add_argument("--per-task", type=int, default=None)
    d.add_argument("--model", default=DECOMPOSE_MODEL)
    d.add_argument("--workers", type=int, default=1, help="parallel requests (match OLLAMA_NUM_PARALLEL)")
    d.add_argument("--out-dir", default=str(OUT_DIR))
    d.set_defaults(func=cmd_decompose)

    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test", "train"], default="dev")
    r.add_argument("--per-task", type=int, default=None)
    r.add_argument("--claims", choices=["sentences", "llm"], required=True)
    r.add_argument("--evidence", choices=list(EVIDENCE), required=True)
    r.add_argument("--model", choices=["base", "large"], default="large")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--fp16", action="store_true", help="half precision (CUDA only)")
    r.add_argument("--out-dir", default=str(OUT_DIR))
    r.set_defaults(func=cmd_run)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--tune", required=True)
    ev.add_argument("--report", required=True)
    ev.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
