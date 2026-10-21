"""RAGTruth loader.

Downloads the official RAGTruth release (ParticleMedia/RAGTruth) at a pinned commit, parses it
into typed records, and validates the span annotations.

Run as a script to print the dataset schema and label statistics, write them to
results/ragtruth_stats/summary.json, and write a stratified inspection sample (train split only)
to notes/ragtruth_samples.md:

    uv run python -m src.data.ragtruth
"""

from __future__ import annotations

import json
import random
import re
import statistics
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.data.linearize import business_facts, review_passages

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data_cache" / "ragtruth"
COMMIT = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
BASE_URL = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{COMMIT}/dataset"
FILES = ("response.jsonl", "source_info.jsonl")

TASKS = ("QA", "Summary", "Data2txt")
LABEL_TYPES = (
    "Evident Conflict",
    "Subtle Conflict",
    "Evident Baseless Info",
    "Subtle Baseless Info",
)


@dataclass(frozen=True)
class HallucinationSpan:
    start: int
    end: int
    text: str
    label_type: str
    implicit_true: bool  # span is factually correct, but not stated in the context
    due_to_null: bool  # Data2txt only: hallucination caused by a null field in the input
    meta: str | None  # free-text annotator comment (None for 86 spans at commit c103204)

    @property
    def is_conflict(self) -> bool:
        """Contradicts the context (vs. baseless = not supported by it)."""
        return self.label_type.endswith("Conflict")

    @property
    def is_subtle(self) -> bool:
        return self.label_type.startswith("Subtle")


@dataclass
class RagTruthExample:
    id: str
    source_id: str
    task_type: str  # QA | Summary | Data2txt
    source_dataset: str  # MARCO | CNN/DM | Recent News | Yelp
    model: str
    temperature: float
    split: str  # train | test (RAGTruth ships no dev split; see make_dev_split)
    quality: str  # good | incorrect_refusal | truncated
    response: str
    context: str  # the grounding text the response must be faithful to
    question: str | None  # QA only
    prompt: str
    spans: list[HallucinationSpan] = field(default_factory=list)
    source_info: dict | str | None = field(default=None, repr=False)  # raw RAGTruth source_info

    @property
    def has_hallucination(self) -> bool:
        return bool(self.spans)


def evidence_passages(e: RagTruthExample, zero_hours_rule: str = "closed") -> list[str]:
    """The grounding evidence for one example, split into separately checkable passages.

    QA: the 3 retrieved passages. Summary: the article (one passage; the NLI wrapper windows it).
    Data2txt: the business record converted to sentences, then one passage per review.
    """
    if e.task_type == "QA":
        parts = re.split(r"(?m)^passage \d+:", e.source_info["passages"])
        return [p.strip() for p in parts if p.strip()]
    if e.task_type == "Summary":
        return [e.source_info.strip()]
    if e.task_type == "Data2txt":
        facts = " ".join(business_facts(e.source_info, zero_hours_rule))
        return [facts, *review_passages(e.source_info)]
    raise ValueError(f"unknown task_type {e.task_type!r}")


def download(force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = CACHE_DIR / name
        if force or not dest.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
    return CACHE_DIR


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _context_and_question(source: dict) -> tuple[str, str | None]:
    info = source["source_info"]
    task = source["task_type"]
    if task == "Summary":
        return info, None
    if task == "QA":
        return info["passages"], info["question"]
    if task == "Data2txt":
        # Raw JSON, kept for display. The verifier uses evidence_passages(), which converts it
        # to sentences (src/data/linearize.py).
        return json.dumps(info, ensure_ascii=False, indent=1), None
    raise ValueError(f"unknown task_type {task!r}")


def _dedupe_spans(spans: list[HallucinationSpan]) -> list[HallucinationSpan]:
    """Collapse spans at identical (start, end) so one hallucination is never counted twice.

    At commit c103204, 64 span pairs share exact offsets: 24 are identical copies, 38 differ only
    in implicit_true (plausibly re-added when the 2024/02 update introduced that flag; not
    confirmed), 2 differ in meta/label_type. Keep the copy with implicit_true=True if any,
    otherwise the first.
    """
    by_pos: dict[tuple[int, int], HallucinationSpan] = {}
    for s in spans:
        kept = by_pos.get((s.start, s.end))
        if kept is None or (s.implicit_true and not kept.implicit_true):
            by_pos[(s.start, s.end)] = s
    return sorted(by_pos.values(), key=lambda s: (s.start, s.end))


def load(split: str | None = None) -> list[RagTruthExample]:
    """Load RAGTruth responses joined with their sources. `split` is 'train', 'test' or None."""
    download()
    sources = {s["source_id"]: s for s in _read_jsonl(CACHE_DIR / "source_info.jsonl")}
    examples = []
    for r in _read_jsonl(CACHE_DIR / "response.jsonl"):
        if split is not None and r["split"] != split:
            continue
        src = sources[r["source_id"]]
        context, question = _context_and_question(src)
        spans = [
            HallucinationSpan(
                start=lab["start"],
                end=lab["end"],
                text=lab["text"],
                label_type=lab["label_type"],
                implicit_true=lab["implicit_true"],
                due_to_null=lab["due_to_null"],
                meta=lab["meta"],
            )
            for lab in r["labels"]
        ]
        examples.append(
            RagTruthExample(
                id=r["id"],
                source_id=r["source_id"],
                task_type=src["task_type"],
                source_dataset=src["source"],
                model=r["model"],
                temperature=r["temperature"],
                split=r["split"],
                quality=r["quality"],
                response=r["response"],
                context=context,
                question=question,
                prompt=src["prompt"],
                spans=_dedupe_spans(spans),
                source_info=src["source_info"],
            )
        )
    return examples


def make_dev_split(
    train: list[RagTruthExample], dev_frac: float = 0.1, seed: int = 0
) -> tuple[list[RagTruthExample], list[RagTruthExample]]:
    """Carve a dev set out of train, grouped by source_id.

    Each source has six responses (one per model); splitting by response would put the same
    source in both halves and leak. Stratified by task so each task keeps its share.
    """
    rng = random.Random(seed)
    by_task: dict[str, list[str]] = defaultdict(list)
    for sid, task in sorted({(e.source_id, e.task_type) for e in train}):
        by_task[task].append(sid)
    dev_sources = set()
    for task in sorted(by_task):
        sids = by_task[task]
        rng.shuffle(sids)
        dev_sources.update(sids[: round(len(sids) * dev_frac)])
    new_train = [e for e in train if e.source_id not in dev_sources]
    dev = [e for e in train if e.source_id in dev_sources]
    return new_train, dev



def select(split: str, per_task: int | None = None, seed: int = 0) -> list[RagTruthExample]:
    """Examples for an experiment: 'train' / 'dev' (source-grouped 10% of train) / 'test',
    optionally `per_task` sampled per task with a fixed seed."""
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

# ---------------------------------------------------------------------------------------------
# Validation and statistics
# ---------------------------------------------------------------------------------------------


def validate_spans(examples: list[RagTruthExample]) -> dict:
    """Check that every span's offsets reproduce its text, and count overlapping spans."""
    mismatched, out_of_bounds, overlapping_pairs, examples_with_overlap = [], [], 0, 0
    for e in examples:
        for s in e.spans:
            if not (0 <= s.start < s.end <= len(e.response)):
                out_of_bounds.append((e.id, s.start, s.end))
            elif e.response[s.start : s.end] != s.text:
                mismatched.append((e.id, s.start, s.end))
        n = sum(
            1
            for i, a in enumerate(e.spans)
            for b in e.spans[i + 1 :]
            if a.start < b.end and b.start < a.end
        )
        overlapping_pairs += n
        examples_with_overlap += n > 0
    return {
        "n_spans": sum(len(e.spans) for e in examples),
        "offset_text_mismatch": len(mismatched),
        "out_of_bounds": len(out_of_bounds),
        "mismatch_examples": mismatched[:5],
        "overlapping_span_pairs": overlapping_pairs,
        "responses_with_overlapping_spans": examples_with_overlap,
    }


def _describe(xs: list[float]) -> dict:
    if not xs:
        return {}
    qs = statistics.quantiles(xs, n=20) if len(xs) > 1 else [xs[0]] * 19
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 1),
        "median": statistics.median(xs),
        "p5": qs[0],
        "p95": qs[-1],
        "max": max(xs),
    }


def _words(text: str) -> int:
    return len(text.split())


def compute_stats(examples: list[RagTruthExample]) -> dict:
    stats: dict = {}
    stats["n_responses"] = len(examples)
    stats["n_sources"] = len({e.source_id for e in examples})
    stats["responses_by_split"] = dict(Counter(e.split for e in examples))
    stats["responses_by_task"] = dict(Counter(e.task_type for e in examples))
    stats["responses_by_model"] = dict(Counter(e.model for e in examples))
    stats["quality"] = dict(Counter(e.quality for e in examples))

    train_src = {e.source_id for e in examples if e.split == "train"}
    test_src = {e.source_id for e in examples if e.split == "test"}
    stats["sources_in_both_train_and_test"] = len(train_src & test_src)

    def rate(group: list[RagTruthExample]) -> float:
        return round(sum(e.has_hallucination for e in group) / len(group), 3)

    stats["hallucinated_response_rate"] = {
        "overall": rate(examples),
        **{
            f"{split}/{task}": rate(g)
            for (split, task), g in sorted(_group(examples, lambda e: (e.split, e.task_type)).items())
        },
    }
    stats["hallucinated_response_rate_by_model"] = {
        m: rate(g) for m, g in sorted(_group(examples, lambda e: e.model).items())
    }

    spans = [(e, s) for e in examples for s in e.spans]
    stats["spans_by_label_type"] = dict(Counter(s.label_type for _, s in spans))
    stats["spans_by_task_and_label_type"] = {
        task: dict(Counter(s.label_type for e, s in spans if e.task_type == task)) for task in TASKS
    }
    stats["conflict_vs_baseless"] = dict(
        Counter("conflict" if s.is_conflict else "baseless" for _, s in spans)
    )
    stats["implicit_true_spans"] = sum(s.implicit_true for _, s in spans)
    stats["due_to_null_spans"] = sum(s.due_to_null for _, s in spans)
    stats["spans_per_hallucinated_response"] = _describe(
        [len(e.spans) for e in examples if e.spans]
    )
    stats["span_length_words"] = {
        task: _describe([_words(s.text) for e, s in spans if e.task_type == task]) for task in TASKS
    }
    stats["response_length_words"] = {
        task: _describe([_words(e.response) for e in examples if e.task_type == task])
        for task in TASKS
    }
    # Word counts, not model tokens. DeBERTa-v3 token counts measured on 5 Summary articles were
    # ~1.13-1.26x the word count. Note: its tokenizer reports no max length, so the NLI wrapper
    # must enforce the 512-token limit itself.
    seen, ctx_len = set(), defaultdict(list)
    for e in examples:
        if e.source_id not in seen:
            seen.add(e.source_id)
            ctx_len[e.task_type].append(_words(e.context))
    stats["context_length_words_per_source"] = {t: _describe(ctx_len[t]) for t in TASKS}
    stats["span_validation"] = validate_spans(examples)
    return stats


def _group(examples, key) -> dict:
    out = defaultdict(list)
    for e in examples:
        out[key(e)].append(e)
    return out


# ---------------------------------------------------------------------------------------------
# Inspection sample
# ---------------------------------------------------------------------------------------------


def _features(e: RagTruthExample) -> set[str]:
    f = {f"type:{s.label_type}" for s in e.spans}
    if not e.spans:
        f.add("clean")
    if any(s.implicit_true for s in e.spans):
        f.add("implicit_true")
    if any(s.due_to_null for s in e.spans):
        f.add("due_to_null")
    spans = e.spans
    if any(a.start < b.end and b.start < a.end for i, a in enumerate(spans) for b in spans[i + 1 :]):
        f.add("overlap")
    return f


# The inspection sample as originally drawn by stratified_sample(train, per_task=5, seed=0) before
# span deduplication was added. Frozen so later loader changes cannot reshuffle examples that have
# already been hand-reviewed in notes/ragtruth_inspection.md.
INSPECTION_SAMPLE_IDS = (
    "12213", "17032", "12024", "16501", "13316",  # QA
    "2300", "4872", "1283", "526", "5395",  # Summary
    "9723", "7842", "9380", "10693", "6755",  # Data2txt
)


def stratified_sample(
    examples: list[RagTruthExample], per_task: int = 5, seed: int = 0
) -> list[RagTruthExample]:
    """Pick `per_task` examples per task, greedily covering label types, a clean response,
    the implicit_true / due_to_null flags, overlapping spans, and as many models as possible.

    Truncated responses and incorrect refusals (~1% of the data) are excluded so the sample shows
    typical cases; they are still counted in compute_stats."""
    rng = random.Random(seed)
    examples = [e for e in examples if e.quality == "good"]
    picked = []
    for task in TASKS:
        pool = [e for e in examples if e.task_type == task]
        rng.shuffle(pool)
        covered: set[str] = set()
        models: set[str] = set()
        chosen: list[RagTruthExample] = []
        for _ in range(per_task):
            best = max(
                (e for e in pool if e not in chosen),
                key=lambda e: (len(_features(e) - covered), e.model not in models),
            )
            chosen.append(best)
            covered |= _features(best)
            models.add(best.model)
        picked.extend(chosen)
    return picked


def _highlight(response: str, spans: list[HallucinationSpan]) -> str:
    """Wrap hallucinated regions in ==...== (overlapping spans merged for display)."""
    merged: list[list[int]] = []
    for s in sorted(spans, key=lambda s: s.start):
        if merged and s.start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s.end)
        else:
            merged.append([s.start, s.end])
    out, pos = [], 0
    for a, b in merged:
        out += [response[pos:a], "==", response[a:b], "=="]
        pos = b
    out.append(response[pos:])
    return "".join(out)


def write_sample_markdown(sample: list[RagTruthExample], path: Path) -> None:
    lines = [
        "# RAGTruth inspection sample",
        "",
        (
            f"Generated by `uv run python -m src.data.ragtruth` (RAGTruth commit `{COMMIT[:7]}`, "
            "train split, quality=good only, frozen IDs). Do not edit by hand; write findings in "
            "`ragtruth_inspection.md`."
        ),
        "",
        "Hallucinated regions are marked `==like this==` in the response.",
        "",
    ]
    for i, e in enumerate(sample, 1):
        lines += [
            f"## {i}. {e.task_type} · {e.model} · id {e.id}",
            "",
            (
                f"- source: {e.source_dataset} (source_id {e.source_id}), "
                f"temperature {e.temperature}, quality: {e.quality}"
            ),
            f"- spans: {len(e.spans)} · features: {', '.join(sorted(_features(e)))}",
            "",
        ]
        if e.question:
            lines += [f"**Question:** {e.question}", ""]
        lines += [
            "<details><summary>Context (click to expand)</summary>",
            "",
            "```",
            e.context.strip(),
            "```",
            "",
            "</details>",
            "",
            "**Response:**",
            "",
            "> " + _highlight(e.response, e.spans).replace("\n", "\n> "),
            "",
        ]
        if e.spans:
            lines += ["| chars | type | implicit_true | due_to_null | text | annotator note |",
                      "|---|---|---|---|---|---|"]
            for s in e.spans:
                note = (s.meta or "").replace("\n", " / ").replace("|", "\\|")
                text = s.text.replace("|", "\\|")
                lines.append(
                    f"| {s.start}–{s.end} | {s.label_type} | {s.implicit_true} | "
                    f"{s.due_to_null} | {text} | {note} |"
                )
            lines.append("")
        lines.append("---")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    examples = load()
    stats = compute_stats(examples)
    print(json.dumps(stats, indent=2))

    out_dir = REPO_ROOT / "results" / "ragtruth_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"config": {"ragtruth_commit": COMMIT}, "stats": stats}, indent=2)
    )

    train = [e for e in examples if e.split == "train"]
    by_id = {e.id: e for e in train}
    sample = [by_id[i] for i in INSPECTION_SAMPLE_IDS]
    write_sample_markdown(sample, REPO_ROOT / "notes" / "ragtruth_samples.md")
    print(f"\nwrote {out_dir / 'summary.json'}")
    print(f"wrote notes/ragtruth_samples.md ({len(sample)} examples)")


if __name__ == "__main__":
    main()
