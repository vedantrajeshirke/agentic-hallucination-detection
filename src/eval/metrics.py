"""Score verifier predictions against RAGTruth span labels.

A claim (here: a sentence) is gold-hallucinated if its character span overlaps any labeled span.
A response is gold-hallucinated if it has any labeled span. Predictions re-aggregate stored
per-passage NLI maxima, so thresholds and rules can change without re-running the model.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from src.data.ragtruth import RagTruthExample
from src.verify.nli import NLIResult
from src.verify.scorer import aggregate

_ALNUM = re.compile(r"[A-Za-z0-9]")
SEVERITY = ("contradicted", "unsupported", "supported")


def gold_for_span(
    e: RagTruthExample, start: int, end: int, exclude_implicit_true: bool = False
) -> dict:
    # an overlap must contain a letter or digit: some labels start with the previous
    # sentence's final "." (error analysis 01, M-QA-04)
    spans = [
        s
        for s in e.spans
        if s.start < end and start < s.end
        and _ALNUM.search(e.response[max(start, s.start) : min(end, s.end)])
    ]
    if exclude_implicit_true:
        spans = [s for s in spans if not s.implicit_true]
    return {
        "hallucinated": bool(spans),
        "conflict": any(s.is_conflict for s in spans),
        "baseless": any(not s.is_conflict for s in spans),
    }


def predict(per_passage: list[list[float]], t_entail: float, t_contradict: float, tie_break: str) -> str:
    """per_passage: [[max_entail, max_contradict], ...] for one claim."""
    results = [[NLIResult(e, max(0.0, 1 - e - c), c, (0, 0))] for e, c in per_passage]
    return aggregate(results, t_entail, t_contradict, tie_break).label


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(
    records: list[dict],
    examples: dict[str, RagTruthExample],
    t_entail: float = 0.5,
    t_contradict: float = 0.5,
    tie_break: str = "max_prob",
    exclude_implicit_true: bool = False,
) -> dict:
    """records: output rows of experiments/01 or 02 (one per response)."""
    sent = defaultdict(Counter)  # task -> tp/fp/fn
    resp = defaultdict(Counter)
    confusion = Counter()  # (gold type, predicted label) for gold-hallucinated sentences
    n_conflicting = 0
    for rec in records:
        e = examples[rec["id"]]
        any_pred = False
        for s in rec["sentences"]:
            gold = gold_for_span(e, s["start"], s["end"], exclude_implicit_true)
            # sentence records carry either one score set (sentence = claim) or a list of claims
            claim_scores = [c["per_passage"] for c in s["claims"]] if "claims" in s else [s["per_passage"]]
            labels = [predict(pp, t_entail, t_contradict, tie_break) for pp in claim_scores]
            # worst claim decides; a sentence with no claims (greeting, header) is never flagged
            label = min(labels, key=SEVERITY.index) if labels else "supported"
            pred = label != "supported"
            any_pred |= pred
            n_conflicting += any(
                max(x[0] for x in pp) >= t_entail and max(x[1] for x in pp) >= t_contradict
                for pp in claim_scores
            )
            for task in (e.task_type, "all"):
                sent[task]["tp" if pred and gold["hallucinated"] else "fp" if pred else "fn" if gold["hallucinated"] else "tn"] += 1
            if gold["hallucinated"]:
                gtype = "both" if gold["conflict"] and gold["baseless"] else "conflict" if gold["conflict"] else "baseless"
                confusion[(gtype, label)] += 1
        gold_resp = any(not exclude_implicit_true or not s.implicit_true for s in e.spans)
        for task in (e.task_type, "all"):
            resp[task]["tp" if any_pred and gold_resp else "fp" if any_pred else "fn" if gold_resp else "tn"] += 1

    def table(counts):
        return {t: prf(c["tp"], c["fp"], c["fn"]) | {"n": sum(c.values())} for t, c in sorted(counts.items())}

    return {
        "params": {
            "t_entail": t_entail,
            "t_contradict": t_contradict,
            "tie_break": tie_break,
            "exclude_implicit_true": exclude_implicit_true,
        },
        "sentence_level": table(sent),
        "response_level": table(resp),
        "type_confusion": {f"{g} -> {p}": n for (g, p), n in sorted(confusion.items())},
        "conflicting_sentences": n_conflicting,
    }


def tune_t_entail(records, examples, grid=None, **kw) -> tuple[float, float]:
    """Pick the entailment threshold maximising overall sentence-level F1 (use on dev only)."""
    grid = grid or [round(0.05 * i, 2) for i in range(1, 20)]
    best = max(grid, key=lambda t: evaluate(records, examples, t_entail=t, **kw)["sentence_level"]["all"]["f1"])
    return best, evaluate(records, examples, t_entail=best, **kw)["sentence_level"]["all"]["f1"]
