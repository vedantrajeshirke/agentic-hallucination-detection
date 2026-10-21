"""Shared bits for experiment scripts: run metadata and summary writing."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    """HEAD commit of the repo, with "-dirty" if there are uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        return head + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata() -> dict:
    import torch

    return {
        "code_commit": git_commit(),
        "torch": torch.__version__,
        "host": platform.node(),
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids(path: Path) -> set[str]:
    return {r["id"] for r in read_jsonl(path)} if path.exists() else set()


def md_table(result: dict, title: str) -> list[str]:
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


def write_summary(tune: str, report: str, title: str) -> dict:
    """Tune t_entail on `tune` (dev), report on `report`; write summary_*.json and .md next to it."""
    from src.data.ragtruth import load
    from src.eval.metrics import evaluate, tune_t_entail

    examples = {e.id: e for e in load()}
    tune_recs, report_recs = read_jsonl(tune), read_jsonl(report)
    t, tune_f1 = tune_t_entail(tune_recs, examples)
    same = Path(tune).resolve() == Path(report).resolve()
    results = {
        "tuned_on": Path(tune).name,
        "reported_on": Path(report).name,
        "n_reported": len(report_recs),
        "t_entail": t,
        "tune_f1": tune_f1,
        "warning": "threshold tuned and reported on the same file: NOT a valid result" if same else None,
        "main": evaluate(report_recs, examples, t_entail=t),
        "excluding_implicit_true": evaluate(report_recs, examples, t_entail=t, exclude_implicit_true=True),
    }
    stem = Path(report).stem.replace("scores_", "")
    out_dir = Path(report).parent
    (out_dir / f"summary_{stem}.json").write_text(json.dumps(results, indent=2))
    md = [f"# {title} ({stem})", ""]
    if same:
        md += ["> **Warning:** threshold tuned on the same data it is reported on. Smoke check only.", ""]
    md += [f"Entailment threshold {t} (tuned on `{Path(tune).name}`, F1 {tune_f1:.3f}); {len(report_recs)} responses.", ""]
    md += md_table(results["main"], "All labels")
    md += md_table(results["excluding_implicit_true"], "Excluding implicit_true labels")
    (out_dir / f"summary_{stem}.md").write_text("\n".join(md))
    print("\n".join(md))
    return results
