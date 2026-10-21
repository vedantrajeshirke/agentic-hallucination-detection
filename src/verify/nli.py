"""DeBERTa-v3 NLI wrapper: (premise, hypothesis) -> P(entailment, neutral, contradiction).

Premises longer than the model's 512-token window are split into overlapping token windows, so
every part of a long source is seen. Each window is scored separately; how window scores are
combined into a claim verdict is the scorer's job (src/verify/scorer.py), not this module's.

Smoke test:
    uv run python -m src.verify.nli
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Pinned revisions so results are reproducible. Both checkpoints share the label order
# {0: entailment, 1: neutral, 2: contradiction}, but other DeBERTa NLI checkpoints differ
# (e.g. cross-encoder/nli-deberta-v3-base), so the order is always read from the config.
MODELS = {
    "base": ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"),
    "large": (
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    ),
}
LABELS = ("entailment", "neutral", "contradiction")

# Both checkpoints use 512 position embeddings, but their tokenizers report no max length, so
# nothing truncates automatically. Enforced here.
MAX_LENGTH = 512


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float
    premise_window: tuple[int, int]  # character span of the premise text actually scored

    @property
    def label(self) -> str:
        probs = (self.entailment, self.neutral, self.contradiction)
        return LABELS[max(range(3), key=probs.__getitem__)]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NLIModel:
    def __init__(
        self,
        size: str = "base",
        device: str | None = None,
        batch_size: int = 16,
        window_stride: int = 128,
        fp16: bool = False,
    ):
        name, revision = MODELS[size]
        self.name, self.revision = name, revision
        self.device = device or default_device()
        self.batch_size = batch_size
        self.window_stride = window_stride  # token overlap between consecutive premise windows
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, revision=revision)
        self.model.to(self.device).eval()
        # Half precision is ~2x faster on CUDA GPUs (e.g. Colab T4); scores can differ slightly.
        self.fp16 = fp16 and self.device == "cuda"
        if self.fp16:
            self.model.half()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if sorted(id2label.values()) != sorted(LABELS):
            raise ValueError(f"{name} has unexpected labels {id2label}")
        # column index in the model output for each of our LABELS, in our order
        self._label_cols = [next(i for i, v in id2label.items() if v == lab) for lab in LABELS]

    # -----------------------------------------------------------------------------------------
    # Windowing
    # -----------------------------------------------------------------------------------------

    def premise_windows(self, premise: str, hypothesis: str) -> list[tuple[int, int]]:
        """Character spans of `premise` such that each span + hypothesis fits in MAX_LENGTH.

        Windows are cut on token boundaries and overlap by `window_stride` tokens, so a fact that
        straddles a boundary appears whole in at least one window (if it is shorter than the
        overlap).
        """
        hyp_len = len(self.tokenizer(hypothesis, add_special_tokens=False)["input_ids"])
        # [CLS] premise [SEP] hypothesis [SEP]
        budget = MAX_LENGTH - hyp_len - 3
        if budget < 64:
            raise ValueError(f"hypothesis too long ({hyp_len} tokens) to leave room for a premise")

        enc = self.tokenizer(premise, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        if len(offsets) <= budget:
            return [(0, len(premise))]

        step = budget - min(self.window_stride, budget // 2)
        windows = []
        for start in range(0, len(offsets), step):
            end = min(start + budget, len(offsets))
            windows.append((offsets[start][0], offsets[end - 1][1]))
            if end == len(offsets):
                break
        return windows

    # -----------------------------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------------------------

    @torch.inference_mode()
    def _score_pairs(self, premises: list[str], hypotheses: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(premises), self.batch_size):
            batch = self.tokenizer(
                premises[i : i + self.batch_size],
                hypotheses[i : i + self.batch_size],
                truncation="only_first",  # safety net; windows are already sized to fit
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**batch).logits.float(), dim=-1)
            out.extend(probs[:, self._label_cols].cpu().tolist())
        return out

    def score(self, premise: str, hypothesis: str) -> list[NLIResult]:
        """Score one hypothesis against every window of one premise."""
        return self.score_many([(premise, hypothesis)])[0]

    def score_many(self, pairs: list[tuple[str, str]]) -> list[list[NLIResult]]:
        """Score many (premise, hypothesis) pairs, batching all windows together.

        Returns one list of window results per input pair, in input order.
        """
        flat_p, flat_h, owners, spans = [], [], [], []
        for idx, (premise, hypothesis) in enumerate(pairs):
            for a, b in self.premise_windows(premise, hypothesis):
                flat_p.append(premise[a:b])
                flat_h.append(hypothesis)
                owners.append(idx)
                spans.append((a, b))
        probs = self._score_pairs(flat_p, flat_h)
        results: list[list[NLIResult]] = [[] for _ in pairs]
        for owner, span, (e, n, c) in zip(owners, spans, probs):
            results[owner].append(NLIResult(e, n, c, span))
        return results


# ---------------------------------------------------------------------------------------------
# Smoke test: 5 hand-written pairs, most taken from cases seen in notes/ragtruth_inspection.md
# ---------------------------------------------------------------------------------------------

SMOKE_PAIRS = [
    # (premise, hypothesis, expected label, why)
    (
        # An earlier version omitted "In the factory system," so the premise never said whose
        # hours these were; the large model then (defensibly) answered neutral.
        ("In the factory system, working hours were as long as they had been for the farmer, "
        "that is, from dawn to dusk, six days per week."),
        "Factory working hours were longer than agricultural working hours.",
        "contradiction",
        "ex. 2: 'as long as' vs 'longer'",
    ),
    (
        ("Hours were regulated with precision by the factory clock to ensure that a full measure "
        "of work was performed."),
        "Factory work hours were regulated with precision.",
        "entailment",
        "ex. 2: the supported part of the same sentence",
    ),
    (
        "The ambience is not intimate. The ambience is divey. The bar has outdoor seating.",
        "The bar offers an intimate atmosphere.",
        "contradiction",
        "ex. 11: JSON field converted to a sentence",
    ),
    (
        ("The Palestinian Authority officially became the 123rd member of the International "
        "Criminal Court on Wednesday."),
        "The Gaza Strip is included in the occupied Palestinian territory.",
        "neutral",
        "RAGTruth README example: baseless addition, not in the premise",
    ),
    (
        ("Five female activists who were detained last month were released by Chinese police on "
        "Monday."),
        "Chinese police freed five women activists on Monday.",
        "entailment",
        "ex. 8: paraphrase that should be supported",
    ),
]


def _smoke(size: str) -> None:
    import time

    model = NLIModel(size)
    print(f"\n=== {model.name} @ {model.revision[:7]} on {model.device}")
    t0 = time.perf_counter()
    results = model.score_many([(p, h) for p, h, _, _ in SMOKE_PAIRS])
    elapsed = time.perf_counter() - t0
    correct = 0
    for (_, hyp, expected, why), res in zip(SMOKE_PAIRS, results):
        r = res[0]
        ok = r.label == expected
        correct += ok
        print(
            f"{'PASS' if ok else 'FAIL'}  expected={expected:13s} got={r.label:13s} "
            f"E={r.entailment:.2f} N={r.neutral:.2f} C={r.contradiction:.2f}  ({why})"
        )
    print(f"{correct}/{len(SMOKE_PAIRS)} correct, {elapsed:.2f}s for {len(SMOKE_PAIRS)} pairs")


if __name__ == "__main__":
    import sys

    for size in sys.argv[1:] or ["base", "large"]:
        _smoke(size)
