"""Claim + evidence passages -> supported / contradicted / unsupported.

Aggregation (PROJECT.md §6.3): supported if any passage entails the claim, contradicted if any
contradicts it, otherwise unsupported. PROJECT.md doesn't say what happens when both hold, which
real sources do produce (RAGTruth ex. 9: the article says Ly never saw her mother again, then
the mother appears at the end). `tie_break` decides; the case is flagged as `conflicting` either
way so it can be counted.

Evidence modes (inspection: ex. 4 needs passages combined, ex. 5 is hurt by combining):
- "per_passage": each passage is checked on its own (its windows, if long).
- "concat": all passages joined into one premise, then windowed.

Thresholds are placeholders until tuned on the dev split (never on test).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.verify.nli import NLIModel, NLIResult

VERDICTS = ("supported", "contradicted", "unsupported")
TIE_BREAKS = ("max_prob", "supported", "contradicted")
MODES = ("per_passage", "concat")


@dataclass(frozen=True)
class ClaimVerdict:
    label: str  # supported | contradicted | unsupported
    p_entail: float  # max entailment over all passages/windows
    p_contradict: float  # max contradiction over all passages/windows
    conflicting: bool  # some window entails AND some window contradicts
    entail_at: tuple[int, tuple[int, int]]  # (passage index, char window) of best entailment
    contradict_at: tuple[int, tuple[int, int]]  # same for best contradiction

    @property
    def hallucination_score(self) -> float:
        """Higher = more likely unsupported or contradicted. For threshold sweeps / PR curves."""
        return 1.0 - self.p_entail


def aggregate(
    per_passage: list[list[NLIResult]],
    t_entail: float = 0.5,
    t_contradict: float = 0.5,
    tie_break: str = "max_prob",
) -> ClaimVerdict:
    """Combine window-level NLI results (grouped by passage) into one verdict."""
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {TIE_BREAKS}")
    flat = [(i, r) for i, results in enumerate(per_passage) for r in results]
    if not flat:
        raise ValueError("no NLI results to aggregate")
    ei, best_e = max(flat, key=lambda x: x[1].entailment)
    ci, best_c = max(flat, key=lambda x: x[1].contradiction)
    supported = best_e.entailment >= t_entail
    contradicted = best_c.contradiction >= t_contradict
    conflicting = supported and contradicted

    if conflicting:
        if tie_break == "max_prob":
            label = "supported" if best_e.entailment >= best_c.contradiction else "contradicted"
        else:
            label = tie_break
    elif supported:
        label = "supported"
    elif contradicted:
        label = "contradicted"
    else:
        label = "unsupported"

    return ClaimVerdict(
        label=label,
        p_entail=best_e.entailment,
        p_contradict=best_c.contradiction,
        conflicting=conflicting,
        entail_at=(ei, best_e.premise_window),
        contradict_at=(ci, best_c.premise_window),
    )


class Verifier:
    def __init__(
        self,
        nli: NLIModel,
        mode: str = "per_passage",
        t_entail: float = 0.5,
        t_contradict: float = 0.5,
        tie_break: str = "max_prob",
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.nli, self.mode = nli, mode
        self.t_entail, self.t_contradict, self.tie_break = t_entail, t_contradict, tie_break

    def config(self) -> dict:
        return {
            "nli_model": self.nli.name,
            "nli_revision": self.nli.revision,
            "device": self.nli.device,
            "fp16": self.nli.fp16,
            "mode": self.mode,
            "t_entail": self.t_entail,
            "t_contradict": self.t_contradict,
            "tie_break": self.tie_break,
        }

    def premises(self, passages: list[str]) -> list[str]:
        return passages if self.mode == "per_passage" else ["\n\n".join(passages)]

    def raw_scores(self, claims: list[str], passages: list[str]) -> list[list[list[NLIResult]]]:
        """NLI results for every claim x premise, batched in one call: [claim][premise][window].

        Kept separate from `verify` so thresholds can be re-applied without re-running NLI.
        """
        premises = self.premises(passages)
        pairs = [(p, c) for c in claims for p in premises]
        flat = self.nli.score_many(pairs)
        k = len(premises)
        return [flat[i * k : (i + 1) * k] for i in range(len(claims))]

    def verify(self, claims: list[str], passages: list[str]) -> list[ClaimVerdict]:
        return [
            aggregate(r, self.t_entail, self.t_contradict, self.tie_break)
            for r in self.raw_scores(claims, passages)
        ]
