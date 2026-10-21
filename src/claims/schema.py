from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Claim:
    text: str
    span: tuple[int, int] | None  # character offsets in the response, if the claim maps to one
    step_idx: int | None = None  # agent step that produced it (Phase 3); None for single-step
    source_ids: tuple[str, ...] = ()  # passages the generator cited, if any
