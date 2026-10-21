"""Response -> claims.

`split_sentences` is the sentence-level baseline: every sentence is one claim. It keeps exact
character offsets, so claims align with RAGTruth's span labels without any fuzzy matching.
The LLM-based decomposer (Llama 3.1 via Ollama) comes next and is compared against it.
"""

from __future__ import annotations

import re

from src.claims.schema import Claim

# A sentence ends at . ! or ? followed by whitespace and an uppercase letter, digit, quote,
# bracket or list marker. Lines are split first, so bullets and numbered steps stay separate.
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[*\-•])")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")
# Don't end a sentence after a list number ("1."), a single initial ("W."), or a common
# abbreviation ("U.S.", "Dr.", "St.").
_NO_BREAK_AFTER = re.compile(
    r"(?:^|\s)(?:\d{1,2}|[A-Z]|(?:[A-Z]\.)+[A-Z]|Dr|Mr|Mrs|Ms|St|Jr|Sr|vs|etc|e\.g|i\.e|No|Inc|Ltd|Co)"
    r"\.[\"')\]]*$"
)


def split_sentences(text: str) -> list[Claim]:
    claims = []
    pos = 0
    for line in text.split("\n"):
        start = 0
        for m in _SENT_END.finditer(line):
            if _NO_BREAK_AFTER.search(line[start : m.start()]):
                continue
            claims.append((pos + start, pos + m.start()))
            start = m.end()
        claims.append((pos + start, pos + len(line)))
        pos += len(line) + 1
    out = []
    for a, b in claims:
        seg = text[a:b]
        # trim whitespace but keep offsets exact
        a += len(seg) - len(seg.lstrip())
        b -= len(seg) - len(seg.rstrip())
        if b > a and _HAS_WORD.search(text[a:b]):
            out.append(Claim(text=text[a:b], span=(a, b)))
    return out
