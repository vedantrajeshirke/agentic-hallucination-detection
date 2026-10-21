"""Response -> claims.

`split_sentences` is the sentence-level baseline: every sentence is one claim. It keeps exact
character offsets, so claims align with RAGTruth's span labels without any fuzzy matching.

`decompose_response` splits each of those sentences into atomic, self-contained claims with
Llama 3.1 (via Ollama). Claims stay attached to their source sentence, so evaluation is still
at sentence level against the same labels: a sentence is flagged if any of its claims is, and a
sentence with no claims (greetings, headers, citations) is never flagged.
"""

from __future__ import annotations

import json
import re

from src.claims.schema import Claim

# A sentence ends at . ! or ? followed by whitespace and an uppercase letter, digit, quote,
# bracket or list marker. Lines are split first, so bullets and numbered steps stay separate.
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[*\-•])")
_HAS_WORD = re.compile(r"[A-Za-z0-9]")
# Don't end a sentence after a list number ("1."), a single initial ("W."), or a common
# abbreviation ("U.S.", "Dr.", "St.").
_NO_BREAK_AFTER = re.compile(
    r"(?:^|\s)(?:\d{1,2}|[A-Z]|(?:[A-Z]\.)+[A-Z]|Dr|Mr|Mrs|Ms|St|Jr|Sr|vs|etc|e\.g|i\.e|No|Inc|Ltd|Co"
    r"|Rep|Sen|Gov|Gen|Lt|Col|Capt|Sgt|Prof|Rev|Hon|Pres|Mt|Ft|Ave|Blvd|Dept|Univ|approx"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
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


DECOMPOSE_MODEL = "llama3.1:8b"
DECOMPOSE_PROMPT = """You split an AI assistant's answer into atomic factual claims for fact-checking.

The answer is given as sentences with IDs (S1, S2, ...). For EACH sentence ID, list the claims
that sentence makes. IDs are labels only; the answer may contain its own numbering.

Rules:
1. One fact per claim. Split compound sentences, lists, comparisons and cause-effect
   ("X, leading to Y") into separate claims.
2. Each claim must be a complete, standalone sentence. Replace pronouns and vague references
   ("it", "they", "she", "the bar", "the company") with the specific name from the answer.
3. Keep the meaning exactly. Do not add, infer, correct, strengthen or weaken anything. Keep
   numbers, dates, names, quantities, qualifiers and hedges ("may", "about", "according to X",
   "customers say") as written. When splitting a list, keep the original verb with every item.
4. Do not drop anything: every fact in the sentence must appear in some claim.
5. Instructions or steps stay as instructions, e.g. "Preheat the oven to 350 degrees."
6. Return an empty list for sentences with no factual claim: greetings ("Sure!"), introductions
   ("Here is a summary:", "Based on the passages, ..."), headers ("Passage 2:"), bare citations
   ("(Passage 1)"), offers to help, refusals, and comments about the answer or the passages.

Example
Answer sentences:
S1: Sure! Here is an overview of Joe's Diner:
S2: 1. Atmosphere: The diner offers a cozy atmosphere and free WiFi, and customers praise its friendly staff.
S3: 2. Hours: Its hours were extended, leading to longer shifts than at nearby cafes or bakeries.
S4: Workers there suffered from fatigue, injuries, and layoffs.
Output:
{{"sentences": [
 {{"id": "S1", "claims": []}},
 {{"id": "S2", "claims": ["Joe's Diner offers a cozy atmosphere.", "Joe's Diner offers free WiFi.", "Customers praise the friendly staff at Joe's Diner."]}},
 {{"id": "S3", "claims": ["Joe's Diner's hours were extended.", "The extended hours at Joe's Diner led to longer shifts.", "Shifts at Joe's Diner are longer than at nearby cafes.", "Shifts at Joe's Diner are longer than at nearby bakeries."]}},
 {{"id": "S4", "claims": ["Workers at Joe's Diner suffered from fatigue.", "Workers at Joe's Diner suffered from injuries.", "Workers at Joe's Diner suffered from layoffs."]}}
]}}
{question}
Answer sentences:
{sentences}"""

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claims": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "claims"],
            },
        }
    },
    "required": ["sentences"],
}


def decompose_response(
    sentences: list[str],
    question: str | None = None,
    model: str = DECOMPOSE_MODEL,
    client=None,
    seed: int = 0,
) -> list[list[str]]:
    """Atomic claims for each sentence, in order. Sentences the model skips get []."""
    import ollama

    client = client or ollama.Client()
    numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sentences, 1))
    prompt = DECOMPOSE_PROMPT.format(
        question=f"\nThe answer responds to this question: {question}\n" if question else "",
        sentences=numbered,
    )
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=DECOMPOSE_SCHEMA,
        options={"temperature": 0, "seed": seed, "num_ctx": 8192},
    )
    data = json.loads(resp.message.content)
    by_id = {str(item["id"]).strip().upper(): [c.strip() for c in item["claims"] if c.strip()] for item in data["sentences"]}
    return [by_id.get(f"S{i}", []) for i in range(1, len(sentences) + 1)]
