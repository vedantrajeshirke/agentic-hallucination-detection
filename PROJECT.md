# Hallucination Detection and Prevention in Agentic AI Systems

Personal research project.
Status: implementation starting. Deliverable cadence: written progress update every two weeks.

---

## 1. The problem

A language model produces text that fits the pattern of a good answer rather than consulting a fact
it holds. A correct statement and an invented one are built the same way and arrive with the same
confidence, so nothing in the output marks which is which. The cost is not that the system is
sometimes wrong, it is that the reader has no way to tell *where*.

Agentic systems sharpen this. An agent searches, reads, decides whether it has enough, and searches
again across many steps. An unsupported claim made at step 2 does not stay one bad sentence — it
becomes a premise the remaining steps build on, so the agent spends its effort elaborating a
conclusion that was never grounded, and the result looks *more* credible for being more thorough.

**Framing:** this is treated as a verification problem, not a generation problem. Check each claim
against the passages actually retrieved, at the step where the claim is made rather than at the end
of the run, and feed the result back into the agent loop.

**Thesis to defend:** most existing work (RAGTruth, LettuceDetect, Vectara HHEM) evaluates only the
final answer and leaves intermediate agent steps unchecked. Step-level verification catches an error
while it is still one step instead of after it has shaped ten.

---

## 2. Objectives

1. **Verification.** Decompose answers into atomic claims; check each against retrieved passages
   using a lightweight NLI model. Validate on RAGTruth.
2. **Detection comparison.** Benchmark five signals against each other, on both final answers and
   intermediate steps: self-critique, token-level uncertainty, semantic entropy, NLI-based evidence
   checking, LLM-as-judge.
3. **Prevention.** Feed the detection signal back into the agent loop to trigger re-search,
   backtracking, or abstention. Measure the tradeoff between fewer false claims and lost usefulness.

The headline result should be a curve, not a number: how far unsupported claims can be reduced
before the system becomes too cautious to be useful.

---

## 3. Stack

| Component | Choice | Notes |
|---|---|---|
| LLM runtime | Ollama | local, no API cost |
| Generator model | Llama 3.1 8B | agent + answer generation |
| Embeddings | `nomic-embed-text` | via Ollama |
| Entailment | DeBERTa-v3-MNLI | claim vs. passage, the core verifier |
| Vector store | ChromaDB | local persistence |
| Language | Python 3.11+ | |

Everything runs locally. No paid API dependency — keep it that way, it matters for reproducibility
and for the cost section of any write-up.

---

## 4. Datasets

- **RAGTruth** — word-level hallucination annotations across QA, summarization, data-to-text.
  Primary validation set for Phase 1.
- **HotpotQA** — multi-hop QA with supporting-fact annotations. Source of agent trajectories.
- **MuSiQue** — multi-hop QA, harder compositional questions, also with supporting facts.
- **2WikiMultiHop** — secondary, if time allows.

The multi-hop sets matter because they mark *which* context sentences are needed for the answer.
That is what makes step-level labelling possible: without it you can only check the final answer.

> **To verify in code, do not assume:** exact split sizes, label schema, and field names for each
> dataset. Load them and print the schema before writing anything that depends on it.

---

## 5. Proposed repo layout

```
.
├── PROJECT.md              # this file
├── pyproject.toml
├── src/
│   ├── claims/             # answer -> atomic claims
│   │   ├── decompose.py
│   │   └── schema.py       # Claim dataclass: text, span, step_idx, source_ids
│   ├── verify/
│   │   ├── nli.py          # DeBERTa entailment wrapper
│   │   └── scorer.py       # claim + passages -> {supported, contradicted, unsupported} + conf
│   ├── signals/            # Phase 2, one module per detection signal
│   │   ├── self_critique.py
│   │   ├── token_uncertainty.py
│   │   ├── semantic_entropy.py
│   │   ├── nli_evidence.py
│   │   └── llm_judge.py
│   ├── agent/              # Phase 3
│   │   ├── loop.py         # search -> read -> decide -> search
│   │   ├── intervene.py    # re-search / backtrack / abstain policy
│   │   └── trace.py        # structured per-step trace for analysis
│   ├── data/
│   │   ├── ragtruth.py
│   │   └── multihop.py
│   └── eval/
│       ├── metrics.py
│       └── report.py
├── experiments/            # numbered, each writes to results/
├── results/                # JSON + plots, gitignored except summaries
└── tests/
```

Keep every experiment reproducible: fixed seeds, config logged alongside results, no notebook-only
findings that cannot be re-run.

---

## 6. Phase 1 — claim-level verifier (current)

Build for **single-step retrieval first**, validated against RAGTruth, to establish a measured
baseline before the agentic layer exists. Without this baseline the step-level numbers later have
nothing to be compared against.

Order of work:

1. **Load RAGTruth.** Parse into a clean internal format. Print label distribution and inspect 10
   annotated examples by hand before writing anything else — the annotation granularity determines
   how claim decomposition has to work.
2. **Claim decomposition.** Split a response into atomic, independently checkable claims. Start with
   a prompt-based decomposer on Llama 3.1 8B. Known failure modes to watch: pronouns losing their
   referent, and compound sentences that should split but do not.
3. **NLI scoring.** For each (claim, retrieved passage) pair, run DeBERTa-v3-MNLI. Aggregate across
   passages — a claim is supported if *any* passage entails it, contradicted if any contradicts it,
   otherwise unsupported. Tune the entailment threshold on a dev split, never on test.
4. **Align to RAGTruth labels.** Map word-level spans onto the decomposed claims so predictions can
   be scored. This mapping is the fiddly part; expect it to need care.
5. **Baseline numbers.** Precision, recall, F1 at claim level. Also report the confusion between
   *contradicted* and *unsupported*, because conflating them is the usual way these systems look
   better than they are.

**Definition of done for Phase 1:** a reproducible script that takes RAGTruth, runs the verifier,
and emits a metrics table — plus a short written error analysis of where it fails.

---

## 7. Phases 2 and 3 (later, do not start early)

**Phase 2 — signal comparison.** Each signal implements the same interface so they can be swapped:
`score(claim, context, generation_state) -> float`. Evaluate on final answers *and* on intermediate
steps. The interesting question is whether signal rankings change between those two settings — if
they do, that is a result worth reporting on its own.

**Phase 3 — intervention.** The agent loop gets a check after each step. On an unsupported claim,
choose: re-search with a reformulated query, backtrack to the last supported state, or abstain.
Metrics must include a usefulness cost, not just a hallucination reduction: abstention rate, task
success rate, extra steps per question, wall-clock overhead. A system that refuses everything scores
perfectly on hallucination and is worthless.

---

## 8. Metrics

- **Claim level:** precision / recall / F1 for unsupported-claim detection.
- **Step level:** detection latency — how many steps pass between an unsupported claim appearing and
  being caught.
- **Run level:** task accuracy, abstention rate, mean steps per question, overhead vs. unverified
  baseline.
- **Tradeoff curve:** hallucination rate against task success as the intervention threshold sweeps.
  This is the main figure.

Always report the unverified baseline alongside. A number without a baseline says nothing.

---

## 9. Working agreements for the agent

- Verify dataset schemas by loading them; do not write code against assumed field names.
- Prefer small runnable scripts over notebooks for anything that produces a reported number.
- Every result lands in `results/` with the config that produced it.
- No fabricated numbers in any document, ever. Given the subject of this project, that would be
  somewhat embarrassing.
- Flag uncertainty explicitly rather than smoothing over it.

---

## 10. Start here

```
1. Scaffold the repo layout above, pyproject with the stack pinned.
2. src/data/ragtruth.py — download, parse, print schema + label distribution.
3. Hand-inspect 10 examples, write findings into a short notes file.
4. src/verify/nli.py — DeBERTa wrapper, smoke-test on 5 hand-written claim/passage pairs.
5. Only then: src/claims/decompose.py.
```

Step 3 is not optional. The annotation format decides the design of steps 4 and 5.
