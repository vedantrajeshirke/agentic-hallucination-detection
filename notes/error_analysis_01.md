# Error analysis: experiment 01 (sentence-level baseline, RAGTruth test)

Sample: `notes/error_analysis_01_samples.md`, drawn by `experiments/01b_error_sample.py` (seed 0)
from the test predictions at the tuned threshold (t_entail 0.25). 60 false alarms (20 per task) and
30 misses (10 per task). First-pass judgements by Claude, each checked against the source text
(full passages / article searches, not truncated views). **Spot-check before citing.**

## Headline

1. **About a third of false alarms are real hallucinations the labels missed** (11/60 clear,
   21/60 including borderline). Corrected sentence precision is roughly **0.37** (clear only) to
   **0.51** (incl. borderline), vs measured 0.234. Uncertainty is wide (n=20 per task): the range
   from per-task 95% Wilson intervals is 0.29–0.53 (clear) and 0.39–0.66 (incl. borderline).
2. **The rest of the false alarms are mostly fixable checker problems, not NLI limits:**
   non-claims (20%), off-topic passages "contradicting" supported claims (22%), evidence split
   across passages (8%), NLI failing on list/imperative formatting (10%).
3. **Most misses are a wrong detail inside a mostly-correct sentence** (11/30), which is the
   case for claim splitting. 6/30 are NLI over-crediting wrong numbers, tenses or entities.
4. Three defects in my own pipeline surfaced: the sentence splitter breaks on "Rep.", review
   passages say "A 5-star customer review", which contradicts claims about the overall rating,
   and gold alignment counted a 1-character (punctuation) overlap as a label (3/1,558 sentences,
   negligible). None of these are fixed yet: they change the sentences or the scoring, so they go
   into the next experiment rather than silently altering the reported baseline.

## False alarms (60)

| Category | QA | Summary | Data2txt | Total |
|---|---|---|---|---|
| **Real hallucination, unlabeled (clear)** | 2 | 4 | 5 | **11** |
| Real but borderline (embellishment, promotional "popular"/"cozy", evaluative synthesis) | 1 | 3 | 6 | 10 |
| Non-claim ("Sure!", "Passage 3:", "(Passage 2)", sign-offs, refusal, meta-comments) | 6 | 6 | 0 | **12** |
| Supported, but another passage/window "contradicts" it (tie-break → contradicted) | 5 | 3 | 5 | **13** |
| Supported only by combining passages | 2 | 0 | 3 | 5 |
| Supported, NLI fails on format (imperative steps, list numbers, citations, category lists, verbatim copy) | 4 | 1 | 1 | 6 |
| Supported, context-dependent (pronoun "Their", splitter fragment, near-threshold inference) | 0 | 3 | 0 | 3 |
| **Total** | 20 | 20 | 20 | 60 |

Corrected precision = (TP + Σ_task FP_task × real-rate_task) / flagged, with TP 1,050, flagged 4,487
and FP populations QA 1,577 / Summary 114 / Data2txt 1,746.

| Task | Real (clear) | 95% CI | Real (incl. borderline) | 95% CI |
|---|---|---|---|---|
| QA | 2/20 | 3–30% | 3/20 | 5–36% |
| Summary | 4/20 | 8–42% | 7/20 | 18–57% |
| Data2txt | 5/20 | 11–47% | 11/20 | 34–74% |

Recall is not corrected: unlabeled hallucinations the checker *also* missed are invisible without
labeling unflagged sentences.

## Misses (30)

| Category | QA | Summary | Data2txt | Total |
|---|---|---|---|---|
| **Wrong detail inside a mostly-correct sentence** | 3 | 2 | 6 | **11** |
| NLI over-credits wrong content (numbers, tense, entity, world knowledge) | 2 | 2 | 2 | 6 |
| Unsupported claim with entailment between 0.25 and 0.5 (threshold) | 3 | 2 | 1 | 6 |
| Conflicting evidence, tie-break → supported | 0 | 1 | 1 | 2 |
| Label questionable (over-broad span, true weakening, 1-char overlap, unclear) | 2 | 3 | 0 | 5 |
| **Total** | 10 | 10 | 10 | 30 |

## What to fix, by evidence

| Fix | Addresses | Share of sampled errors | Cost |
|---|---|---|---|
| Claim splitting (Llama), dropping non-claims, rewriting steps as declaratives | sub-sentence misses, non-claims, format failures | 11/30 misses; 12+6/60 FAs | next experiment |
| Prefer entailment when a relevant passage entails (tie_break="supported"), or count contradictions only from passages relevant to the claim | off-topic contradictions | 13/60 FAs (but 2/30 misses go the other way) | rule change, tunable on dev offline |
| Review passages without star counts ("A customer review from …") | Data2txt rating contradictions (FA-D2T-02, -09, -14) | 3/20 Data2txt FAs | re-run Data2txt NLI |
| Splitter abbreviations (Rep., Sen., Gov., …); alphanumeric-overlap gold rule | splitter fragments; alignment artifact | 1/60 FA; 3 test sentences | trivial |
| Concatenated-passage check as a second signal | multi-passage support | 5/60 FAs | extra NLI pass |

The tie-break evidence points both ways (13 FAs from contradiction winning, 2 misses from
entailment winning), so it should be tuned on dev, not decided from this sample.

## Per-case judgements

Format: tag · category · note.

**QA false alarms**
- FA-QA-01 · multi-passage · "take it out" (P1) + seasonings list (P2)
- FA-QA-02 · off-topic contradiction · P2 supports (float in boiling water); P3 says simmering
- FA-QA-03 · format · list-item fragment ("Creating unique textures…"), supported by P3
- FA-QA-04 · real (clear) · claims P3 gives no cooking instructions; P3 has a bake step
- FA-QA-05 · format · verbatim copy of P1 step 5, entailment 0.05
- FA-QA-06 · non-claim · "(Passage 2)"
- FA-QA-07 · real (borderline) · applies HHR rear-door steps to "a 2006 Chevy" (P2 is Impala 06–12)
- FA-QA-08 · non-claim · sign-off
- FA-QA-09 · format · near-verbatim P2 step, step-number/citation noise, contradiction 0.98
- FA-QA-10 · non-claim · "Passage 3:"
- FA-QA-11 · off-topic contradiction · P1 entails 0.83; P2's different "1." item contradicts
- FA-QA-12 · real (clear) · "published by him along with Bob": source says "Published by." (blank)
- FA-QA-13 · format · imperative paraphrase of P2 ("may use graph paper")
- FA-QA-14 · non-claim · incorrect refusal ("Unable to answer…"); an error, not a hallucination span
- FA-QA-15 · off-topic contradiction · P2 entails 0.85; P3's different procedure contradicts
- FA-QA-16 · multi-passage · House Seasoning (P1) or salt+pepper (P2)
- FA-QA-17 · off-topic contradiction · P2 entails 0.94; P3 contradicts
- FA-QA-18 · non-claim · "(Passage 2)"
- FA-QA-19 · off-topic contradiction · P1 step 4 verbatim; P2 contradicts
- FA-QA-20 · non-claim · meta-comment ("it does not explain…"), true

**Summary false alarms**
- FA-Summary-01/02/03/11/14/16 · non-claim · "Sure!"
- FA-Summary-04 · context (splitter) · fragment after "Rep." split; content supported
- FA-Summary-05 · context (pronoun) · "Their sentences will run concurrently", supported
- FA-Summary-06 · real (clear) · reverses who sets the precondition (Houthis demand a halt), and the order
- FA-Summary-07 · off-topic contradiction · supported (life without parole, max-security)
- FA-Summary-08 · off-topic contradiction · verbatim ("declined to comment"); other window says the lawyer spoke
- FA-Summary-09 · off-topic contradiction · Reba + Garth both supported
- FA-Summary-10 · context (threshold) · Survivor 45 rule supported; entailment 0.23
- FA-Summary-12 · real (clear) · "law enforcement agencies applauded": not in article
- FA-Summary-13 · real (borderline) · "may not go as planned": speculation
- FA-Summary-15 · real (borderline) · "information will remain classified" vs unclassified summary provided
- FA-Summary-17 · real (clear) · "according to an article in the Hill": the Hill is never mentioned
- FA-Summary-18 · format · five names verbatim, contradiction 0.95 (pure NLI failure)
- FA-Summary-19 · real (borderline) · father "in critical condition": only the boys are
- FA-Summary-20 · real (clear) · "analysts predict Ramadi may soon fall"; official says "not imminent"

**Data2txt false alarms**
- FA-D2T-01 · multi-passage · coffee (R3) + service/ambiance/Miracle Whip (R4)
- FA-D2T-02 · off-topic contradiction · 4.0 stars entailed 1.00; "A 5-star customer review" contradicts (linearization flaw)
- FA-D2T-03 · real (borderline) · "their goal is to make the process stress-free": promotional
- FA-D2T-04 · real (borderline) · "popular… in the heart of Santa Barbara"
- FA-D2T-05 · real (clear) · "generous portion sizes": no review mentions portions; R3 says "barely anything"
- FA-D2T-06 · real (borderline) · "cozy… for locals and visitors alike"
- FA-D2T-07 · real (borderline) · "ample parking": only a lot
- FA-D2T-08 · real (clear) · praise for staff helpfulness/atmosphere not in reviews (food poisoning, glass)
- FA-D2T-09 · off-topic contradiction · hours + 4.0 stars entailed; review star counts contradict
- FA-D2T-10 · off-topic contradiction · R2 "fast, efficient, pleasant"; R4 complaints contradict
- FA-D2T-11 · multi-passage · fries + corn dog (R3), malt (R4)
- FA-D2T-12 · real (borderline) · "popular Mexican restaurant"
- FA-D2T-13 · format · category list paraphrase (Bubble Tea, Food Stands, Sushi)
- FA-D2T-14 · off-topic contradiction · 3.5 stars entailed; "A 5-star customer review" contradicts
- FA-D2T-15 · real (clear) · "majority agree it's worth the cost": R3 says "completely overpriced… not worth $24"
- FA-D2T-16 · real (borderline) · "popular spot… may require improvement": evaluative synthesis
- FA-D2T-17 · real (clear) · "great service": no review mentions service
- FA-D2T-18 · multi-passage · tasty (R3) + fresh tasty (R4)
- FA-D2T-19 · off-topic contradiction · Warren praised (R3); same review calls service substandard
- FA-D2T-20 · real (clear) · "lively atmosphere": R2 says "always borderline dead"

**Misses**
- M-QA-01 · detail · Mississippi lowest-pay figures unsupported; Alaska part supported (also conflicting)
- M-QA-02 · threshold · invented example, entailment 0.27
- M-QA-03 · NLI over-credit · "between eighth and ninth day" vs "after the ninth day" (numbers)
- M-QA-04 · label artifact · 1-character (".") overlap with the next sentence's span
- M-QA-05 · detail · "(or until desired doneness)" implicit_true, entailment 0.44
- M-QA-06 · threshold · unsupported conclusion, entailment 0.33
- M-QA-07 · detail · invented phone number inside a supported step
- M-QA-08 · label questionable · passages discuss cycling; unclear why "no mention"
- M-QA-09 · NLI over-credit · world-knowledge filler (protein synthesis etc.) not in passages, entailment 0.86
- M-QA-10 · threshold · implicit_true tip, entailment 0.47
- M-Summary-01 · threshold · implicit_true, entailment 0.45
- M-Summary-02 · label questionable · span covers several sentences; this one is supported ("afraid to come forward… accused of being a liar")
- M-Summary-03 · label questionable · same over-broad span; "sparked a conversation" is supported
- M-Summary-04 · NLI over-credit · "guilty of witnessing" vs first-degree murder, entailment 0.42
- M-Summary-05 · detail · "firearms" vs "a firearm"
- M-Summary-06 · NLI over-credit · Cheryl is a logic problem, not a student (entity)
- M-Summary-07 · threshold · invented defense argument, entailment 0.43
- M-Summary-08 · label questionable · "free April 18 and 19" is true if free April 18–26 (weakening)
- M-Summary-09 · detail · "just days later" invented timing
- M-Summary-10 · conflicting evidence · "imperative" clause contradicted (0.53) but rest entailed 1.00
- M-D2T-01 · detail · "excellent takeout options" praise invented; good food supported
- M-D2T-02 · detail · "rooftop lounge" inside a list
- M-D2T-03 · NLI over-credit · "has closed" vs "is closing" (tense)
- M-D2T-04 · detail · WiFi is null; the parking half is supported
- M-D2T-05 · threshold · invented praise, entailment 0.47
- M-D2T-06 · detail · "overcooked" invented; "lacking in quality" supported
- M-D2T-07 · NLI over-credit · miscommunication details mismatched
- M-D2T-08 · detail · "good balance of ingredients" invented; burger praise supported
- M-D2T-09 · detail · "He": gender assumed (1 word)
- M-D2T-10 · conflicting evidence · structured data says not casual/classy; a review entails it
