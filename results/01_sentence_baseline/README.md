# Experiment 01: sentence-level baseline

Each response sentence is one claim; DeBERTa-v3-large NLI checks it against the example's evidence
passages (per passage, max over 512-token windows). A sentence is flagged if no passage entails it
above the threshold. Script: `experiments/01_sentence_baseline.py`.

## Run

| | |
|---|---|
| Code | commit `a7d39dc` (pinned via the Colab notebook `REF`; these configs predate the `code_commit` field) |
| Data | RAGTruth commit `c103204`; dev = 1,506 responses (10% of train sources, seed 0); test = 2,700 |
| Model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` @ `b3546ea`, fp16 |
| Hardware | Google Colab T4, torch 2.14.1+cu130 (2026-10-21) |
| Threshold | entailment 0.25, tuned on dev for overall sentence F1; contradiction 0.5; tie-break max_prob |

The `t_entail: 0.5` in the config files is the run-time default and does not affect the stored
scores; the threshold actually used is the tuned one in `summary_test_large_per_passage.json`.

## Results (test)

| level | precision | recall | F1 |
|---|---|---|---|
| response | 0.506 | 0.838 | **0.631** |
| sentence | 0.234 | 0.674 | **0.347** |

Per task (response F1): Data2txt 0.793, Summary 0.467, QA 0.403. Full tables, the type confusion,
and the variant excluding `implicit_true` labels are in `summary_test_large_per_passage.md`.

## Caveats

- Precision is measured against labels that miss hallucinations (see `notes/ragtruth_inspection.md`),
  so it is likely understated; by how much is not yet measured.
- Sentence-level scores are not comparable to RAGTruth's published span-level scores; compare at
  response level only.
- Single run, single seed; threshold tuned for overall sentence F1, not per task.

Raw per-sentence scores (`scores_*.jsonl`) are kept out of git; they are in Google Drive under
`agentic-hallucination/results/01_sentence_baseline/`.
