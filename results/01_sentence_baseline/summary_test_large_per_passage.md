# Experiment 01: sentence baseline (test_large_per_passage)

Entailment threshold 0.25 (tuned on `scores_dev_large_per_passage.jsonl`, F1 0.373).

### All labels

| level | task | n | precision | recall | F1 |
|---|---|---|---|---|---|
| sentence | Data2txt | 8218 | 0.291 | 0.765 | 0.421 |
| sentence | QA | 5718 | 0.143 | 0.723 | 0.239 |
| sentence | Summary | 4946 | 0.384 | 0.275 | 0.321 |
| sentence | all | 18882 | 0.234 | 0.674 | 0.347 |
| response | Data2txt | 900 | 0.674 | 0.962 | 0.793 |
| response | QA | 900 | 0.257 | 0.931 | 0.403 |
| response | Summary | 900 | 0.538 | 0.412 | 0.467 |
| response | all | 2700 | 0.506 | 0.838 | 0.631 |

Gold type -> predicted label (gold-hallucinated sentences):

- baseless -> contradicted: 155
- baseless -> supported: 319
- baseless -> unsupported: 495
- both -> contradicted: 21
- both -> supported: 7
- both -> unsupported: 8
- conflict -> contradicted: 274
- conflict -> supported: 182
- conflict -> unsupported: 97

Sentences with conflicting evidence: 2033

### Excluding implicit_true labels

| level | task | n | precision | recall | F1 |
|---|---|---|---|---|---|
| sentence | Data2txt | 8218 | 0.279 | 0.770 | 0.410 |
| sentence | QA | 5718 | 0.125 | 0.744 | 0.214 |
| sentence | Summary | 4946 | 0.384 | 0.287 | 0.329 |
| sentence | all | 18882 | 0.220 | 0.682 | 0.333 |
| response | Data2txt | 900 | 0.657 | 0.965 | 0.782 |
| response | QA | 900 | 0.212 | 0.925 | 0.345 |
| response | Summary | 900 | 0.532 | 0.423 | 0.472 |
| response | all | 2700 | 0.480 | 0.840 | 0.611 |

Gold type -> predicted label (gold-hallucinated sentences):

- baseless -> contradicted: 147
- baseless -> supported: 272
- baseless -> unsupported: 443
- both -> contradicted: 21
- both -> supported: 7
- both -> unsupported: 8
- conflict -> contradicted: 274
- conflict -> supported: 182
- conflict -> unsupported: 96

Sentences with conflicting evidence: 2033
