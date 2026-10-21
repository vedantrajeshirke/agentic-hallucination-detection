"""Tests for src/verify/nli.py. Loads the base model (cached after first download)."""

import itertools

import pytest

from src.verify.nli import MAX_LENGTH, NLIModel


@pytest.fixture(scope="module")
def model():
    return NLIModel("base", device="cpu")


def test_short_premise_is_one_window(model):
    assert model.premise_windows("The sky is blue.", "The sky is blue.") == [(0, 16)]


def test_windows_fit_and_cover_long_premise(model):
    premise = " ".join(f"Sentence number {i} mentions item {i * 7}." for i in range(400))
    hypothesis = "Sentence number 399 mentions item 2793."
    windows = model.premise_windows(premise, hypothesis)
    assert len(windows) > 1
    # every window, paired with the hypothesis, fits the model
    for a, b in windows:
        assert len(model.tokenizer(premise[a:b], hypothesis)["input_ids"]) <= MAX_LENGTH
    # windows start at the beginning, end at the end, and overlap (no gaps)
    assert windows[0][0] == 0
    assert windows[-1][1] == len(premise.rstrip())
    for (_, prev_end), (next_start, _) in itertools.pairwise(windows):
        assert next_start < prev_end


def test_label_order_read_from_config(model):
    result = model.score("The ambience is not intimate.", "The bar is intimate.")[0]
    assert result.label == "contradiction"
