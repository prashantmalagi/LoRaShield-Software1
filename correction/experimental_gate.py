"""Research-only conservative gate for offline Morse proposals.

This module is intentionally not imported by the receiver.  Seq2seq V1's
current DecodeResult exposes EOS/termination and emitted tokens, but does not
retain decoder probabilities; no confidence score is invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import morse_seq2seq_v1_data as data


@dataclass(frozen=True)
class InferenceEvidence:
    terminated_by_eos: bool
    termination_reason: str
    output_tokens: tuple[str, ...] | None = None
    # Optional future evidence. Values must be calibrated before thresholding.
    mean_token_probability: float | None = None
    minimum_token_probability: float | None = None
    mean_probability_margin: float | None = None


@dataclass(frozen=True)
class GatePolicy:
    max_symbol_edits: int
    max_boundary_edits: int
    min_mean_token_probability: float | None = None


@dataclass(frozen=True)
class GateDecision:
    output_morse: str
    accepted: bool
    reason: str


def _boundary_edits(left: str, right: str) -> int:
    # Compare ordered token streams after removing dot/dash tokens.
    def boundaries(value: str) -> tuple[str, ...]:
        return tuple("W" if token == "<WORD>" else "L"
                     for token in data.encode_morse(value)[1:-1]
                     if token in ("<WORD>", "<LETTER>"))
    a, b = boundaries(left), boundaries(right)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1]


def _symbol_edits(left: str, right: str) -> int:
    a = "".join(c for c in left if c in ".-")
    b = "".join(c for c in right if c in ".-")
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1]


def decide(original_morse: str, proposed_morse: str | None,
           evidence: InferenceEvidence, policy: GatePolicy) -> GateDecision:
    """Return the proposal only when syntax, termination and fixed policy pass."""
    def reject(reason: str) -> GateDecision:
        return GateDecision(original_morse, False, reason)

    try:
        data.encode_morse(original_morse)
    except (TypeError, ValueError):
        return reject("invalid_original")
    if not isinstance(evidence, InferenceEvidence):
        return reject("missing_evidence")
    if proposed_morse is None or not isinstance(proposed_morse, str):
        return reject("missing_proposal")
    try:
        data.encode_morse(proposed_morse)
    except (TypeError, ValueError):
        return reject("invalid_proposal")
    if evidence.output_tokens is not None:
        try:
            token_proposal = data.decode_morse(("<BOS>", *evidence.output_tokens, "<EOS>"))
        except (TypeError, ValueError):
            return reject("invalid_evidence_tokens")
        if token_proposal != proposed_morse:
            return reject("proposal_evidence_mismatch")
    if not evidence.terminated_by_eos or evidence.termination_reason != "eos":
        return reject("incomplete_decode")
    if policy.max_symbol_edits < 0 or policy.max_boundary_edits < 0:
        return reject("invalid_policy")
    if proposed_morse == original_morse:
        return GateDecision(original_morse, False, "no_change")
    if _symbol_edits(original_morse, proposed_morse) > policy.max_symbol_edits:
        return reject("symbol_edit_limit")
    if _boundary_edits(original_morse, proposed_morse) > policy.max_boundary_edits:
        return reject("boundary_edit_limit")
    threshold = policy.min_mean_token_probability
    if threshold is not None:
        score = evidence.mean_token_probability
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            return reject("uncalibrated_or_missing_score")
        if score < threshold:
            return reject("score_below_threshold")
    return GateDecision(proposed_morse, True, "accepted")
