"""Offline selective-correction research experiment for the saved V3 model.

This script calibrates a conservative abstention gate on deterministic synthetic
dot/dash-substitution data, then evaluates the frozen policy once on disjoint
locked-test sources. References exist only in offline records/scoring; proposal
generation and runtime_gate receive no reference fields. Unknown channel
profiles abstain. The experiment does not read real-capture data, train a model,
write results, or establish real radio-channel performance.

The V3 head emits one token per input position. This experiment only permits
same-length dot/dash substitutions with unchanged boundaries and valid Morse
groups. Insertions, deletions, boundary edits, unsupported input, and overlength
words are out of scope and abstain. A channel profile is an explicit trust
assumption, not an automatically inferred corruption label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
SEED = 4819207
CALIBRATION_MESSAGES = 300
LOCKED_TEST_MESSAGES = 300
SEQUENCE_LENGTH = 120
VOCABULARY = ["<PAD>", "<UNK>", ".", "-", " "]
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)
NOISE_LEVELS = (0, 5, 10, 15, 20)
SUBSTITUTION_TRAINING_HARM_CAP = 0.05
MIN_ACCEPTED_FOR_INFERENCE = 300
MAX_HARM_RATE_FOR_PASS = 0.01
MIN_RELATIVE_ERROR_REDUCTION_FOR_PASS = 0.10
Z_95 = 1.959963984540054

from evaluate_morse_correction_v3 import MORSE  # noqa: E402
from evaluate_morse_correction_v3_independent import _old_partition_messages  # noqa: E402
from evaluate_morse_correction_v3_synthetic import (  # noqa: E402
    DEFAULT_MESSAGES as PREVIOUS_SYNTHETIC_COUNT,
    SEED as PREVIOUS_SYNTHETIC_SEED,
    generate_source_corpus as generate_previous_synthetic_corpus,
)
from evaluate_morse_correction_v3_corruption import (  # noqa: E402
    DEFAULT_MESSAGES as PREVIOUS_CORRUPTION_COUNT,
    SEED as PREVIOUS_CORRUPTION_SEED,
    generate_source_corpus as generate_previous_corruption_corpus,
)


@dataclass(frozen=True)
class OfflineCase:
    sample_id: str
    source_text: str
    reference_morse: str
    received_morse: str
    noise_percent: int
    corruption_type: str


@dataclass(frozen=True)
class TokenChange:
    word_index: int
    position_in_word: int
    original_token: str
    proposed_token: str
    original_probability: float
    proposed_probability: float
    probability_margin: float


@dataclass(frozen=True)
class ProposalFeatures:
    """Only runtime-available input/proposal features; deliberately no references."""
    received_morse: str
    proposed_morse: str
    changes: tuple[TokenChange, ...]
    minimum_changed_token_margin: float
    mean_changed_token_margin: float
    maximum_changed_token_margin: float
    candidate_morse_consistent: bool
    same_character_length: bool
    same_letter_boundaries: bool
    same_word_boundaries: bool


@dataclass(frozen=True)
class GatePolicy:
    minimum_changed_token_margin: float
    mean_changed_token_margin: float
    maximum_changes: int
    require_consistent_candidate: bool = True
    allow_channel_profiles: tuple[str, ...] = ("clean", "dot_dash_substitution")
    selected_from: str = "calibration_only"


@dataclass(frozen=True)
class GateDecision:
    received_morse: str
    output_morse: str
    accepted: bool
    reason: str
    features: ProposalFeatures | None = None


@dataclass(frozen=True)
class CalibrationRecord:
    features: ProposalFeatures
    reference_morse: str
    noise_percent: int
    corruption_type: str


@lru_cache(maxsize=1)
def prior_source_words() -> frozenset[str]:
    """Recreate prior source partitions/corpora solely to exclude exact word overlap."""
    previous = set(_old_partition_messages())
    synth_sources = generate_previous_synthetic_corpus(
        PREVIOUS_SYNTHETIC_SEED, PREVIOUS_SYNTHETIC_COUNT
    )
    corruption_sources = generate_previous_corruption_corpus(
        PREVIOUS_CORRUPTION_SEED, PREVIOUS_CORRUPTION_COUNT
    )
    previous.update(word for source in synth_sources + corruption_sources for word in source.split())
    return frozenset(previous)


def encode_alphanumeric(text: str) -> str:
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError("source text must be nonempty without edge whitespace")
    words = text.split(" ")
    if any(not word for word in words):
        raise ValueError("source words must use single spaces")
    if any(char not in MORSE for word in words for char in word):
        raise ValueError("source text supports A-Z and 0-9 only")
    return " / ".join(" ".join(MORSE[char] for char in word) for word in words)


def split_words(morse: str) -> list[str]:
    if not isinstance(morse, str) or not morse:
        raise ValueError("Morse input must be nonempty")
    if morse != morse.strip():
        raise ValueError("leading/trailing whitespace is invalid")
    if ":" in morse:
        raise ValueError("ciphertext-shaped input is unsupported")
    if any(char not in ".- /" for char in morse):
        raise ValueError("unsupported input character")
    words = morse.split(" / ")
    if any(not word or "/" in word for word in words):
        raise ValueError("word boundary must be exactly ' / ' with nonempty words")
    for word in words:
        if word.startswith(" ") or word.endswith(" ") or "  " in word:
            raise ValueError("letter boundaries must be single spaces")
        if len(word) > SEQUENCE_LENGTH:
            raise ValueError(f"word length {len(word)} exceeds {SEQUENCE_LENGTH}; no truncation")
    return words


def generate_source_partition(seed: int, count: int, used_words: set[str] | None = None) -> list[str]:
    if count < 1:
        raise ValueError("partition size must be positive")
    rng = random.Random(seed)
    used = set(used_words or ())
    alphabet = tuple(MORSE)
    sources: list[str] = []
    seen_sources: set[str] = set()
    while len(sources) < count:
        words = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 5)))
            for _ in range(rng.randint(2, 3))
        ]
        text = " ".join(words)
        if text in seen_sources or set(words) & used:
            continue
        reference = encode_alphanumeric(text)
        if any(len(word) > SEQUENCE_LENGTH for word in split_words(reference)):
            raise AssertionError("generated source exceeds model word limit")
        sources.append(text)
        seen_sources.add(text)
        used.update(words)
    return sources


def generate_partitions(seed: int = SEED, calibration_count: int = CALIBRATION_MESSAGES,
                        locked_count: int = LOCKED_TEST_MESSAGES) -> tuple[list[str], list[str]]:
    """Partition source messages first; only callers then generate noisy variants."""
    used = set(prior_source_words())
    calibration = generate_source_partition(seed + 11, calibration_count, used)
    used.update(word for source in calibration for word in source.split())
    locked = generate_source_partition(seed + 29, locked_count, used)
    cal_words = {word for source in calibration for word in source.split()}
    locked_words = {word for source in locked for word in source.split()}
    if cal_words & locked_words or cal_words & prior_source_words() or locked_words & prior_source_words():
        raise AssertionError("calibration/locked/prior source words overlap")
    return calibration, locked


def flip_dot_dash_noise(morse: str, percentage: int, rng: random.Random) -> str:
    if percentage not in NOISE_LEVELS:
        raise ValueError(f"noise percentage must be in {NOISE_LEVELS}")
    split_words(morse)
    positions = [index for index, char in enumerate(morse) if char in ".-"]
    flips = int(round(len(positions) * percentage / 100.0))
    selected = set(rng.sample(positions, flips))
    return "".join(("-" if char == "." else ".") if index in selected else char
                   for index, char in enumerate(morse))


def generate_cases(sources: list[str], seed: int, partition_label: str) -> list[OfflineCase]:
    cases = []
    for noise_percent in NOISE_LEVELS:
        rng = random.Random(seed + noise_percent + 311)
        for source_index, source in enumerate(sources):
            reference = encode_alphanumeric(source)
            received = flip_dot_dash_noise(reference, noise_percent, rng)
            cases.append(OfflineCase(
                sample_id=f"{partition_label}-{source_index:05d}-{noise_percent:02d}",
                source_text=source,
                reference_morse=reference,
                received_morse=received,
                noise_percent=noise_percent,
                corruption_type="clean" if noise_percent == 0 else "dot_dash_substitution",
            ))
    return cases


def word_boundary_positions(morse: str) -> tuple[int, ...]:
    words = morse.split(" / ")
    cumulative = 0
    result = []
    for word in words[:-1]:
        cumulative += len(word.split())
        result.append(cumulative)
    return tuple(result)


def morse_is_consistent(morse: str) -> bool:
    try:
        words = split_words(morse)
    except ValueError:
        return False
    known_codes = set(MORSE.values())
    return all(group in known_codes for word in words for group in word.split(" "))


def encode_padded(word: str) -> list[int]:
    if not word or len(word) > SEQUENCE_LENGTH or any(char not in ".- " for char in word):
        raise ValueError("model accepts only slash-free dot/dash/space words up to 120 characters")
    ids = [TOKEN_IDS[char] for char in word]
    return ids + [PAD_ID] * (SEQUENCE_LENGTH - len(ids))


def propose_many(model: Any, messages: list[str]) -> list[ProposalFeatures]:
    """Run inference and retain probabilities; references are not accepted here."""
    import numpy as np

    split_messages = [split_words(message) for message in messages]
    flattened = [(message_index, word_index, word)
                 for message_index, words in enumerate(split_messages)
                 for word_index, word in enumerate(words)]
    batch = np.asarray([encode_padded(word) for _, _, word in flattened], dtype=np.int32)
    probabilities = model.predict(batch, verbose=0)
    expected = (len(flattened), SEQUENCE_LENGTH, len(VOCABULARY))
    if probabilities.shape != expected:
        raise ValueError(f"model output {probabilities.shape}, expected {expected}")
    id_to_token = {index: token for token, index in TOKEN_IDS.items()}
    proposed_words: list[list[str]] = [[] for _ in messages]
    changes: list[list[TokenChange]] = [[] for _ in messages]
    for row_index, (message_index, word_index, word) in enumerate(flattened):
        scores = probabilities[row_index]
        output_chars = []
        for char_index, original_char in enumerate(word):
            token_id = int(scores[char_index].argmax())
            predicted_char = id_to_token.get(token_id, "<UNK>")
            if predicted_char in ("<PAD>", "<UNK>"):
                predicted_char = original_char
            output_chars.append(predicted_char)
            if predicted_char != original_char:
                original_id = TOKEN_IDS[original_char]
                proposed_probability = float(scores[char_index, token_id])
                original_probability = float(scores[char_index, original_id])
                changes[message_index].append(TokenChange(
                    word_index=word_index,
                    position_in_word=char_index,
                    original_token=original_char,
                    proposed_token=predicted_char,
                    original_probability=original_probability,
                    proposed_probability=proposed_probability,
                    probability_margin=proposed_probability - original_probability,
                ))
        proposed_words[message_index].append("".join(output_chars))

    results = []
    for message, words, message_changes in zip(messages, proposed_words, changes):
        candidate = " / ".join(words)
        margins = [change.probability_margin for change in message_changes]
        candidate_words = candidate.split(" / ")
        received_words = split_words(message)
        same_letter_boundaries = (
            len(candidate_words) == len(received_words)
            and all([i for i, char in enumerate(a) if char == " "]
                    == [i for i, char in enumerate(b) if char == " "]
                    for a, b in zip(received_words, candidate_words))
        )
        results.append(ProposalFeatures(
            received_morse=message,
            proposed_morse=candidate,
            changes=tuple(message_changes),
            minimum_changed_token_margin=min(margins) if margins else 0.0,
            mean_changed_token_margin=sum(margins) / len(margins) if margins else 0.0,
            maximum_changed_token_margin=max(margins) if margins else 0.0,
            candidate_morse_consistent=morse_is_consistent(candidate),
            same_character_length=(len(candidate) == len(message)),
            same_letter_boundaries=same_letter_boundaries,
            same_word_boundaries=(word_boundary_positions(candidate) == word_boundary_positions(message)),
        ))
    return results


def propose_one(model: Any, received_morse: str) -> ProposalFeatures:
    return propose_many(model, [received_morse])[0]


def decide(features: ProposalFeatures, policy: GatePolicy, channel_profile: str) -> GateDecision:
    """Reference-free runtime rule; no reference label exists in this signature."""
    raw = features.received_morse
    if channel_profile not in policy.allow_channel_profiles:
        return GateDecision(raw, raw, False, "out_of_scope_channel_profile", features)
    if not features.changes:
        return GateDecision(raw, raw, False, "no_change_proposed", features)
    if not features.same_character_length:
        return GateDecision(raw, raw, False, "length_change", features)
    if not features.same_letter_boundaries or not features.same_word_boundaries:
        return GateDecision(raw, raw, False, "boundary_change", features)
    if any(change.original_token not in ".-" or change.proposed_token not in ".-"
           for change in features.changes):
        return GateDecision(raw, raw, False, "non_symbol_or_boundary_change", features)
    if policy.require_consistent_candidate and not features.candidate_morse_consistent:
        return GateDecision(raw, raw, False, "candidate_morse_inconsistent", features)
    if len(features.changes) > policy.maximum_changes:
        return GateDecision(raw, raw, False, "too_many_changes", features)
    if features.minimum_changed_token_margin < policy.minimum_changed_token_margin:
        return GateDecision(raw, raw, False, "minimum_margin_below_threshold", features)
    if features.mean_changed_token_margin < policy.mean_changed_token_margin:
        return GateDecision(raw, raw, False, "mean_margin_below_threshold", features)
    return GateDecision(raw, features.proposed_morse, True, "accepted", features)


def runtime_gate(model: Any, received_morse: str, policy: GatePolicy,
                 channel_profile: str = "unknown") -> GateDecision:
    """Abstain unchanged by default; channel profile is not a reference label."""
    if channel_profile not in policy.allow_channel_profiles:
        return GateDecision(received_morse, received_morse, False,
                            "out_of_scope_channel_profile", None)
    try:
        split_words(received_morse)
        features = propose_one(model, received_morse)
    except ValueError as exc:
        reason = "overlength_word" if "exceeds" in str(exc) else "unsupported_or_malformed_input"
        return GateDecision(received_morse, received_morse, False, reason, None)
    return decide(features, policy, channel_profile)


def symbol_stream(morse: str) -> str:
    return morse.replace(" ", "").replace("/", "")


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def errors(reference: str, candidate: str) -> int:
    return levenshtein(symbol_stream(reference), symbol_stream(candidate))


def _calibration_outcome(record: CalibrationRecord, decision: GateDecision) -> tuple[int, int]:
    return errors(record.reference_morse, record.features.received_morse), errors(
        record.reference_morse, decision.output_morse
    )


def _margin_grid() -> tuple[float, ...]:
    return tuple(index / 10 for index in range(10)) + (1.01,)


def select_policy(calibration_records: list[CalibrationRecord]) -> tuple[GatePolicy, dict[str, Any]]:
    """Select maximum calibration coverage subject to clean-safe and 5% harm limits."""
    if not calibration_records:
        raise ValueError("calibration records are required")
    candidates = []
    for min_margin in _margin_grid():
        for mean_margin in _margin_grid():
            for max_changes in (1, 2, 3, 4, 6, 8, 12, 24):
                policy = GatePolicy(min_margin, mean_margin, max_changes)
                accepted = harmful = clean_altered = 0
                for record in calibration_records:
                    profile = "clean" if record.noise_percent == 0 else "dot_dash_substitution"
                    decision = decide(record.features, policy, profile)
                    if decision.accepted:
                        accepted += 1
                        before, after = _calibration_outcome(record, decision)
                        harmful += after > before
                        if record.noise_percent == 0 and decision.output_morse != record.reference_morse:
                            clean_altered += 1
                harm_rate = harmful / accepted if accepted else 0.0
                if clean_altered == 0 and harm_rate <= SUBSTITUTION_TRAINING_HARM_CAP:
                    candidates.append((accepted, harmful, min_margin, mean_margin,
                                       max_changes, policy))
    if not candidates:
        policy = GatePolicy(1.01, 1.01, 0)
        return policy, {
            "selected_policy": "abstain_all",
            "accepted": 0,
            "harmful": 0,
            "calibration_harm_rate": 0.0,
            "calibration_clean_alterations": 0,
            "selection_rule": "No candidate policy met calibration harm/clean constraints.",
        }
    # Max coverage first, then fewer harms, stricter margins, and lower edit budget.
    chosen = sorted(candidates, key=lambda row: (
        -row[0], row[1], -row[2], -row[3], row[4]
    ))[0]
    accepted, harmful, min_margin, mean_margin, max_changes, policy = chosen
    if accepted == 0:
        policy = GatePolicy(1.01, 1.01, 0)
        return policy, {
            "selected_policy": "abstain_all",
            "accepted": 0,
            "harmful": 0,
            "calibration_harm_rate": 0.0,
            "calibration_clean_alterations": 0,
            "selection_rule": "No corrections met calibration constraints with nonzero coverage.",
        }
    clean_altered = 0
    return policy, {
        "selected_policy": "calibrated_selective_gate",
        "minimum_changed_token_margin": min_margin,
        "mean_changed_token_margin": mean_margin,
        "maximum_changes": max_changes,
        "accepted": accepted,
        "harmful": harmful,
        "calibration_harm_rate": harmful / accepted if accepted else 0.0,
        "calibration_clean_alterations": clean_altered,
        "selection_rule": "Maximize accepted calibration coverage with zero accepted clean alterations and <=5% calibration harmful rate.",
    }


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple[float, float] | None:
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = (z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5)
              / denominator)
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_locked(cases: list[OfflineCase], features: list[ProposalFeatures],
                    policy: GatePolicy) -> dict[str, Any]:
    if len(cases) != len(features):
        raise ValueError("locked cases/features count mismatch")
    decisions = []
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_outcomes = Counter()
    raw_symbol_errors = corrected_symbol_errors = 0
    raw_symbol_denominator = corrected_symbol_denominator = 0
    raw_boundaries = corrected_boundaries = 0
    clean_count = clean_altered = 0
    harmful_severity = []
    grouped = defaultdict(lambda: {
        "n": 0, "harmful": 0, "accepted": 0, "improved": 0,
        "unchanged": 0, "worsened": 0, "raw_symbol_errors": 0,
        "selected_symbol_errors": 0, "raw_symbol_denominator": 0,
        "selected_symbol_denominator": 0, "raw_boundaries_preserved": 0,
        "selected_boundaries_preserved": 0, "clean_messages": 0,
        "clean_altered": 0,
    })
    source_diffs: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)

    for case, proposal in zip(cases, features):
        profile = "clean" if case.noise_percent == 0 else "dot_dash_substitution"
        decision = decide(proposal, policy, profile)
        decisions.append(decision)
        reasons[case.corruption_type][decision.reason] += 1
        group = grouped[case.corruption_type]
        group["n"] += 1
        raw_error = errors(case.reference_morse, case.received_morse)
        corrected_error = errors(case.reference_morse, decision.output_morse)
        raw_len = max(len(symbol_stream(case.reference_morse)),
                      len(symbol_stream(case.received_morse)), 1)
        corrected_len = max(len(symbol_stream(case.reference_morse)),
                            len(symbol_stream(decision.output_morse)), 1)
        raw_symbol_errors += raw_error
        corrected_symbol_errors += corrected_error
        raw_symbol_denominator += raw_len
        corrected_symbol_denominator += corrected_len
        raw_boundary_match = word_boundary_positions(case.reference_morse) == word_boundary_positions(case.received_morse)
        corrected_boundary_match = word_boundary_positions(case.reference_morse) == word_boundary_positions(decision.output_morse)
        raw_boundaries += raw_boundary_match
        corrected_boundaries += corrected_boundary_match
        group["raw_symbol_errors"] += raw_error
        group["selected_symbol_errors"] += corrected_error
        group["raw_symbol_denominator"] += raw_len
        group["selected_symbol_denominator"] += corrected_len
        group["raw_boundaries_preserved"] += raw_boundary_match
        group["selected_boundaries_preserved"] += corrected_boundary_match
        source_diffs[case.source_text].append((raw_error, corrected_error, raw_len, corrected_len))
        if case.noise_percent == 0:
            clean_count += 1
            altered = decision.output_morse != case.reference_morse
            clean_altered += altered
            group["clean_messages"] += 1
            group["clean_altered"] += altered
        if decision.accepted:
            group["accepted"] += 1
            if corrected_error < raw_error:
                accepted_outcomes["improved"] += 1
                group["improved"] += 1
            elif corrected_error > raw_error:
                accepted_outcomes["worsened"] += 1
                group["worsened"] += 1
                group["harmful"] += 1
                harmful_severity.append(corrected_error - raw_error)
            else:
                accepted_outcomes["unchanged"] += 1
                group["unchanged"] += 1

    accepted_count = sum(decision.accepted for decision in decisions)
    harmful_count = accepted_outcomes["worsened"]
    interval = wilson_interval(harmful_count, accepted_count)
    n_sources = len(source_diffs)
    rng = random.Random(SEED + 927)
    bootstrap_differences = []
    source_keys = list(source_diffs)
    for _ in range(1000):
        selected = [source_diffs[rng.choice(source_keys)] for _ in range(n_sources)]
        raw_num = corrected_num = raw_den = corrected_den = 0
        for rows in selected:
            for raw_error, corrected_error, raw_len, corrected_len in rows:
                raw_num += raw_error
                corrected_num += corrected_error
                raw_den += raw_len
                corrected_den += corrected_len
        bootstrap_differences.append(raw_num / max(raw_den, 1) - corrected_num / max(corrected_den, 1))
    bootstrap_differences.sort()
    bootstrap_ci = (bootstrap_differences[24], bootstrap_differences[974]) if bootstrap_differences else None
    raw_rate = raw_symbol_errors / max(raw_symbol_denominator, 1)
    corrected_rate = corrected_symbol_errors / max(corrected_symbol_denominator, 1)
    relative_reduction = ((raw_rate - corrected_rate) / raw_rate) if raw_rate else 0.0
    conclusive_count = accepted_count >= MIN_ACCEPTED_FOR_INFERENCE
    passes_harm = interval is not None and interval[1] <= MAX_HARM_RATE_FOR_PASS
    passes_clean = clean_count > 0 and clean_altered == 0
    passes_improvement = (bootstrap_ci is not None and bootstrap_ci[0] > 0
                          and relative_reduction >= MIN_RELATIVE_ERROR_REDUCTION_FOR_PASS)
    if not conclusive_count:
        status = "INCONCLUSIVE: fewer than 300 accepted test corrections"
    elif not passes_harm:
        status = "NOT PASSED: 95% harmful-rate upper bound exceeds 1%"
    elif not passes_clean:
        status = "NOT PASSED: locked clean inputs were altered"
    elif not passes_improvement:
        status = "NOT PASSED: required locked-set error reduction was not demonstrated"
    else:
        status = "CRITERIA MET for this synthetic substitution experiment only"
    return {
        "eligible_messages": len(cases),
        "accepted_corrections": accepted_count,
        "coverage": accepted_count / max(len(cases), 1),
        "accepted_outcomes": dict(accepted_outcomes),
        "harmful_accepted": harmful_count,
        "harmful_accepted_rate": harmful_count / accepted_count if accepted_count else None,
        "harmful_accepted_wilson_95_percent": list(interval) if interval else None,
        "harmful_error_severity": {
            "sum_extra_symbol_errors": sum(harmful_severity),
            "mean_extra_symbol_errors": sum(harmful_severity) / len(harmful_severity)
            if harmful_severity else 0.0,
            "maximum_extra_symbol_errors": max(harmful_severity, default=0),
        },
        "clean_messages": clean_count,
        "clean_altered": clean_altered,
        "clean_message_alteration_rate": clean_altered / clean_count if clean_count else None,
        "raw_symbol_error_rate": raw_rate,
        "selected_output_symbol_error_rate": corrected_rate,
        "relative_symbol_error_reduction": relative_reduction,
        "paired_source_bootstrap_95_percent_rate_difference": list(bootstrap_ci) if bootstrap_ci else None,
        "raw_word_boundaries_preserved": raw_boundaries,
        "selected_output_word_boundaries_preserved": corrected_boundaries,
        "word_boundary_message_count": len(cases),
        "abstentions_by_corruption_type_and_reason": {
            key: dict(value) for key, value in sorted(reasons.items())
        },
        "outcomes_by_corruption_type": {key: value for key, value in sorted(grouped.items())},
        "metrics_by_corruption_type": {
            key: {
                **value,
                "raw_symbol_error_rate": value["raw_symbol_errors"] / max(value["raw_symbol_denominator"], 1),
                "selected_output_symbol_error_rate": value["selected_symbol_errors"] / max(value["selected_symbol_denominator"], 1),
                "raw_word_boundary_preservation_rate": value["raw_boundaries_preserved"] / max(value["n"], 1),
                "selected_output_word_boundary_preservation_rate": value["selected_boundaries_preserved"] / max(value["n"], 1),
                "clean_message_alteration_rate": value["clean_altered"] / value["clean_messages"] if value["clean_messages"] else None,
            }
            for key, value in sorted(grouped.items())
        },
        "evidence_status": status,
    }


def validate_model_metadata(model: Any, metadata: dict[str, Any]) -> None:
    expected_ids = {token: index for index, token in enumerate(VOCABULARY)}
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("metadata model_file mismatch")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("metadata sequence_length mismatch")
    if metadata.get("vocabulary") != VOCABULARY or metadata.get("token_ids") != expected_ids:
        raise ValueError("metadata vocabulary/token IDs mismatch")
    if metadata.get("padding_id") != PAD_ID:
        raise ValueError("metadata padding ID mismatch")
    if tuple(model.input_shape) != (None, SEQUENCE_LENGTH):
        raise ValueError(f"V3 input shape mismatch: {model.input_shape}")
    if tuple(model.output_shape) != (None, SEQUENCE_LENGTH, len(VOCABULARY)):
        raise ValueError(f"V3 output shape mismatch: {model.output_shape}")
    embedding = model.get_layer("token_embedding")
    if embedding.input_dim != len(VOCABULARY) or not embedding.mask_zero:
        raise ValueError("V3 embedding/PAD contract mismatch")


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("V3 model and metadata are required")
    with METADATA_PATH.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    import tensorflow as tf

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    validate_model_metadata(model, metadata)
    return model, metadata


def run_experiment(model: Any, calibration_cases: list[OfflineCase],
                   locked_cases: list[OfflineCase]) -> tuple[GatePolicy, dict[str, Any], dict[str, Any]]:
    """Calibration proposes/selects first; locked proposals are evaluated exactly once after freeze."""
    calibration_features = propose_many(model, [case.received_morse for case in calibration_cases])
    calibration_records = [CalibrationRecord(
        features=features,
        reference_morse=case.reference_morse,
        noise_percent=case.noise_percent,
        corruption_type=case.corruption_type,
    ) for case, features in zip(calibration_cases, calibration_features)]
    policy, selection_report = select_policy(calibration_records)
    locked_features = propose_many(model, [case.received_morse for case in locked_cases])
    locked_report = evaluate_locked(locked_cases, locked_features, policy)
    return policy, selection_report, locked_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-messages", type=int, default=CALIBRATION_MESSAGES)
    parser.add_argument("--locked-test-messages", type=int, default=LOCKED_TEST_MESSAGES)
    args = parser.parse_args()
    if args.calibration_messages < 1 or args.locked_test_messages < 1:
        parser.error("partition sizes must be positive")
    try:
        calibration_sources, locked_sources = generate_partitions(
            SEED, args.calibration_messages, args.locked_test_messages
        )
        # No noisy variants are generated until source partitions are disjoint.
        calibration_cases = generate_cases(calibration_sources, SEED + 101, "cal")
        locked_cases = generate_cases(locked_sources, SEED + 303, "locked")
        model, metadata = load_artifacts()
        policy, selection, report = run_experiment(model, calibration_cases, locked_cases)
    except Exception as exc:
        print(f"ERROR: selective V3 experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("V3 conservative selective-correction experiment (synthetic, offline only)")
    print(f"Model: {MODEL_PATH.name}; metadata: {METADATA_PATH.name}; seed: {SEED}")
    print(f"Disjoint source messages: calibration={len(calibration_sources)}, locked-test={len(locked_sources)}; "
          f"known prior source words excluded={len(prior_source_words())}")
    print("Reference labels are used only by calibration selection and offline scoring; runtime_gate receives ProposalFeatures, policy, and a channel profile only.")
    print("Frozen calibration policy:")
    print(json.dumps({
        "minimum_changed_token_margin": policy.minimum_changed_token_margin,
        "mean_changed_token_margin": policy.mean_changed_token_margin,
        "maximum_changes": policy.maximum_changes,
        "allow_channel_profiles": policy.allow_channel_profiles,
        **selection,
    }, indent=2))
    print("Locked-test report (single evaluation after policy freeze):")
    print(json.dumps(report, indent=2))
    print("Limitations: synthetic substitution-only channel, softmax-derived features are not inherently calibrated, and no evidence here establishes real radio performance or detects unknown insertions/deletions in arbitrary traffic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
