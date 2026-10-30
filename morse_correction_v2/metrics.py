"""Offline metrics with explicit case and edit denominators."""
from __future__ import annotations

from typing import Sequence

from .calibration import combined_errors
from .schema import V2Example


def _levenshtein(left, right) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _symbol_stream(value: str) -> str:
    return "".join(ch for ch in value if ch in ".-")


def _boundary_offsets(value: str, kind: str) -> tuple[int, ...]:
    offsets = []
    symbol_offset = 0
    words = value.split(" / ")
    letter_count = 0
    for wi, word in enumerate(words):
        letters = word.split(" ")
        for li, letter in enumerate(letters):
            symbol_offset += len(letter)
            letter_count += 1
            if kind == "letter" and li < len(letters) - 1:
                offsets.append(symbol_offset)
        if kind == "word" and wi < len(words) - 1:
            offsets.append(letter_count)
    return tuple(offsets)


def _system_scores(examples: Sequence[V2Example], outputs: Sequence[str]) -> dict:
    if len(examples) != len(outputs):
        raise ValueError("example and output counts differ")
    symbol_errors = symbol_denominator = 0
    letter_errors = letter_denominator = 0
    word_errors = word_denominator = 0
    exact = 0
    for example, output in zip(examples, outputs):
        ref, out = example.reference_morse, output
        rs, os = _symbol_stream(ref), _symbol_stream(out)
        symbol_errors += _levenshtein(rs, os)
        symbol_denominator += max(len(rs), len(os), 1)
        rlb, olb = _boundary_offsets(ref, "letter"), _boundary_offsets(out, "letter")
        rwb, owb = _boundary_offsets(ref, "word"), _boundary_offsets(out, "word")
        letter_errors += _levenshtein(rlb, olb)
        letter_denominator += max(len(rlb), len(olb), 1)
        word_errors += _levenshtein(rwb, owb)
        word_denominator += max(len(rwb), len(owb), 1)
        exact += ref == out
    return {"messages": len(examples), "exact_messages": exact,
            "symbol_errors": symbol_errors, "symbol_denominator": symbol_denominator,
            "symbol_error_rate": symbol_errors / max(symbol_denominator, 1),
            "letter_boundary_errors": letter_errors,
            "letter_boundary_denominator": letter_denominator,
            "letter_boundary_error_rate": letter_errors / max(letter_denominator, 1),
            "word_boundary_errors": word_errors,
            "word_boundary_denominator": word_denominator,
            "word_boundary_error_rate": word_errors / max(word_denominator, 1)}


def evaluate_gate(examples: Sequence[V2Example], outputs: Sequence[str],
                  accepted: Sequence[bool]) -> dict:
    """Report baseline/gated metrics and selection-relevant denominators."""
    if len(examples) != len(outputs) or len(examples) != len(accepted):
        raise ValueError("all metric inputs must have the same length")
    clean_indices = [i for i, row in enumerate(examples) if row.is_clean]
    corrupted_indices = [i for i, row in enumerate(examples) if not row.is_clean]
    eligible_corrupted = [i for i in corrupted_indices
                          if combined_errors(examples[i].reference_morse,
                                             examples[i].received_morse) > 0]
    accepted_indices = [i for i, value in enumerate(accepted) if value]
    clean_false = sum(accepted[i] and outputs[i] != examples[i].received_morse
                      for i in clean_indices)
    harmful = sum(combined_errors(examples[i].reference_morse, outputs[i]) >
                  combined_errors(examples[i].reference_morse, examples[i].received_morse)
                  for i in accepted_indices)
    corrected = sum(combined_errors(examples[i].reference_morse, outputs[i]) <
                    combined_errors(examples[i].reference_morse, examples[i].received_morse)
                    for i in eligible_corrupted)
    baseline_outputs = [row.received_morse for row in examples]
    return {
        "denominators": {"all_cases": len(examples), "clean_cases": len(clean_indices),
                         "corrupted_cases": len(corrupted_indices),
                         "corrupted_cases_with_input_error": len(eligible_corrupted),
                         "accepted": len(accepted_indices), "abstentions": len(examples) - len(accepted_indices)},
        "clean_false_accepts": clean_false,
        "clean_false_accept_rate": clean_false / max(len(clean_indices), 1),
        "harmful_accepted_corrections": harmful,
        "harmful_accepted_rate": harmful / len(accepted_indices) if accepted_indices else None,
        "corrupted_cases_corrected": corrected,
        "corrupted_case_correction_rate": corrected / len(eligible_corrupted) if eligible_corrupted else None,
        "unchanged": _system_scores(examples, baseline_outputs),
        "gated": _system_scores(examples, outputs),
    }

