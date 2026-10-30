"""Constrained edit and separator decoders for the V2 research model."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .schema import boundary_targets, validate_morse


@dataclass(frozen=True)
class EditResult:
    symbols: str
    alignment: tuple[int, ...]
    insertions: int
    deletions: int
    substitutions: int


def _argmax(values: Sequence[float]) -> int:
    if len(values) == 0 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("decoder scores must be finite and nonempty")
    return max(range(len(values)), key=lambda i: float(values[i]))


def decode_symbol_edits(input_symbols: str, symbol_logits, insertion_logits, *,
                        max_output_symbols: int = 512,
                        max_insertions_per_gap: int = 2) -> EditResult:
    """Decode insertions at gaps and output/delete actions at source symbols.

    Insertion slots are left-packed: after the first NONE action, later slots
    in that gap are ignored. Symbol classes are DELETE, DOT, DASH.
    """
    if not input_symbols or any(ch not in ".-" for ch in input_symbols):
        raise ValueError("input symbol stream must be nonempty dot/dash text")
    n = len(input_symbols)
    if len(symbol_logits) != n or len(insertion_logits) != n + 1:
        raise ValueError("symbol/insertion score dimensions do not match input")
    if max_insertions_per_gap < 0 or max_output_symbols < 1:
        raise ValueError("decoder limits must be positive")
    output = []
    alignment = []
    insertions = deletions = substitutions = 0
    for i in range(n + 1):
        slots = insertion_logits[i]
        if len(slots) < max_insertions_per_gap:
            raise ValueError("not enough insertion-slot scores")
        open_slots = True
        for slot in range(max_insertions_per_gap):
            if not open_slots:
                continue
            action = _argmax(slots[slot])
            if action == 0:
                open_slots = False
                continue
            output.append("." if action == 1 else "-")
            alignment.append(-1)
            insertions += 1
            if len(output) > max_output_symbols:
                raise ValueError("corrected symbol stream exceeds output limit")
        if i == n:
            break
        action = _argmax(symbol_logits[i])
        if action == 0:
            deletions += 1
            continue
        symbol = "." if action == 1 else "-"
        output.append(symbol)
        alignment.append(i)
        substitutions += symbol != input_symbols[i]
        if len(output) > max_output_symbols:
            raise ValueError("corrected symbol stream exceeds output limit")
    if not output:
        raise ValueError("decoder produced an empty symbol stream")
    return EditResult("".join(output), tuple(alignment), insertions, deletions, substitutions)


def _separator_label(scores) -> int:
    return _argmax(scores)


def decode_boundaries(symbols: str, boundary_logits, *, max_tokens: int = 256) -> str:
    """Predict one NONE/LETTER/WORD label per symbol gap and render canonical Morse."""
    if not symbols or any(ch not in ".-" for ch in symbols):
        raise ValueError("boundary decoder requires a nonempty dot/dash stream")
    if len(boundary_logits) != len(symbols) + 1:
        raise ValueError("boundary score count must be symbol count plus one")
    labels = [_separator_label(row) for row in boundary_logits]
    if any(label not in (0, 1, 2) for label in labels):
        raise ValueError("unknown boundary label")
    labels[0] = labels[-1] = 0  # Edge sentinels cannot introduce empty groups.
    result = []
    for i, symbol in enumerate(symbols):
        result.append(symbol)
        if i + 1 < len(symbols):
            result.append(" " if labels[i + 1] == 1 else " / " if labels[i + 1] == 2 else "")
    rendered = "".join(result)
    validate_morse(rendered, max_tokens=max_tokens)
    return rendered


def map_boundary_prior(received_morse: str, alignment: Sequence[int]) -> tuple[int, ...]:
    """Map observed source boundaries onto candidate output gaps via alignment."""
    source_symbols = "".join(ch for ch in received_morse if ch in ".-")
    source_gaps = boundary_targets(received_morse)
    priors = []
    for gap in range(len(alignment) + 1):
        left = alignment[gap - 1] if gap else None
        right = alignment[gap] if gap < len(alignment) else None
        if left is not None and right is not None:
            source_gap = max(left + 1, min(right, len(source_symbols)))
        elif right is not None:
            source_gap = right
        elif left is not None:
            source_gap = left + 1
        else:
            source_gap = 0
        priors.append(source_gaps[min(max(source_gap, 0), len(source_symbols))])
    return tuple(priors)

