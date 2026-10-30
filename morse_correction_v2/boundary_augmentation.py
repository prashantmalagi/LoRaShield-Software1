"""Deterministic candidate corruption for boundary-stage robustness."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from .schema import boundary_targets, validate_morse


@dataclass(frozen=True)
class BoundaryCandidate:
    symbols: str
    alignment: tuple[int, ...]
    gap_labels: tuple[int, ...]


def make_boundary_candidate(reference_morse: str, *, seed: int,
                            corruption_rate: float = 0.12) -> BoundaryCandidate:
    """Create a noisy symbol candidate aligned to reference symbol positions.

    Applies deterministic flips, deletions, and insertions; projected boundary
    labels are retained only when adjacent candidate symbols align to adjacent
    reference symbols. Insertions/deletion gaps receive NONE supervision.
    """
    if not 0.0 <= corruption_rate <= 0.5:
        raise ValueError("corruption_rate must be between zero and 0.5")
    source = "".join(ch for ch in validate_morse(reference_morse) if ch in ".-")
    gold_gaps = boundary_targets(reference_morse)
    rng = random.Random(seed)
    symbols: list[str] = []
    alignment: list[int] = []
    for index, symbol in enumerate(source):
        if rng.random() < corruption_rate * 0.25:
            symbols.append("." if symbol == "-" else "-")
            alignment.append(index)
        else:
            symbols.append(symbol)
            alignment.append(index)
        if rng.random() < corruption_rate * 0.15:
            symbols.append("." if rng.random() < 0.5 else "-")
            alignment.append(-1)
        if rng.random() < corruption_rate * 0.10 and len(symbols) > 1:
            symbols.pop()
            alignment.pop()
    if not symbols:
        symbols, alignment = [source[0]], [0]
    labels = [0] * (len(symbols) + 1)
    for gap in range(1, len(symbols)):
        left, right = alignment[gap - 1], alignment[gap]
        if left >= 0 and right == left + 1:
            labels[gap] = gold_gaps[right]
    candidate = BoundaryCandidate("".join(symbols), tuple(alignment), tuple(labels))
    # The projected labels must render a syntactically valid Morse surface.
    rendered = []
    for i, symbol in enumerate(candidate.symbols):
        rendered.append(symbol)
        if i + 1 < len(candidate.symbols):
            rendered.append(" " if candidate.gap_labels[i + 1] == 1 else
                            " / " if candidate.gap_labels[i + 1] == 2 else "")
    validate_morse("".join(rendered))
    return candidate


def validate_candidate_alignment(candidate: BoundaryCandidate, source_symbol_count: int) -> None:
    if len(candidate.symbols) != len(candidate.alignment):
        raise ValueError("candidate symbols and alignment lengths differ")
    if len(candidate.gap_labels) != len(candidate.symbols) + 1:
        raise ValueError("candidate boundary labels must cover every gap")
    if not candidate.symbols or any(c not in ".-" for c in candidate.symbols):
        raise ValueError("candidate symbol stream is malformed")
    if candidate.gap_labels[0] != 0 or candidate.gap_labels[-1] != 0:
        raise ValueError("edge gaps must use NONE boundary labels")
    if any(label not in (0, 1, 2) for label in candidate.gap_labels):
        raise ValueError("unknown boundary label")
    if any(index < -1 or index >= source_symbol_count for index in candidate.alignment):
        raise ValueError("alignment index is outside source symbol stream")


def evaluate_predicted_boundary_candidates(model, examples, *, batch_builder,
                                            max_output_symbols: int = 512) -> dict:
    """Measure boundary predictions after the model's own Stage-A symbol edits."""
    import numpy as np
    from .decoding import decode_symbol_edits, decode_boundaries
    from .schema import boundary_targets
    symbol_total = boundary_correct = valid_outputs = 0
    for example in examples:
        initial = batch_builder([example])
        stage_a = model(initial.inputs, training=False)
        source = "".join(c for c in example.received_morse if c in ".-")
        edits = decode_symbol_edits(source, stage_a["symbol_logits"][0].numpy(),
                                     stage_a["insertion_logits"][0].numpy(),
                                     max_output_symbols=max_output_symbols)
        gold_gaps = boundary_targets(example.reference_morse)
        source_to_reference = {source_index: reference_index
                               for reference_index, source_index in enumerate(example.output_alignment)
                               if source_index >= 0}
        labels = [0] * (len(edits.symbols) + 1)
        for gap in range(1, len(edits.symbols)):
            left, right = edits.alignment[gap - 1], edits.alignment[gap]
            left_ref, right_ref = source_to_reference.get(left), source_to_reference.get(right)
            if left_ref is not None and right_ref == left_ref + 1:
                labels[gap] = gold_gaps[right_ref]
        candidate = BoundaryCandidate(edits.symbols, edits.alignment, tuple(labels))
        validate_candidate_alignment(candidate, len(source))
        second = batch_builder([example], boundary_candidates=[candidate])
        output = model(second.inputs, training=False)["boundary_logits"][0].numpy()
        prediction = np.argmax(output, axis=-1).tolist()
        prediction[0] = prediction[-1] = 0
        boundary_correct += sum(a == b for a, b in zip(prediction, labels))
        symbol_total += len(labels)
        try:
            decode_boundaries(edits.symbols, output)
            valid_outputs += 1
        except ValueError:
            pass
    return {"candidate_source": "model_predicted_stage_a", "boundary_gap_count": symbol_total,
            "boundary_accuracy": boundary_correct / max(symbol_total, 1),
            "predicted_candidate_count": len(examples), "syntactically_valid_outputs": valid_outputs,
            "syntactic_validity_rate": valid_outputs / max(len(examples), 1)}
