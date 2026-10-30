"""Padded batches for the edit, boundary, clean, and risk objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .decoding import map_boundary_prior
from .schema import V2Example


@dataclass(frozen=True)
class V2Batch:
    inputs: dict
    targets: dict
    masks: dict


def _symbols(morse: str) -> str:
    return "".join(ch for ch in morse if ch in ".-")


def prepare_batch(examples: Sequence[V2Example], *, boundary_augmentation_seed: int | None = None,
                  boundary_candidates=None) -> V2Batch:
    import numpy as np
    if not examples:
        raise ValueError("at least one example is required")
    partitions = {row.partition for row in examples}
    if len(partitions) != 1:
        raise ValueError("a batch may contain exactly one data partition")
    if "locked_test" in partitions:
        raise ValueError("locked-test examples are forbidden in training batches")
    if boundary_candidates is not None and boundary_augmentation_seed is not None:
        raise ValueError("provide explicit candidates or augmentation seed, not both")
    overrides = list(boundary_candidates) if boundary_candidates is not None else None
    if boundary_augmentation_seed is not None:
        from .boundary_augmentation import make_boundary_candidate
        overrides = [make_boundary_candidate(row.reference_morse,
                    seed=boundary_augmentation_seed + i) for i, row in enumerate(examples)]
    if overrides is not None and len(overrides) != len(examples):
        raise ValueError("one boundary candidate is required per example")
    if overrides is not None:
        from .boundary_augmentation import validate_candidate_alignment
        for row, candidate in zip(examples, overrides):
            validate_candidate_alignment(candidate, len(_symbols(row.received_morse)))
    max_input = max(len(_symbols(row.received_morse)) for row in examples)
    max_target = max(len(overrides[i].symbols if overrides else _symbols(row.reference_morse))
                     for i, row in enumerate(examples))
    max_insertions = max(len(gap) for row in examples for gap in row.insertions_before)
    symbol_ids = np.zeros((len(examples), max_input), dtype=np.int32)
    input_gap_ids = np.zeros((len(examples), max_input + 1), dtype=np.int32)
    candidate_ids = np.zeros((len(examples), max_target), dtype=np.int32)
    candidate_alignment = np.full((len(examples), max_target), -2, dtype=np.int32)
    candidate_gap_ids = np.zeros((len(examples), max_target + 1), dtype=np.int32)
    candidate_boundary_ids = np.zeros((len(examples), max_target + 1), dtype=np.int32)
    symbol_ops = np.zeros((len(examples), max_input), dtype=np.int32)
    insertion_ops = np.zeros((len(examples), max_input + 1, max_insertions), dtype=np.int32)
    boundary_labels = np.zeros((len(examples), max_target + 1), dtype=np.int32)
    risk_labels = np.full((len(examples),), -1, dtype=np.int32)
    clean_labels = np.zeros((len(examples),), dtype=np.int32)
    insertion_mask = np.zeros(insertion_ops.shape, dtype=np.float32)
    boundary_mask = np.zeros(boundary_labels.shape, dtype=np.float32)
    for bi, row in enumerate(examples):
        source = _symbols(row.received_morse)
        target = _symbols(row.reference_morse)
        candidate = overrides[bi] if overrides else None
        candidate_symbols = candidate.symbols if candidate else target
        candidate_alignment_values = candidate.alignment if candidate else row.output_alignment
        candidate_boundary_values = candidate.gap_labels if candidate else row.boundary_labels
        symbol_ids[bi, :len(source)] = [1 if c == "." else 2 for c in source]
        # For symbol-edit context, pass observed separators at gaps.
        from .schema import boundary_targets
        input_gap_ids[bi, :len(source) + 1] = boundary_targets(row.received_morse)
        candidate_ids[bi, :len(candidate_symbols)] = [1 if c == "." else 2 for c in candidate_symbols]
        candidate_alignment[bi, :len(candidate_symbols)] = candidate_alignment_values
        priors = map_boundary_prior(row.received_morse, candidate_alignment_values)
        candidate_gap_ids[bi, :len(priors)] = priors
        candidate_boundary_ids[bi, :len(candidate_boundary_values)] = candidate_boundary_values
        symbol_ops[bi, :len(source)] = row.symbol_operations
        insertion_ops[bi, :len(row.insertions_before), :] = row.insertions_before
        boundary_labels[bi, :len(candidate_boundary_values)] = candidate_boundary_values
        insertion_mask[bi, :len(source) + 1, :] = 1.0
        boundary_mask[bi, :len(candidate_symbols) + 1] = 1.0
        clean_labels[bi] = int(row.is_clean)
    return V2Batch(
        inputs={"symbol_ids": symbol_ids, "input_gap_ids": input_gap_ids,
                 "candidate_symbol_ids": candidate_ids,
                 "candidate_alignment": candidate_alignment,
                 "candidate_gap_ids": candidate_gap_ids,
                 "candidate_boundary_ids": candidate_boundary_ids},
        targets={"symbol_ops": symbol_ops, "insertion_ops": insertion_ops,
                 "boundary_labels": boundary_labels, "risk_labels": risk_labels,
                 "clean_labels": clean_labels},
        masks={"symbol": symbol_ids != 0,
               "insertion": insertion_mask,
               "boundary": boundary_mask,
               "candidate": candidate_ids != 0},
    )

