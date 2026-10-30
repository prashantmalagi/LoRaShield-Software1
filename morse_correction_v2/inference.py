"""Reference-free V2 proposal generation and selective abstention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .calibration import TemperatureCalibrator
from .decoding import decode_boundaries, decode_symbol_edits, map_boundary_prior
from .gate import GateDecision, GateEvidence, GatePolicy, decide
from .schema import preprocess_received, validate_morse


@dataclass(frozen=True)
class V2InferenceResult:
    received_morse: str
    proposed_morse: str | None
    output_morse: str
    accepted: bool
    reason: str
    evidence: GateEvidence | None
    symbol_alignment: tuple[int, ...] = ()
    risk_logits: tuple[float, float, float] | None = None


def _symbols(value: str) -> str:
    return "".join(ch for ch in value if ch in ".-")


def _distance(left, right) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _numpy_outputs(model: Any, inputs: dict) -> dict:
    outputs = model(inputs, training=False)
    return {key: value.numpy() if hasattr(value, "numpy") else value
            for key, value in outputs.items()}


def propose(model: Any, received_morse: str, *, policy: GatePolicy | None = None,
            clean_calibrator: TemperatureCalibrator | None = None,
            risk_calibrator: TemperatureCalibrator | None = None,
            max_output_symbols: int = 512) -> V2InferenceResult:
    """Generate a candidate, then return it only when calibrated gate accepts."""
    try:
        validate_morse(received_morse)
    except (TypeError, ValueError):
        return V2InferenceResult(received_morse, None, received_morse, False,
                                 "invalid_received_input", None)
    import numpy as np
    from .schema import boundary_targets
    prepared = preprocess_received(received_morse)
    source = _symbols(received_morse)
    symbol_ids = np.asarray([prepared.symbol_ids], dtype=np.int32)
    input_gaps = np.asarray([prepared.input_gap_ids], dtype=np.int32)
    identity_alignment = np.arange(len(source), dtype=np.int32)
    identity_ids = symbol_ids.copy()
    identity_candidate_gaps = np.asarray([map_boundary_prior(received_morse, identity_alignment)], dtype=np.int32)
    identity_boundaries = np.asarray([boundary_targets(received_morse)], dtype=np.int32)
    base_inputs = {"symbol_ids": symbol_ids, "input_gap_ids": input_gaps,
                   "candidate_symbol_ids": identity_ids,
                   "candidate_alignment": identity_alignment[None, :],
                   "candidate_gap_ids": identity_candidate_gaps,
                   "candidate_boundary_ids": identity_boundaries}
    try:
        first = _numpy_outputs(model, base_inputs)
        edit = decode_symbol_edits(source, first["symbol_logits"][0], first["insertion_logits"][0],
                                   max_output_symbols=max_output_symbols)
        candidate_ids = np.asarray([[1 if c == "." else 2 for c in edit.symbols]], dtype=np.int32)
        alignment = np.asarray([edit.alignment], dtype=np.int32)
        candidate_gaps = np.asarray([map_boundary_prior(received_morse, edit.alignment)], dtype=np.int32)
        second = _numpy_outputs(model, {**base_inputs, "candidate_symbol_ids": candidate_ids,
                                        "candidate_alignment": alignment,
                                        "candidate_gap_ids": candidate_gaps,
                                        "candidate_boundary_ids": candidate_gaps})
        proposed = decode_boundaries(edit.symbols, second["boundary_logits"][0])
        proposed_boundary_ids = np.asarray([boundary_targets(proposed)], dtype=np.int32)
        risk_outputs = _numpy_outputs(model, {**base_inputs, "candidate_symbol_ids": candidate_ids,
            "candidate_alignment": alignment, "candidate_gap_ids": candidate_gaps,
            "candidate_boundary_ids": proposed_boundary_ids})
        clean_probs = clean_calibrator.probabilities(second["clean_logits"][0]) if clean_calibrator else None
        risk_probs = risk_calibrator.probabilities(risk_outputs["risk_logits"][0]) if risk_calibrator else None
        boundary_edit_count = _distance(
            boundary_targets(received_morse), boundary_targets(proposed))
        evidence = GateEvidence(
            terminated=True, syntax_valid=True,
            symbol_edit_count=_distance(source, edit.symbols),
            boundary_edit_count=boundary_edit_count,
            p_clean=clean_probs[1] if clean_probs else None,
            p_improve=risk_probs[0] if risk_probs else None,
            p_tie=risk_probs[1] if risk_probs else None,
            p_harm=risk_probs[2] if risk_probs else None,
        )
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        return V2InferenceResult(received_morse, None, received_morse, False,
                                 f"decode_abstention:{type(exc).__name__}", None)
    decision: GateDecision = decide(received_morse, proposed, evidence,
                                    policy or GatePolicy())
    return V2InferenceResult(received_morse, proposed, decision.output_morse,
                             decision.accepted, decision.reason, evidence,
                             tuple(edit.alignment),
                             tuple(float(x) for x in risk_outputs["risk_logits"][0]))

