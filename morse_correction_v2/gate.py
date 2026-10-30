"""Explicit abstention gate; rejected candidates return the exact input."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .schema import validate_morse


@dataclass(frozen=True)
class GateEvidence:
    terminated: bool
    syntax_valid: bool
    symbol_edit_count: int
    boundary_edit_count: int
    p_clean: float | None
    p_improve: float | None
    p_tie: float | None
    p_harm: float | None


@dataclass(frozen=True)
class GatePolicy:
    min_improve_probability: float = 1.0
    max_harm_probability: float = 0.0
    max_symbol_edits: int = 0
    max_boundary_edits: int = 0
    max_clean_probability: float = 0.0
    selected_from: str = "abstain_all"


@dataclass(frozen=True)
class GateDecision:
    received_morse: str
    output_morse: str
    accepted: bool
    reason: str


def _valid_probability(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0.0 <= value <= 1.0


def decide(received_morse: str, proposed_morse: str | None,
           evidence: GateEvidence, policy: GatePolicy) -> GateDecision:
    def abstain(reason: str) -> GateDecision:
        return GateDecision(received_morse, received_morse, False, reason)

    try:
        validate_morse(received_morse)
    except (TypeError, ValueError):
        return abstain("invalid_received_input")
    if proposed_morse is None or not isinstance(proposed_morse, str):
        return abstain("missing_proposal")
    try:
        validate_morse(proposed_morse)
    except (TypeError, ValueError):
        return abstain("invalid_proposal")
    if proposed_morse == received_morse:
        return abstain("identity_pass_through")
    if not evidence.terminated or not evidence.syntax_valid:
        return abstain("incomplete_or_invalid_decode")
    if (not 0.0 <= policy.min_improve_probability <= 1.0
            or not 0.0 <= policy.max_harm_probability <= 1.0
            or not 0.0 <= policy.max_clean_probability <= 1.0
            or type(policy.max_symbol_edits) is not int or policy.max_symbol_edits < 0
            or type(policy.max_boundary_edits) is not int or policy.max_boundary_edits < 0):
        return abstain("invalid_policy")
    if evidence.symbol_edit_count < 0 or evidence.boundary_edit_count < 0:
        return abstain("invalid_edit_evidence")
    if evidence.symbol_edit_count > policy.max_symbol_edits:
        return abstain("symbol_edit_budget")
    if evidence.boundary_edit_count > policy.max_boundary_edits:
        return abstain("boundary_edit_budget")
    scores = (evidence.p_clean, evidence.p_improve, evidence.p_tie, evidence.p_harm)
    if not all(_valid_probability(x) for x in scores):
        return abstain("uncalibrated_or_missing_risk")
    if abs(sum(scores[1:]) - 1.0) > 1e-3:
        return abstain("invalid_risk_distribution")
    if evidence.p_clean > policy.max_clean_probability:
        return abstain("clean_or_uncertain_input")
    if evidence.p_improve < policy.min_improve_probability:
        return abstain("insufficient_improvement_probability")
    if evidence.p_harm > policy.max_harm_probability:
        return abstain("harm_probability_too_high")
    return GateDecision(received_morse, proposed_morse, True, "accepted")

