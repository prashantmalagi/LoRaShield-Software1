"""Calibration-only probability scaling and gate policy selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import hashlib
import json

from .gate import GateEvidence, GatePolicy, decide
from .provenance import VerifiedPartitionManifest
from .schema import validate_morse


@dataclass(frozen=True)
class RiskRecord:
    partition: str
    received_morse: str
    proposed_morse: str | None
    reference_morse: str
    evidence: GateEvidence
    corruption_type: str


@dataclass(frozen=True)
class RiskCalibrationCase:
    source_id: str
    partition: str
    logits: tuple[float, float, float]
    label: int
    is_clean: bool
    proposal_changed_input: bool


def fingerprint_model(model) -> str:
    """Fingerprint actual model weights; caller-provided strings are not accepted."""
    weights = model.get_weights()
    if not weights:
        raise ValueError("model must be built and have weights before calibration")
    digest = hashlib.sha256()
    for weight in weights:
        digest.update(str(weight.shape).encode("ascii"))
        digest.update(weight.tobytes(order="C"))
    return digest.hexdigest()


def _provenance_digest(manifest: VerifiedPartitionManifest, model_fingerprint: str) -> str:
    body = {"manifest_fingerprint": manifest.manifest_fingerprint,
            "calibration_content_fingerprint": manifest.content_fingerprints["calibration"],
            "model_fingerprint": model_fingerprint}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RiskCalibrationResult:
    calibrator: TemperatureCalibrator
    improve_threshold: float
    harm_threshold: float
    selected_all_abstain: bool
    provenance_digest: str
    criteria: str


def calibrate_risk_and_select_threshold(cases: Sequence[RiskCalibrationCase], *,
        manifest: VerifiedPartitionManifest, model) -> RiskCalibrationResult:
    """Temperature fit and fixed safety-policy selection on calibration only.

    Predeclared criteria: zero altered clean accepts, >=0.95 calibrated improve
    probability, <=0.01 calibrated harm probability, and zero observed harmful
    accepted calibration proposals. If no nonzero policy qualifies, abstain all.
    """
    if not isinstance(manifest, VerifiedPartitionManifest) or not manifest.is_file_verified:
        raise ValueError("calibration requires a manifest verified against partition files")
    if not cases or any(row.partition != "calibration" for row in cases):
        raise ValueError("cases must be nonempty and calibration-only")
    if {row.source_id for row in cases} != set(manifest.source_ids["calibration"]):
        raise ValueError("case source IDs must exactly match verified calibration partition")
    if any(row.label not in (0, 1, 2) for row in cases):
        raise ValueError("risk labels must be improve/tie/harm classes 0/1/2")
    calibrator = fit_temperature([row.logits for row in cases], [row.label for row in cases],
                                 partition="calibration")
    accepted = []
    for row in cases:
        probs = calibrator.probabilities(row.logits)
        if probs[0] >= 0.95 and probs[2] <= 0.01:
            accepted.append((row, probs))
    safe = bool(accepted) and not any(row.is_clean and row.proposal_changed_input
                                     for row, _ in accepted) and \
           not any(row.label == 2 for row, _ in accepted)
    return RiskCalibrationResult(calibrator, 0.95, 0.01, not safe,
                                 _provenance_digest(manifest, fingerprint_model(model)),
                                 "zero altered clean accepts; zero observed harmful accepts; "
                                 "P(improve)>=0.95; P(harm)<=0.01")


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float
    fitted_on: str = "calibration"

    def probabilities(self, logits: Sequence[float]) -> tuple[float, ...]:
        if not logits or any(not math.isfinite(float(x)) for x in logits):
            raise ValueError("logits must be finite and nonempty")
        values = [float(x) / self.temperature for x in logits]
        peak = max(values)
        exps = [math.exp(x - peak) for x in values]
        total = sum(exps)
        return tuple(x / total for x in exps)


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int], *,
                    partition: str) -> TemperatureCalibrator:
    if partition != "calibration":
        raise ValueError("temperature fitting is restricted to calibration data")
    if not logits or len(logits) != len(labels):
        raise ValueError("matching nonempty logits and labels are required")
    if any(label < 0 or label >= len(row) for row, label in zip(logits, labels)):
        raise ValueError("calibration class label is out of range")
    candidates = [0.25 + i * 0.05 for i in range(156)]
    scored = []
    for temperature in candidates:
        calibrator = TemperatureCalibrator(temperature)
        nll = 0.0
        for row, label in zip(logits, labels):
            probability = calibrator.probabilities(row)[label]
            nll -= math.log(max(probability, 1e-12))
        scored.append((nll / len(labels), temperature))
    return TemperatureCalibrator(min(scored)[1])


def _distance(left: Sequence, right: Sequence) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def combined_errors(reference: str, candidate: str) -> int:
    def symbols(value): return tuple(ch for ch in value if ch in ".-")
    def boundaries(value):
        rows = []
        offset = 0
        for word_i, word in enumerate(value.split(" / ")):
            for letter_i, letter in enumerate(word.split(" ")):
                offset += len(letter)
                if letter_i < len(word.split(" ")) - 1:
                    rows.append((offset, "L"))
            if word_i < len(value.split(" / ")) - 1:
                rows.append((offset, "W"))
        return tuple(rows)
    return _distance(symbols(reference), symbols(candidate)) + _distance(
        boundaries(reference), boundaries(candidate))


def derive_risk_label(reference_morse: str, received_morse: str,
                      proposed_morse: str) -> int:
    """Offline-only target: 0 improves, 1 ties, 2 harms vs received input."""
    validate_morse(reference_morse)
    validate_morse(received_morse)
    validate_morse(proposed_morse)
    before = combined_errors(reference_morse, received_morse)
    after = combined_errors(reference_morse, proposed_morse)
    return 0 if after < before else 2 if after > before else 1


def select_gate_policy(records: Sequence[RiskRecord], *, partition: str,
                       max_harm_rate: float = 0.05) -> tuple[GatePolicy, dict]:
    if partition != "calibration" or any(row.partition != "calibration" for row in records):
        raise ValueError("gate thresholds may be selected from calibration records only")
    if not records:
        raise ValueError("calibration records are required")
    choices = []
    for max_symbols in range(1, 5):
        for max_boundaries in range(0, 4):
            for min_improve in (0.50, 0.60, 0.70, 0.80, 0.90):
                for max_harm in (0.05, 0.10, 0.20):
                    if max_harm > max_harm_rate:
                        continue
                    for max_clean in (0.10, 0.25, 0.50):
                        policy = GatePolicy(min_improve, max_harm, max_symbols,
                                            max_boundaries, max_clean, "calibration")
                        accepted = harmful = clean_altered = 0
                        for row in records:
                            decision = decide(row.received_morse, row.proposed_morse,
                                              row.evidence, policy)
                            if not decision.accepted:
                                continue
                            accepted += 1
                            before = combined_errors(row.reference_morse, row.received_morse)
                            after = combined_errors(row.reference_morse, decision.output_morse)
                            harmful += after > before
                            clean_altered += row.corruption_type == "clean" and decision.output_morse != row.reference_morse
                        if accepted and clean_altered == 0 and harmful / accepted <= max_harm_rate:
                            choices.append((accepted, -harmful, -max_symbols,
                                            -max_boundaries, policy))
    if not choices:
        return GatePolicy(), {"selected": "abstain_all", "accepted": 0,
                              "safety_criteria_met": False,
                              "criteria": "zero accepted clean alterations and harmful accepted rate <= configured cap"}
    selected = sorted(choices, key=lambda row: (
        -row[0], -row[1], -row[2], -row[3],
        -row[4].min_improve_probability, row[4].max_harm_probability,
        row[4].max_clean_probability,
    ))[0][-1]
    return selected, {"selected": "calibration_policy", "accepted": max(row[0] for row in choices),
                      "policy": selected, "safety_criteria_met": True}

