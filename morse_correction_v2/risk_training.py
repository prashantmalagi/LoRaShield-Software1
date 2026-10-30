"""Leakage-safe source-held-out proposal generation for risk-head datasets.

This module defines orchestration contracts only; it does not train or load a
model. Callbacks supplied by a future approved runner are responsible for
fitting one base model per fold and decoding held-out train examples.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Sequence

from .calibration import derive_risk_label
from .schema import V2Example, boundary_targets, validate_morse
from .boundary_augmentation import BoundaryCandidate, validate_candidate_alignment
from .batching import prepare_batch

_OOF_TOKEN = object()


@dataclass(frozen=True)
class RiskExample:
    source_id: str
    partition: str
    received_morse: str
    proposed_morse: str
    reference_morse: str
    risk_label: int
    fold: int
    symbol_alignment: tuple[int, ...]
    source_example: V2Example
    fold_provenance: "OOFFoldProvenance"


@dataclass(frozen=True)
class ProposalInput:
    """Reference-free heldout input exposed to the base-model callback."""
    source_id: str
    received_morse: str
    corruption_type: str


@dataclass(frozen=True)
class CandidateProposal:
    proposed_morse: str | None
    symbol_alignment: tuple[int, ...]
    rejection_reason: str | None = None


@dataclass(frozen=True)
class RejectedProposal:
    source_id: str
    fold: int
    reason: str


@dataclass(frozen=True)
class OOFFoldProvenance:
    fold: int
    base_fit_source_ids: tuple[str, ...]
    heldout_source_ids: tuple[str, ...]
    digest: str
    _token: object

    @property
    def is_verified(self):
        return self._token is _OOF_TOKEN


@dataclass(frozen=True)
class OOFGenerationResult:
    examples: tuple[RiskExample, ...]
    rejected: tuple[RejectedProposal, ...]


@dataclass(frozen=True)
class VerifiedRiskBatch:
    inputs: dict
    risk_labels: tuple[int, ...]
    source_ids: tuple[str, ...]
    fold_provenance: OOFFoldProvenance
    partition: str
    _token: object

    @property
    def is_verified(self):
        return self._token is _OOF_TOKEN


def source_held_out_folds(rows: Sequence[V2Example], folds: int = 5,
                          seed: int = 884201) -> tuple[tuple[tuple[V2Example, ...], tuple[V2Example, ...]], ...]:
    """Return (base-fit, risk-heldout) folds grouped by source_id."""
    if folds < 2 or not rows or any(r.partition != "train" for r in rows):
        raise ValueError("OOF risk folds require at least two folds of train rows only")
    groups: dict[str, list[V2Example]] = {}
    for row in rows:
        groups.setdefault(row.source_id, []).append(row)
    if len(groups) < folds:
        raise ValueError("there must be at least one distinct source per fold")
    # Stable deterministic shuffle without depending on global random state.
    import random
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    buckets = [[] for _ in range(folds)]
    for index, source_id in enumerate(ids):
        buckets[index % folds].append(source_id)
    result = []
    for bucket in buckets:
        held_ids = set(bucket)
        heldout = tuple(row for row in rows if row.source_id in held_ids)
        fit = tuple(row for row in rows if row.source_id not in held_ids)
        if not fit or held_ids & {r.source_id for r in fit}:
            raise AssertionError("OOF source leakage")
        result.append((fit, heldout))
    return tuple(result)


def generate_oof_risk_examples(
    rows: Sequence[V2Example],
    fit_and_propose: Callable[[Sequence[V2Example], Sequence[ProposalInput]], Sequence[CandidateProposal]],
    *, folds: int = 5, seed: int = 884201,
) -> OOFGenerationResult:
    """Fit callback gets train-fold rows and heldout rows; labels use train refs."""
    result, rejected = [], []
    for fold_index, (fit_rows, heldout_rows) in enumerate(source_held_out_folds(rows, folds, seed)):
        held_ids = {r.source_id for r in heldout_rows}
        if held_ids & {r.source_id for r in fit_rows}:
            raise ValueError("a risk validation source appeared in its base-model fit fold")
        fit_ids = tuple(sorted({r.source_id for r in fit_rows}))
        held_ids_tuple = tuple(sorted(held_ids))
        digest = hashlib.sha256(json.dumps(
            {"fold": fold_index, "fit": fit_ids, "heldout": held_ids_tuple},
            sort_keys=True).encode("utf-8")).hexdigest()
        provenance = OOFFoldProvenance(fold_index, fit_ids, held_ids_tuple, digest, _OOF_TOKEN)
        proposal_inputs = tuple(ProposalInput(r.source_id, r.received_morse, r.corruption_type)
                               for r in heldout_rows)
        proposals = tuple(fit_and_propose(fit_rows, proposal_inputs))
        if len(proposals) != len(heldout_rows):
            raise ValueError("proposal callback must return one proposal per heldout row")
        for row, proposal in zip(heldout_rows, proposals):
            if row.partition != "train":
                raise ValueError("only permitted train references may label risk examples")
            try:
                if not isinstance(proposal, CandidateProposal):
                    raise ValueError("proposal must include output Morse and symbol alignment")
                if proposal.proposed_morse is None:
                    raise ValueError(proposal.rejection_reason or "proposal generation failed")
                validate_morse(proposal.proposed_morse)
                candidate_symbols = "".join(c for c in proposal.proposed_morse if c in ".-")
                source_symbols = "".join(c for c in row.received_morse if c in ".-")
                candidate = BoundaryCandidate(candidate_symbols, proposal.symbol_alignment,
                                               boundary_targets(proposal.proposed_morse))
                validate_candidate_alignment(candidate, len(source_symbols))
                previous = -1
                for source_index in proposal.symbol_alignment:
                    if source_index >= 0:
                        if source_index <= previous:
                            raise ValueError("alignment source positions must increase strictly")
                        previous = source_index
            except (TypeError, ValueError) as exc:
                rejected.append(RejectedProposal(row.source_id, fold_index, str(exc)))
                continue
            result.append(RiskExample(row.source_id, "train", row.received_morse,
                                      proposal.proposed_morse, row.reference_morse,
                                      derive_risk_label(row.reference_morse,
                                                        row.received_morse, proposal.proposed_morse),
                                      fold_index, tuple(proposal.symbol_alignment), row, provenance))
    return OOFGenerationResult(tuple(result), tuple(rejected))


def build_verified_risk_batches(examples: Sequence[RiskExample], *, batch_size: int = 32):
    """Materialize proposal tensors only from validated OOF results."""
    if batch_size < 1 or not examples:
        raise ValueError("nonempty OOF examples and positive batch size are required")
    groups: dict[int, list[RiskExample]] = {}
    for row in examples:
        provenance = row.fold_provenance
        expected_digest = hashlib.sha256(json.dumps(
            {"fold": provenance.fold, "fit": provenance.base_fit_source_ids,
             "heldout": provenance.heldout_source_ids}, sort_keys=True).encode("utf-8")).hexdigest()
        if not provenance.is_verified or provenance.digest != expected_digest:
            raise ValueError("unverified OOF fold provenance")
        if row.partition != "train" or row.source_id != row.source_example.source_id or \
           row.source_example.partition != "train" or row.source_id not in provenance.heldout_source_ids or \
           set(provenance.heldout_source_ids) & set(provenance.base_fit_source_ids):
            raise ValueError("invalid OOF fold provenance")
        if row.received_morse != row.source_example.received_morse or \
           row.reference_morse != row.source_example.reference_morse:
            raise ValueError("risk row does not match its heldout source example")
        validate_morse(row.proposed_morse)
        if derive_risk_label(row.reference_morse, row.received_morse,
                             row.proposed_morse) != row.risk_label:
            raise ValueError("risk label does not match candidate/reference outcome")
        groups.setdefault(row.fold, []).append(row)
    batches = []
    for fold, group in sorted(groups.items()):
        provenance = group[0].fold_provenance
        if any(row.fold_provenance != provenance for row in group):
            raise ValueError("mixed OOF fold provenance in batch group")
        for start in range(0, len(group), batch_size):
            chunk = group[start:start + batch_size]
            candidate_rows = [BoundaryCandidate(
                "".join(c for c in row.proposed_morse if c in ".-"), row.symbol_alignment,
                boundary_targets(row.proposed_morse)) for row in chunk]
            batch = prepare_batch([row.source_example for row in chunk],
                                  boundary_candidates=candidate_rows)
            batches.append(VerifiedRiskBatch(batch.inputs,
                tuple(row.risk_label for row in chunk),
                tuple(row.source_id for row in chunk), provenance, "train", _OOF_TOKEN))
    return tuple(batches)


def train_risk_head(model, batches, optimizer, *, epochs: int = 1) -> dict:
    """Train only risk_hidden/risk_logits from OOF train batches.

    Accepts only internally minted `VerifiedRiskBatch` instances generated by
    the OOF tensor builder. Validation/calibration/locked partitions are rejected.
    """
    import tensorflow as tf
    if epochs < 1 or not batches:
        raise ValueError("positive epochs and nonempty OOF batches are required")
    allowed = {"risk_hidden", "risk_logits"}
    if not {layer.name for layer in model.layers if layer.name in allowed} == allowed:
        raise ValueError("model has no isolated risk-head layers")
    original_trainable = {layer.name: layer.trainable for layer in model.layers}
    losses = []
    try:
        for layer in model.layers:
            layer.trainable = layer.name in allowed
        variables = [v for layer in model.layers if layer.trainable for v in layer.trainable_variables]
        for batch in batches:
            if not isinstance(batch, VerifiedRiskBatch) or not batch.is_verified:
                raise ValueError("risk training requires internally verified OOF batches")
            if batch.partition != "train" or set(batch.source_ids) - set(batch.fold_provenance.heldout_source_ids):
                raise ValueError("risk batch sources are not heldout train sources")
            if set(batch.fold_provenance.heldout_source_ids) & set(batch.fold_provenance.base_fit_source_ids):
                raise ValueError("heldout sources overlap base-model training fold")
            labels = tf.convert_to_tensor(batch.risk_labels, dtype=tf.int32)
            if bool(tf.reduce_any(tf.logical_or(labels < 0, labels > 2)).numpy()):
                raise ValueError("risk labels must be improve/tie/harm classes 0/1/2")
            with tf.GradientTape() as tape:
                logits = model(batch.inputs, training=True)["risk_logits"]
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True))
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients((g, v) for g, v in zip(gradients, variables) if g is not None)
            losses.append(float(loss.numpy()))
    finally:
        for layer in model.layers:
            layer.trainable = original_trainable[layer.name]
    return {"risk_loss": sum(losses) / len(losses), "risk_head_layers": sorted(allowed),
            "training_partition": "train", "locked_test_loaded": False}
