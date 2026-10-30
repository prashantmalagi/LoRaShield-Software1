"""Source-grouped OOF proposal and risk-head training runner.

The CLI requires `--execute-training`; merely importing or inspecting it does
not train. This module was not invoked during implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import tempfile
from typing import Sequence

from .checkpointing import CheckpointStore
from .inference import propose
from .model import MODEL_CONFIG, build_model
from .provenance import VerifiedPartitionManifest, partition_content_fingerprint, verify_manifest_files
from .risk_training import (CandidateProposal, ProposalInput, build_verified_risk_batches,
                            generate_oof_risk_examples, train_risk_head)
from .train import _epoch, load_partition


def fit_oof_fold(fit_rows, *, epochs: int, batch_size: int, seed: int):
    """Fit one fold model using fit rows only; no heldout data enters gradients."""
    import tensorflow as tf
    if not fit_rows or any(row.partition != "train" for row in fit_rows):
        raise ValueError("OOF base-model fitting accepts train rows only")
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    for epoch in range(epochs):
        _epoch(model, fit_rows, optimizer=optimizer, batch_size=batch_size, seed=seed + epoch)
    return model


def predict_oof_fold(model, heldout: Sequence[ProposalInput]) -> tuple[CandidateProposal, ...]:
    """Generate one reference-free decoded candidate per heldout input."""
    outputs = []
    for row in heldout:
        result = propose(model, row.received_morse)
        outputs.append(CandidateProposal(result.proposed_morse, result.symbol_alignment,
            None if result.proposed_morse is not None else result.reason))
    return tuple(outputs)


def run_oof_risk_training(train_rows, base_model, risk_optimizer, output_checkpoint, *,
                          verified_manifest: VerifiedPartitionManifest,
                          folds=5, fold_epochs=8, risk_epochs=5, batch_size=32,
                          seed=884201, fold_fit=fit_oof_fold, fold_predict=predict_oof_fold):
    """Create OOF proposal batches, train only the supplied model's risk head,
    then atomically save a new checkpoint. Caller supplies the frozen base model.
    """
    if not train_rows or any(row.partition != "train" for row in train_rows):
        raise ValueError("OOF runner accepts train partition rows only")
    if not isinstance(verified_manifest, VerifiedPartitionManifest) or \
       not verified_manifest.is_file_verified:
        raise ValueError("OOF runner requires a manifest verified against all partition files")
    if partition_content_fingerprint(train_rows) != verified_manifest.content_fingerprints["train"] or \
       {row.source_id for row in train_rows} != set(verified_manifest.source_ids["train"]):
        raise ValueError("OOF train rows do not match verified train partition content")
    target = Path(output_checkpoint)
    if target.exists():
        raise FileExistsError("risk checkpoint path exists; refusing overwrite")
    cache = {}

    def fit_and_propose(fit_rows, heldout_inputs):
        fit_ids = {row.source_id for row in fit_rows}
        held_ids = {row.source_id for row in heldout_inputs}
        if fit_ids & held_ids:
            raise ValueError("heldout OOF source appeared in base-model fit rows")
        # All corruption variants of one heldout source travel together.
        key = tuple(sorted(fit_ids))
        if key not in cache:
            cache[key] = fold_fit(fit_rows, epochs=fold_epochs, batch_size=batch_size,
                                  seed=seed + len(cache) * 1009)
        return fold_predict(cache[key], heldout_inputs)

    generated = generate_oof_risk_examples(train_rows, fit_and_propose,
                                           folds=folds, seed=seed)
    if not generated.examples:
        raise ValueError("OOF runner produced no valid candidate proposals")
    verified_batches = build_verified_risk_batches(generated.examples, batch_size=batch_size)
    # train_risk_head restores the base model's trainability flags after updates.
    risk_report = train_risk_head(base_model, verified_batches, risk_optimizer, epochs=risk_epochs)
    metadata = {
        "model_name": "Morse Correction V2 candidate-conditioned risk model",
        "architecture": MODEL_CONFIG,
        "risk_head_trained": True,
        "automatic_correction_ready": False,
        "training_partition": "train",
        "training_partition": "train",
        "oof_folds": folds,
        "oof_examples": len(generated.examples),
        "rejected_proposals": [{"source_id": row.source_id, "fold": row.fold,
                                "reason": row.reason} for row in generated.rejected],
        "risk_training": risk_report,
        "locked_test_loaded": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v2-risk-staging-", dir=target.parent) as staging:
        staging_checkpoint = Path(staging) / "checkpoint"
        store = CheckpointStore(staging_checkpoint)
        store.initialize(resume=False)
        store.commit(model=base_model, optimizer=risk_optimizer, epoch=1, metadata=metadata)
        if target.exists():
            raise FileExistsError("risk checkpoint path appeared during commit")
        os.replace(staging_checkpoint, target)
    return generated, metadata


def restore_base_model(checkpoint_dir: str | Path):
    """Restore stage-one checkpoint state for OOF risk training; no update here."""
    import tensorflow as tf
    model = build_model()
    # Build the model variables from a tiny synthetic, reference-free input.
    import numpy as np
    zeros = np.zeros((1, 1), dtype=np.int32)
    model({"symbol_ids": np.ones((1, 1), dtype=np.int32),
           "input_gap_ids": zeros, "candidate_symbol_ids": np.ones((1, 1), dtype=np.int32),
           "candidate_alignment": zeros, "candidate_gap_ids": zeros,
           "candidate_boundary_ids": zeros}, training=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)
    epoch, metadata = CheckpointStore(checkpoint_dir).restore(model=model, optimizer=optimizer)
    if not epoch or metadata.get("training_partition") != "train":
        raise ValueError("base checkpoint is not a completed train-partition checkpoint")
    return model, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-epochs", type=int, default=8)
    parser.add_argument("--risk-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=884201)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute_training:
        parser.error("OOF/risk optimizer updates require --execute-training")
    rows = load_partition(args.train_data, "train")
    manifest = verify_manifest_files(args.manifest)
    model, base_metadata = restore_base_model(args.base_checkpoint)
    if base_metadata.get("risk_head_trained"):
        parser.error("base checkpoint must be a stage-one model before OOF risk training")
    import tensorflow as tf
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    generated, metadata = run_oof_risk_training(rows, model, optimizer, args.output_checkpoint,
        verified_manifest=manifest,
        folds=args.folds, fold_epochs=args.fold_epochs, risk_epochs=args.risk_epochs,
        batch_size=args.batch_size, seed=args.seed)
    print(json.dumps({"oof_examples": len(generated.examples),
                      "rejected_proposals": len(generated.rejected), "metadata": metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
