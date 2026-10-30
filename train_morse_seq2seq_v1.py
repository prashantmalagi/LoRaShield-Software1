"""Reproducible Phase 3 training pipeline for the standalone Morse seq2seq model.

No mode is implicit. ``--audit-sources-only`` generates only clean source
partitions and persists a completed overlap manifest. ``--dry-run`` uses a tiny
synthetic fixture and a temporary checkpoint. ``--train`` is the explicit full
training switch and has not been run as part of this research phase.

Training pairs are generated only for train and validation. Calibration and
locked-test sources are audited and fingerprinted but are never passed into
training, validation callbacks, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import morse_seq2seq_v1_data as data
import morse_seq2seq_v1_model as prototype

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = ROOT / "morse_seq2seq_v1_completed_manifest.json"
DEFAULT_CHECKPOINT_PATH = ROOT / "models" / "morse_seq2seq_v1_best.keras"
DEFAULT_CANDIDATE_PATH = ROOT / "models" / "morse_seq2seq_v1_candidate.keras"
DEFAULT_METADATA_PATH = ROOT / "models" / "morse_seq2seq_v1_training_metadata.json"
MODEL_SEED = data.ROOT_SEED + 1
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 4
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_ENCODER_UNITS = 24


def configure_determinism(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, and TensorFlow; enable deterministic ops if supported."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    deterministic_ops = False
    try:
        tf.config.experimental.enable_op_determinism()
        deterministic_ops = True
    except (AttributeError, RuntimeError):
        deterministic_ops = False
    return {
        "seed": seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "tensorflow_version": tf.__version__,
        "deterministic_ops_enabled": deterministic_ops,
    }


def audit_full_sources(*, seed: int = data.ROOT_SEED) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Generate only source messages, then require zero complete/word overlaps."""
    exclusions = data.existing_source_corpora()
    partitions = data.generate_source_partitions(
        seed, data.PLANNED_PARTITION_COUNTS, exclusions
    )
    overlap = data.overlap_report(partitions, exclusions)
    failed = [name for name in (
        "complete_message_overlap_count", "word_overlap_count",
        "cross_partition_message_collision_count", "cross_partition_word_collision_count",
    ) if overlap[name] != 0]
    if failed:
        raise RuntimeError(f"source overlap audit failed: {failed}")
    manifest = data.build_manifest(partitions, seed=seed, exclusions=exclusions)
    return partitions, manifest


def persist_manifest_without_overwrite(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Persist an audited manifest once; repeated runs must reproduce it exactly."""
    target = Path(path)
    serialized = data.manifest_json(manifest) + "\n"
    if target.exists():
        current = target.read_text(encoding="utf-8")
        if current != serialized:
            raise FileExistsError(
                f"manifest already exists and differs; choose a new path: {target}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized, encoding="utf-8")


def make_partition_pairs(partition: str, sources: Sequence[str], *, seed: int) -> list[data.TrainingPair]:
    """Generate deterministic variants for exactly one already-assigned split."""
    if partition not in data.PARTITION_NAMES:
        raise ValueError(f"unknown source partition: {partition}")
    partition_index = data.PARTITION_NAMES.index(partition)
    pairs = []
    for source_index, source in enumerate(sources):
        source_id = f"{partition}-{source_index:06d}"
        pair_seed = seed + partition_index * 1_000_003 + source_index * 10_007
        pairs.extend(data.make_variants_for_source(source_id, partition, source, pair_seed))
    return pairs


def make_train_validation_pairs(
    partitions: Mapping[str, Sequence[str]], *, seed: int = data.ROOT_SEED,
) -> tuple[list[data.TrainingPair], list[data.TrainingPair]]:
    """Return only train/validation pairs; calibration/test labels are untouched."""
    if set(partitions) != set(data.PARTITION_NAMES):
        raise ValueError(f"partitions must define exactly {data.PARTITION_NAMES}")
    train, validation = partitions["train"], partitions["validation"]
    train_words = {word for message in train for word in message.split(" ")}
    val_words = {word for message in validation for word in message.split(" ")}
    if set(train) & set(validation) or train_words & val_words:
        raise ValueError("train and validation source messages/words must be disjoint")
    return (
        make_partition_pairs("train", train, seed=seed),
        make_partition_pairs("validation", validation, seed=seed),
    )


def assert_pair_partition(pairs: Sequence[data.TrainingPair], expected: str) -> None:
    if any(pair.partition != expected for pair in pairs):
        raise ValueError(f"expected {expected} pairs only")


def best_epoch_from_validation(history: Mapping[str, Sequence[float]]) -> int:
    """Select the one-based epoch with minimum validation loss only."""
    values = list(history.get("val_loss", ()))
    if not values or not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("finite val_loss history is required for checkpoint selection")
    return int(np.argmin(values)) + 1


def validation_callbacks(checkpoint_path: str | Path, patience: int = DEFAULT_PATIENCE):
    """Create checkpoint/early-stop callbacks monitored only on val_loss."""
    if patience < 1:
        raise ValueError("patience must be positive")
    import tensorflow as tf

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=False, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience,
            restore_best_weights=True, verbose=0,
        ),
    ]


def _fit_batches(model: Any, train_batch: prototype.TrainingBatch,
                 validation_batch: prototype.TrainingBatch, *, epochs: int,
                 batch_size: int, callbacks: Sequence[Any], verbose: int):
    return model.fit(
        [train_batch.encoder_input_ids, train_batch.decoder_input_ids],
        train_batch.decoder_target_ids,
        sample_weight=train_batch.decoder_target_weights,
        validation_data=(
            [validation_batch.encoder_input_ids, validation_batch.decoder_input_ids],
            validation_batch.decoder_target_ids,
            validation_batch.decoder_target_weights,
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=list(callbacks),
        shuffle=True,
        verbose=verbose,
    )


def run_dry_run(seed: int = MODEL_SEED) -> dict[str, Any]:
    """One tiny synthetic train step, validation pass, checkpoint save/reload."""
    env = configure_determinism(seed)
    train_pairs = [
        data.make_training_pair("dry-train-0", "train", "E", "clean", seed + 1),
        data.make_training_pair("dry-train-1", "train", "SOS TEST",
                                "dot_dash_substitution_05", seed + 2),
    ]
    validation_pairs = [
        data.make_training_pair("dry-val-0", "validation", "A B", "clean", seed + 3),
        data.make_training_pair("dry-val-1", "validation", "HELLO",
                                "dot_dash_substitution_05", seed + 4),
    ]
    assert_pair_partition(train_pairs, "train")
    assert_pair_partition(validation_pairs, "validation")
    train_batch = prototype.prepare_training_batch(train_pairs)
    val_batch = prototype.prepare_training_batch(validation_pairs)
    model = prototype.build_model(embedding_dim=8, encoder_units=4)
    with tempfile.TemporaryDirectory(prefix="morse_seq2seq_v1_dry_run_") as temp_dir:
        checkpoint = Path(temp_dir) / "dry_run_best.keras"
        history = _fit_batches(
            model, train_batch, val_batch, epochs=1, batch_size=2,
            callbacks=validation_callbacks(checkpoint, patience=1), verbose=0,
        )
        if not checkpoint.is_file():
            raise RuntimeError("validation checkpoint was not written")
        best_model = prototype.load_model(checkpoint)
        decode = prototype.greedy_decode(best_model, "... --- ...", max_output_tokens=8)
        if len(decode.output_tokens) > 7:
            raise AssertionError("bounded decoder exceeded output limit")
        return {
            "seed": seed,
            "environment": env,
            "train_fixture_pairs": len(train_pairs),
            "validation_fixture_pairs": len(validation_pairs),
            "train_batch_shapes": {
                "encoder": list(train_batch.encoder_input_ids.shape),
                "decoder_input": list(train_batch.decoder_input_ids.shape),
                "decoder_target": list(train_batch.decoder_target_ids.shape),
            },
            "validation_loss": [float(value) for value in history.history["val_loss"]],
            "best_epoch": best_epoch_from_validation(history.history),
            "checkpoint_reload": True,
            "bounded_decode_reason": decode.termination_reason,
        }


def _metadata_for_full_training(manifest: Mapping[str, Any], environment: Mapping[str, Any],
                                history: Mapping[str, Sequence[float]], *,
                                seed: int, epochs: int, batch_size: int,
                                patience: int, embedding_dim: int,
                                encoder_units: int, train_pairs: int,
                                validation_pairs: int, candidate_path: str | Path,
                                manifest_path: str | Path) -> dict[str, Any]:
    return {
        "model_name": "LoRaShield Morse seq2seq V1 research candidate",
        "model_file": Path(candidate_path).name,
        "spec_version": data.SPEC_VERSION,
        "token_vocabulary_ids": data.TOKEN_TO_ID,
        "maximum_sequence_tokens_including_bos_eos": data.MAX_SEQUENCE_TOKENS,
        "architecture": {
            "encoder": f"Bidirectional GRU ({encoder_units} units per direction)",
            "decoder": f"Autoregressive GRU ({2 * encoder_units} units) with additive attention",
            "embedding_dim": embedding_dim,
        },
        "determinism_and_environment": dict(environment),
        "training_configuration": {
            "seed": seed,
            "epochs_requested": epochs,
            "best_epoch_by_validation_loss": best_epoch_from_validation(history),
            "batch_size": batch_size,
            "early_stopping_patience": patience,
            "checkpoint_monitor": "val_loss",
            "train_pair_count": train_pairs,
            "validation_pair_count": validation_pairs,
            "padding_loss_mask": "target token ID 0 (PAD) contributes zero loss and zero sample weight",
            "locked_test_used_for_training_or_checkpoint_selection": False,
            "calibration_used_for_training_or_checkpoint_selection": False,
        },
        "source_manifest": {
            "file": Path(manifest_path).name,
            "partition_counts": manifest["partition_audit"]["source_counts"],
            "partition_fingerprints_sha256": manifest["partition_audit"]["source_fingerprints_sha256"],
        },
        "locked_test_evaluation": "not run by the training script",
        "limitations": [
            "Synthetic alphanumeric Morse only; results do not establish real radio-channel performance.",
            "Variable-length output does not guarantee recovery of ambiguous symbol or boundary errors.",
        ],
    }


def run_full_training(*, manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
                      checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
                      candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
                      metadata_path: str | Path = DEFAULT_METADATA_PATH,
                      seed: int = MODEL_SEED, epochs: int = DEFAULT_EPOCHS,
                      batch_size: int = DEFAULT_BATCH_SIZE, patience: int = DEFAULT_PATIENCE,
                      embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                      encoder_units: int = DEFAULT_ENCODER_UNITS) -> dict[str, Any]:
    """Full pipeline implementation; this command is intentionally not run here."""
    for output in (checkpoint_path, candidate_path, metadata_path):
        if Path(output).exists():
            raise FileExistsError(f"refusing to overwrite existing research artifact: {output}")
    partitions, manifest = audit_full_sources(seed=data.ROOT_SEED)
    persist_manifest_without_overwrite(manifest, manifest_path)
    environment = configure_determinism(seed)
    train_pairs, val_pairs = make_train_validation_pairs(partitions, seed=data.ROOT_SEED)
    assert_pair_partition(train_pairs, "train")
    assert_pair_partition(val_pairs, "validation")
    train_batch = prototype.prepare_training_batch(train_pairs)
    val_batch = prototype.prepare_training_batch(val_pairs)
    model = prototype.build_model(embedding_dim=embedding_dim, encoder_units=encoder_units)
    history = _fit_batches(
        model, train_batch, val_batch, epochs=epochs, batch_size=batch_size,
        callbacks=validation_callbacks(checkpoint_path, patience), verbose=2,
    )
    if not Path(checkpoint_path).is_file():
        raise RuntimeError("no validation-selected checkpoint was produced")
    best_model = prototype.load_model(checkpoint_path)
    prototype.save_model(best_model, candidate_path)
    metadata = _metadata_for_full_training(
        manifest, environment, history.history, seed=seed, epochs=epochs,
        batch_size=batch_size, patience=patience, embedding_dim=embedding_dim,
        encoder_units=encoder_units, train_pairs=len(train_pairs),
        validation_pairs=len(val_pairs), candidate_path=candidate_path,
        manifest_path=manifest_path,
    )
    Path(metadata_path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-sources-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train", action="store_true", help="explicitly launch full synthetic training")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    args = parser.parse_args()
    try:
        if args.audit_sources_only:
            partitions, manifest = audit_full_sources()
            persist_manifest_without_overwrite(manifest, args.manifest_path)
            print("Source-only overlap audit completed; no corruption variants or model training generated.")
            print(json.dumps({
                "manifest": str(args.manifest_path),
                "partition_counts": manifest["partition_audit"]["source_counts"],
                "overlap_results": manifest["partition_audit"]["overlap_check"]["results"],
            }, indent=2))
        elif args.dry_run:
            print("Tiny deterministic synthetic dry run; no full source corpus or model artifact written.")
            print(json.dumps(run_dry_run(args.seed), indent=2))
        else:
            result = run_full_training(
                manifest_path=args.manifest_path,
                checkpoint_path=args.checkpoint_path,
                candidate_path=args.candidate_path,
                metadata_path=args.metadata_path,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
            )
            print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: seq2seq research pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
