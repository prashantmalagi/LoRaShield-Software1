"""Explicit train/validation-only V2 training entry point.

This code is implemented but was not run during package implementation. A
trained risk head requires separate out-of-fold candidate outcomes; until that
stage is completed and calibrated, the gate fails closed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from .batching import prepare_batch
from .boundary_augmentation import evaluate_predicted_boundary_candidates
from .losses import compute_loss_components
from .model import MODEL_CONFIG, build_model
from .schema import V2Example
from .checkpointing import CheckpointStore
from .provenance import partition_content_fingerprint


def load_partition(path: str | Path, expected_partition: str) -> list[V2Example]:
    if expected_partition not in ("train", "validation", "calibration"):
        raise ValueError("loader only supports train, validation, or calibration")
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        raw = json.loads(line)
        partition = raw.get("partition")
        if partition != expected_partition:
            raise ValueError(f"expected {expected_partition} rows; found {partition!r}")
        rows.append(V2Example(
            source_id=raw["source_id"], partition=partition,
            corruption_type=raw["corruption_type"], severity=float(raw["severity"]),
            seed=int(raw["seed"]), source_text=raw["source_text"],
            received_morse=raw["received_morse"], reference_morse=raw["reference_morse"],
            symbol_operations=tuple(raw["symbol_operations"]),
            insertions_before=tuple(tuple(x) for x in raw["insertions_before"]),
            output_alignment=tuple(raw["output_alignment"]),
            boundary_labels=tuple(raw["boundary_labels"]), is_clean=bool(raw["is_clean"]),
        ))
    if not rows:
        raise ValueError(f"{expected_partition} dataset is empty")
    return rows


def balanced_batch_indices(rows: Sequence[V2Example], batch_size: int, seed: int) -> list[list[int]]:
    """Shuffle examples while placing a clean identity case in each feasible batch."""
    if batch_size < 2:
        raise ValueError("batch size must be at least two for clean/corrupt mixing")
    clean = [i for i, row in enumerate(rows) if row.is_clean]
    noisy = [i for i, row in enumerate(rows) if not row.is_clean]
    if not clean or not noisy:
        raise ValueError("training requires both clean and corrupted examples")
    rng = random.Random(seed)
    rng.shuffle(clean); rng.shuffle(noisy)
    batch_count = (len(rows) + batch_size - 1) // batch_size
    if len(clean) < batch_count:
        raise ValueError("not enough clean examples to include one in every training batch")
    batches = []
    for _ in range(batch_count):
        batch = [clean.pop()]
        while len(batch) < batch_size and noisy:
            batch.append(noisy.pop())
        while len(batch) < batch_size and clean:
            batch.append(clean.pop())
        rng.shuffle(batch)
        batches.append(batch)
    # If corruption count is small, any remaining clean examples become further batches.
    while clean:
        batches.append(clean[:batch_size])
        del clean[:batch_size]
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(rows))):
        raise AssertionError("balanced batch plan must include each example exactly once")
    return batches


def _epoch(model, rows: Sequence[V2Example], *, optimizer=None, batch_size: int,
           seed: int) -> float:
    import tensorflow as tf
    if optimizer is not None:
        batch_plan = balanced_batch_indices(rows, batch_size, seed)
    else:
        batch_plan = [list(range(start, min(start + batch_size, len(rows))))
                      for start in range(0, len(rows), batch_size)]
    total, batch_total = 0.0, 0
    for batch_indices in batch_plan:
        batch = prepare_batch([rows[i] for i in batch_indices],
                              boundary_augmentation_seed=seed if optimizer is not None else None)
        if optimizer is None:
            outputs = model(batch.inputs, training=False)
            components = compute_loss_components(outputs, batch.targets, batch.masks)
        else:
            with tf.GradientTape() as tape:
                outputs = model(batch.inputs, training=True)
                components = compute_loss_components(outputs, batch.targets, batch.masks)
            gradients = tape.gradient(components["total"], model.trainable_variables)
            optimizer.apply_gradients((g, v) for g, v in zip(gradients, model.trainable_variables)
                                      if g is not None)
        total += float(components["total"].numpy())
        batch_total += 1
    return total / max(batch_total, 1)


def train_model(train_rows: Sequence[V2Example], validation_rows: Sequence[V2Example],
                output_weights: str | Path, *, epochs: int = 20,
                batch_size: int = 32, seed: int = 884201, resume: bool = False) -> dict:
    if not train_rows or any(row.partition != "train" for row in train_rows):
        raise ValueError("gradient updates accept train partition only")
    if not validation_rows or any(row.partition != "validation" for row in validation_rows):
        raise ValueError("checkpoint selection accepts validation partition only")
    train_ids = {row.source_id for row in train_rows}
    validation_ids = {row.source_id for row in validation_rows}
    train_words = {word for row in train_rows for word in row.source_text.split()}
    validation_words = {word for row in validation_rows for word in row.source_text.split()}
    if train_ids & validation_ids or train_words & validation_words:
        raise ValueError("train and validation sources must be message/word disjoint")
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs must be positive and batch size must be at least two")
    target = Path(output_weights)
    store = CheckpointStore(target)
    store.initialize(resume=resume)
    import tensorflow as tf
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    first_batch = prepare_batch([train_rows[0]])
    model(first_batch.inputs, training=False)
    if hasattr(optimizer, "build"):
        optimizer.build(model.trainable_variables)
    train_fingerprint = partition_content_fingerprint(train_rows)
    validation_fingerprint = partition_content_fingerprint(validation_rows)
    start_epoch = 0
    previous = {}
    if resume:
        start_epoch, previous = store.restore(model=model, optimizer=optimizer)
        if previous.get("train_source_fingerprint") != train_fingerprint or \
           previous.get("validation_source_fingerprint") != validation_fingerprint:
            raise ValueError("resume partition fingerprints do not match checkpoint")
    best_validation = float("inf")
    best_epoch = None
    history = []
    if resume:
        best_validation = float(previous.get("best_validation_loss", float("inf")))
        best_epoch = previous.get("best_epoch")
        history = list(previous.get("history", []))
        if start_epoch >= epochs:
            return previous
    for epoch in range(start_epoch, epochs):
        train_loss = _epoch(model, train_rows, optimizer=optimizer,
                            batch_size=batch_size, seed=seed + epoch)
        validation_loss = _epoch(model, validation_rows, optimizer=None,
                                 batch_size=batch_size, seed=seed)
        predicted_boundary = evaluate_predicted_boundary_candidates(
            model, validation_rows,
            batch_builder=lambda examples, **kwargs: prepare_batch(examples, **kwargs))
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "validation_predicted_boundary": predicted_boundary})
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
        metadata = {"model_name": "Morse Correction V2 research candidate",
                "architecture": MODEL_CONFIG, "best_epoch": best_epoch,
                "latest_epoch": epoch + 1, "best_validation_loss": best_validation,
                "history": history,
                "risk_head_trained": False,
                "automatic_correction_ready": False,
                "training_partition": "train", "checkpoint_selection_partition": "validation",
                "train_source_fingerprint": train_fingerprint,
                "validation_source_fingerprint": validation_fingerprint,
                "calibration_loaded": False, "locked_test_loaded": False}
        store.commit(model=model, optimizer=optimizer, epoch=epoch + 1, metadata=metadata)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output-checkpoint", "--output-weights", dest="output_weights",
                        type=Path, required=True,
                        help="new checkpoint directory; existing runs require --resume")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=884201)
    parser.add_argument("--execute-training", action="store_true",
                        help="explicitly run optimizer updates; omit for help/preflight only")
    parser.add_argument("--resume", action="store_true",
                        help="resume optimizer/model state from the checkpoint directory")
    args = parser.parse_args(argv)
    if not args.execute_training:
        parser.error("training is disabled unless --execute-training is explicitly supplied")
    train_rows = load_partition(args.train_data, "train")
    validation_rows = load_partition(args.validation_data, "validation")
    result = train_model(train_rows, validation_rows, args.output_weights,
                         epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                         resume=args.resume)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

