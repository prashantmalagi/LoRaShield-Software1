"""Train and evaluate a standalone V3 synthetic Morse denoiser.

This script writes only models/lorashield_morse_correction_v3.keras and its
separate metadata JSON (full run). ``--smoke`` performs a tiny one-epoch run
in a temporary directory and never writes a project model artifact.

The generated data contains clean messages and independent random dot/dash
substitutions only. It is not a model of real LoRa channels, burst errors,
timing variation, insertions, deletions, or realistic language frequencies.
The resulting candidate is experimental and is not integrated into the app.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
SEQUENCE_LENGTH = 120
VOCAB = ["<PAD>", "<UNK>", ".", "-", " "]
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)
NOISE_LEVELS = (0, 5, 10, 15, 20)
EVAL_SEED = 20261029  # Same deterministic alphanumeric corpus seed as V3 evaluator.
TRAIN_SEED, VALIDATION_SEED = 731941, 562813

# Reuse the exact Morse conventions and edit metrics exercised by the existing
# standalone evaluator. This import has no model-loading or logging side effect.
from evaluate_morse_correction_v3 import (  # noqa: E402
    CHARSET,
    MORSE,
    _boundary_edit_distance,
    _edit_rate,
    _encode,
    _noise,
)


def _messages(seed: int, count: int, exclude: set[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    result = []
    seen: set[str] = set(exclude or ())
    while len(result) < count:
        text = "".join(rng.choice(CHARSET) for _ in range(rng.randint(3, 8)))
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _encoded_messages(texts: list[str]) -> list[str]:
    encoded = [_encode(text) for text in texts]
    too_long = [text for text, morse in zip(texts, encoded) if len(morse) > SEQUENCE_LENGTH]
    if too_long:
        raise ValueError(f"{len(too_long)} generated messages exceed sequence length {SEQUENCE_LENGTH}")
    return encoded


def _split_sources(train_n: int, validation_n: int, test_n: int) -> dict[str, list[str]]:
    # Independent deterministic streams; uniqueness is checked before variants
    # are created so no source message can leak across partitions.
    test = _messages(EVAL_SEED, test_n)
    validation = _messages(VALIDATION_SEED, validation_n, set(test))
    train = _messages(TRAIN_SEED, train_n, set(test) | set(validation))
    result = {"train": train, "validation": validation, "test": test}
    sets = {name: set(values) for name, values in result.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("Unique source messages overlap across data partitions")
    return result


def _encode_ids(morse: str) -> list[int]:
    mapping = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, ".": DOT_ID, "-": DASH_ID, " ": SPACE_ID}
    return [mapping.get(ch, UNK_ID) for ch in morse]


def _make_pairs(texts: list[str], seed: int) -> tuple[Any, Any, Any, list[str], list[str]]:
    import numpy as np

    clean_strings = _encoded_messages(texts)
    rng = random.Random(seed)
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    weights: list[list[float]] = []
    pair_clean: list[str] = []
    pair_received: list[str] = []
    for clean in clean_strings:
        # Each source yields one clean self-pair and one pair at every noise
        # level. All variants remain inside this source's assigned partition.
        for noise_pct in NOISE_LEVELS:
            received = _noise(clean, noise_pct / 100.0, rng)
            source_ids = _encode_ids(received)
            target_ids = _encode_ids(clean)
            actual_length = len(source_ids)
            pad_count = SEQUENCE_LENGTH - actual_length
            if len(target_ids) != actual_length or pad_count < 0:
                raise AssertionError("Input/target alignment or sequence length mismatch")
            input_row = source_ids + [PAD_ID] * pad_count
            target_row = target_ids + [PAD_ID] * pad_count
            weight_row = [1.0] * actual_length + [0.0] * pad_count
            # Verify token IDs and masked suffix against the serialized contract.
            if any(token not in range(len(VOCAB)) for token in input_row + target_row):
                raise AssertionError("Token ID outside metadata vocabulary")
            if input_row[actual_length:] != [PAD_ID] * pad_count:
                raise AssertionError("Input PAD encoding mismatch")
            if target_row[actual_length:] != [PAD_ID] * pad_count or any(weight_row[actual_length:]):
                raise AssertionError("Target PAD mask mismatch")
            inputs.append(input_row)
            targets.append(target_row)
            weights.append(weight_row)
            pair_clean.append(clean)
            pair_received.append(received)
    return (np.asarray(inputs, dtype="int32"), np.asarray(targets, dtype="int32"),
            np.asarray(weights, dtype="float32"), pair_clean, pair_received)


def _build_model(tf):
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int32", name="morse_token_ids")
    x = tf.keras.layers.Embedding(input_dim=len(VOCAB), output_dim=32,
                                  mask_zero=True, name="token_embedding")(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(48, return_sequences=True, zero_output_for_mask=True),
        name="bidirectional_gru",
    )(x)
    outputs = tf.keras.layers.Dense(len(VOCAB), activation="softmax", name="token_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="lorashield_morse_correction_v3")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
    )
    return model


def _predict_strings(model, received_strings: list[str]) -> list[str]:
    import numpy as np

    inputs = []
    for received in received_strings:
        ids = _encode_ids(received)
        if len(ids) > SEQUENCE_LENGTH:
            raise ValueError("Evaluation message exceeds model sequence length")
        inputs.append(ids + [PAD_ID] * (SEQUENCE_LENGTH - len(ids)))
    probabilities = model.predict(np.asarray(inputs, dtype="int32"), verbose=0)
    outputs: list[str] = []
    id_to_char = {PAD_ID: "<PAD>", UNK_ID: "<UNK>", DOT_ID: ".", DASH_ID: "-", SPACE_ID: " "}
    for received, scores in zip(received_strings, probabilities):
        decoded = []
        for index, predicted_id in enumerate(scores[:len(received)].argmax(axis=-1)):
            char = id_to_char[int(predicted_id)]
            decoded.append(received[index] if char in ("<PAD>", "<UNK>") else char)
        outputs.append("".join(decoded))
    return outputs


def _evaluate(model, clean_strings: list[str], seed: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for noise_pct in NOISE_LEVELS:
        rng = random.Random(seed + noise_pct)
        noisy = [_noise(clean, noise_pct / 100.0, rng) for clean in clean_strings]
        corrected = _predict_strings(model, noisy)
        base_errors = cand_errors = base_symbol_errors = cand_symbol_errors = 0
        base_den = cand_den = base_symbol_den = cand_symbol_den = 0
        improved = unchanged = worsened = 0
        for clean, received, candidate in zip(clean_strings, noisy, corrected):
            base, _ = _edit_rate(clean, received)
            cand, _ = _edit_rate(clean, candidate)
            base_sym, _ = _edit_rate(clean, received, symbols_only=True)
            cand_sym, _ = _edit_rate(clean, candidate, symbols_only=True)
            base_errors += base
            cand_errors += cand
            base_symbol_errors += base_sym
            cand_symbol_errors += cand_sym
            base_den += max(len(clean), len(received), 1)
            cand_den += max(len(clean), len(candidate), 1)
            base_symbol_den += max(len(clean.replace(" ", "")), len(received.replace(" ", "")), 1)
            cand_symbol_den += max(len(clean.replace(" ", "")), len(candidate.replace(" ", "")), 1)
            if cand < base:
                improved += 1
            elif cand > base:
                worsened += 1
            else:
                unchanged += 1
        rows[f"noise_{noise_pct}_percent"] = {
            "messages": len(clean_strings),
            "baseline_boundary_edit_errors": base_errors,
            "baseline_boundary_edit_rate": base_errors / base_den,
            "candidate_boundary_edit_errors": cand_errors,
            "candidate_boundary_edit_rate": cand_errors / cand_den,
            "baseline_dot_dash_edit_errors": base_symbol_errors,
            "baseline_dot_dash_edit_rate": base_symbol_errors / base_symbol_den,
            "candidate_dot_dash_edit_errors": cand_symbol_errors,
            "candidate_dot_dash_edit_rate": cand_symbol_errors / cand_symbol_den,
            "improved": improved,
            "unchanged": unchanged,
            "worsened": worsened,
        }
    clean_outputs = _predict_strings(model, clean_strings)
    rows["clean_preservation"] = {
        "messages": len(clean_strings),
        "exactly_preserved": sum(a == b for a, b in zip(clean_strings, clean_outputs)),
        "rate": sum(a == b for a, b in zip(clean_strings, clean_outputs)) / max(len(clean_strings), 1),
    }
    return rows


def _assert_metadata_compatibility(model, metadata: dict[str, Any]) -> None:
    in_shape, out_shape = model.input_shape, model.output_shape
    if list(metadata["vocabulary"]) != VOCAB:
        raise AssertionError("Saved metadata vocabulary mismatch")
    if metadata["token_ids"] != {token: i for i, token in enumerate(VOCAB)}:
        raise AssertionError("Saved metadata token ID mapping mismatch")
    if in_shape != (None, metadata["sequence_length"]):
        raise AssertionError(f"Model input {in_shape} does not match metadata")
    if out_shape != (None, metadata["sequence_length"], len(metadata["vocabulary"])):
        raise AssertionError(f"Model output {out_shape} does not match metadata")


def _run(args: argparse.Namespace) -> int:
    try:
        import numpy as np
        import tensorflow as tf
    except Exception as exc:
        print(f"ERROR: TensorFlow/NumPy could not be imported: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.smoke:
        train_count, validation_count, test_count = 40, 12, 24
        epochs, batch_size = 1, 8
        model_path: Path
        with tempfile.TemporaryDirectory(prefix="lorashield_v3_smoke_") as tmpdir:
            model_path = Path(tmpdir) / "smoke.keras"
            return _train_evaluate(tf, np, model_path, train_count, validation_count,
                                   test_count, epochs, batch_size, args.seed, smoke=True)

    if MODEL_PATH.exists() or METADATA_PATH.exists():
        print("ERROR: V3 output already exists; refusing to overwrite model or metadata.", file=sys.stderr)
        print(f"  model: {MODEL_PATH} (exists={MODEL_PATH.exists()})", file=sys.stderr)
        print(f"  metadata: {METADATA_PATH} (exists={METADATA_PATH.exists()})", file=sys.stderr)
        return 2
    return _train_evaluate(tf, np, MODEL_PATH, args.train_messages,
                           args.validation_messages, args.test_messages,
                           args.epochs, args.batch_size, args.seed, smoke=False)


def _train_evaluate(tf, np, model_path: Path, train_n: int, validation_n: int,
                    test_n: int, epochs: int, batch_size: int, seed: int,
                    smoke: bool) -> int:
    tf.keras.utils.set_random_seed(seed)
    partitions = _split_sources(train_n, validation_n, test_n)
    train_x, train_y, train_w, _, _ = _make_pairs(partitions["train"], seed + 1)
    val_x, val_y, val_w, _, _ = _make_pairs(partitions["validation"], seed + 2)
    test_clean = _encoded_messages(partitions["test"])
    model = _build_model(tf)

    # Architectural/data contract checks before fitting.
    if model.input_shape != (None, SEQUENCE_LENGTH) or model.output_shape != (None, SEQUENCE_LENGTH, len(VOCAB)):
        raise AssertionError("Model input/output shapes do not match V3 token metadata")
    if train_x.shape != train_y.shape or train_x.shape != train_w.shape:
        raise AssertionError("Training input, target, and mask shapes differ")
    if not np.isin(train_x, np.arange(len(VOCAB))).all() or not np.isin(train_y, np.arange(len(VOCAB))).all():
        raise AssertionError("Generated token ID outside V3 vocabulary")

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                   restore_best_weights=True)]
    history = model.fit(
        train_x, train_y, sample_weight=train_w,
        validation_data=(val_x, val_y, val_w),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
        verbose=2,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    reloaded = tf.keras.models.load_model(model_path, compile=False)
    preliminary_eval = _evaluate(reloaded, test_clean, EVAL_SEED)
    metadata: dict[str, Any] = {
        "model_name": "LoRaShield Morse Correction V3 candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": model_path.name,
        "sequence_length": SEQUENCE_LENGTH,
        "vocabulary": VOCAB,
        "token_ids": {token: i for i, token in enumerate(VOCAB)},
        "padding_id": PAD_ID,
        "input_format": "Fixed-length int32 character IDs for dot, dash, spaces; right padded",
        "output_format": "Per-position softmax over V3 vocabulary",
        "masking": "PAD id 0 has zero sample weight for loss and weighted token accuracy; embedding mask_zero=True",
        "architecture": {"embedding_dim": 32, "bidirectional_gru_units_per_direction": 48,
                         "output": "Dense(5, softmax)"},
        "training": {
            "seed": seed, "train_source_seed": TRAIN_SEED, "validation_source_seed": VALIDATION_SEED,
            "evaluation_corpus_seed": EVAL_SEED, "epochs_requested": epochs,
            "epochs_completed": len(history.history.get("loss", [])), "batch_size": batch_size,
            "optimizer": "Adam(0.001)", "loss": "SparseCategoricalCrossentropy with zero sample weights for PAD",
            "noise_levels_percent": list(NOISE_LEVELS),
            "pairs_per_source": len(NOISE_LEVELS),
            "clean_pairs_included": True,
            "partition_source_message_counts": {key: len(value) for key, value in partitions.items()},
            "pair_counts": {"train": int(len(train_x)), "validation": int(len(val_x))},
            "train_unique_message_count": len(set(partitions["train"])),
            "validation_unique_message_count": len(set(partitions["validation"])),
            "test_unique_message_count": len(set(partitions["test"])),
            "partition_overlap_checked": True,
        },
        "synthetic_data_limitations": [
            "Only alphanumeric A-Z and 0-9 messages encoded with the project's Morse mapping.",
            "Noise consists of independent random dot/dash substitutions; no insertions, deletions, bursts, timing, or radio-channel effects.",
            "Synthetic metrics do not establish performance on real LoRa traffic or realistic language distributions.",
        ],
        "held_out_evaluation": preliminary_eval,
        "smoke_run": smoke,
    }
    _assert_metadata_compatibility(reloaded, metadata)
    if not smoke:
        # Metadata is written only after the candidate successfully saves,
        # reloads, validates, and completes held-out evaluation.
        with METADATA_PATH.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
    print(f"\nCandidate model: {model_path}")
    print(f"Training completed epochs: {metadata['training']['epochs_completed']}")
    print(f"Unique source messages: train={train_n}, validation={validation_n}, test={test_n} (disjoint)")
    print("Held-out test metrics (candidate versus unchanged received input):")
    print(f"Clean preservation: {preliminary_eval['clean_preservation']['exactly_preserved']}/"
          f"{preliminary_eval['clean_preservation']['messages']} "
          f"({preliminary_eval['clean_preservation']['rate']:.2%})")
    for noise_pct in NOISE_LEVELS:
        row = preliminary_eval[f"noise_{noise_pct}_percent"]
        print(f"{noise_pct:>2}%: boundary {row['baseline_boundary_edit_rate']:.3%} -> "
              f"{row['candidate_boundary_edit_rate']:.3%}; dot/dash "
              f"{row['baseline_dot_dash_edit_rate']:.3%} -> {row['candidate_dot_dash_edit_rate']:.3%}; "
              f"improved/unchanged/worsened={row['improved']}/{row['unchanged']}/{row['worsened']}")
    print("Interpretation: lower held-out error and clean preservation are required; training loss alone is not evidence of usefulness.")
    if smoke:
        print("Smoke model and metadata were temporary; no project artifacts were written.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny one-epoch train/save/reload/evaluate check; temporary outputs only")
    parser.add_argument("--seed", type=int, default=48621, help="TensorFlow and noise-generation seed")
    parser.add_argument("--train-messages", type=int, default=5000)
    parser.add_argument("--validation-messages", type=int, default=600)
    parser.add_argument("--test-messages", type=int, default=200,
                        help="held-out messages; generated with the same seed/corpus method as evaluate_morse_correction_v3.py")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    for name in ("train_messages", "validation_messages", "test_messages", "epochs", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.train_messages + args.validation_messages + args.test_messages > 100000:
        parser.error("total unique source message count must be <= 100000")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
