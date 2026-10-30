"""Read-only offline evaluation for real LoRa receiver Morse captures.

Dataset: UTF-8 JSONL, one object per sample. Required fields are sample_id,
source_session_id, captured_at_utc, capture_source, received_raw_morse_payload,
quality_label, noise_type, reference_text, reference_morse, reference_source,
reference_independent, reference_reliable, crc_status, and notes. There are no
optional fields; unknown keys are rejected. Only captured_at_utc, reference_text,
reference_morse, and reference_source may be null. quality_label is clean, noisy,
or unknown. noise_type is dot_dash_substitution, burst, insertion_deletion,
timing, other, or unknown. crc_status is valid, invalid, unknown, or
not_applicable. A sample is scored only with an independently sourced reliable
reference and a nonempty reference_source.

The evaluator reads only the dataset, V3 metadata, and V3 model. It prints to
stdout and does not write logs or reports. It does not open hardware, parse
packets, access AES/CRC, import the backend, or use the V2 corrector.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
VOCAB = ["<PAD>", "<UNK>", ".", "-", " "]
TOKEN_IDS = {token: idx for idx, token in enumerate(VOCAB)}
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)
SEQUENCE_LENGTH = 120
QUALITY_LABELS = {"clean", "noisy", "unknown"}
NOISE_TYPES = {"dot_dash_substitution", "burst", "insertion_deletion", "timing", "other", "unknown"}
CRC_STATUSES = {"valid", "invalid", "unknown", "not_applicable"}
REQUIRED_FIELDS = {
    "sample_id", "source_session_id", "captured_at_utc", "capture_source",
    "received_raw_morse_payload", "quality_label", "noise_type", "reference_text",
    "reference_morse", "reference_source", "reference_independent",
    "reference_reliable", "crc_status", "notes",
}
OPTIONAL_FIELDS: set[str] = set()
NULLABLE_FIELDS = {"captured_at_utc", "reference_text", "reference_morse", "reference_source"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS

# The inspected existing V3 evaluator uses the project ITU Morse alphabet for
# A-Z/0-9 and its canonical separators: one letter-space and " / " per word.
from evaluate_morse_correction_v3 import MORSE  # noqa: E402


class DatasetValidationError(ValueError):
    """Raised when any dataset row is malformed; inference must not start."""


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def split_words(morse: str) -> list[str]:
    """Validate canonical project separators and return slash-free Morse words."""
    if not isinstance(morse, str) or not morse:
        raise ValueError("Morse value must be a nonempty string")
    if morse != morse.strip():
        raise ValueError("Leading/trailing whitespace is not canonical Morse payload")
    if ":" in morse:
        raise ValueError("Colon indicates ciphertext-shaped input; ciphertext is not accepted")
    if any(char not in ".- /" for char in morse):
        raise ValueError("Morse input contains unsupported characters")
    words = morse.split(" / ")
    if any(not word for word in words) or any("/" in word for word in words):
        raise ValueError("Word separators must be exactly ' / ' with nonempty words")
    for word in words:
        if word.startswith(" ") or word.endswith(" ") or "  " in word:
            raise ValueError("Letter separators must be single internal spaces")
        if len(word) > SEQUENCE_LENGTH:
            raise ValueError(
                f"Morse word is {len(word)} characters; V3 limit is {SEQUENCE_LENGTH}; no truncation is allowed"
            )
        if not word or any(ch not in ".- " for ch in word):
            raise ValueError("Each word may contain only dots, dashes, and letter spaces")
    return words


def encode_reference_text(text: str) -> str:
    """Encode independently sourced A-Z/0-9 text using project separators."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("reference_text must be null or a nonempty string")
    words = text.upper().split()
    encoded_words = []
    for word in words:
        if any(char not in MORSE for char in word):
            raise ValueError("reference_text consistency encoding supports A-Z and 0-9 only")
        encoded_words.append(" ".join(MORSE[char] for char in word))
    return " / ".join(encoded_words)


def _validate_reference_fields(row: dict[str, Any], row_num: int) -> tuple[str | None, str | None]:
    text = row["reference_text"]
    morse = row["reference_morse"]
    for field, value in (("reference_text", text), ("reference_morse", morse)):
        if value is not None and (not isinstance(value, str) or not value):
            raise DatasetValidationError(f"line {row_num}: {field} must be null or nonempty text")
    if morse is not None:
        try:
            split_words(morse)
        except ValueError as exc:
            raise DatasetValidationError(f"line {row_num}: invalid reference_morse: {exc}") from exc
    encoded_text = None
    if text is not None:
        try:
            encoded_text = encode_reference_text(text)
        except ValueError as exc:
            raise DatasetValidationError(f"line {row_num}: invalid reference_text: {exc}") from exc
    if text is not None and morse is not None and encoded_text != morse:
        raise DatasetValidationError(
            f"line {row_num}: reference_text and reference_morse disagree; expected {encoded_text!r}"
        )
    resolved_morse = morse if morse is not None else encoded_text
    return text, resolved_morse


def _validate_row(row: Any, row_num: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DatasetValidationError(f"line {row_num}: each JSONL row must be an object")
    missing = REQUIRED_FIELDS - row.keys()
    if missing:
        raise DatasetValidationError(f"line {row_num}: missing required fields {sorted(missing)}")
    unknown = row.keys() - ALLOWED_FIELDS
    if unknown:
        raise DatasetValidationError(f"line {row_num}: unknown fields {sorted(unknown)}")
    if not isinstance(row["sample_id"], str) or not row["sample_id"].strip():
        raise DatasetValidationError(f"line {row_num}: sample_id must be nonempty text")
    if not isinstance(row["source_session_id"], str) or not row["source_session_id"].strip():
        raise DatasetValidationError(f"line {row_num}: source_session_id must be nonempty text")
    for field in ("capture_source", "notes"):
        if not isinstance(row[field], str):
            raise DatasetValidationError(f"line {row_num}: {field} must be text")
    captured = row["captured_at_utc"]
    if captured is not None and not isinstance(captured, str):
        raise DatasetValidationError(f"line {row_num}: captured_at_utc must be text or null")
    if not isinstance(row["quality_label"], str) or row["quality_label"] not in QUALITY_LABELS:
        raise DatasetValidationError(f"line {row_num}: quality_label must be one of {sorted(QUALITY_LABELS)}")
    if not isinstance(row["noise_type"], str) or row["noise_type"] not in NOISE_TYPES:
        raise DatasetValidationError(f"line {row_num}: noise_type must be one of {sorted(NOISE_TYPES)}")
    if not isinstance(row["crc_status"], str) or row["crc_status"] not in CRC_STATUSES:
        raise DatasetValidationError(f"line {row_num}: crc_status must be one of {sorted(CRC_STATUSES)}")
    for field in ("reference_independent", "reference_reliable"):
        if not isinstance(row[field], bool):
            raise DatasetValidationError(f"line {row_num}: {field} must be a JSON boolean")
    ref_source = row["reference_source"]
    if ref_source is not None and not isinstance(ref_source, str):
        raise DatasetValidationError(f"line {row_num}: reference_source must be text or null")

    received = row["received_raw_morse_payload"]
    try:
        split_words(received)
    except ValueError as exc:
        raise DatasetValidationError(f"line {row_num}: invalid received_raw_morse_payload: {exc}") from exc
    reference_text, reference_morse = _validate_reference_fields(row, row_num)
    scored = bool(
        reference_morse is not None
        and row["reference_independent"] is True
        and row["reference_reliable"] is True
        and isinstance(ref_source, str)
        and ref_source.strip()
    )
    reasons = []
    if reference_morse is None:
        reasons.append("no reference text or Morse supplied")
    if not row["reference_independent"]:
        reasons.append("reference is not marked independent")
    if not row["reference_reliable"]:
        reasons.append("reference is not marked reliable")
    if not isinstance(ref_source, str) or not ref_source.strip():
        reasons.append("reference_source is missing")
    normalized = dict(row)
    normalized["_reference_text"] = reference_text
    normalized["_reference_morse"] = reference_morse
    normalized["_scored"] = scored
    normalized["_unscored_reason"] = "; ".join(reasons) if reasons else ""
    return normalized


def validate_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetValidationError(f"dataset file not found: {path}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for row_num, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetValidationError(f"line {row_num}: invalid JSON: {exc.msg}") from exc
                row = _validate_row(raw, row_num)
                if row["sample_id"] in seen_ids:
                    raise DatasetValidationError(f"line {row_num}: duplicate sample_id {row['sample_id']!r}")
                seen_ids.add(row["sample_id"])
                rows.append(row)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"dataset is not UTF-8: {exc}") from exc
    return rows


def _letter_boundary_distance(reference_word: str, candidate_word: str) -> int:
    left = reference_word.split(" ") if reference_word else []
    right = candidate_word.split(" ") if candidate_word else []
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(1, len(left) + 1):
        dp[i][0] = dp[i - 1][0] + len(left[i - 1])
    for j in range(1, len(right) + 1):
        dp[0][j] = dp[0][j - 1] + len(right[j - 1])
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + len(left[i - 1]),
                dp[i][j - 1] + len(right[j - 1]),
                dp[i - 1][j - 1] + _levenshtein(left[i - 1], right[j - 1]),
            )
    return dp[-1][-1] + abs(max(len(left) - 1, 0) - max(len(right) - 1, 0))


def boundary_edit_distance(reference: str, candidate: str) -> int:
    """Hierarchical alignment: Morse letter groups within aligned word groups."""
    left, right = reference.split(" / "), candidate.split(" / ")
    dp = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    word_symbols = lambda word: len(word.replace(" ", ""))
    for i in range(1, len(left) + 1):
        dp[i][0] = dp[i - 1][0] + word_symbols(left[i - 1])
    for j in range(1, len(right) + 1):
        dp[0][j] = dp[0][j - 1] + word_symbols(right[j - 1])
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + word_symbols(left[i - 1]),
                dp[i][j - 1] + word_symbols(right[j - 1]),
                dp[i - 1][j - 1] + _letter_boundary_distance(left[i - 1], right[j - 1]),
            )
    return dp[-1][-1] + abs(max(len(left) - 1, 0) - max(len(right) - 1, 0))


def symbol_edit_distance(reference: str, candidate: str) -> int:
    return _levenshtein(reference.replace(" ", "").replace("/", ""),
                        candidate.replace(" ", "").replace("/", ""))


def _word_boundary_positions(morse: str) -> tuple[int, ...]:
    """Return ordered slash positions as counts of Morse letter groups before each slash."""
    words = morse.split(" / ")
    positions = []
    cumulative = 0
    for word in words[:-1]:
        cumulative += len(word.split())
        positions.append(cumulative)
    return tuple(positions)


def _word_boundary_info(reference: str, candidate: str) -> dict[str, Any]:
    ref_words, candidate_words = reference.split(" / "), candidate.split(" / ")
    ref_positions = _word_boundary_positions(reference)
    candidate_positions = _word_boundary_positions(candidate)
    ref_count, candidate_count = len(ref_positions), len(candidate_positions)
    return {
        "reference_word_count": len(ref_words),
        "candidate_word_count": len(candidate_words),
        "reference_boundary_count": ref_count,
        "candidate_boundary_count": candidate_count,
        "reference_boundary_positions_letter_groups": list(ref_positions),
        "candidate_boundary_positions_letter_groups": list(candidate_positions),
        "boundaries_preserved": ref_positions == candidate_positions,
        "boundary_position_error_count": sum(
            left != right for left, right in zip(ref_positions, candidate_positions)
        ) + abs(ref_count - candidate_count),
    }


def _rate(errors: int, reference: str, candidate: str) -> float:
    return errors / max(len(reference), len(candidate), 1)


def _infer_words(model: Any, received: str) -> str:
    import numpy as np

    words = split_words(received)
    input_rows = []
    for word in words:
        # No slash is allowed in an inference row; validation occurs before all inference.
        if "/" in word:
            raise AssertionError("slash reached V3 tokenizer")
        ids = [TOKEN_IDS.get(char) for char in word]
        if any(token_id is None for token_id in ids):
            raise AssertionError("unsupported symbol reached V3 tokenizer")
        input_rows.append(ids + [PAD_ID] * (SEQUENCE_LENGTH - len(ids)))
    batch = np.asarray(input_rows, dtype=np.int32)
    expected_input = tuple(model.input_shape[1:])
    if batch.shape[1:] != expected_input:
        raise ValueError(f"Encoded input does not match V3 model shape {model.input_shape}")
    predictions = model.predict(batch, verbose=0)
    expected_output = (len(words), SEQUENCE_LENGTH, len(VOCAB))
    if predictions.shape != expected_output:
        raise ValueError(f"V3 output shape {predictions.shape}, expected {expected_output}")
    id_to_token = {i: token for i, token in enumerate(VOCAB)}
    output_words = []
    for word, scores in zip(words, predictions):
        output = []
        for position, class_id in enumerate(scores.argmax(axis=-1)[:len(word)]):
            char = id_to_token[int(class_id)]
            output.append(word[position] if char in ("<PAD>", "<UNK>") else char)
        output_words.append("".join(output))
    return " / ".join(output_words)


def score_sample(row: dict[str, Any], corrected: str) -> dict[str, Any]:
    reference = row["_reference_morse"]
    raw_boundary = boundary_edit_distance(reference, row["received_raw_morse_payload"])
    corrected_boundary = boundary_edit_distance(reference, corrected)
    raw_symbols = symbol_edit_distance(reference, row["received_raw_morse_payload"])
    corrected_symbols = symbol_edit_distance(reference, corrected)
    if corrected_boundary < raw_boundary:
        outcome = "improved"
    elif corrected_boundary > raw_boundary:
        outcome = "worsened"
    else:
        outcome = "unchanged"
    return {
        "sample_id": row["sample_id"],
        "source_session_id": row["source_session_id"],
        "quality_label": row["quality_label"],
        "noise_type": row["noise_type"],
        "raw_boundary_errors": raw_boundary,
        "corrected_boundary_errors": corrected_boundary,
        "raw_boundary_error_rate": _rate(raw_boundary, reference, row["received_raw_morse_payload"]),
        "corrected_boundary_error_rate": _rate(corrected_boundary, reference, corrected),
        "raw_symbol_errors": raw_symbols,
        "corrected_symbol_errors": corrected_symbols,
        "raw_symbol_error_rate": raw_symbols / max(len(reference.replace(" ", "").replace("/", "")), 1),
        "corrected_symbol_error_rate": corrected_symbols / max(len(reference.replace(" ", "").replace("/", "")), 1),
        "raw_word_boundaries": _word_boundary_info(reference, row["received_raw_morse_payload"]),
        "corrected_word_boundaries": _word_boundary_info(reference, corrected),
        "outcome": outcome,
        "clean_exact_preserved": (corrected == row["received_raw_morse_payload"])
        if row["quality_label"] == "clean" else None,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"scored_samples": 0, "improved": 0, "unchanged": 0, "worsened": 0}
    clean_rows = [row for row in rows if row.get("clean_exact_preserved") is not None]
    return {
        "scored_samples": len(rows),
        "improved": sum(row["outcome"] == "improved" for row in rows),
        "unchanged": sum(row["outcome"] == "unchanged" for row in rows),
        "worsened": sum(row["outcome"] == "worsened" for row in rows),
        "raw_boundary_error_rate": sum(r["raw_boundary_errors"] for r in rows)
        / max(sum(max(len(r["_reference_morse"]), len(r["received_raw_morse_payload"]), 1) for r in rows), 1),
        "corrected_boundary_error_rate": sum(r["corrected_boundary_errors"] for r in rows)
        / max(sum(max(len(r["_reference_morse"]), len(r["corrected_morse"]), 1) for r in rows), 1),
        "raw_symbol_error_rate": sum(r["raw_symbol_errors"] for r in rows)
        / max(sum(max(len(r["_reference_morse"].replace(" ", "").replace("/", "")), 1) for r in rows), 1),
        "corrected_symbol_error_rate": sum(r["corrected_symbol_errors"] for r in rows)
        / max(sum(max(len(r["_reference_morse"].replace(" ", "").replace("/", "")), 1) for r in rows), 1),
        "raw_word_boundary_count_preserved": sum(
            r["raw_word_boundaries"]["boundaries_preserved"] for r in rows
        ),
        "corrected_word_boundary_count_preserved": sum(
            r["corrected_word_boundaries"]["boundaries_preserved"] for r in rows
        ),
        "word_boundary_samples": len(rows),
        "clean_samples": len(clean_rows),
        "clean_exactly_preserved": sum(bool(r["clean_exact_preserved"]) for r in clean_rows),
        "clean_exact_preservation_rate": (
            sum(bool(r["clean_exact_preserved"]) for r in clean_rows) / len(clean_rows)
            if clean_rows else None
        ),
    }


def _group_report(scored: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[str(row[key] or "unknown")].append(row)
    return {group_name: _aggregate(group_rows) for group_name, group_rows in sorted(groups.items())}


def _load_model() -> Any:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("V3 model and metadata must both exist")
    with METADATA_PATH.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("V3 metadata model_file mismatch")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH or metadata.get("vocabulary") != VOCAB:
        raise ValueError("V3 metadata length or vocabulary mismatch")
    if metadata.get("token_ids") != TOKEN_IDS or metadata.get("padding_id") != PAD_ID:
        raise ValueError("V3 metadata token ID mapping mismatch")
    import tensorflow as tf

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if tuple(model.input_shape) != (None, SEQUENCE_LENGTH):
        raise ValueError(f"V3 model input shape mismatch: {model.input_shape}")
    if tuple(model.output_shape) != (None, SEQUENCE_LENGTH, len(VOCAB)):
        raise ValueError(f"V3 model output shape mismatch: {model.output_shape}")
    embedding = model.get_layer("token_embedding")
    if embedding.input_dim != len(VOCAB) or not embedding.mask_zero:
        raise ValueError("V3 embedding does not match vocabulary/PAD mapping")
    return model


def evaluate_rows(rows: list[dict[str, Any]], model: Any) -> dict[str, Any]:
    # Validation is complete before this function is called.
    scored: list[dict[str, Any]] = []
    unscored: list[dict[str, str]] = []
    for row in rows:
        if not row["_scored"]:
            unscored.append({"sample_id": row["sample_id"], "reason": row["_unscored_reason"]})
            continue
        corrected = _infer_words(model, row["received_raw_morse_payload"])
        metrics = score_sample(row, corrected)
        metrics["_reference_morse"] = row["_reference_morse"]
        metrics["corrected_morse"] = corrected
        metrics["received_raw_morse_payload"] = row["received_raw_morse_payload"]
        scored.append(metrics)
    clean_rows = [row for row in scored if row["quality_label"] == "clean"]
    clean_preservation = {
        "scored_clean_samples": len(clean_rows),
        "exactly_preserved": sum(bool(row["clean_exact_preserved"]) for row in clean_rows),
        "rate": (sum(bool(row["clean_exact_preserved"]) for row in clean_rows) / len(clean_rows))
        if clean_rows else None,
    }
    return {
        "scored": scored,
        "unscored": unscored,
        "overall": _aggregate(scored),
        "clean_preservation": clean_preservation,
        "by_source_session": _group_report(scored, "source_session_id"),
        "by_quality_label": _group_report(scored, "quality_label"),
        "by_noise_type": _group_report(scored, "noise_type"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "morse_receiver_samples.jsonl")
    parser.add_argument("--validate-only", action="store_true", help="validate JSONL and exit without loading the model")
    args = parser.parse_args()
    try:
        rows = validate_dataset(args.dataset)
    except DatasetValidationError as exc:
        print(f"DATASET INVALID: {exc}", file=sys.stderr)
        return 2
    scored_count = sum(row["_scored"] for row in rows)
    print(f"Dataset valid: samples={len(rows)}, reference-scored={scored_count}, unscored={len(rows) - scored_count}")
    if args.validate_only:
        print("Validation-only mode: no model loaded and no inference performed.")
        return 0
    if not rows:
        print("No receiver samples are present; no evaluation was performed.")
        return 0
    if scored_count == 0:
        print("No samples have dependable independent references; no scored evaluation was performed.")
        for row in rows:
            print(f"UNSCORED {row['sample_id']}: {row['_unscored_reason']}")
        return 0
    try:
        model = _load_model()
        report = evaluate_rows(rows, model)
    except Exception as exc:
        print(f"EVALUATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print("\nModel: models/lorashield_morse_correction_v3.keras")
    print("Word handling: split on ' / ', infer words separately, rejoin with ' / '.")
    print("Metrics: custom hierarchical group-aligned boundary-aware score (not ordinary character edit distance); its rate uses maximum full Morse length. Dot/dash-only Levenshtein rate uses reference dot/dash length. Word-boundary preservation compares ordered slash positions counted in Morse letter groups.")
    print("Outcome counts compare corrected boundary error with raw received boundary error.")
    print(json.dumps({
        "overall": report["overall"],
        "clean_exact_preservation": report["clean_preservation"],
        "by_source_session": report["by_source_session"],
        "by_quality_label": report["by_quality_label"],
        "by_noise_type": report["by_noise_type"],
    }, indent=2))
    for row in report["scored"]:
        print(f"SCORED {row['sample_id']}: raw={row['received_raw_morse_payload']!r}; "
              f"corrected={row['corrected_morse']!r}; outcome={row['outcome']}; "
              f"raw_boundary_rate={row['raw_boundary_error_rate']:.4%}; "
              f"corrected_boundary_rate={row['corrected_boundary_error_rate']:.4%}; "
              f"raw_symbol_rate={row['raw_symbol_error_rate']:.4%}; "
              f"corrected_symbol_rate={row['corrected_symbol_error_rate']:.4%}; "
              f"word_boundaries_preserved={row['corrected_word_boundaries']['boundaries_preserved']}")
    for row in report["unscored"]:
        print(f"UNSCORED {row['sample_id']}: {row['reason']}")
    print("Synthetic or unverified inputs are not counted as real receiver evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
