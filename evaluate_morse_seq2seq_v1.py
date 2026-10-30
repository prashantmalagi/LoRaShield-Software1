"""Locked-test comparison for Morse seq2seq V1, V3, and unchanged input.

References are used only after inference to score the locked cases. The
candidate's runtime decoder receives received Morse only. This evaluator does
not select thresholds, tune the model, modify artifacts, or access hardware.
The locked test is generated from the completed source manifest and evaluated
once; synthetic results do not establish real radio performance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import morse_seq2seq_v1_data as data
import morse_seq2seq_v1_model as prototype
from train_morse_seq2seq_v1 import (
    DEFAULT_MANIFEST_PATH,
    audit_full_sources,
    make_partition_pairs,
)

ROOT = Path(__file__).resolve().parent
V3_MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
V3_METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
SEQ2SEQ_MODEL_PATH = ROOT / "models" / "morse_seq2seq_v1_candidate.keras"
SEQ2SEQ_METADATA_PATH = ROOT / "morse_seq2seq_v1_training_metadata.json"
V3_SEQUENCE_LENGTH = 120
V3_VOCABULARY = ["<PAD>", "<UNK>", ".", "-", " "]


def levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
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


def symbol_stream(morse: str) -> str:
    return "".join(char for char in morse if char in ".-")


def boundary_positions(morse: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return letter-boundary symbol offsets and word-boundary letter offsets."""
    words = morse.split(" / ")
    letter_positions: list[int] = []
    word_positions: list[int] = []
    symbol_offset = 0
    letter_offset = 0
    for word_index, word in enumerate(words):
        groups = [group for group in word.split(" ") if group]
        for group_index, group in enumerate(groups):
            symbol_offset += sum(char in ".-" for char in group)
            letter_offset += 1
            if group_index < len(groups) - 1:
                letter_positions.append(symbol_offset)
        if word_index < len(words) - 1:
            word_positions.append(letter_offset)
    return tuple(letter_positions), tuple(word_positions)


def case_metrics(reference: str, candidate: str) -> dict[str, Any]:
    ref_symbols, out_symbols = symbol_stream(reference), symbol_stream(candidate)
    ref_letters, ref_words = boundary_positions(reference)
    out_letters, out_words = boundary_positions(candidate)
    symbol_errors = levenshtein(ref_symbols, out_symbols)
    letter_errors = levenshtein(ref_letters, out_letters)
    word_errors = levenshtein(ref_words, out_words)
    return {
        "symbol_errors": symbol_errors,
        "symbol_denominator": max(len(ref_symbols), len(out_symbols), 1),
        "letter_boundary_errors": letter_errors,
        "letter_boundary_denominator": max(len(ref_letters), len(out_letters), 1),
        "word_boundary_errors": word_errors,
        "word_boundary_denominator": max(len(ref_words), len(out_words), 1),
        "letter_boundaries_exact": ref_letters == out_letters,
        "word_boundaries_exact": ref_words == out_words,
        "exact_morse": reference == candidate,
        "combined_errors": symbol_errors + letter_errors + word_errors,
    }


def greedy_decode_batch(model: Any, received_messages: Sequence[str], *,
                        max_output_tokens: int = prototype.MAX_OUTPUT_TOKENS,
                        batch_size: int = 32) -> list[prototype.DecodeResult]:
    """Batched greedy decoding; references are not accepted by this interface."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(max_output_tokens) is not int or not 2 <= max_output_tokens <= data.MAX_SEQUENCE_TOKENS:
        raise ValueError("max_output_tokens must include BOS/EOS and fit the sequence limit")
    results: list[prototype.DecodeResult] = []
    for start in range(0, len(received_messages), batch_size):
        messages = list(received_messages[start:start + batch_size])
        runtime_rows = [prototype.preprocess_received(message).input_ids for message in messages]
        enc_width = max(map(len, runtime_rows))
        encoder_ids = np.asarray([
            list(row) + [prototype.PAD_ID] * (enc_width - len(row)) for row in runtime_rows
        ], dtype=np.int32)
        decoder_rows = [[prototype.BOS_ID] for _ in messages]
        emitted: list[list[str]] = [[] for _ in messages]
        active = [True] * len(messages)
        reasons = [""] * len(messages)
        eos_flags = [False] * len(messages)
        for _step in range(max_output_tokens - 1):
            if not any(active):
                break
            decoder_width = max(map(len, decoder_rows))
            decoder_batch = np.asarray([
                row + [prototype.PAD_ID] * (decoder_width - len(row))
                for row in decoder_rows
            ], dtype=np.int32)
            probabilities = model.predict([encoder_ids, decoder_batch], verbose=0)
            expected = (len(messages), decoder_width, prototype.VOCAB_SIZE)
            if tuple(probabilities.shape) != expected:
                raise ValueError(f"model output shape {probabilities.shape}; expected {expected}")
            for row_index, is_active in enumerate(active):
                if not is_active:
                    continue
                token_id = int(np.asarray(probabilities)[row_index, len(decoder_rows[row_index]) - 1].argmax())
                token = data.ID_TO_TOKEN[token_id]
                if token == "<EOS>":
                    active[row_index] = False
                    eos_flags[row_index] = True
                    reasons[row_index] = "eos"
                elif token in ("<PAD>", "<UNK>", "<BOS>"):
                    active[row_index] = False
                    reasons[row_index] = "invalid_output_token"
                else:
                    emitted[row_index].append(token)
                    decoder_rows[row_index].append(token_id)
        for index, message in enumerate(messages):
            if active[index]:
                reasons[index] = "max_output_tokens"
            output = None
            if eos_flags[index]:
                try:
                    output = data.decode_morse(("<BOS>", *emitted[index], "<EOS>"),
                                               max_tokens=max_output_tokens)
                except ValueError:
                    reasons[index] = "invalid_sequence"
            results.append(prototype.DecodeResult(
                received_morse=message,
                output_morse=output,
                output_tokens=tuple(emitted[index]),
                terminated_by_eos=eos_flags[index],
                termination_reason=reasons[index],
            ))
    return results


def _system_accumulator() -> dict[str, Any]:
    return {
        "messages": 0,
        "symbol_errors": 0,
        "symbol_denominator": 0,
        "letter_boundary_errors": 0,
        "letter_boundary_denominator": 0,
        "word_boundary_errors": 0,
        "word_boundary_denominator": 0,
        "letter_boundaries_exact": 0,
        "word_boundaries_exact": 0,
        "exact_clean_messages": 0,
        "clean_messages": 0,
        "improved_vs_unchanged": 0,
        "unchanged_vs_unchanged": 0,
        "worsened_vs_unchanged": 0,
        "harmful_extra_combined_errors": [],
        "decode_failures": 0,
    }


def score_outputs(cases: Sequence[data.TrainingPair],
                  v3_outputs: Sequence[str],
                  candidate_outputs: Sequence[prototype.DecodeResult]) -> dict[str, Any]:
    """Offline reference scoring for unchanged, V3, and seq2seq on identical cases."""
    if len(cases) != len(v3_outputs) or len(cases) != len(candidate_outputs):
        raise ValueError("case and output lengths must match")
    systems = {name: _system_accumulator() for name in ("unchanged", "v3", "seq2seq_v1")}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {name: _system_accumulator() for name in systems}
    )
    for case, v3_output, candidate_result in zip(cases, v3_outputs, candidate_outputs):
        candidate_output = candidate_result.output_morse or case.received_morse
        values = {
            "unchanged": case.received_morse,
            "v3": v3_output,
            "seq2seq_v1": candidate_output,
        }
        reference_metrics = case_metrics(case.reference_morse, case.received_morse)
        for name, output in values.items():
            metrics = case_metrics(case.reference_morse, output)
            row = systems[name]
            by_type = grouped[case.corruption_type][name]
            for target in (row, by_type):
                target["messages"] += 1
                target["symbol_errors"] += metrics["symbol_errors"]
                target["symbol_denominator"] += metrics["symbol_denominator"]
                target["letter_boundary_errors"] += metrics["letter_boundary_errors"]
                target["letter_boundary_denominator"] += metrics["letter_boundary_denominator"]
                target["word_boundary_errors"] += metrics["word_boundary_errors"]
                target["word_boundary_denominator"] += metrics["word_boundary_denominator"]
                target["letter_boundaries_exact"] += metrics["letter_boundaries_exact"]
                target["word_boundaries_exact"] += metrics["word_boundaries_exact"]
                if case.corruption_type == "clean":
                    target["clean_messages"] += 1
                    target["exact_clean_messages"] += output == case.reference_morse
            if name == "seq2seq_v1" and not candidate_result.output_morse:
                row["decode_failures"] += 1
                by_type["decode_failures"] += 1
            if name != "unchanged":
                delta = metrics["combined_errors"] - reference_metrics["combined_errors"]
                if delta < 0:
                    row["improved_vs_unchanged"] += 1
                    by_type["improved_vs_unchanged"] += 1
                elif delta > 0:
                    row["worsened_vs_unchanged"] += 1
                    by_type["worsened_vs_unchanged"] += 1
                    row["harmful_extra_combined_errors"].append(delta)
                    by_type["harmful_extra_combined_errors"].append(delta)
                else:
                    row["unchanged_vs_unchanged"] += 1
                    by_type["unchanged_vs_unchanged"] += 1

    def finalize(row: dict[str, Any]) -> dict[str, Any]:
        harms = row["harmful_extra_combined_errors"]
        return {
            **{key: value for key, value in row.items() if key != "harmful_extra_combined_errors"},
            "symbol_edit_rate": row["symbol_errors"] / max(row["symbol_denominator"], 1),
            "letter_boundary_error_rate": row["letter_boundary_errors"] / max(row["letter_boundary_denominator"], 1),
            "word_boundary_error_rate": row["word_boundary_errors"] / max(row["word_boundary_denominator"], 1),
            "letter_boundary_preservation_rate": row["letter_boundaries_exact"] / max(row["messages"], 1),
            "word_boundary_preservation_rate": row["word_boundaries_exact"] / max(row["messages"], 1),
            "clean_preservation_rate": row["exact_clean_messages"] / row["clean_messages"] if row["clean_messages"] else None,
            "mean_harmful_extra_combined_errors": sum(harms) / len(harms) if harms else 0.0,
            "maximum_harmful_extra_combined_errors": max(harms, default=0),
            "harmful_message_count": len(harms),
        }

    return {
        "metric_definitions": {
            "symbol_edit_rate": "Levenshtein dot/dash distance divided by max reference/output dot-dash length",
            "letter_boundary_error": "Levenshtein distance between ordered letter-separator symbol offsets",
            "word_boundary_error": "Levenshtein distance between ordered word-separator letter offsets",
            "outcome_counts": "Compare combined symbol + letter-boundary + word-boundary errors against unchanged received input",
            "harmful_severity": "Extra combined errors on worsened messages",
        },
        "systems": {name: finalize(row) for name, row in systems.items()},
        "by_corruption_type": {
            kind: {name: finalize(row) for name, row in system_rows.items()}
            for kind, system_rows in sorted(grouped.items())
        },
    }


def _load_v3_model():
    if not V3_MODEL_PATH.is_file() or not V3_METADATA_PATH.is_file():
        raise FileNotFoundError("saved V3 model and metadata are required for comparison")
    metadata = json.loads(V3_METADATA_PATH.read_text(encoding="utf-8"))
    expected_ids = {token: index for index, token in enumerate(V3_VOCABULARY)}
    if (metadata.get("model_file") != V3_MODEL_PATH.name
            or metadata.get("sequence_length") != V3_SEQUENCE_LENGTH
            or metadata.get("vocabulary") != V3_VOCABULARY
            or metadata.get("token_ids") != expected_ids):
        raise ValueError("V3 metadata vocabulary/sequence/model contract mismatch")
    import tensorflow as tf

    model = tf.keras.models.load_model(str(V3_MODEL_PATH), compile=False)
    if tuple(model.input_shape) != (None, V3_SEQUENCE_LENGTH):
        raise ValueError(f"unexpected V3 input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, V3_SEQUENCE_LENGTH, len(V3_VOCABULARY)):
        raise ValueError(f"unexpected V3 output shape: {model.output_shape}")
    return model


def _verify_source_manifest(manifest_path: str | Path):
    stored = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    seed = stored["planned_settings"]["root_seed"]
    partitions, computed = audit_full_sources(seed=seed)
    stored_audit = stored["partition_audit"]
    computed_audit = computed["partition_audit"]
    if stored_audit["source_counts"] != computed_audit["source_counts"]:
        raise ValueError("completed manifest source counts do not reproduce")
    if stored_audit["source_fingerprints_sha256"] != computed_audit["source_fingerprints_sha256"]:
        raise ValueError("completed manifest source fingerprints do not reproduce")
    if stored["exclusion_audit"]["measured"] != computed["exclusion_audit"]["measured"]:
        raise ValueError("completed manifest exclusion counts/fingerprints do not reproduce")
    if stored_audit["overlap_check"]["results"] != computed_audit["overlap_check"]["results"]:
        raise ValueError("completed manifest overlap results do not reproduce")
    if computed_audit["overlap_check"]["results"]["word_overlap_count"] != 0:
        raise ValueError("reproduced source partitions have word overlap")
    return partitions, seed


def evaluate_locked(candidate_model: Any, v3_model: Any,
                    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
                    *, inference_batch_size: int = 32) -> dict[str, Any]:
    """Evaluate one frozen candidate on locked cases; no calibration or tuning."""
    partitions, source_seed = _verify_source_manifest(manifest_path)
    locked_sources = partitions["locked_test"]
    cases = make_partition_pairs("locked_test", locked_sources, seed=source_seed)
    received = [case.received_morse for case in cases]
    from evaluate_morse_correction_v3_corruption import predict_messages as predict_v3

    v3_outputs = predict_v3(v3_model, received)
    candidate_outputs = []
    for start in range(0, len(received), inference_batch_size):
        candidate_outputs.extend(greedy_decode_batch(
            candidate_model, received[start:start + inference_batch_size],
            max_output_tokens=data.MAX_SEQUENCE_TOKENS,
            batch_size=inference_batch_size,
        ))
    return {
        "evaluation": "locked synthetic test; no threshold selection or tuning",
        "manifest": Path(manifest_path).name,
        "locked_source_messages": len(locked_sources),
        "locked_cases": len(cases),
        "corruption_types": list(data.CORRUPTION_KINDS),
        "references_used_only_for_offline_scoring": True,
        "report": score_outputs(cases, v3_outputs, candidate_outputs),
        "limitations": [
            "Synthetic random alphanumeric messages and modeled corruption only.",
            "No claim of real radio-channel performance or unique recovery of ambiguous Morse.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--candidate-model", type=Path, default=SEQ2SEQ_MODEL_PATH)
    parser.add_argument("--candidate-metadata", type=Path, default=SEQ2SEQ_METADATA_PATH)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        if not args.manifest.is_file():
            raise FileNotFoundError(f"completed source manifest required: {args.manifest}")
        if not args.candidate_model.is_file() or not args.candidate_metadata.is_file():
            raise FileNotFoundError("trained V1 candidate model and metadata are required")
        candidate_metadata = json.loads(args.candidate_metadata.read_text(encoding="utf-8"))
        if (candidate_metadata.get("model_file") != args.candidate_model.name
                or candidate_metadata.get("spec_version") != data.SPEC_VERSION
                or candidate_metadata.get("token_vocabulary_ids") != data.TOKEN_TO_ID
                or candidate_metadata.get("maximum_sequence_tokens_including_bos_eos")
                != data.MAX_SEQUENCE_TOKENS):
            raise ValueError("V1 candidate metadata/model filename mismatch")
        candidate_model = prototype.load_model(args.candidate_model)
        input_shapes = [tuple(shape) for shape in candidate_model.input_shape]
        if input_shapes != [(None, None), (None, None)]:
            raise ValueError(f"unexpected V1 candidate input shapes: {input_shapes}")
        if tuple(candidate_model.output_shape) != (None, None, len(data.TOKEN_TO_ID)):
            raise ValueError(f"unexpected V1 candidate output shape: {candidate_model.output_shape}")
        v3_model = _load_v3_model()
        result = evaluate_locked(
            candidate_model, v3_model, args.manifest,
            inference_batch_size=args.inference_batch_size,
        )
    except Exception as exc:
        print(f"ERROR: locked seq2seq evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
