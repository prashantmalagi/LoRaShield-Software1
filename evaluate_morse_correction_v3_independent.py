"""Independent, read-only evaluation of the saved LoRaShield V3 candidate.

This program loads the V3 model and its metadata explicitly. It does not
import correction.config, write project artifacts, or change live settings.
The synthetic tests use alphanumeric Morse messages and random dot/dash flips;
they do not represent real LoRa noise, insertions, deletions, or language use.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
FRESH_SEED = 20261029141537
DEFAULT_MESSAGES = 500
NOISE_LEVELS = (0, 5, 10, 15, 20)
SEQUENCE_LENGTH = 120
VOCABULARY = ["<PAD>", "<UNK>", ".", "-", " "]
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)

from evaluate_morse_correction_v3 import CHARSET, MORSE, _encode, _noise  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    """Unit-cost character edit distance, including substitutions/insertions/deletions."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def boundary_edit_distance(reference: str, candidate: str) -> int:
    """Align Morse letter groups with weighted insertion/deletion/substitution.

    Spaces delimit letters and are never substituted for dot/dash symbols.
    Inserting/deleting a whole letter group costs its symbol count;
    substitutions of aligned groups cost their dot/dash Levenshtein distance.
    The absolute difference in boundary counts is added once, so deleting the
    only letter does not incur a nonexistent separator. Dynamic programming
    aligns groups so later letters do not become spuriously misaligned.
    """
    left = reference.split(" ") if reference else []
    right = candidate.split(" ") if candidate else []
    rows, cols = len(left), len(right)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        dp[i][0] = dp[i - 1][0] + len(left[i - 1])
    for j in range(1, cols + 1):
        dp[0][j] = dp[0][j - 1] + len(right[j - 1])
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            delete_group = dp[i - 1][j] + len(left[i - 1])
            insert_group = dp[i][j - 1] + len(right[j - 1])
            substitute_group = dp[i - 1][j - 1] + levenshtein(left[i - 1], right[j - 1])
            dp[i][j] = min(delete_group, insert_group, substitute_group)
    left_boundaries = max(rows - 1, 0)
    right_boundaries = max(cols - 1, 0)
    return dp[rows][cols] + abs(left_boundaries - right_boundaries)


def edit_errors(reference: str, candidate: str, *, symbols_only: bool = False) -> tuple[int, float]:
    if symbols_only:
        ref_symbols = reference.replace(" ", "")
        cand_symbols = candidate.replace(" ", "")
        errors = levenshtein(ref_symbols, cand_symbols)
        denominator = max(len(ref_symbols), len(cand_symbols), 1)
    else:
        errors = boundary_edit_distance(reference, candidate)
        denominator = max(len(reference), len(candidate), 1)
    return errors, errors / denominator


def assert_disjoint(partitions: dict[str, Iterable[str]]) -> None:
    owners: dict[str, str] = {}
    for partition_name, messages in partitions.items():
        for message in messages:
            previous = owners.get(message)
            if previous is not None:
                raise AssertionError(
                    f"Source message overlaps partitions {previous!r} and {partition_name!r}"
                )
            owners[message] = partition_name


def _unique_messages(seed: int, count: int, excluded: set[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    seen = set(excluded or ())
    messages: list[str] = []
    while len(messages) < count:
        message = "".join(rng.choice(CHARSET) for _ in range(rng.randint(3, 8)))
        if message not in seen:
            seen.add(message)
            messages.append(message)
    return messages


def _old_partition_messages() -> set[str]:
    """Recreate known prior V3 train/validation/test and evaluation sources."""
    prior_eval = _unique_messages(20261029, 200)
    prior_val = _unique_messages(562813, 600, set(prior_eval))
    prior_train = _unique_messages(731941, 5000, set(prior_eval) | set(prior_val))
    prior_smoke_test = _unique_messages(20261029, 24)
    prior_smoke_val = _unique_messages(562813, 12, set(prior_smoke_test))
    prior_smoke_train = _unique_messages(731941, 40, set(prior_smoke_test) | set(prior_smoke_val))
    prior_v2_eval = _unique_messages(999, 200)
    return set(prior_eval + prior_val + prior_train + prior_smoke_test +
               prior_smoke_val + prior_smoke_train + prior_v2_eval)


def make_fresh_corpus(seed: int = FRESH_SEED, count: int = DEFAULT_MESSAGES) -> list[str]:
    """Create unique source texts excluded from known previous synthetic corpora."""
    sources = _unique_messages(seed, count, _old_partition_messages())
    assert_disjoint({"prior_sources": _old_partition_messages(), "new_evaluation": sources})
    return sources


def encode_padded(morse: str, sequence_length: int = SEQUENCE_LENGTH) -> list[int]:
    token_ids = {"<PAD>": PAD_ID, "<UNK>": UNK_ID, ".": DOT_ID, "-": DASH_ID, " ": SPACE_ID}
    ids = [token_ids.get(char, UNK_ID) for char in morse]
    if len(ids) > sequence_length:
        raise ValueError(f"Input length {len(ids)} exceeds model length {sequence_length}")
    return ids + [PAD_ID] * (sequence_length - len(ids))


def decode_real_positions(received: str, predicted_ids: Iterable[int]) -> str:
    """Decode only non-PAD input positions; special predictions fall back to input."""
    id_to_token = {PAD_ID: "<PAD>", UNK_ID: "<UNK>", DOT_ID: ".", DASH_ID: "-", SPACE_ID: " "}
    ids = list(predicted_ids)
    if len(ids) < len(received):
        raise ValueError("Model prediction is shorter than the unpadded input")
    decoded = []
    for index, token_id in enumerate(ids[:len(received)]):
        char = id_to_token.get(int(token_id), "<UNK>")
        decoded.append(received[index] if char in ("<PAD>", "<UNK>") else char)
    return "".join(decoded)


def validate_metadata_and_model(model: Any, metadata: dict[str, Any]) -> None:
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("Metadata model_file does not identify the V3 artifact")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("V3 metadata sequence_length mismatch")
    if metadata.get("vocabulary") != VOCABULARY:
        raise ValueError("V3 metadata vocabulary mismatch")
    if metadata.get("token_ids") != {token: index for index, token in enumerate(VOCABULARY)}:
        raise ValueError("V3 metadata token ID mapping mismatch")
    if metadata.get("padding_id") != PAD_ID:
        raise ValueError("V3 metadata padding ID mismatch")
    if tuple(model.input_shape) != (None, SEQUENCE_LENGTH):
        raise ValueError(f"V3 model input shape mismatch: {model.input_shape}")
    if tuple(model.output_shape) != (None, SEQUENCE_LENGTH, len(VOCABULARY)):
        raise ValueError(f"V3 model output shape mismatch: {model.output_shape}")
    embedding = model.get_layer("token_embedding")
    if embedding.input_dim != len(VOCABULARY) or not embedding.mask_zero:
        raise ValueError("V3 embedding vocabulary or PAD mask does not match metadata")


def _predict(model: Any, messages: list[str]) -> list[str]:
    import numpy as np

    batch = np.asarray([encode_padded(message) for message in messages], dtype=np.int32)
    predictions = model.predict(batch, verbose=0)
    result = []
    for received, score_rows in zip(messages, predictions):
        ids = score_rows.argmax(axis=-1)
        result.append(decode_real_positions(received, ids))
    return result


def evaluate(model: Any, source_texts: list[str], seed: int) -> dict[str, Any]:
    clean = [_encode(text) for text in source_texts]
    if any(len(message) > SEQUENCE_LENGTH for message in clean):
        raise ValueError("Generated test message exceeds model sequence length")
    result: dict[str, Any] = {}
    for noise_pct in NOISE_LEVELS:
        rng = random.Random(seed + noise_pct)
        received = [_noise(message, noise_pct / 100.0, rng) for message in clean]
        corrected = _predict(model, received)
        before_boundary = after_boundary = before_symbols = after_symbols = 0
        before_boundary_den = after_boundary_den = before_symbol_den = after_symbol_den = 0
        improved = unchanged = worsened = 0
        for reference, noisy, candidate in zip(clean, received, corrected):
            before, _ = edit_errors(reference, noisy)
            after, _ = edit_errors(reference, candidate)
            before_sym, _ = edit_errors(reference, noisy, symbols_only=True)
            after_sym, _ = edit_errors(reference, candidate, symbols_only=True)
            before_boundary += before
            after_boundary += after
            before_symbols += before_sym
            after_symbols += after_sym
            before_boundary_den += max(len(reference), len(noisy), 1)
            after_boundary_den += max(len(reference), len(candidate), 1)
            before_symbol_den += max(len(reference.replace(" ", "")), len(noisy.replace(" ", "")), 1)
            after_symbol_den += max(len(reference.replace(" ", "")), len(candidate.replace(" ", "")), 1)
            if after < before:
                improved += 1
            elif after > before:
                worsened += 1
            else:
                unchanged += 1
        result[f"noise_{noise_pct}_percent"] = {
            "baseline_boundary_errors": before_boundary,
            "baseline_boundary_rate": before_boundary / before_boundary_den,
            "candidate_boundary_errors": after_boundary,
            "candidate_boundary_rate": after_boundary / after_boundary_den,
            "baseline_dot_dash_errors": before_symbols,
            "baseline_dot_dash_rate": before_symbols / before_symbol_den,
            "candidate_dot_dash_errors": after_symbols,
            "candidate_dot_dash_rate": after_symbols / after_symbol_den,
            "improved": improved,
            "unchanged": unchanged,
            "worsened": worsened,
        }
    clean_outputs = _predict(model, clean)
    preserved = sum(reference == prediction for reference, prediction in zip(clean, clean_outputs))
    result["clean_preservation"] = {
        "messages": len(clean), "exactly_preserved": preserved,
        "rate": preserved / max(len(clean), 1),
    }
    return result


def run_self_tests(model: Any | None = None, metadata: dict[str, Any] | None = None) -> None:
    assert levenshtein("...", "...") == 0
    assert levenshtein("...", ".-.") == 1
    # Insert/delete one whole letter in the middle; following groups align.
    assert boundary_edit_distance("... --- ...", "... .... --- ...") == 5
    assert boundary_edit_distance("... .... --- ...", "... --- ...") == 5
    assert boundary_edit_distance("... ---", "... --.") == 1
    assert encode_padded(".-")[:2] == [DOT_ID, DASH_ID]
    assert encode_padded(".-")[2:] == [PAD_ID] * (SEQUENCE_LENGTH - 2)
    assert decode_real_positions(".-", [DOT_ID, DASH_ID] + [PAD_ID] * 118) == ".-"
    assert decode_real_positions(".-", [PAD_ID, UNK_ID] + [PAD_ID] * 118) == ".-"
    try:
        assert_disjoint({"train": ["ABC"], "test": ["ABC"]})
    except AssertionError:
        pass
    else:
        raise AssertionError("Partition-overlap check did not reject a duplicate source")
    fresh = make_fresh_corpus(FRESH_SEED, 500)
    assert len(fresh) == len(set(fresh)) == 500
    assert_disjoint({"prior": _old_partition_messages(), "fresh": fresh})
    if model is not None and metadata is not None:
        validate_metadata_and_model(model, metadata)


def _load_artifacts():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"V3 model not found: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"V3 metadata not found: {METADATA_PATH}")
    import tensorflow as tf

    with METADATA_PATH.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    # Explicit V3 path: no MorseCorrector and no correction.config import.
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    validate_metadata_and_model(model, metadata)
    return model, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES,
                        help=f"independent test messages (default: {DEFAULT_MESSAGES})")
    parser.add_argument("--self-test", action="store_true", help="validate metrics, partitions, padding, and model compatibility")
    args = parser.parse_args()
    if args.messages < 1:
        parser.error("--messages must be positive")
    try:
        model, metadata = _load_artifacts()
        run_self_tests(model, metadata)
        print("Self-tests passed: group insertion/deletion alignment, vocabulary/shapes, padding, and disjoint sources.")
        if args.self_test:
            return 0
        prior = _old_partition_messages()
        sources = make_fresh_corpus(FRESH_SEED, args.messages)
        assert_disjoint({"prior_corpora": prior, "new_test": sources})
        clean_messages = [_encode(text) for text in sources]
        metrics = evaluate(model, sources, FRESH_SEED)
    except Exception as exc:
        print(f"ERROR: independent V3 evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print("\nIndependent V3 evaluation (synthetic data)")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Fresh seed: {FRESH_SEED}")
    print(f"Unique source messages: {len(sources)}; prior source messages excluded: {len(prior)}")
    print(f"Input lengths: {SEQUENCE_LENGTH}; vocabulary IDs: {metadata['token_ids']}")
    print("Metrics: boundary edit distance aligns space-delimited Morse letter groups by dynamic programming; group insert/delete costs symbol count, with absolute boundary-count difference added once; aligned-group substitution costs dot/dash Levenshtein distance.")
    print("Dot/dash-only metric: Levenshtein distance after removing spaces. Rate denominator: sum of per-message max(reference length, candidate length, 1).")
    print("Counts compare candidate errors with unchanged received-input errors; clean preservation is exact-string match on the held-out clean corpus.")
    print("Noise: independent dot/dash substitutions only; no insertion/deletion channel noise, bursts, timing effects, or real LoRa traffic.")
    clean = metrics["clean_preservation"]
    print(f"Clean preservation: {clean['exactly_preserved']}/{clean['messages']} ({clean['rate']:.2%})")
    for noise_pct in NOISE_LEVELS:
        row = metrics[f"noise_{noise_pct}_percent"]
        print(f"{noise_pct:>2}% noise: boundary {row['baseline_boundary_rate']:.3%} -> {row['candidate_boundary_rate']:.3%}; "
              f"dot/dash {row['baseline_dot_dash_rate']:.3%} -> {row['candidate_dot_dash_rate']:.3%}; "
              f"improved/unchanged/worsened={row['improved']}/{row['unchanged']}/{row['worsened']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
