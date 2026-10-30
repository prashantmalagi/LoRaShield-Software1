"""Reproducible software-only evaluation of the saved LoRaShield V3 model.

This evaluator creates synthetic A-Z/0-9 messages in memory and applies fixed-
count dot/dash flips. It never reads the real-capture JSONL dataset and never
writes results. Synthetic substitution tests do not establish real radio-channel
performance, language-distribution performance, or insertion/deletion recovery.
The corpus is evaluation-only; no training or model-selection step is performed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
SEED = 8631052
DEFAULT_MESSAGES = 200
NOISE_LEVELS = (0, 5, 10, 15, 20)
SEQUENCE_LENGTH = 120
VOCABULARY = ["<PAD>", "<UNK>", ".", "-", " "]
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)

# This module holds the project's established A-Z/0-9 Morse mapping and has no
# model, logger, backend, serial, radio, AES, or CRC import at module scope.
from evaluate_morse_correction_v3 import MORSE  # noqa: E402
from evaluate_morse_correction_v3_independent import _old_partition_messages  # noqa: E402


@dataclass(frozen=True)
class EvaluationCase:
    source_text: str
    reference_morse: str
    received_morse: str
    noise_percent: int


@lru_cache(maxsize=1)
def _prior_source_messages() -> frozenset[str]:
    """Recreate known earlier V3 partitions only to exclude exact word overlap."""
    return frozenset(_old_partition_messages())


def encode_alphanumeric(text: str) -> str:
    """Encode uppercase A-Z/0-9 words using canonical project separators."""
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError("source text must be nonempty text without edge whitespace")
    words = text.split(" ")
    if any(not word for word in words):
        raise ValueError("source text must use single spaces between nonempty words")
    if any(any(char not in MORSE for char in word) for word in words):
        raise ValueError("source text supports A-Z and 0-9 only")
    return " / ".join(" ".join(MORSE[char] for char in word) for word in words)


def split_words(morse: str) -> list[str]:
    """Validate canonical Morse and split words before model inference."""
    if not isinstance(morse, str) or not morse:
        raise ValueError("Morse input must be a nonempty string")
    if morse != morse.strip():
        raise ValueError("leading or trailing whitespace is not canonical Morse")
    if ":" in morse:
        raise ValueError("ciphertext-shaped input is not accepted")
    if any(char not in ".- /" for char in morse):
        raise ValueError("Morse input contains unsupported characters")
    words = morse.split(" / ")
    if any(not word or "/" in word for word in words):
        raise ValueError("word boundaries must use exactly ' / ' with nonempty words")
    for word in words:
        if word.startswith(" ") or word.endswith(" ") or "  " in word:
            raise ValueError("letter boundaries must be single spaces")
        if len(word) > SEQUENCE_LENGTH:
            raise ValueError(
                f"Morse word length {len(word)} exceeds {SEQUENCE_LENGTH}; truncation is forbidden"
            )
        if any(char not in ".- " for char in word):
            raise ValueError("Morse words may contain only dot, dash, and letter spaces")
    return words


def generate_source_corpus(seed: int = SEED, count: int = DEFAULT_MESSAGES) -> list[str]:
    """Generate unique 2-3-word sources; prior V3 training sources had no spaces."""
    if count < 1:
        raise ValueError("message count must be positive")
    rng = random.Random(seed)
    alphabet = tuple(MORSE)
    excluded_words = _prior_source_messages()
    seen: set[str] = set()
    sources: list[str] = []
    while len(sources) < count:
        word_count = rng.randint(2, 3)
        words = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 5)))
            for _ in range(word_count)
        ]
        text = " ".join(words)
        if text not in seen and not (set(words) & excluded_words):
            reference = encode_alphanumeric(text)
            if any(len(word) > SEQUENCE_LENGTH for word in split_words(reference)):
                raise AssertionError("generated message exceeds V3 word length")
            seen.add(text)
            sources.append(text)
    if (len(set(sources)) != count or any(" " not in source for source in sources)
            or any(set(source.split()) & excluded_words for source in sources)):
        raise AssertionError("source corpus uniqueness or training-corpus disjointness failed")
    return sources


def apply_substitution_noise(morse: str, noise_percent: int, rng: random.Random) -> str:
    """Flip exactly round(rate * dot/dash symbols) selected symbols, preserving spaces."""
    if noise_percent not in NOISE_LEVELS:
        raise ValueError(f"noise percent must be one of {NOISE_LEVELS}")
    split_words(morse)
    symbol_positions = [index for index, char in enumerate(morse) if char in ".-"]
    flip_count = int(round(len(symbol_positions) * noise_percent / 100.0))
    chosen = set(rng.sample(symbol_positions, flip_count))
    return "".join(("-" if char == "." else ".") if index in chosen else char
                   for index, char in enumerate(morse))


def generate_cases(seed: int = SEED, count: int = DEFAULT_MESSAGES) -> dict[int, list[EvaluationCase]]:
    """Generate clean references first, then independent deterministic noise variants."""
    sources = generate_source_corpus(seed, count)
    clean_pairs = [(source, encode_alphanumeric(source)) for source in sources]
    cases: dict[int, list[EvaluationCase]] = {}
    for noise_percent in NOISE_LEVELS:
        rng = random.Random(seed + noise_percent + 101)
        condition = []
        for source_text, reference in clean_pairs:
            received = apply_substitution_noise(reference, noise_percent, rng)
            condition.append(EvaluationCase(source_text, reference, received, noise_percent))
        cases[noise_percent] = condition
    return cases


def levenshtein(left: str, right: str) -> int:
    """Standard unit-cost edit distance for dot/dash streams."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def dot_dash_stream(morse: str) -> str:
    return morse.replace(" ", "").replace("/", "")


def symbol_error_count(reference: str, candidate: str) -> int:
    return levenshtein(dot_dash_stream(reference), dot_dash_stream(candidate))


def word_boundary_positions(morse: str) -> tuple[int, ...]:
    """Ordered slash positions, measured in Morse letter groups before each slash."""
    words = morse.split(" / ")
    cumulative = 0
    positions = []
    for word in words[:-1]:
        cumulative += len(word.split())
        positions.append(cumulative)
    return tuple(positions)


def word_boundaries_preserved(reference: str, candidate: str) -> bool:
    return word_boundary_positions(reference) == word_boundary_positions(candidate)


def encode_padded(word: str) -> list[int]:
    if not word or len(word) > SEQUENCE_LENGTH or any(char not in ".- " for char in word):
        raise ValueError("model input must be a valid slash-free Morse word of length <= 120")
    ids = [TOKEN_IDS[char] for char in word]
    return ids + [PAD_ID] * (SEQUENCE_LENGTH - len(ids))


def decode_prediction(input_word: str, token_ids: Iterable[int]) -> str:
    id_to_token = {index: token for token, index in TOKEN_IDS.items()}
    predictions = list(token_ids)
    if len(predictions) < len(input_word):
        raise ValueError("model output is shorter than the unpadded input")
    result = []
    for index, token_id in enumerate(predictions[:len(input_word)]):
        char = id_to_token.get(int(token_id), "<UNK>")
        result.append(input_word[index] if char in ("<PAD>", "<UNK>") else char)
    return "".join(result)


def validate_model_metadata(model: Any, metadata: dict[str, Any]) -> None:
    expected_ids = {token: index for index, token in enumerate(VOCABULARY)}
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("V3 metadata does not identify the requested model artifact")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("V3 metadata sequence length mismatch")
    if metadata.get("vocabulary") != VOCABULARY or metadata.get("token_ids") != expected_ids:
        raise ValueError("V3 metadata vocabulary or token ID mapping mismatch")
    if metadata.get("padding_id") != PAD_ID:
        raise ValueError("V3 metadata PAD ID mismatch")
    if tuple(model.input_shape) != (None, SEQUENCE_LENGTH):
        raise ValueError(f"V3 input shape mismatch: {model.input_shape}")
    if tuple(model.output_shape) != (None, SEQUENCE_LENGTH, len(VOCABULARY)):
        raise ValueError(f"V3 output shape mismatch: {model.output_shape}")
    embedding = model.get_layer("token_embedding")
    if embedding.input_dim != len(VOCABULARY) or not embedding.mask_zero:
        raise ValueError("V3 embedding vocabulary/PAD mask mismatch")


def predict_messages(model: Any, morse_messages: list[str]) -> list[str]:
    """Validate every message before inference; infer words only, never slash."""
    import numpy as np

    split_messages = [split_words(message) for message in morse_messages]
    all_words = [word for words in split_messages for word in words]
    batch = np.asarray([encode_padded(word) for word in all_words], dtype=np.int32)
    predictions = model.predict(batch, verbose=0)
    expected = (len(all_words), SEQUENCE_LENGTH, len(VOCABULARY))
    if predictions.shape != expected:
        raise ValueError(f"V3 prediction shape {predictions.shape}, expected {expected}")
    corrected_words = []
    for word, scores in zip(all_words, predictions):
        corrected_words.append(decode_prediction(word, scores.argmax(axis=-1)))
    result = []
    offset = 0
    for words in split_messages:
        count = len(words)
        result.append(" / ".join(corrected_words[offset:offset + count]))
        offset += count
    return result


def score_condition(cases: list[EvaluationCase], corrected: list[str]) -> dict[str, Any]:
    if len(cases) != len(corrected):
        raise ValueError("case and prediction counts do not match")
    raw_errors = corrected_errors = raw_symbols = corrected_symbols = 0
    raw_denominator = corrected_denominator = 0
    raw_boundaries = corrected_boundaries = 0
    improved = unchanged = worsened = 0
    for case, candidate in zip(cases, corrected):
        reference_symbols = dot_dash_stream(case.reference_morse)
        received_symbols = dot_dash_stream(case.received_morse)
        candidate_symbols = dot_dash_stream(candidate)
        before = levenshtein(reference_symbols, received_symbols)
        after = levenshtein(reference_symbols, candidate_symbols)
        raw_symbols += before
        corrected_symbols += after
        raw_denominator += max(len(reference_symbols), len(received_symbols), 1)
        corrected_denominator += max(len(reference_symbols), len(candidate_symbols), 1)
        raw_errors += before
        corrected_errors += after
        raw_boundaries += word_boundaries_preserved(case.reference_morse, case.received_morse)
        corrected_boundaries += word_boundaries_preserved(case.reference_morse, candidate)
        if after < before:
            improved += 1
        elif after > before:
            worsened += 1
        else:
            unchanged += 1
    total = len(cases)
    return {
        "messages": total,
        "raw_symbol_errors": raw_errors,
        "corrected_symbol_errors": corrected_errors,
        "raw_symbol_error_rate": raw_symbols / max(raw_denominator, 1),
        "corrected_symbol_error_rate": corrected_symbols / max(corrected_denominator, 1),
        "raw_word_boundaries_preserved": raw_boundaries,
        "corrected_word_boundaries_preserved": corrected_boundaries,
        "word_boundary_preservation_rate_raw": raw_boundaries / max(total, 1),
        "word_boundary_preservation_rate_corrected": corrected_boundaries / max(total, 1),
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
    }


def evaluate(model: Any, cases_by_noise: dict[int, list[EvaluationCase]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    clean_preserved = 0
    clean_cases = cases_by_noise[0]
    for noise_percent in NOISE_LEVELS:
        cases = cases_by_noise[noise_percent]
        predictions = predict_messages(model, [case.received_morse for case in cases])
        results[f"noise_{noise_percent}_percent"] = score_condition(cases, predictions)
        if noise_percent == 0:
            clean_preserved = sum(
                candidate == case.reference_morse
                for case, candidate in zip(cases, predictions)
            )
    results["clean_message_preservation"] = {
        "messages": len(clean_cases),
        "exactly_preserved": clean_preserved,
        "rate": clean_preserved / max(len(clean_cases), 1),
    }
    return results


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"V3 model not found: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"V3 metadata not found: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    import tensorflow as tf

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    validate_model_metadata(model, metadata)
    return model, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES,
                        help=f"unique source messages per noise level (default {DEFAULT_MESSAGES})")
    args = parser.parse_args()
    if args.messages < 1:
        parser.error("--messages must be positive")
    try:
        cases_by_noise = generate_cases(SEED, args.messages)
        sources = [case.source_text for case in cases_by_noise[0]]
        model, metadata = load_artifacts()
        results = evaluate(model, cases_by_noise)
    except Exception as exc:
        print(f"ERROR: synthetic V3 evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("Synthetic V3 Morse correction evaluation (software-only; no real-channel claim)")
    print(f"Model: {MODEL_PATH.name}; metadata: {METADATA_PATH.name}")
    print(f"Fixed seed: {SEED}; unique source messages: {len(sources)}; per-level cases: {args.messages}")
    print(f"Prior train/validation/test source words excluded: {len(_prior_source_messages())}")
    print(f"Noise: exact rounded count of independent random dot/dash flips; levels={NOISE_LEVELS}")
    print("References are encoded from source text before corruption; each slash-delimited word is inferred separately.")
    print("Rates: dot/dash Levenshtein errors divided by summed per-message max(reference, received/candidate symbol lengths).")
    print("Outcome counts compare corrected dot/dash errors to unchanged received-input dot/dash errors.")
    clean = results["clean_message_preservation"]
    print(f"Clean preservation: {clean['exactly_preserved']}/{clean['messages']} ({clean['rate']:.2%})")
    for level in NOISE_LEVELS:
        row = results[f"noise_{level}_percent"]
        print(f"{level:>2}% noise: symbol error rate {row['raw_symbol_error_rate']:.3%} -> "
              f"{row['corrected_symbol_error_rate']:.3%}; word boundaries preserved raw/corrected="
              f"{row['raw_word_boundaries_preserved']}/{row['messages']} -> "
              f"{row['corrected_word_boundaries_preserved']}/{row['messages']}; "
              f"improved/unchanged/worsened={row['improved']}/{row['unchanged']}/{row['worsened']}")
    print("Limitations: synthetic A-Z/0-9 strings with substitution flips only; no insertions, deletions, bursts, timing, RF, packet, or language effects. Earlier source messages are recreated only to exclude overlap; this evaluation does not tune or select the model and does not establish real radio-channel performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
