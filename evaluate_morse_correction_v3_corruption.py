"""Structured synthetic corruption stress test for the saved V3 candidate.

This is a software-only experiment. It generates independent A-Z/0-9 references
and applies dot/dash substitutions, symbol insertions/deletions, short bursts,
letter-space edits, and slash-boundary edits. All results are synthetic and do
not establish real radio-channel performance. The evaluator does not train or
select the model, access the real-capture dataset, or write result files.

Capability notes: substitutions are in-vocabulary and similar to V3 training.
Symbols and spaces are also representable, but the model emits one token per
input position; it cannot safely restore sequence length after insertions or
deletions. Slash is not a model token: word boundaries are split before
inference and rejoined as received, so slash-boundary correction is unsupported.
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
SEED = 2748193
DEFAULT_MESSAGES = 100
SEQUENCE_LENGTH = 120
VOCABULARY = ["<PAD>", "<UNK>", ".", "-", " "]
TOKEN_IDS = {token: index for index, token in enumerate(VOCABULARY)}
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)
CORRUPTION_TYPES = (
    "clean",
    "dot_dash_substitution",
    "dot_dash_insertion",
    "dot_dash_deletion",
    "short_burst_substitution",
    "letter_boundary_insertion",
    "letter_boundary_deletion",
    "word_boundary_insertion",
    "word_boundary_deletion",
    "word_boundary_shift",
)
CAPABILITY_BY_TYPE = {
    "clean": "clean baseline",
    "dot_dash_substitution": "in vocabulary and similar to synthetic training noise",
    "dot_dash_insertion": "input alphabet supported; output length cannot safely repair the insertion",
    "dot_dash_deletion": "input alphabet supported; output length cannot safely restore the deletion",
    "short_burst_substitution": "in vocabulary; burst distribution differs from independent training flips",
    "letter_boundary_insertion": "space is in vocabulary; fixed-position output cannot safely remove an inserted position",
    "letter_boundary_deletion": "space is in vocabulary; fixed-position output cannot safely restore a deleted position",
    "word_boundary_insertion": "slash is not a model token; splitter/rejoiner carries observed segmentation through",
    "word_boundary_deletion": "slash is not a model token; splitter/rejoiner carries observed segmentation through",
    "word_boundary_shift": "slash is not a model token; splitter/rejoiner carries observed segmentation through",
}
SUBSTITUTION_PERCENT = 10
BURST_MIN_SYMBOLS = 2
BURST_MAX_SYMBOLS = 3

from evaluate_morse_correction_v3 import MORSE  # noqa: E402
from evaluate_morse_correction_v3_independent import _old_partition_messages  # noqa: E402
from evaluate_morse_correction_v3_synthetic import (  # noqa: E402
    DEFAULT_MESSAGES as PREVIOUS_SYNTHETIC_COUNT,
    SEED as PREVIOUS_SYNTHETIC_SEED,
    generate_source_corpus as generate_previous_synthetic_corpus,
)


@dataclass(frozen=True)
class CorruptionCase:
    source_text: str
    reference_morse: str
    received_morse: str
    corruption_type: str


@lru_cache(maxsize=1)
def _excluded_source_words() -> frozenset[str]:
    words = set(_old_partition_messages())
    previous_messages = generate_previous_synthetic_corpus(
        PREVIOUS_SYNTHETIC_SEED, PREVIOUS_SYNTHETIC_COUNT
    )
    words.update(word for message in previous_messages for word in message.split())
    return frozenset(words)


def encode_alphanumeric(text: str) -> str:
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError("source text must be nonempty without edge whitespace")
    words = text.split(" ")
    if any(not word for word in words):
        raise ValueError("source words must be separated by one space")
    if any(any(char not in MORSE for char in word) for word in words):
        raise ValueError("source text supports A-Z and 0-9 only")
    return " / ".join(" ".join(MORSE[char] for char in word) for word in words)


def split_words(morse: str) -> list[str]:
    if not isinstance(morse, str) or not morse:
        raise ValueError("Morse input must be a nonempty string")
    if morse != morse.strip():
        raise ValueError("leading or trailing whitespace is invalid")
    if ":" in morse:
        raise ValueError("ciphertext-shaped input is not accepted")
    if any(char not in ".- /" for char in morse):
        raise ValueError("unsupported Morse character")
    words = morse.split(" / ")
    if any(not word or "/" in word for word in words):
        raise ValueError("word separator must be exactly ' / ' and words cannot be empty")
    for word in words:
        if word.startswith(" ") or word.endswith(" ") or "  " in word:
            raise ValueError("letter boundaries must be single spaces")
        if len(word) > SEQUENCE_LENGTH:
            raise ValueError(f"word length {len(word)} exceeds {SEQUENCE_LENGTH}; no truncation")
        if any(char not in ".- " for char in word):
            raise ValueError("word contains symbols outside the model input alphabet")
    return words


def generate_source_corpus(seed: int = SEED, count: int = DEFAULT_MESSAGES) -> list[str]:
    if count < 1:
        raise ValueError("message count must be positive")
    rng = random.Random(seed)
    alphabet = tuple(MORSE)
    excluded = _excluded_source_words()
    seen: set[str] = set()
    sources = []
    while len(sources) < count:
        words = []
        for _ in range(rng.randint(2, 3)):
            word = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 5)))
            # All generated sources include a code with at least 3 symbols so
            # burst and letter-boundary insertion conditions are applicable.
            if not any(len(MORSE[char]) >= BURST_MIN_SYMBOLS for char in word):
                word = word[:-1] + rng.choice(tuple(char for char in alphabet if len(MORSE[char]) >= 3))
            words.append(word)
        text = " ".join(words)
        if text in seen or set(words) & excluded:
            continue
        reference = encode_alphanumeric(text)
        if any(len(word) > SEQUENCE_LENGTH for word in split_words(reference)):
            raise AssertionError("generated reference exceeds model word limit")
        seen.add(text)
        sources.append(text)
    if len(set(sources)) != count or any(set(source.split()) & excluded for source in sources):
        raise AssertionError("source uniqueness or known-corpus disjointness failed")
    return sources


def _replace_word(morse: str, word_index: int, replacement: list[str]) -> str:
    words = split_words(morse)
    words[word_index:word_index + 1] = replacement
    result = " / ".join(words)
    split_words(result)
    return result


def corrupt_symbols(morse: str, kind: str, rng: random.Random) -> str:
    words = split_words(morse)
    if kind == "dot_dash_insertion":
        index = rng.randrange(len(words))
        word = words[index]
        insertion_at = rng.randrange(len(word) + 1)
        symbol = rng.choice(".-")
        return _replace_word(morse, index, [word[:insertion_at] + symbol + word[insertion_at:]])
    if kind == "dot_dash_deletion":
        choices = []
        for word_index, word in enumerate(words):
            groups = word.split(" ")
            offset = 0
            for group in groups:
                if len(group) > 1:
                    choices.extend((word_index, offset + pos) for pos in range(len(group)))
                offset += len(group) + 1
        word_index, position = rng.choice(choices)
        word = words[word_index]
        return _replace_word(morse, word_index, [word[:position] + word[position + 1:]])
    if kind == "short_burst_substitution":
        runs = []
        for word_index, word in enumerate(words):
            start = 0
            while start < len(word):
                if word[start] not in ".-":
                    start += 1
                    continue
                end = start
                while end < len(word) and word[end] in ".-":
                    end += 1
                if end - start >= BURST_MIN_SYMBOLS:
                    runs.append((word_index, start, end))
                start = end
        word_index, run_start, run_end = rng.choice(runs)
        run_len = run_end - run_start
        burst_len = rng.randint(BURST_MIN_SYMBOLS, min(BURST_MAX_SYMBOLS, run_len))
        burst_start = rng.randint(run_start, run_end - burst_len)
        word = words[word_index]
        chars = list(word)
        for pos in range(burst_start, burst_start + burst_len):
            chars[pos] = "-" if chars[pos] == "." else "."
        return _replace_word(morse, word_index, ["".join(chars)])
    raise ValueError(f"unsupported symbol corruption: {kind}")


def corrupt_letter_boundary(morse: str, kind: str, rng: random.Random) -> str:
    words = split_words(morse)
    if kind == "letter_boundary_insertion":
        choices = [(wi, pos) for wi, word in enumerate(words)
                   for pos in range(1, len(word))
                   if word[pos - 1] in ".-" and word[pos] in ".-"]
        wi, pos = rng.choice(choices)
        word = words[wi]
        return _replace_word(morse, wi, [word[:pos] + " " + word[pos:]])
    if kind == "letter_boundary_deletion":
        choices = [(wi, pos) for wi, word in enumerate(words)
                   for pos, char in enumerate(word) if char == " "]
        wi, pos = rng.choice(choices)
        word = words[wi]
        return _replace_word(morse, wi, [word[:pos] + word[pos + 1:]])
    raise ValueError(f"unsupported letter-boundary corruption: {kind}")


def corrupt_word_boundary(morse: str, kind: str, rng: random.Random) -> str:
    words = split_words(morse)
    if kind == "word_boundary_deletion":
        index = rng.randrange(len(words) - 1)
        words[index:index + 2] = [words[index] + " " + words[index + 1]]
    elif kind == "word_boundary_insertion":
        choices = [(wi, group_index) for wi, word in enumerate(words)
                   for group_index in range(1, len(word.split(" ")))]
        wi, group_index = rng.choice(choices)
        groups = words[wi].split(" ")
        words[wi:wi + 1] = [" ".join(groups[:group_index]), " ".join(groups[group_index:])]
    elif kind == "word_boundary_shift":
        shiftable = [index for index in range(len(words) - 1)
                     if len(words[index].split()) > 1 or len(words[index + 1].split()) > 1]
        index = rng.choice(shiftable)
        left_groups, right_groups = words[index].split(), words[index + 1].split()
        if len(left_groups) > 1:
            moved = left_groups.pop()
            right_groups.insert(0, moved)
        else:
            moved = right_groups.pop(0)
            left_groups.append(moved)
        words[index:index + 2] = [" ".join(left_groups), " ".join(right_groups)]
    else:
        raise ValueError(f"unsupported word-boundary corruption: {kind}")
    result = " / ".join(words)
    split_words(result)
    return result


def make_corruption_case(source: str, kind: str, rng: random.Random) -> CorruptionCase:
    reference = encode_alphanumeric(source)
    if kind == "clean":
        received = reference
    elif kind == "dot_dash_substitution":
        positions = [index for index, char in enumerate(reference) if char in ".-"]
        flip_count = max(1, int(round(len(positions) * SUBSTITUTION_PERCENT / 100)))
        selected = set(rng.sample(positions, flip_count))
        received = "".join(("-" if char == "." else ".") if index in selected else char
                           for index, char in enumerate(reference))
    elif kind in {"dot_dash_insertion", "dot_dash_deletion", "short_burst_substitution"}:
        received = corrupt_symbols(reference, kind, rng)
    elif kind in {"letter_boundary_insertion", "letter_boundary_deletion"}:
        received = corrupt_letter_boundary(reference, kind, rng)
    elif kind in {"word_boundary_insertion", "word_boundary_deletion", "word_boundary_shift"}:
        received = corrupt_word_boundary(reference, kind, rng)
    else:
        raise ValueError(f"unknown corruption type: {kind}")
    split_words(received)
    return CorruptionCase(source, reference, received, kind)


def generate_corpus(seed: int = SEED, count: int = DEFAULT_MESSAGES) -> dict[str, list[CorruptionCase]]:
    sources = generate_source_corpus(seed, count)
    return {
        kind: [make_corruption_case(source, kind, random.Random(seed + condition_index * 100003 + i))
               for i, source in enumerate(sources)]
        for condition_index, kind in enumerate(CORRUPTION_TYPES)
    }


def levenshtein(left: str, right: str) -> int:
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


def dot_dash_stream(morse: str) -> str:
    return morse.replace(" ", "").replace("/", "")


def symbol_error_count(reference: str, candidate: str) -> int:
    return levenshtein(dot_dash_stream(reference), dot_dash_stream(candidate))


def word_boundary_positions(morse: str) -> tuple[int, ...]:
    words = morse.split(" / ")
    cumulative = 0
    positions = []
    for word in words[:-1]:
        cumulative += len(word.split())
        positions.append(cumulative)
    return tuple(positions)


def boundaries_preserved(reference: str, candidate: str) -> bool:
    return word_boundary_positions(reference) == word_boundary_positions(candidate)


def encode_padded(word: str) -> list[int]:
    if not word or len(word) > SEQUENCE_LENGTH or any(char not in ".- " for char in word):
        raise ValueError("inference accepts only slash-free dot/dash/space words up to 120 chars")
    ids = [TOKEN_IDS[char] for char in word]
    return ids + [PAD_ID] * (SEQUENCE_LENGTH - len(ids))


def decode_prediction(word: str, prediction_ids: Iterable[int]) -> str:
    id_to_token = {index: token for token, index in TOKEN_IDS.items()}
    ids = list(prediction_ids)
    if len(ids) < len(word):
        raise ValueError("model output shorter than input word")
    return "".join(
        word[index] if id_to_token.get(int(token), "<UNK>") in ("<PAD>", "<UNK>")
        else id_to_token[int(token)]
        for index, token in enumerate(ids[:len(word)])
    )


def predict_messages(model: Any, messages: list[str]) -> list[str]:
    import numpy as np

    split_messages = [split_words(message) for message in messages]
    all_words = [word for words in split_messages for word in words]
    batch = np.asarray([encode_padded(word) for word in all_words], dtype=np.int32)
    scores = model.predict(batch, verbose=0)
    expected = (len(all_words), SEQUENCE_LENGTH, len(VOCABULARY))
    if scores.shape != expected:
        raise ValueError(f"model output shape {scores.shape}; expected {expected}")
    corrected_words = [decode_prediction(word, score.argmax(axis=-1))
                       for word, score in zip(all_words, scores)]
    result = []
    offset = 0
    for words in split_messages:
        result.append(" / ".join(corrected_words[offset:offset + len(words)]))
        offset += len(words)
    return result


def score_condition(cases: list[CorruptionCase], corrected: list[str]) -> dict[str, Any]:
    if len(cases) != len(corrected):
        raise ValueError("case/prediction count mismatch")
    raw_errors = corrected_errors = raw_den = corrected_den = 0
    raw_boundary_count = corrected_boundary_count = 0
    improved = unchanged = worsened = 0
    for case, output in zip(cases, corrected):
        reference_stream = dot_dash_stream(case.reference_morse)
        raw_stream = dot_dash_stream(case.received_morse)
        output_stream = dot_dash_stream(output)
        before = levenshtein(reference_stream, raw_stream)
        after = levenshtein(reference_stream, output_stream)
        raw_errors += before
        corrected_errors += after
        raw_den += max(len(reference_stream), len(raw_stream), 1)
        corrected_den += max(len(reference_stream), len(output_stream), 1)
        raw_boundary_count += boundaries_preserved(case.reference_morse, case.received_morse)
        corrected_boundary_count += boundaries_preserved(case.reference_morse, output)
        if after < before:
            improved += 1
        elif after > before:
            worsened += 1
        else:
            unchanged += 1
    count = len(cases)
    return {
        "messages": count,
        "raw_symbol_errors": raw_errors,
        "corrected_symbol_errors": corrected_errors,
        "raw_symbol_error_rate": raw_errors / max(raw_den, 1),
        "corrected_symbol_error_rate": corrected_errors / max(corrected_den, 1),
        "raw_word_boundaries_preserved": raw_boundary_count,
        "corrected_word_boundaries_preserved": corrected_boundary_count,
        "raw_word_boundary_preservation_rate": raw_boundary_count / max(count, 1),
        "corrected_word_boundary_preservation_rate": corrected_boundary_count / max(count, 1),
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
    }


def evaluate(model: Any, cases_by_type: dict[str, list[CorruptionCase]]) -> dict[str, Any]:
    report = {}
    for kind in CORRUPTION_TYPES:
        cases = cases_by_type[kind]
        outputs = predict_messages(model, [case.received_morse for case in cases])
        report[kind] = score_condition(cases, outputs)
        if kind == "clean":
            report[kind]["exactly_preserved"] = sum(
                case.reference_morse == output for case, output in zip(cases, outputs)
            )
            report[kind]["clean_preservation_rate"] = (
                report[kind]["exactly_preserved"] / max(len(cases), 1)
            )
    return report


def validate_model_metadata(model: Any, metadata: dict[str, Any]) -> None:
    expected_ids = {token: i for i, token in enumerate(VOCABULARY)}
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("metadata does not identify V3 model")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("metadata sequence length mismatch")
    if metadata.get("vocabulary") != VOCABULARY or metadata.get("token_ids") != expected_ids:
        raise ValueError("metadata vocabulary/token IDs mismatch")
    if metadata.get("padding_id") != PAD_ID:
        raise ValueError("metadata PAD ID mismatch")
    if tuple(model.input_shape) != (None, SEQUENCE_LENGTH):
        raise ValueError(f"V3 input shape mismatch: {model.input_shape}")
    if tuple(model.output_shape) != (None, SEQUENCE_LENGTH, len(VOCABULARY)):
        raise ValueError(f"V3 output shape mismatch: {model.output_shape}")
    layer = model.get_layer("token_embedding")
    if layer.input_dim != len(VOCABULARY) or not layer.mask_zero:
        raise ValueError("V3 embedding/PAD mapping mismatch")


def load_artifacts() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("V3 model and metadata files are required")
    with METADATA_PATH.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    import tensorflow as tf

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    validate_model_metadata(model, metadata)
    return model, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES,
                        help=f"independent source messages per corruption type (default {DEFAULT_MESSAGES})")
    args = parser.parse_args()
    if args.messages < 1:
        parser.error("--messages must be positive")
    try:
        cases = generate_corpus(SEED, args.messages)
        model, metadata = load_artifacts()
        report = evaluate(model, cases)
    except Exception as exc:
        print(f"ERROR: V3 corruption evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print("LoRaShield V3 structured corruption experiment (synthetic only)")
    print(f"Model: {MODEL_PATH.name}; metadata: {METADATA_PATH.name}")
    print(f"Seed: {SEED}; unique source messages: {args.messages}; excluded prior source words: {len(_excluded_source_words())}")
    print("Metric: dot/dash Levenshtein errors / summed per-message max(reference, raw/corrected symbol length).")
    print("Outcome counts compare corrected symbol errors with uncorrected received symbol errors.")
    print("Boundary preservation compares ordered slash positions measured in Morse letter groups.")
    print("Capability by type:")
    for kind in CORRUPTION_TYPES:
        print(f"  {kind}: {CAPABILITY_BY_TYPE[kind]}")
    print(f"Clean exact preservation: {report['clean']['exactly_preserved']}/{args.messages} ({report['clean']['clean_preservation_rate']:.2%})")
    for kind in CORRUPTION_TYPES[1:]:
        row = report[kind]
        print(f"{kind}: symbol error rate {row['raw_symbol_error_rate']:.3%} -> {row['corrected_symbol_error_rate']:.3%}; "
              f"word boundaries raw/corrected={row['raw_word_boundaries_preserved']}/{row['messages']} -> "
              f"{row['corrected_word_boundaries_preserved']}/{row['messages']}; "
              f"improved/unchanged/worsened={row['improved']}/{row['unchanged']}/{row['worsened']}")
    print("Limitations: all corruption is programmatically generated; this does not model RF timing, packet loss, hardware CRC, or natural language and is not evidence of real radio performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
