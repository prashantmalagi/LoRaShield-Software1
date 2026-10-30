"""Canonical Morse surfaces and V2 token/label contracts."""
from __future__ import annotations

from dataclasses import dataclass

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..", "0": "-----", "1": ".----", "2": "..---",
    "3": "...--", "4": "....-", "5": ".....", "6": "-....",
    "7": "--...", "8": "---..", "9": "----.",
}
CHARSET = tuple(MORSE)
EVENT_VOCAB = ("<PAD>", "<DOT>", "<DASH>", "<LETTER>", "<WORD>")
EVENT_TO_ID = {token: i for i, token in enumerate(EVENT_VOCAB)}
BOUNDARY_LABELS = {"NONE": 0, "LETTER": 1, "WORD": 2}
SYMBOL_OPS = {"DELETE": 0, "DOT": 1, "DASH": 2}
INSERT_OPS = {"NONE": 0, "DOT": 1, "DASH": 2}
EDIT_ACTIONS = ("KEEP_DOT", "KEEP_DASH", "FLIP", "DELETE", "INSERT_DOT", "INSERT_DASH")
PARTITIONS = ("train", "validation", "calibration")
CORRUPTION_KINDS = (
    "clean", "substitution", "burst_substitution", "symbol_insertion",
    "symbol_deletion", "letter_boundary_insertion", "letter_boundary_deletion",
    "letter_boundary_shift", "word_boundary_insertion", "word_boundary_deletion",
    "word_boundary_shift", "combined",
)
MAX_SEQUENCE_TOKENS = 256


def split_morse(value: str) -> list[list[list[str]]]:
    """Parse canonical text into [word][letter][symbol] lists."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("Morse must be nonempty and have no edge whitespace")
    if any(ch not in ".- /" for ch in value):
        raise ValueError("unsupported Morse character")
    words = value.split(" / ")
    parsed = []
    for word in words:
        if not word or word.startswith(" ") or word.endswith(" ") or "  " in word or "/" in word:
            raise ValueError("malformed Morse boundary")
        letters = word.split(" ")
        if any(not letter or any(ch not in ".-" for ch in letter) for letter in letters):
            raise ValueError("empty or malformed Morse letter")
        parsed.append([list(letter) for letter in letters])
    return parsed


def validate_morse(value: str, *, max_tokens: int = MAX_SEQUENCE_TOKENS) -> str:
    words = split_morse(value)
    tokens = 2 + sum(len(letter) for word in words for letter in word)
    tokens += sum(max(0, len(word) - 1) for word in words) + len(words) - 1
    if tokens > max_tokens:
        raise ValueError("Morse sequence exceeds configured token limit")
    return value


def encode_text(text: str) -> str:
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError("source text must be nonempty with no edge whitespace")
    words = text.split(" ")
    if any(not word for word in words) or any(ch not in MORSE for word in words for ch in word):
        raise ValueError("source text supports uppercase A-Z/0-9 and single spaces")
    return " / ".join(" ".join(MORSE[ch] for ch in word) for word in words)


def encode_events(morse: str) -> tuple[int, ...]:
    words = split_morse(morse)
    result = []
    for wi, word in enumerate(words):
        if wi:
            result.append(EVENT_TO_ID["<WORD>"])
        for li, letter in enumerate(word):
            if li:
                result.append(EVENT_TO_ID["<LETTER>"])
            result.extend(EVENT_TO_ID["<DOT>" if symbol == "." else "<DASH>"] for symbol in letter)
    return tuple(result)


@dataclass(frozen=True)
class PreprocessedInput:
    received_morse: str
    symbol_ids: tuple[int, ...]
    input_gap_ids: tuple[int, ...]


def preprocess_received(received_morse: str) -> PreprocessedInput:
    """Reference-free runtime preprocessing for the separate V2 inference path."""
    validate_morse(received_morse)
    symbols = tuple(1 if ch == "." else 2 for ch in received_morse if ch in ".-")
    gaps = boundary_targets(received_morse)
    if len(gaps) != len(symbols) + 1:
        raise AssertionError("canonical input gap count invariant failed")
    return PreprocessedInput(received_morse, symbols, gaps)


def boundary_targets(morse: str) -> tuple[int, ...]:
    """One label per gap in the flattened dot/dash stream, including edge gaps."""
    words = split_morse(morse)
    labels = [BOUNDARY_LABELS["NONE"]]
    for wi, word in enumerate(words):
        for li, letter in enumerate(word):
            for _ in letter:
                labels.append(BOUNDARY_LABELS["NONE"])
            if li < len(word) - 1:
                labels[-1] = BOUNDARY_LABELS["LETTER"]
        if wi < len(words) - 1:
            labels[-1] = BOUNDARY_LABELS["WORD"]
    return tuple(labels)


@dataclass(frozen=True)
class V2Example:
    source_id: str
    partition: str
    corruption_type: str
    severity: float
    seed: int
    source_text: str
    received_morse: str
    reference_morse: str
    symbol_operations: tuple[int, ...]
    insertions_before: tuple[tuple[int, ...], ...]
    output_alignment: tuple[int, ...]
    boundary_labels: tuple[int, ...]
    is_clean: bool

