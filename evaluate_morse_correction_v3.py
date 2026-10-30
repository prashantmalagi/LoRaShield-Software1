"""Standalone evaluation for the existing LoRaShield Morse correction model.

Run from the project root with ``python evaluate_morse_correction_v3.py``.
This script does not change the model or the application's correction setting.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20261029
DEFAULT_MESSAGES = 100
NOISE_LEVELS = (0, 5, 10, 15, 20)
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


def _levenshtein(a: str, b: str) -> int:
    """Unit-cost Levenshtein distance using O(min(len(a), len(b))) memory."""
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


def _boundary_edit_distance(reference: str, candidate: str) -> int:
    """Edit distance that aligns space-delimited Morse letters separately.

    Spaces are boundaries, not symbols that can be substituted for dots or
    dashes. Corresponding letter groups are aligned by position; unmatched
    groups cost their symbol count, and differing boundary counts cost one.
    """
    left, right = reference.split(" "), candidate.split(" ")
    paired = min(len(left), len(right))
    distance = sum(_levenshtein(left[i], right[i]) for i in range(paired))
    distance += sum(map(len, left[paired:])) + sum(map(len, right[paired:]))
    distance += abs((len(left) - 1) - (len(right) - 1))
    return distance


def _edit_rate(reference: str, candidate: str, *, symbols_only: bool = False) -> tuple[int, float]:
    if symbols_only:
        reference = reference.replace(" ", "")
        candidate = candidate.replace(" ", "")
        errors = _levenshtein(reference, candidate)
        denominator = max(len(reference), len(candidate), 1)
    else:
        errors = _boundary_edit_distance(reference, candidate)
        denominator = max(len(reference), len(candidate), 1)
    return errors, errors / denominator


def _encode(text: str) -> str:
    return " ".join(MORSE[ch] for ch in text)


def _noise(morse: str, rate: float, rng: random.Random) -> str:
    return "".join(("-" if ch == "." else ".") if ch in ".-" and rng.random() < rate else ch
                   for ch in morse)


def _self_test() -> None:
    cases = [
        ("...", "...", 0),
        ("...", ".-.", 1),
        ("...", "....", 1),
        ("....", "...", 1),
    ]
    for left, right, expected in cases:
        actual = _levenshtein(left, right)
        assert actual == expected, f"distance({left!r}, {right!r})={actual}, expected {expected}"
    assert _boundary_edit_distance("... ---", "... .... ---") == 8
    print("Self-tests passed: identical, substitution, insertion, deletion, and boundary handling.")


def _run(messages_count: int) -> int:
    try:
        from correction.morse_corrector import MorseCorrector

        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            print(f"ERROR: model unavailable: {corrector._load_error}", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"ERROR: could not load corrector/model: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    corpus_rng = random.Random(SEED)
    corpus = ["".join(corpus_rng.choice(CHARSET) for _ in range(corpus_rng.randint(3, 8)))
              for _ in range(messages_count)]
    encoded = [_encode(message) for message in corpus]
    too_long = sum(len(message) > corrector._seq_len for message in encoded)
    if too_long:
        print(f"ERROR: {too_long} corpus messages exceed model limit {corrector._seq_len}.", file=sys.stderr)
        return 2

    print(f"Model: {corrector._model_path}")
    print(f"Seed: {SEED}; messages per condition: {messages_count}; sequence length: {corrector._seq_len}")
    print("All error comparisons use the clean encoded message as reference.")
    print("Rates: boundary-aware edit distance / max string length; symbol-only edit distance / max dot-dash count.")

    for noise_pct in NOISE_LEVELS:
        noise_rng = random.Random(SEED + noise_pct)
        before_edits = after_edits = before_symbols = after_symbols = 0
        before_chars = after_chars = before_symbol_count = after_symbol_count = 0
        improved = unchanged = worse = errors = 0
        clean_preserved = 0
        for clean in encoded:
            received = _noise(clean, noise_pct / 100, noise_rng)
            try:
                result = corrector.correct(received)
                if result.error:
                    raise RuntimeError(result.error)
                corrected = result.corrected_morse
            except Exception as exc:
                errors += 1
                print(f"Inference error at {noise_pct}% noise; using received input for this case: {exc}", file=sys.stderr)
                corrected = received

            before, _ = _edit_rate(clean, received)
            after, _ = _edit_rate(clean, corrected)
            before_sym, _ = _edit_rate(clean, received, symbols_only=True)
            after_sym, _ = _edit_rate(clean, corrected, symbols_only=True)
            before_edits += before
            after_edits += after
            before_symbols += before_sym
            after_symbols += after_sym
            before_chars += max(len(clean), len(received), 1)
            after_chars += max(len(clean), len(corrected), 1)
            before_symbol_count += max(len(clean.replace(" ", "")), len(received.replace(" ", "")), 1)
            after_symbol_count += max(len(clean.replace(" ", "")), len(corrected.replace(" ", "")), 1)
            if after < before:
                improved += 1
            elif after > before:
                worse += 1
            else:
                unchanged += 1
            if noise_pct == 0 and corrected == clean:
                clean_preserved += 1

        print(f"\n{noise_pct}% noise:")
        print(f"  Before correction: boundary edits={before_edits}, edit rate={before_edits / before_chars:.3%}; "
              f"dot/dash edits={before_symbols}, symbol rate={before_symbols / before_symbol_count:.3%}")
        print(f"  After correction:  boundary edits={after_edits}, edit rate={after_edits / after_chars:.3%}; "
              f"dot/dash edits={after_symbols}, symbol rate={after_symbols / after_symbol_count:.3%}")
        print(f"  Compared with received input: improved={improved}, unchanged={unchanged}, worse={worse}; "
              f"inference errors={errors}")
        if noise_pct == 0:
            print(f"  Clean-input preservation: {clean_preserved}/{messages_count} "
                  f"({clean_preserved / messages_count:.1%}) exact messages preserved")
        if after_edits < before_edits:
            interpretation = "aggregate boundary-aware edit errors decreased."
        elif after_edits > before_edits:
            interpretation = "aggregate boundary-aware edit errors increased."
        else:
            interpretation = "aggregate boundary-aware edit errors were unchanged."
        print(f"  Interpretation: {interpretation}")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES,
                        help=f"messages per condition (default: {DEFAULT_MESSAGES})")
    parser.add_argument("--self-test", action="store_true", help="run metric self-tests and exit")
    args = parser.parse_args()
    if args.messages < 1:
        parser.error("--messages must be positive")
    _self_test()
    if args.self_test:
        return 0
    return _run(args.messages)


if __name__ == "__main__":
    raise SystemExit(main())
