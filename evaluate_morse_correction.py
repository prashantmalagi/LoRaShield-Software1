"""
LoRaShield – V2 Morse Correction Evaluation Script
====================================================
Independently evaluates the trained V2 Keras model on freshly generated
test messages that were NOT used during training.

Run from the project root:
    python evaluate_morse_correction.py

Optional flags:
    --noise-levels 0 5 10 15 20   (noise % values to test; default as above)
    --messages 200                (number of test messages per noise level)
    --seed 999                    (random seed for reproducibility)
    --enable-correction           (default: correction ON for evaluation)

Evaluation protocol
-------------------
1.  Generate *N* random messages using the same character set as
    training (A-Z, 0-9).  These are INDEPENDENT of any training data.
2.  Encode each message to Morse using the standard ITU dictionary.
3.  For each noise level:
    a.  Inject dot/dash flips at the specified probability.
    b.  Compute SER(clean, noisy)  -- baseline without correction.
    c.  Apply the V2 corrector.
    d.  Compute SER(clean, corrected) -- after correction.
    e.  Report improvement = SER_before - SER_after.
    f.  Flag cases where correction makes results worse.
4.  Exclude all <PAD> positions from the SER calculation.
5.  Report a table of results and a per-noise-level summary.

Limitations
-----------
* These are synthetic channel-noise results (random dot/dash flips).
* Real LoRa channel errors include burst errors, insertions, deletions,
  and RSSI-dependent patterns that are NOT captured here.
* Do NOT use these results as evidence of real-world radio performance.
* The model was shown to have a generalization problem in fresh-message
  tests (see README.md for details).

Important note on evaluation honesty
--------------------------------------
The V2 model's original test results came from a held-out split of the
same synthetic dataset used for training.  This script uses INDEPENDENTLY
GENERATED messages to give a more realistic picture of generalization.
Results here may differ substantially from the original reported figures.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# ── Ensure project root is on path ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── ITU Morse dictionary (A-Z, 0-9) ───────────────────────────────────────────
_MORSE_DICT = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..",  "9": "----.",
}
_CHARSET = list(_MORSE_DICT.keys())
_MAX_SEQ_LEN = 120  # Updated from corrector at runtime if model loads


def _random_message(rng: random.Random, min_len: int = 3, max_len: int = 8) -> str:
    """Generate a random word of 3–8 characters from the Morse charset."""
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(_CHARSET) for _ in range(length))


def _text_to_morse(text: str) -> str:
    """Encode text to Morse. Words separated by ' / ', letters by ' '."""
    words = text.split()
    morse_words = []
    for word in words:
        codes = [_MORSE_DICT[ch] for ch in word if ch in _MORSE_DICT]
        if codes:
            morse_words.append(" ".join(codes))
    return " / ".join(morse_words)


def _inject_noise(morse: str, noise_rate: float, rng: random.Random) -> str:
    """Flip dots and dashes in *morse* with probability *noise_rate*."""
    out = []
    for ch in morse:
        if ch == "." and rng.random() < noise_rate:
            out.append("-")
        elif ch == "-" and rng.random() < noise_rate:
            out.append(".")
        else:
            out.append(ch)
    return "".join(out)


def _symbol_error_rate(original: str, received: str) -> float:
    """
    Fraction of dot/dash positions that differ.
    Spaces and slashes are excluded (matching training evaluation).
    Returns 0.0 when there are no symbol positions.
    """
    orig_syms = [ch for ch in original if ch in (".", "-")]
    recv_syms = [ch for ch in received if ch in (".", "-")]
    n = min(len(orig_syms), len(recv_syms))
    if n == 0:
        return 0.0
    errors = sum(1 for a, b in zip(orig_syms[:n], recv_syms[:n]) if a != b)
    return errors / n


def _format_table(rows: list[tuple], headers: list[str]) -> str:
    """Format a simple ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    lines = [sep, header_row, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def _run_evaluation(
    noise_levels: list[int],
    n_messages: int,
    seed: int,
    enable_correction: bool,
) -> None:
    global _MAX_SEQ_LEN

    print()
    print("=" * 70)
    print("  LoRaShield V2 Morse Correction – Independent Evaluation")
    print("=" * 70)
    print(f"  Test messages per noise level : {n_messages}")
    print(f"  Noise levels (%)              : {noise_levels}")
    print(f"  Random seed                   : {seed}")
    print(f"  ML correction                 : {'ENABLED' if enable_correction else 'DISABLED (baseline)'}")
    print("=" * 70)
    print()
    print("  !! IMPORTANT: Results are from SYNTHETIC Morse data only.")
    print("     These figures do NOT represent real LoRa channel performance.")
    print("     Real-world radio hardware validation has NOT been performed.")
    print()

    # ── Load corrector ────────────────────────────────────────────────────────
    try:
        from correction.morse_corrector import MorseCorrector
        corrector = MorseCorrector(enabled=enable_correction)
        if enable_correction and not corrector.is_loaded:
            print("  ERROR: ML correction requested but model failed to load.")
            print(f"         {corrector._load_error}")
            print("         Running evaluation with correction DISABLED.")
            enable_correction = False
        elif enable_correction:
            _MAX_SEQ_LEN = corrector._seq_len
            print(f"  Model loaded. seq_len={_MAX_SEQ_LEN}")
        else:
            print("  Running in pass-through (no-correction baseline) mode.")
    except Exception as exc:
        print(f"  ERROR loading MorseCorrector: {exc}")
        print("  Running evaluation with correction DISABLED.")
        corrector = None
        enable_correction = False

    print()

    # ── Generate test messages ────────────────────────────────────────────────
    rng = random.Random(seed)
    messages = []
    while len(messages) < n_messages:
        word = _random_message(rng)
        morse = _text_to_morse(word)
        # Only include messages that fit in the model's input window
        if len(morse) <= _MAX_SEQ_LEN:
            messages.append((word, morse))

    print(f"  Generated {len(messages)} test messages (all fit within {_MAX_SEQ_LEN}-symbol limit).")
    print()

    # ── Evaluate at each noise level ──────────────────────────────────────────
    table_rows = []
    for noise_pct in noise_levels:
        noise_rate = noise_pct / 100.0
        noise_rng = random.Random(seed + noise_pct)

        ser_before_list = []
        ser_after_list = []
        worse_count = 0
        skipped = 0
        t_start = time.perf_counter()

        for word, clean_morse in messages:
            noisy = _inject_noise(clean_morse, noise_rate, noise_rng)
            ser_before = _symbol_error_rate(clean_morse, noisy)

            if corrector is not None and enable_correction:
                try:
                    result = corrector.correct(noisy)
                    corrected = result.corrected_morse
                    if result.error:
                        corrected = noisy  # fall back on error
                except ValueError:
                    # Input too long (shouldn't happen; we filter above)
                    corrected = noisy
                    skipped += 1
            else:
                corrected = noisy

            ser_after = _symbol_error_rate(clean_morse, corrected)

            ser_before_list.append(ser_before)
            ser_after_list.append(ser_after)

            if ser_after > ser_before + 1e-9:
                worse_count += 1

        elapsed = time.perf_counter() - t_start
        n = len(ser_before_list)

        mean_before = sum(ser_before_list) / n if n > 0 else 0.0
        mean_after  = sum(ser_after_list)  / n if n > 0 else 0.0
        improvement = mean_before - mean_after
        worse_pct   = (worse_count / n * 100) if n > 0 else 0.0

        table_rows.append((
            f"{noise_pct}%",
            f"{mean_before:.4%}",
            f"{mean_after:.4%}",
            f"{improvement:+.4%}",
            f"{worse_count} ({worse_pct:.1f}%)",
            f"{elapsed:.1f}s",
        ))

    headers = [
        "Noise",
        "SER Before",
        "SER After",
        "Improvement",
        "Worse cases",
        "Elapsed",
    ]
    print(_format_table(table_rows, headers))
    print()
    print("  Notes:")
    print("  - SER = Symbol Error Rate (dot/dash positions only; spaces excluded)")
    print("  - 'Worse cases' = messages where correction raised SER above baseline")
    print("  - Improvement < 0 means correction made overall SER worse")
    print()
    print("  Reported synthetic evaluation figures from V2 vocab metadata:")
    print("   0% noise -> 0.010% SER  |  5% -> 0.165%  | 10% -> 0.332%")
    print("  15% noise -> 0.532% SER  | 20% -> 0.744%")
    print()
    print("  These figures were from the original split, NOT independent messages.")
    print("  Differences between the above and this run indicate generalization issues.")
    print()
    print("=" * 70)
    print("  Evaluation complete.  No hardware validation has been performed.")
    print("=" * 70)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V2 Morse correction model on independent synthetic data."
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=int,
        default=[0, 5, 10, 15, 20],
        metavar="PCT",
        help="Noise levels in percent (default: 0 5 10 15 20)",
    )
    parser.add_argument(
        "--messages",
        type=int,
        default=200,
        metavar="N",
        help="Number of test messages per noise level (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=999,
        metavar="SEED",
        help="Random seed for reproducibility (default: 999)",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        default=False,
        help="Disable correction (run baseline-only evaluation)",
    )

    args = parser.parse_args()
    _run_evaluation(
        noise_levels=args.noise_levels,
        n_messages=args.messages,
        seed=args.seed,
        enable_correction=not args.no_correction,
    )


if __name__ == "__main__":
    main()
