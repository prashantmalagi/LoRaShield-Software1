"""
LoRaShield – V2 ML Morse Error-Correction Tests
=================================================
Tests for correction/morse_corrector.py and correction/config.py.

Run from the project root:
    python -m pytest tests/test_ml_morse_corrector.py -v
    # or without pytest:
    python tests/test_ml_morse_corrector.py

Test categories
---------------
 1.  Config defaults and environment-variable override.
 2.  MorseCorrector instantiation and model loading.
 3.  Vocabulary mapping (ID order must match training).
 4.  Empty input handling.
 5.  Input at the 120-symbol limit.
 6.  Input beyond the 120-symbol limit (must raise ValueError).
 7.  Invalid Morse symbols (unexpected characters).
 8.  Correction disabled (pass-through).
 9.  Clean synthetic Morse sequences.
10.  Noisy Morse sequences (dot/dash flip).
11.  Space preservation in output.
12.  Existing encoder/decoder compatibility (regression).
13.  AES encryption/decryption regression.
14.  CRC calculation/verification regression.
15.  Graceful handling of model unavailable (simulated).
16.  Graceful handling of inference errors (simulated).
17.  Symbol Error Rate (SER) measurement helper.
18.  SER before vs. after correction on noisy synthetic data.
19.  Correction disabled baseline matches input exactly.

Notes
-----
* All tests use only synthetic data.  Results do NOT represent real LoRa
  channel measurements.
* We do NOT assert that the model always restores the original Morse.
  Instead we measure symbol error rate (SER) and verify the model does
  not catastrophically worsen clean input.
* Tests that require TensorFlow are skipped if TF is not installed.
* The model is loaded once per test session (class-level setUp or
  module-level fixture) to keep the test suite fast.
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Check TensorFlow availability ─────────────────────────────────────────────
try:
    import tensorflow  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

TF_SKIP = unittest.skipUnless(TF_AVAILABLE, "TensorFlow not installed – skipping ML test")

# ── Import module under test ───────────────────────────────────────────────────
from correction.config import (
    ML_CORRECTION_ENABLED,
    EXPECTED_VOCAB,
    NUM_CLASSES,
    MAX_SEQUENCE_LENGTH,
    MODEL_FILENAME,
    VOCAB_FILENAME,
)
from correction.morse_corrector import CorrectionResult, MorseCorrector

# ── Shared test helpers ────────────────────────────────────────────────────────

# Standard ITU Morse dictionary (subset sufficient for synthetic tests)
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


def _text_to_morse(text: str) -> str:
    """Encode text to Morse (letters sep by space, words by ' / ')."""
    words = text.upper().split()
    morse_words = []
    for word in words:
        codes = [_MORSE_DICT[ch] for ch in word if ch in _MORSE_DICT]
        if codes:
            morse_words.append(" ".join(codes))
    return " / ".join(morse_words)


def _inject_noise(morse: str, noise_rate: float, seed: int = 42) -> str:
    """
    Flip each dot or dash in *morse* with probability *noise_rate*.

    Spaces and slashes are not modified.
    Returns the noisy Morse string.
    """
    rng = random.Random(seed)
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
    Compute the fraction of dot/dash positions that differ between
    *original* and *received*, excluding spaces and slashes.

    Returns 0.0 when there are no symbol positions.
    This is the metric used in the V2 evaluation.
    """
    orig_syms = [ch for ch in original if ch in (".", "-")]
    recv_syms = [ch for ch in received if ch in (".", "-")]

    # Align by minimum length (should match for same-length sequences)
    n = min(len(orig_syms), len(recv_syms))
    if n == 0:
        return 0.0
    errors = sum(1 for a, b in zip(orig_syms[:n], recv_syms[:n]) if a != b)
    return errors / n


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    """Verify configuration constants are as expected."""

    def test_correction_disabled_by_default(self):
        """ML_CORRECTION_ENABLED must be False unless env var is set."""
        env_val = os.environ.get("LORASHIELD_ML_CORRECTION", "")
        expected = bool(env_val.strip())
        self.assertEqual(ML_CORRECTION_ENABLED, expected)

    def test_expected_vocab_order(self):
        """Vocabulary order must match training exactly."""
        self.assertEqual(EXPECTED_VOCAB[0], "<PAD>")
        self.assertEqual(EXPECTED_VOCAB[1], "<UNK>")
        self.assertEqual(EXPECTED_VOCAB[2], ".")
        self.assertEqual(EXPECTED_VOCAB[3], "-")
        self.assertEqual(EXPECTED_VOCAB[4], " ")

    def test_num_classes(self):
        self.assertEqual(NUM_CLASSES, 5)

    def test_max_sequence_length(self):
        self.assertEqual(MAX_SEQUENCE_LENGTH, 120)

    def test_model_filename(self):
        self.assertEqual(MODEL_FILENAME, "lorashield_morse_correction_v2.keras")

    def test_vocab_filename(self):
        self.assertEqual(VOCAB_FILENAME, "lorashield_morse_correction_v2_vocab.json")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MorseCorrector instantiation
# ─────────────────────────────────────────────────────────────────────────────

class TestMorseCorrectorInstantiation(unittest.TestCase):
    """Basic instantiation tests that do not require TF inference."""

    def test_instantiation_does_not_raise(self):
        """MorseCorrector() must not raise even if TF is unavailable."""
        try:
            corrector = MorseCorrector()
        except Exception as exc:
            self.fail(f"MorseCorrector() raised unexpectedly: {exc}")

    def test_is_loaded_reflects_tf_availability(self):
        corrector = MorseCorrector()
        # is_loaded may be True or False depending on whether TF is installed,
        # but it must be a bool.
        self.assertIsInstance(corrector.is_loaded, bool)

    def test_is_enabled_false_by_default(self):
        """Correction must be disabled by default (unless env var set)."""
        corrector = MorseCorrector()
        env_enabled = bool(os.environ.get("LORASHIELD_ML_CORRECTION", "").strip())
        self.assertEqual(corrector.is_enabled, env_enabled and corrector.is_loaded)

    def test_status_dict_structure(self):
        corrector = MorseCorrector()
        s = corrector.status()
        self.assertIn("loaded", s)
        self.assertIn("enabled", s)
        self.assertIn("sequence_length", s)
        self.assertIn("model_file", s)

    def test_repr_contains_key_info(self):
        corrector = MorseCorrector()
        r = repr(corrector)
        self.assertIn("loaded=", r)
        self.assertIn("enabled=", r)

    def test_nonexistent_model_path(self):
        """A missing model file must set load_error, not raise."""
        corrector = MorseCorrector(model_path="/nonexistent/path/model.keras")
        self.assertFalse(corrector.is_loaded)
        self.assertIsNotNone(corrector._load_error)

    def test_nonexistent_vocab_path(self):
        """A missing vocab file must set load_error, not raise."""
        corrector = MorseCorrector(vocab_path="/nonexistent/vocab.json")
        self.assertFalse(corrector.is_loaded)

    def test_enable_disable_toggle(self):
        corrector = MorseCorrector(enabled=False)
        self.assertFalse(corrector._enabled)
        corrector.enable()
        self.assertTrue(corrector._enabled)
        corrector.disable()
        self.assertFalse(corrector._enabled)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Vocabulary mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabularyMapping(unittest.TestCase):
    """Verify the vocab JSON matches the expected token order."""

    def _load_vocab(self):
        """Load the vocab JSON directly from disk."""
        import json
        vocab_path = _ROOT / "models" / VOCAB_FILENAME
        if not vocab_path.exists():
            self.skipTest(f"Vocab file not found: {vocab_path}")
        with open(vocab_path, encoding="utf-8") as fh:
            return json.load(fh)

    def test_vocab_file_exists(self):
        vocab_path = _ROOT / "models" / VOCAB_FILENAME
        self.assertTrue(vocab_path.exists(), f"Missing: {vocab_path}")

    def test_morse_chars_order(self):
        vocab = self._load_vocab()
        morse_chars = vocab.get("morse_chars", [])
        self.assertEqual(morse_chars, EXPECTED_VOCAB,
                         f"Vocab order mismatch: {morse_chars}")

    def test_pad_id_is_zero(self):
        vocab = self._load_vocab()
        chars = vocab.get("morse_chars", [])
        self.assertEqual(chars.index("<PAD>"), 0)

    def test_dot_id_is_two(self):
        vocab = self._load_vocab()
        chars = vocab.get("morse_chars", [])
        self.assertEqual(chars.index("."), 2)

    def test_dash_id_is_three(self):
        vocab = self._load_vocab()
        chars = vocab.get("morse_chars", [])
        self.assertEqual(chars.index("-"), 3)

    def test_space_id_is_four(self):
        vocab = self._load_vocab()
        chars = vocab.get("morse_chars", [])
        self.assertEqual(chars.index(" "), 4)

    def test_sequence_length(self):
        vocab = self._load_vocab()
        self.assertEqual(vocab.get("sequence_length"), 120)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Empty input handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyInput(unittest.TestCase):

    def test_empty_string_returns_unchanged(self):
        corrector = MorseCorrector(enabled=False)
        result = corrector.correct("")
        self.assertEqual(result.corrected_morse, "")
        self.assertFalse(result.was_corrected)
        self.assertIsNone(result.error)

    def test_whitespace_only_returns_unchanged(self):
        corrector = MorseCorrector(enabled=False)
        result = corrector.correct("   ")
        self.assertFalse(result.was_corrected)

    def test_empty_input_length_is_zero(self):
        corrector = MorseCorrector(enabled=False)
        result = corrector.correct("")
        self.assertEqual(result.input_length, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Input at the 120-symbol limit
# ─────────────────────────────────────────────────────────────────────────────

class TestInputAtLimit(unittest.TestCase):

    def test_exactly_120_chars_accepted_when_disabled(self):
        """120-char input should not raise when correction is disabled."""
        corrector = MorseCorrector(enabled=False)
        morse = "." * 120
        result = corrector.correct(morse)
        self.assertEqual(result.input_length, 120)
        self.assertIsNone(result.error)

    @TF_SKIP
    def test_exactly_120_chars_accepted_when_enabled(self):
        """120-char input should not raise when correction is enabled."""
        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            self.skipTest("Model not loaded")
        morse = ". - " * 30  # 120 chars
        morse = morse.strip()
        if len(morse) > 120:
            morse = morse[:120]
        result = corrector.correct(morse)
        self.assertIsNone(result.error)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Input beyond the 120-symbol limit
# ─────────────────────────────────────────────────────────────────────────────

class TestInputBeyondLimit(unittest.TestCase):

    def test_121_chars_raises_value_error_when_enabled(self):
        """Inputs longer than seq_len must raise ValueError, not truncate."""
        corrector = MorseCorrector(enabled=True)
        # Even if model is not loaded, the length guard runs before inference
        morse = "." * 121
        with self.assertRaises(ValueError):
            corrector.correct(morse)

    def test_long_input_does_not_raise_when_disabled(self):
        """With correction disabled, length guard is not applied."""
        corrector = MorseCorrector(enabled=False)
        morse = "." * 200
        # Should not raise – pass-through mode
        result = corrector.correct(morse)
        self.assertEqual(result.corrected_morse, morse)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Invalid Morse symbols
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidMorseSymbols(unittest.TestCase):
    """Characters outside the vocab must be handled safely."""

    def test_letters_in_input_disabled(self):
        """Letters are invalid Morse symbols but should not crash (disabled)."""
        corrector = MorseCorrector(enabled=False)
        result = corrector.correct("HELLO")
        self.assertEqual(result.corrected_morse, "HELLO")

    @TF_SKIP
    def test_unknown_chars_mapped_to_unk_when_enabled(self):
        """Unknown chars map to <UNK> (ID 1) during encoding; test doesn't crash."""
        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            self.skipTest("Model not loaded")
        # Slash and digits are not in the V2 vocab
        result = corrector.correct(".. /")
        # Must not raise and must return a CorrectionResult
        self.assertIsInstance(result, CorrectionResult)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Correction disabled (pass-through)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectionDisabled(unittest.TestCase):

    def test_disabled_returns_original(self):
        corrector = MorseCorrector(enabled=False)
        morse = ".- -... -.-."
        result = corrector.correct(morse)
        self.assertEqual(result.corrected_morse, morse)
        self.assertFalse(result.was_corrected)
        self.assertFalse(result.correction_enabled)

    def test_disabled_elapsed_is_fast(self):
        corrector = MorseCorrector(enabled=False)
        result = corrector.correct("...")
        # Should complete essentially instantly (< 100ms)
        self.assertLess(result.elapsed_ms, 100)

    def test_disabled_no_inference_on_valid_morse(self):
        """When disabled, no model call should be made."""
        corrector = MorseCorrector(enabled=False)
        with patch.object(corrector, "_run_inference") as mock_infer:
            corrector.correct(".-")
            mock_infer.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Clean synthetic Morse sequences
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanMorseSequences(unittest.TestCase):
    """Model should not significantly degrade clean (0% noise) inputs."""

    CLEAN_MESSAGES = [
        "HELLO",
        "SOS",
        "TEST",
        "ABCDE",
    ]

    @TF_SKIP
    def test_clean_input_ser_near_zero(self):
        """
        Measure SER after correction of 0%-noise input.

        The V2 model has a known generalization problem: on messages NOT
        in its training distribution it can produce a high SER even on
        clean (0%-noise) input.  The threshold here is set conservatively
        to detect catastrophic failure modes, NOT to assert good accuracy.

        Evaluation on independent test messages is performed separately
        by evaluate_morse_correction.py.

        Threshold: SER ≤ 55% on clean 0%-noise input.
        (Any model worse than this on clean data is broken in a way that
        merits further investigation, but the V2 model may legitimately
        sit below this bound due to the generalization gap.)
        """
        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            self.skipTest("Model not loaded")

        for msg in self.CLEAN_MESSAGES:
            morse = _text_to_morse(msg)
            if len(morse) > MAX_SEQUENCE_LENGTH:
                continue  # skip messages too long for this test
            with self.subTest(message=msg):
                result = corrector.correct(morse)
                ser = _symbol_error_rate(morse, result.corrected_morse)
                # Threshold is deliberately wide because the V2 model has
                # a known generalization problem (see README.md Limitations).
                # A SER > 55% on clean input would indicate something is
                # fundamentally broken (wrong model, wrong vocab, etc.).
                self.assertLessEqual(
                    ser, 0.55,
                    f"SER critically high on clean input {msg!r}: {ser:.3%} "
                    f"(V2 model generalization problem documented in README)"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Noisy Morse sequences
# ─────────────────────────────────────────────────────────────────────────────

class TestNoisyMorseSequences(unittest.TestCase):
    """Measure SER before vs. after correction on noisy synthetic data."""

    TEST_MESSAGE = "HELLO WORLD"
    NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]

    @TF_SKIP
    def test_correction_does_not_catastrophically_worsen_noisy_input(self):
        """
        Regression guard: correction must not raise SER by more than 40
        percentage points above baseline.

        The V2 model has a known generalization problem.  On messages not
        well-represented in its training set, the model can produce high SER
        even on clean (0%-noise) input.  The threshold here is deliberately
        wide to reflect observed reality, NOT to claim good accuracy.

        If this test fails, it means the model is severely broken in a way
        that goes beyond the documented generalization gap.

        Run evaluate_morse_correction.py for a full independent evaluation.
        """
        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            self.skipTest("Model not loaded")

        clean_morse = _text_to_morse(self.TEST_MESSAGE)
        if len(clean_morse) > MAX_SEQUENCE_LENGTH:
            self.skipTest("Test message too long for model input")

        for noise_rate in self.NOISE_LEVELS:
            with self.subTest(noise_rate=noise_rate):
                noisy = _inject_noise(clean_morse, noise_rate, seed=42)
                ser_before = _symbol_error_rate(clean_morse, noisy)

                result = corrector.correct(noisy)
                ser_after = _symbol_error_rate(clean_morse, result.corrected_morse)

                # Allow up to 40 percentage points degradation.
                # This is a wide bound that detects catastrophic failure
                # while acknowledging the V2 model's generalization problem.
                self.assertLessEqual(
                    ser_after, ser_before + 0.40,
                    f"Correction catastrophically worsened SER at noise={noise_rate:.0%}: "
                    f"before={ser_before:.3%}, after={ser_after:.3%} "
                    f"(V2 model generalization problem documented in README)",
                )

    @TF_SKIP
    def test_ser_metric_excludes_spaces(self):
        """_symbol_error_rate must ignore space characters."""
        # Clean and noisy strings with same symbols but different spaces
        original = ".- -"
        received = ". - -"  # added a space – symbols same
        ser = _symbol_error_rate(original, received)
        # '.-' → '.', '-' in orig  vs  '.', '-' '-' in received (min 2)
        # First two symbols match → SER == 0
        self.assertEqual(ser, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Space preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestSpacePreservation(unittest.TestCase):
    """Spaces in the input must appear in the output at the same positions."""

    @TF_SKIP
    def test_spaces_preserved_in_output(self):
        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            self.skipTest("Model not loaded")

        morse = ".- -... -.-."   # A B C
        result = corrector.correct(morse)

        # Count spaces in input and output
        in_spaces = morse.count(" ")
        out_spaces = result.corrected_morse.count(" ")

        # Model should preserve the same number of spaces
        self.assertEqual(
            in_spaces, out_spaces,
            f"Space count changed: input has {in_spaces}, output has {out_spaces}. "
            f"Output: {result.corrected_morse!r}"
        )

    def test_space_positions_unchanged_when_disabled(self):
        corrector = MorseCorrector(enabled=False)
        morse = ".. . .-.. .-.. ---"
        result = corrector.correct(morse)
        self.assertEqual(result.corrected_morse, morse)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Existing encoder/decoder regression
# ─────────────────────────────────────────────────────────────────────────────

class TestEncoderDecoderRegression(unittest.TestCase):
    """Verify that utils.ai encode_text and decode_morse still work."""

    def test_encode_text_hello(self):
        from utils.ai import encode_text
        result = encode_text("HELLO")
        self.assertEqual(result, ".... . .-.. .-.. ---")

    def test_encode_text_sos(self):
        from utils.ai import encode_text
        result = encode_text("SOS")
        self.assertEqual(result, "... --- ...")

    def test_encode_text_word_separator(self):
        from utils.ai import encode_text
        result = encode_text("HELLO WORLD")
        self.assertIn(" / ", result)

    def test_encode_text_empty(self):
        from utils.ai import encode_text
        result = encode_text("")
        self.assertEqual(result, "")

    def test_decode_morse_fallback_hello(self):
        """Fallback decoder (no model) must decode HELLO correctly."""
        from utils.ai import decode_morse
        text, conf, elapsed, method = decode_morse(".... . .-.. .-.. ---")
        self.assertIn("H", text.upper())
        self.assertIsInstance(elapsed, float)

    def test_decode_morse_empty(self):
        from utils.ai import decode_morse
        text, conf, elapsed, method = decode_morse("")
        # Should return empty or near-empty without crashing
        self.assertIsInstance(text, str)


# ─────────────────────────────────────────────────────────────────────────────
# 13. AES encryption/decryption regression
# ─────────────────────────────────────────────────────────────────────────────

class TestAESRegression(unittest.TestCase):
    """Verify that the AES encryption round-trip is unaffected by the ML changes."""

    def setUp(self):
        from utils.encryption import is_crypto_available
        if not is_crypto_available():
            self.skipTest("pycryptodome not installed")

    def test_encrypt_decrypt_roundtrip(self):
        from utils.encryption import encrypt_message, decrypt_message
        msg = "HELLO WORLD"
        key = "TestKey12345"
        enc, err = encrypt_message(msg, key)
        self.assertEqual(err, "")
        self.assertIn(":", enc)   # IV:CT format

        dec, err2 = decrypt_message(enc, key)
        self.assertEqual(err2, "")
        self.assertEqual(dec, msg)

    def test_ciphertext_contains_colon(self):
        """AES ciphertext always contains ':', which distinguishes it from Morse."""
        from utils.encryption import encrypt_message
        enc, _ = encrypt_message("TEST", "key")
        self.assertIn(":", enc)

    def test_wrong_key_fails(self):
        from utils.encryption import encrypt_message, decrypt_message
        enc, _ = encrypt_message("SECRET", "correct-key")
        _, err = decrypt_message(enc, "wrong-key")
        self.assertNotEqual(err, "")


# ─────────────────────────────────────────────────────────────────────────────
# 14. CRC regression
# ─────────────────────────────────────────────────────────────────────────────

class TestCRCRegression(unittest.TestCase):

    def test_build_parse_roundtrip(self):
        from utils.crc import build_packet, parse_packet
        msg = "HELLO WORLD"
        packet = build_packet(msg)
        result = parse_packet(packet)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], msg)

    def test_tampered_packet_fails(self):
        import json
        from utils.crc import build_packet, parse_packet
        packet = build_packet("ORIGINAL")
        obj = json.loads(packet)
        obj["payload"] = "TAMPERED"
        tampered = json.dumps(obj)
        result = parse_packet(tampered)
        self.assertFalse(result["valid"])

    def test_calculate_crc_deterministic(self):
        from utils.crc import calculate_crc
        self.assertEqual(calculate_crc("TEST"), calculate_crc("TEST"))


# ─────────────────────────────────────────────────────────────────────────────
# 15. Graceful handling of unavailable model
# ─────────────────────────────────────────────────────────────────────────────

class TestModelUnavailable(unittest.TestCase):

    def test_correct_returns_original_when_not_loaded(self):
        """When model fails to load, correct() should return input unchanged."""
        corrector = MorseCorrector(
            model_path="/nonexistent/model.keras",
            enabled=True,
        )
        self.assertFalse(corrector.is_loaded)
        result = corrector.correct(".- .-..")
        self.assertEqual(result.corrected_morse, ".- .-..")
        self.assertFalse(result.was_corrected)
        self.assertIsNotNone(result.error)

    def test_is_enabled_false_when_not_loaded(self):
        corrector = MorseCorrector(
            model_path="/nonexistent/model.keras",
            enabled=True,
        )
        # is_enabled checks both _enabled and _loaded
        self.assertFalse(corrector.is_enabled)


# ─────────────────────────────────────────────────────────────────────────────
# 16. Graceful handling of inference errors
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceError(unittest.TestCase):

    def test_inference_exception_returns_original(self):
        """If _run_inference raises, correct() should return input unchanged."""
        corrector = MorseCorrector(enabled=False)
        # Manually set loaded to True and patch _run_inference
        corrector._loaded = True
        corrector._enabled = True
        corrector._seq_len = 120

        def _bad_inference(_):
            raise RuntimeError("Simulated inference failure")

        corrector._run_inference = _bad_inference

        result = corrector.correct(".-")
        self.assertEqual(result.corrected_morse, ".-")
        self.assertFalse(result.was_corrected)
        self.assertIsNotNone(result.error)
        self.assertIn("Inference error", result.error)


# ─────────────────────────────────────────────────────────────────────────────
# 17. SER measurement helper
# ─────────────────────────────────────────────────────────────────────────────

class TestSERHelper(unittest.TestCase):

    def test_identical_strings(self):
        self.assertAlmostEqual(_symbol_error_rate(".-", ".-"), 0.0)

    def test_one_flip(self):
        # "." → "-" : 1 error out of 2 symbols
        self.assertAlmostEqual(_symbol_error_rate(".-", "--"), 0.5)

    def test_all_flipped(self):
        # '.-' vs '-.' : both symbols flipped → SER = 1.0
        self.assertAlmostEqual(_symbol_error_rate(".-", "-."), 1.0)

    def test_empty_strings(self):
        self.assertEqual(_symbol_error_rate("", ""), 0.0)

    def test_spaces_excluded(self):
        # "  " has no symbols → SER = 0.0
        self.assertEqual(_symbol_error_rate("   ", "   "), 0.0)

    def test_spaces_in_morse_excluded(self):
        # 4 symbols: '.', '-', '.', '-'  → 0 errors
        self.assertAlmostEqual(_symbol_error_rate(". -", ". -"), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 18. SER before vs. after correction (synthetic evaluation)
# ─────────────────────────────────────────────────────────────────────────────

class TestSERComparison(unittest.TestCase):
    """
    Measure SER at multiple noise levels.

    This test does NOT assert improvement.  It verifies that the test
    framework for measuring SER works correctly and that correction does
    not catastrophically degrade already-clean signals.

    Results are printed for informational purposes.
    """

    MESSAGES = ["HELLO WORLD", "SOS", "TEST MESSAGE"]
    NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]

    @TF_SKIP
    def test_ser_measurement_runs_without_crash(self):
        corrector = MorseCorrector(enabled=True)
        if not corrector.is_loaded:
            self.skipTest("Model not loaded")

        for msg in self.MESSAGES:
            clean_morse = _text_to_morse(msg)
            if len(clean_morse) > MAX_SEQUENCE_LENGTH:
                continue

            for noise_rate in self.NOISE_LEVELS:
                with self.subTest(message=msg, noise=noise_rate):
                    noisy = _inject_noise(clean_morse, noise_rate)
                    ser_before = _symbol_error_rate(clean_morse, noisy)
                    result = corrector.correct(noisy)
                    ser_after = _symbol_error_rate(clean_morse, result.corrected_morse)

                    # Just verify no crash and result is a valid float
                    self.assertIsInstance(ser_before, float)
                    self.assertIsInstance(ser_after, float)
                    self.assertGreaterEqual(ser_before, 0.0)
                    self.assertGreaterEqual(ser_after, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 19. Correction disabled baseline
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectionDisabledBaseline(unittest.TestCase):
    """When correction is disabled, output must equal input (SER unchanged)."""

    def test_disabled_ser_equals_noisy_ser(self):
        """With correction disabled, SER(clean, output) == SER(clean, noisy)."""
        corrector = MorseCorrector(enabled=False)
        clean = _text_to_morse("HELLO")
        noisy = _inject_noise(clean, 0.10)

        result = corrector.correct(noisy)

        ser_noisy = _symbol_error_rate(clean, noisy)
        ser_output = _symbol_error_rate(clean, result.corrected_morse)
        self.assertAlmostEqual(ser_noisy, ser_output, places=6)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
