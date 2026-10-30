"""
LoRaShield – V2 Morse Symbol Error-Correction Module
=====================================================
Wraps the trained ``lorashield_morse_correction_v2.keras`` model to
provide denoising of Morse symbol sequences received over LoRa.

Architecture
------------
The V2 model is a **Morse-level seq2seq denoiser**, NOT a text decoder.

* Input:  integer-encoded Morse token sequence of length 120.
          Token IDs: {<PAD>:0, <UNK>:1, '.':2, '-':3, ' ':4}.
* Output: per-position softmax over the same 5-token vocabulary.
* Effect: dots/dashes flipped by channel noise are corrected.

The model does NOT operate on AES ciphertext, plain text, or any
representation that is not a space-separated Morse symbol string.

Integration point in the pipeline
----------------------------------
::

    [ESP32 serial RX]
       └─ parse_packet()       ← CRC integrity check (keep unchanged)
           └─ decrypt_message()  ← AES decryption (keep unchanged)
               └─ [optional] MorseCorrector.correct()
                              ← only if the payload is Morse, not
                                 encrypted ciphertext (contains ':')

By default correction is DISABLED (see correction/config.py).
Enable via the environment variable LORASHIELD_ML_CORRECTION=1
or set correction.config.ML_CORRECTION_ENABLED = True at runtime.

Usage
-----
::

    from correction.morse_corrector import MorseCorrector

    corrector = MorseCorrector()          # loads model once
    result = corrector.correct(".- .--.") # returns CorrectionResult
    print(result.corrected_morse)
    print(result.error)                   # None on success

Error handling
--------------
* Input length > MAX_SEQUENCE_LENGTH (120 symbols): raises ValueError.
  Do not silently truncate — this hides data loss.
* Empty input: returns the input unchanged with a note in the log.
* Model unavailable: returns the input unchanged; logs a warning.
* Inference error: returns the input unchanged; logs the exception.

The correction model makes probabilistic predictions.  It can
introduce errors even on clean input.  Never assert that the
corrected output is definitively correct.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Project logger ─────────────────────────────────────────────────────────────
try:
    from utils.logger import get_logger as _get_logger
    _log = _get_logger(__name__)
except Exception:
    import logging
    _log = logging.getLogger(__name__)

# ── Local config ───────────────────────────────────────────────────────────────
from correction.config import (
    ML_CORRECTION_ENABLED,
    MODEL_FILENAME,
    VOCAB_FILENAME,
    MAX_SEQUENCE_LENGTH,
    EXPECTED_VOCAB,
    NUM_CLASSES,
)

# ── Locate models directory relative to this file's grandparent (project root) ─
_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Public result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrectionResult:
    """
    Result returned by :py:meth:`MorseCorrector.correct`.

    Attributes
    ----------
    original_morse:   The input passed to correct() before any changes.
    corrected_morse:  The output after model inference (equals original
                      when correction is disabled, skipped, or failed).
    input_length:     Number of Morse *characters* (not tokens) in the input.
    elapsed_ms:       Wall-clock time for inference, in milliseconds.
    correction_enabled: Whether the ML model was active for this call.
    was_corrected:    True when the corrected string differs from the input.
    error:            Error/warning string, or None on full success.
    """
    original_morse: str
    corrected_morse: str
    input_length: int
    elapsed_ms: float
    correction_enabled: bool
    was_corrected: bool
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class MorseCorrector:
    """
    Single-instance wrapper around the V2 Keras denoising model.

    Load the model once at startup::

        corrector = MorseCorrector()

    Then call :py:meth:`correct` for each received Morse string.

    Parameters
    ----------
    model_path:    Override the default model file path.
    vocab_path:    Override the default vocab JSON path.
    enabled:       Override correction.config.ML_CORRECTION_ENABLED.
                   Pass ``None`` (default) to use the config value.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        vocab_path: Optional[Path | str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._model_path: Path = (
            Path(model_path) if model_path else _MODELS_DIR / MODEL_FILENAME
        )
        self._vocab_path: Path = (
            Path(vocab_path) if vocab_path else _MODELS_DIR / VOCAB_FILENAME
        )
        self._enabled: bool = (
            enabled if enabled is not None else ML_CORRECTION_ENABLED
        )

        # Internals – populated by _load()
        self._model = None
        self._vocab: dict = {}
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._seq_len: int = MAX_SEQUENCE_LENGTH
        self._loaded: bool = False
        self._load_error: Optional[str] = None

        # Attempt to load on construction
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the Keras model and vocab. Sets self._loaded on success."""
        _log.info(
            "[MorseCorrector] Initialising. enabled=%s  model=%s",
            self._enabled, self._model_path.name,
        )

        if not self._model_path.exists():
            msg = f"Model file not found: {self._model_path}"
            _log.error("[MorseCorrector] %s", msg)
            self._load_error = msg
            return

        if not self._vocab_path.exists():
            msg = f"Vocab file not found: {self._vocab_path}"
            _log.error("[MorseCorrector] %s", msg)
            self._load_error = msg
            return

        # ── Load vocab ────────────────────────────────────────────────────────
        try:
            with open(self._vocab_path, "r", encoding="utf-8") as fh:
                self._vocab = json.load(fh)
        except Exception as exc:
            msg = f"Failed to load vocab JSON: {exc}"
            _log.error("[MorseCorrector] %s", msg)
            self._load_error = msg
            return

        # ── Validate vocab ────────────────────────────────────────────────────
        morse_chars: list = self._vocab.get("morse_chars", [])
        if morse_chars != EXPECTED_VOCAB:
            msg = (
                f"Vocab mismatch. Expected {EXPECTED_VOCAB}, "
                f"got {morse_chars}. Model may not match training."
            )
            _log.error("[MorseCorrector] %s", msg)
            self._load_error = msg
            return

        # Build encoding maps
        self._char_to_id = {ch: idx for idx, ch in enumerate(morse_chars)}
        self._id_to_char = {idx: ch for idx, ch in enumerate(morse_chars)}

        # Override sequence length from vocab if present
        self._seq_len = int(self._vocab.get("sequence_length", MAX_SEQUENCE_LENGTH))
        _log.info(
            "[MorseCorrector] Vocab OK. seq_len=%d  vocab=%s",
            self._seq_len, morse_chars,
        )

        # ── Load Keras model ──────────────────────────────────────────────────
        try:
            import numpy as np  # noqa: F401 – confirm numpy available
        except ImportError:
            msg = "numpy not installed; cannot use MorseCorrector."
            _log.error("[MorseCorrector] %s", msg)
            self._load_error = msg
            return

        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import tensorflow as tf  # noqa: F401
                self._model = tf.keras.models.load_model(
                    str(self._model_path), compile=False
                )
            # Validate shape
            in_shape = self._model.input_shape    # e.g. (None, 120)
            out_shape = self._model.output_shape  # e.g. (None, 120, 5)

            _log.info(
                "[MorseCorrector] Model loaded. input_shape=%s  output_shape=%s",
                in_shape, out_shape,
            )

            # Check sequence length compatibility
            if len(in_shape) >= 2 and in_shape[1] is not None:
                if in_shape[1] != self._seq_len:
                    _log.warning(
                        "[MorseCorrector] Model input dim (%d) differs from "
                        "vocab sequence_length (%d). Using model dim.",
                        in_shape[1], self._seq_len,
                    )
                    self._seq_len = in_shape[1]

            # Check output classes
            if len(out_shape) >= 3 and out_shape[2] is not None:
                if out_shape[2] != NUM_CLASSES:
                    msg = (
                        f"Model output classes ({out_shape[2]}) != "
                        f"expected {NUM_CLASSES}. Vocab mismatch."
                    )
                    _log.error("[MorseCorrector] %s", msg)
                    self._load_error = msg
                    self._model = None
                    return

            self._loaded = True
            _log.info(
                "[MorseCorrector] Ready. correction_enabled=%s", self._enabled
            )

        except ImportError:
            msg = "TensorFlow not installed; cannot use MorseCorrector."
            _log.error("[MorseCorrector] %s", msg)
            self._load_error = msg
        except Exception as exc:
            tb = traceback.format_exc()
            msg = f"Keras model load failed: {type(exc).__name__}: {exc}"
            _log.error("[MorseCorrector] %s\n%s", msg, tb)
            self._load_error = msg

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True when the model and vocab loaded successfully."""
        return self._loaded

    @property
    def is_enabled(self) -> bool:
        """True when ML correction is both loaded and enabled."""
        return self._loaded and self._enabled

    def enable(self) -> None:
        """Enable ML correction at runtime."""
        self._enabled = True
        _log.info("[MorseCorrector] Correction ENABLED.")

    def disable(self) -> None:
        """Disable ML correction at runtime (pass-through mode)."""
        self._enabled = False
        _log.info("[MorseCorrector] Correction DISABLED (pass-through).")

    def status(self) -> dict:
        """Return a diagnostic status dict (safe to serialize and log)."""
        return {
            "loaded": self._loaded,
            "enabled": self._enabled,
            "model_file": self._model_path.name,
            "vocab_file": self._vocab_path.name,
            "sequence_length": self._seq_len,
            "vocab": list(self._char_to_id.keys()),
            "load_error": self._load_error,
        }

    def correct(self, morse_input: str) -> CorrectionResult:
        """
        Apply V2 denoising to a Morse symbol string.

        Parameters
        ----------
        morse_input : str
            Morse string composed of ``.``, ``-``, and ``' '`` (space) as
            the symbol separator.  Word boundary markers (``' / '``) are
            treated as spaces in this model because the V2 vocab does not
            have a slash token — callers should convert ``' / '`` to ``' '``
            before passing in, or use the vocab's space encoding directly.

        Returns
        -------
        CorrectionResult
            Always returns a result; never raises (errors populate
            ``result.error`` and ``corrected_morse`` equals ``original_morse``).

        Raises
        ------
        ValueError
            If ``morse_input`` contains more characters than
            ``self._seq_len`` (120).  Callers must not silently truncate.
        """
        t0 = time.perf_counter()
        original = morse_input  # preserve for diagnostics

        # ── Empty input ───────────────────────────────────────────────────────
        if not morse_input or not morse_input.strip():
            _log.debug("[MorseCorrector] Empty input – returning unchanged.")
            return CorrectionResult(
                original_morse=original,
                corrected_morse=original,
                input_length=0,
                elapsed_ms=0.0,
                correction_enabled=self._enabled,
                was_corrected=False,
                error=None,
            )

        morse_input = morse_input.strip()
        input_len = len(morse_input)

        # ── Correction disabled or model unavailable ──────────────────────────
        # When disabled, act as a pure pass-through regardless of input length.
        # The length guard only applies when the model will actually be invoked.
        if not self._enabled:
            _log.debug("[MorseCorrector] Correction disabled – pass-through.")
            return CorrectionResult(
                original_morse=original,
                corrected_morse=original,
                input_length=input_len,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                correction_enabled=False,
                was_corrected=False,
                error=None,
            )

        # ── Length guard (no silent truncation) ───────────────────────────────
        # Only enforce the limit when correction is active and will actually call
        # the model.  Do NOT silently truncate — that hides data loss.
        if input_len > self._seq_len:
            msg = (
                f"Input length {input_len} exceeds model limit {self._seq_len}. "
                f"Split into chunks of ≤{self._seq_len} characters before calling "
                f"correct(), or disable ML correction for this message."
            )
            _log.error("[MorseCorrector] %s", msg)
            raise ValueError(msg)

        if not self._loaded:
            msg = f"Model not loaded: {self._load_error}"
            _log.warning("[MorseCorrector] %s – returning input unchanged.", msg)
            return CorrectionResult(
                original_morse=original,
                corrected_morse=original,
                input_length=input_len,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                correction_enabled=True,
                was_corrected=False,
                error=msg,
            )

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            corrected = self._run_inference(morse_input)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            was_corrected = corrected != morse_input

            _log.info(
                "[MorseCorrector] input_len=%d  elapsed_ms=%.1f  changed=%s",
                input_len, elapsed_ms, was_corrected,
            )

            return CorrectionResult(
                original_morse=original,
                corrected_morse=corrected,
                input_length=input_len,
                elapsed_ms=elapsed_ms,
                correction_enabled=True,
                was_corrected=was_corrected,
                error=None,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            tb = traceback.format_exc()
            # Log without exposing the full morse content (may contain
            # decrypted message fragments)
            _log.error(
                "[MorseCorrector] Inference error after %.1f ms: %s\n%s",
                elapsed_ms, exc, tb,
            )
            return CorrectionResult(
                original_morse=original,
                corrected_morse=original,
                input_length=input_len,
                elapsed_ms=elapsed_ms,
                correction_enabled=True,
                was_corrected=False,
                error=f"Inference error: {type(exc).__name__}: {exc}",
            )

    # ── Internal inference ────────────────────────────────────────────────────

    def _run_inference(self, morse_input: str) -> str:
        """
        Tokenise, pad, predict, and decode the output.

        This is a private method and not part of the public API.

        The model vocabulary only has 5 tokens: <PAD>, <UNK>, '.', '-', ' '.
        Any character not in this set (including '/', letters, digits) is
        mapped to <UNK> (ID 1).

        Padding ID is 0 (<PAD>); padding positions are excluded from the
        output reconstruction.
        """
        import numpy as np

        unk_id = self._char_to_id.get("<UNK>", 1)
        pad_id = self._char_to_id.get("<PAD>", 0)

        # ── Step 1: Tokenise ──────────────────────────────────────────────────
        tokens = [self._char_to_id.get(ch, unk_id) for ch in morse_input]
        n_real = len(tokens)

        _log.debug(
            "[MorseCorrector._run_inference] input_chars=%d  first10_tokens=%s",
            n_real, tokens[:10],
        )

        # ── Step 2: Pad to seq_len ────────────────────────────────────────────
        padded = tokens + [pad_id] * (self._seq_len - n_real)
        x = np.array([padded], dtype=np.int32)  # shape (1, seq_len)

        # ── Step 3: Model inference ───────────────────────────────────────────
        preds = self._model.predict(x, verbose=0)  # (1, seq_len, num_classes)
        pred_ids = preds[0].argmax(axis=-1)        # (seq_len,)

        # ── Step 4: Reconstruct – only real (non-padded) positions ───────────
        reconstructed = []
        for i in range(n_real):
            pred_id = int(pred_ids[i])
            ch = self._id_to_char.get(pred_id, "<UNK>")
            if ch in ("<PAD>", "<UNK>"):
                # Fallback to original character when model outputs a
                # special token for a real position
                ch = morse_input[i]
            reconstructed.append(ch)

        corrected = "".join(reconstructed)

        _log.debug(
            "[MorseCorrector._run_inference] first10_pred_ids=%s",
            pred_ids[:10].tolist(),
        )

        return corrected

    def __repr__(self) -> str:
        return (
            f"MorseCorrector(loaded={self._loaded}, enabled={self._enabled}, "
            f"seq_len={self._seq_len})"
        )
