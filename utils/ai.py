"""
LoRaShield - AI Utility Module (v4)
=====================================
TensorFlow/Keras integration for Morse-to-Text AI decoding.

Model Architecture (morse_decoder_v4.keras):
    - Vocab     : vocab_v4.json
                  CHAR_TO_IDX : {'<PAD>': 0, '<UNK>': 1, '.': 2, '-': 3, ' ': 4, '/': 5}
                  TARGET_CHARS: ['<PAD>', 'A'-'Z', '0'-'9', ' ']  (38 classes)
    - Model     : Direct Morse-to-text decoder
                  input_shape  = (None, MAX_MORSE_LEN)   -- Morse char tokens
                  output_shape = (None, MAX_TEXT_LEN, 38) -- softmax over target chars

Inference Strategy (v4):
    1. Tokenise the full Morse string character-by-character using CHAR_TO_IDX.
    2. Pad/truncate to MAX_MORSE_LEN (120).
    3. model.predict() -> decoded text token sequence (MAX_TEXT_LEN, 38).
    4. argmax -> target character indices.
    5. Map indices via TARGET_CHARS, strip padding.
    6. Confidence = mean(max_softmax) over non-padding positions.

No pickle tokenizer required — vocab is loaded from vocab_v4.json.
All events logged via utils.logger AND printed to stdout.
"""

import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_PATH     = BASE_DIR / "models" / "morse_decoder_v4 (1).keras"
VOCAB_PATH     = BASE_DIR / "models" / "vocab_v4.json"

# Sequence lengths (must match training — loaded from vocab at runtime)
MAX_MORSE_LEN = 120   # default; overridden by vocab_v4.json
MAX_TEXT_LEN  = 25    # default; overridden by vocab_v4.json

LOW_CONFIDENCE_THRESHOLD = 0.60

_model        = None
_vocab        = None   # dict loaded from vocab_v4.json
_model_loaded = False

MORSE_CODE_DICT = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..",  "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.",
    "!": "-.-.--", "/": "-..-.",  "(": "-.--.", ")": "-.--.-",
    "&": ".-...",  ":": "---...", ";": "-.-.-.",  "=": "-...-",
    "+": ".-.-.",  "-": "-....-", "_": "..--..",  '"': ".-..-.",
    "$": "...-..-", "@": ".--.-.",
}

MORSE_REVERSE_DICT = {v: k for k, v in MORSE_CODE_DICT.items()}
SUPPORTED_CHARS    = set(MORSE_CODE_DICT.keys())


def _diag_print(msg: str) -> None:
    """Print msg to stdout AND log at INFO level."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe, flush=True)
    log.info(msg)


# ---------------------------------------------------------------------------
# Levenshtein fuzzy Morse correction
# ---------------------------------------------------------------------------
# ROOT CAUSE NOTE (2026-08-07 diagnostic):
#   The model performs PURE IDENTITY MAPPING on 100% of test inputs.
#   Every output token equals the input token -- no correction occurs.
#   This is a training failure where the model learned the trivial copy
#   solution.  See module docstring for recommended training fixes.
#
# INFERENCE MITIGATION:
#   1. Identity mapping is detected and logged as a warning per call.
#   2. Unrecognized Morse codes (e.g. '.-.-') are fuzzy-corrected via
#      Levenshtein nearest-neighbour (edit distance <= 1).
#   3. VALID but wrong codes (e.g. '--.' when '---' expected) cannot
#      be fixed in inference; they require model retraining.

def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _fuzzy_decode_symbol(code: str) -> tuple:
    """
    Decode one Morse symbol to a character.

    Strategy:
      1. Exact ITU dictionary lookup (always preferred).
      2. Levenshtein nearest-neighbour over all valid codes
         (accepted if edit distance <= 1).
      3. Returns '?' if no match within threshold.

    Returns:
        (character, method_tag, edit_distance)
        method_tag: 'exact' | 'fuzzy' | 'unknown'
    """
    MAX_EDIT_DIST = 1
    code = code.strip()
    if not code:
        return ("", "empty", 0)

    if code in MORSE_REVERSE_DICT:
        return (MORSE_REVERSE_DICT[code], "exact", 0)

    best_char = None
    best_dist = MAX_EDIT_DIST + 1
    best_code = None
    for valid_morse, char in MORSE_REVERSE_DICT.items():
        d = _levenshtein(code, valid_morse)
        if d < best_dist or (d == best_dist and len(valid_morse) == len(code)):
            best_dist = d
            best_char = char
            best_code = valid_morse

    if best_dist <= MAX_EDIT_DIST:
        return (best_char, "fuzzy", best_dist)
    return ("?", "unknown", best_dist)


def load_ai_model(model_path=MODEL_PATH, tokenizer_path=None):
    """
    Load the TensorFlow v4 Keras model and vocab_v4.json vocabulary.

    tokenizer_path is accepted but ignored (kept for API compatibility).
    The vocab is loaded from VOCAB_PATH (vocab_v4.json).

    Returns:
        (True, "AI Loaded") on success.
        (False, full_exception_string) on failure.
    """
    global _model, _vocab, _model_loaded, MAX_MORSE_LEN, MAX_TEXT_LEN

    if _model_loaded:
        _diag_print("AI model already loaded - skipping reload.")
        return (True, "AI Loaded")

    model_path = Path(model_path)
    vocab_path = VOCAB_PATH

    sep = "-" * 50
    _diag_print(sep)
    _diag_print("Loading AI Model (v4 - Direct Morse-to-Text)...")
    _diag_print(f"Model path:       {model_path}")
    _diag_print(f"Vocab path:       {vocab_path}")
    _diag_print(f"Current Working Directory: {os.getcwd()}")

    model_exists = model_path.exists()
    vocab_exists = vocab_path.exists()
    _diag_print(f"Model exists:     {model_exists}")
    _diag_print(f"Vocab exists:     {vocab_exists}")
    if model_exists:
        _diag_print(f"Model file size:  {model_path.stat().st_size:,} bytes")

    if not model_exists:
        msg = f"Model file not found: {model_path}"
        _diag_print(f"ERROR: {msg}")
        _diag_print(sep)
        return (False, msg)
    if not vocab_exists:
        msg = f"Vocab file not found: {vocab_path}"
        _diag_print(f"ERROR: {msg}")
        _diag_print(sep)
        return (False, msg)

    _diag_print("")
    _diag_print("Importing TensorFlow...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import tensorflow as tf
        _diag_print(f"TensorFlow Version: {tf.__version__}")
        try:
            _diag_print(f"Keras Version:      {tf.keras.__version__}")
        except Exception:
            pass
    except ImportError:
        tb = traceback.format_exc()
        msg = f"TensorFlow not installed.\n\nTraceback:\n{tb}"
        _diag_print("ERROR: TensorFlow not installed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"TensorFlow import error: {exc}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: TensorFlow import failed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    _diag_print("")
    _diag_print("Loading vocab (vocab_v4.json)...")
    try:
        import json
        with open(vocab_path, "r", encoding="utf-8") as fh:
            _vocab = json.load(fh)
        # Override sequence length constants from vocab file
        MAX_MORSE_LEN = _vocab.get("MAX_MORSE_LEN", MAX_MORSE_LEN)
        MAX_TEXT_LEN  = _vocab.get("MAX_TEXT_LEN",  MAX_TEXT_LEN)
        _diag_print(f"CHAR_TO_IDX:       {_vocab['CHAR_TO_IDX']}")
        _diag_print(f"Target vocab size: {len(_vocab['TARGET_CHARS'])}")
        _diag_print(f"MAX_MORSE_LEN:     {MAX_MORSE_LEN}")
        _diag_print(f"MAX_TEXT_LEN:      {MAX_TEXT_LEN}")
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"Failed to load vocab: {exc}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: Vocab load failed:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    _diag_print("")
    _diag_print("Loading Keras model (.keras)...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _model = tf.keras.models.load_model(str(model_path), compile=False)
        _diag_print(f"Model Input Shape:  {_model.input_shape}")
        _diag_print(f"Model Output Shape: {_model.output_shape}")
        _diag_print("Model Type:         Direct Morse-to-Text Decoder (v4)")
    except Exception as exc:
        tb = traceback.format_exc()
        full_err = f"Failed to load model: {type(exc).__name__}: {exc}"
        msg = f"{full_err}\n\nTraceback:\n{tb}"
        _diag_print("ERROR: Keras model load failed - full traceback:")
        _diag_print(tb)
        _diag_print(sep)
        return (False, msg)

    _model_loaded = True
    _diag_print("")
    _diag_print("SUCCESS: AI model (v4) fully loaded and ready.")
    _diag_print("  Architecture : Direct Morse-to-Text Decoder")
    _diag_print(f"  Input length : {MAX_MORSE_LEN} (Morse char tokens)")
    _diag_print(f"  Output length: {MAX_TEXT_LEN} (text chars)")
    _diag_print(f"  Output vocab : {len(_vocab['TARGET_CHARS'])} classes")
    _diag_print(sep)
    return (True, "AI Loaded")


def decode_morse(morse_input: str) -> tuple:
    """
    Decode Morse code to plain text.

    When AI model is loaded  -> Seq2Seq denoising + dictionary decode.
    When not yet loaded      -> dictionary fallback (no noise correction).

    Returns:
        (decoded_text, confidence, elapsed_seconds, method)
    """
    t0 = time.perf_counter()
    if not _model_loaded:
        text, conf = _fallback_decode(morse_input)
        elapsed = time.perf_counter() - t0
        log.debug("Dictionary decode: '%s' -> '%s' (%.3fs)",
                  morse_input[:30], text[:30], elapsed)
        return (text, conf, elapsed, "Dictionary")

    try:
        text, conf = _ai_decode(morse_input)
        elapsed = time.perf_counter() - t0
        log.debug("AI decode: '%s' -> '%s' conf=%.2f (%.3fs)",
                  morse_input[:30], text[:30], conf, elapsed)
        return (text, conf, elapsed, "AI")
    except Exception:
        log.exception("AI inference error.")
        elapsed = time.perf_counter() - t0
        return ("Prediction failed", 0.0, elapsed, "AI (error)")


def _ai_decode(morse_input: str) -> tuple:
    """
    Run the v4 Keras model to directly decode a Morse string to plain text.

    Steps:
      1. Tokenise Morse string char-by-char using CHAR_TO_IDX from vocab.
      2. Pad/truncate to MAX_MORSE_LEN.
      3. model.predict() -> shape (1, MAX_TEXT_LEN, vocab_size).
      4. argmax -> TARGET_CHARS indices.
      5. Map to characters, strip <PAD> tokens.
      6. Confidence = mean(max_softmax) over non-pad positions.

    Returns:
        (decoded_text, confidence)
    """
    morse_input = morse_input.strip()
    if not morse_input:
        return ("", 0.0)

    char_to_idx  = _vocab["CHAR_TO_IDX"]
    target_chars = _vocab["TARGET_CHARS"]
    unk_idx      = char_to_idx.get("<UNK>", 1)
    pad_idx      = char_to_idx.get("<PAD>", 0)

    # ── Step 1: Tokenise ──────────────────────────────────────────────────────
    token_seq = [char_to_idx.get(ch, unk_idx) for ch in morse_input]
    log.debug("[AI v4] Step 1 | Input: %r", morse_input)
    log.debug("[AI v4] Step 1 | Tokens (%d): %s", len(token_seq), token_seq[:20])

    if not token_seq:
        log.warning("[AI v4] Tokeniser produced 0 tokens for input %r", morse_input)
        return ("", 0.0)

    # ── Step 2: Pad to MAX_MORSE_LEN ──────────────────────────────────────────
    seq = token_seq[:MAX_MORSE_LEN]
    seq += [pad_idx] * (MAX_MORSE_LEN - len(seq))
    padded = np.array([seq], dtype=np.int32)  # shape (1, MAX_MORSE_LEN)
    log.debug("[AI v4] Step 2 | Padded shape: %s  first20: %s",
              padded.shape, padded[0][:20].tolist())

    # ── Step 3: Predict -> shape (1, MAX_TEXT_LEN, vocab_size) ───────────────
    raw_preds = _model.predict(padded, verbose=0)
    pred_0    = raw_preds[0]                        # (MAX_TEXT_LEN, vocab_size)

    # ── Step 4: argmax -> target char indices ─────────────────────────────────
    pred_indices = np.argmax(pred_0, axis=-1)       # (MAX_TEXT_LEN,)
    pred_probs   = np.max(pred_0, axis=-1)          # (MAX_TEXT_LEN,)

    # ── Step 5: Map indices to characters, stop at <PAD> ─────────────────────
    chars = []
    active_probs = []
    for idx, prob in zip(pred_indices, pred_probs):
        ch = target_chars[int(idx)] if int(idx) < len(target_chars) else "?"
        if ch == "<PAD>":
            break
        chars.append(ch)
        active_probs.append(float(prob))

    decoded_text = "".join(chars).strip()
    log.debug("[AI v4] Step 5 | Decoded: %r", decoded_text)

    # ── Step 6: Confidence ────────────────────────────────────────────────────
    confidence = float(np.mean(active_probs)) if active_probs else 0.0

    log.debug(
        "[AI v4] Result | morse=%r -> text=%r | conf=%.3f",
        morse_input[:40], decoded_text[:40], confidence,
    )
    return (decoded_text, confidence)


def _decode_corrected_morse(corrected_morse: str) -> str:
    """
    Convert a corrected character-level Morse string to plain text.

    Decoding strategy per symbol:
      1. Exact ITU dictionary lookup (preferred).
      2. Levenshtein fuzzy correction (edit distance <= 1) for invalid codes.
      3. Returns '?' if no valid match found within threshold.

    Word separator:   ' / '  (space-slash-space)
    Letter separator: single space ' '

    NOTE: This fuzzy layer handles cases where the model passes through
    invalid Morse codes unchanged (identity-mapping failure).  It cannot
    fix valid-but-wrong codes (e.g. '--.' vs '---') -- those require
    a retrained model.
    """
    corrected_morse = corrected_morse.strip()
    if not corrected_morse:
        return ""

    if " / " in corrected_morse:
        word_chunks = corrected_morse.split(" / ")
    else:
        word_chunks = [corrected_morse]

    decoded_words = []
    fuzzy_count   = 0
    unknown_count = 0

    for chunk in word_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        letters = []
        for code in chunk.split(" "):
            code = code.strip()
            if not code:
                continue
            char, method, dist = _fuzzy_decode_symbol(code)
            if method == "fuzzy":
                fuzzy_count += 1
                log.debug("[Fuzzy] %r not in dict -> %r (edit_dist=%d)", code, char, dist)
            elif method == "unknown":
                unknown_count += 1
                log.debug("[Unknown] %r has no valid Morse match (edit_dist=%d)", code, dist)
            letters.append(char)
        if letters:
            decoded_words.append("".join(letters))

    if fuzzy_count > 0:
        log.debug("[AI v2] Fuzzy corrections applied: %d symbols.", fuzzy_count)
    if unknown_count > 0:
        log.debug(
            "[AI v2] %d unrecognized symbols output as '?' "
            "(invalid Morse codes the model passed through unchanged).",
            unknown_count,
        )

    return " ".join(decoded_words)


def encode_text(text: str) -> str:
    """
    Encode plain text to Morse code using the standard ITU dictionary.

    Supported: A-Z, 0-9, space, and punctuation (see MORSE_CODE_DICT).
    Unsupported characters are silently skipped.

    Returns:
        Morse string with letters separated by spaces, words by ' / '.
    """
    words = text.upper().strip().split()
    morse_words = []
    for word in words:
        codes = [MORSE_CODE_DICT[ch] for ch in word if ch in MORSE_CODE_DICT]
        if codes:
            morse_words.append(" ".join(codes))
    result = " / ".join(morse_words)
    log.debug("Encoded text (len=%d) -> morse (len=%d).", len(text), len(result))
    return result


def is_model_loaded() -> bool:
    """Return True when the AI model is loaded and ready for inference."""
    return _model_loaded


def get_model_info() -> dict:
    """Return a debug summary of the loaded model state."""
    info = {
        "model_loaded":   _model_loaded,
        "model_path":     str(MODEL_PATH),
        "vocab_path":     str(VOCAB_PATH),
        "model_version":  "v4 (Direct Morse-to-Text)",
        "max_morse_len":  MAX_MORSE_LEN,
        "max_text_len":   MAX_TEXT_LEN,
    }
    if _vocab is not None:
        info["char_to_idx"]    = _vocab.get("CHAR_TO_IDX", {})
        info["target_vocab_size"] = len(_vocab.get("TARGET_CHARS", []))
    if _model is not None:
        if hasattr(_model, "input_shape"):
            info["input_shape"]  = str(_model.input_shape)
        if hasattr(_model, "output_shape"):
            info["output_shape"] = str(_model.output_shape)
    return info


def _fallback_decode(morse_input: str) -> tuple:
    """
    Dictionary-based Morse decoding used when the AI model is not loaded.

    Uses fuzzy Levenshtein correction for unrecognized Morse codes.
    Returns (decoded_text, 0.0) -- confidence is always 0.0.
    """
    words = morse_input.strip().split(" / ")
    decoded_words = []
    for word in words:
        tokens = word.strip().split(" ")
        letters = []
        for t in tokens:
            t = t.strip()
            if t:
                char, method, dist = _fuzzy_decode_symbol(t)
                letters.append(char)
        decoded_words.append("".join(letters))
    return (" ".join(decoded_words), 0.0)
