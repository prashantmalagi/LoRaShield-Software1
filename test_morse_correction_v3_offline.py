"""Read-only, offline compatibility smoke test for the LoRaShield V3 model.

Loads the V3 artifact directly and runs in-memory inference only. It does not
import the backend, V2 MorseCorrector, AES, CRC, or serial modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "lorashield_morse_correction_v3.keras"
METADATA_PATH = ROOT / "models" / "lorashield_morse_correction_v3_metadata.json"
HISTORY_PATH = ROOT / "data" / "message_history.json"
SEQUENCE_LENGTH = 120
VOCAB = ["<PAD>", "<UNK>", ".", "-", " "]
TOKEN_IDS = {token: idx for idx, token in enumerate(VOCAB)}
PAD_ID, UNK_ID, DOT_ID, DASH_ID, SPACE_ID = range(5)

# Inspected utils.ai.encode_text: it uses the standard Morse map, one space
# between letters and " / " between words. Its decode helper can fuzzy-correct
# invalid codes, so this test uses exact dictionary lookup for reference text.
# The map comes from the existing standalone evaluator's inspected Morse table.
from evaluate_morse_correction_v3 import MORSE  # noqa: E402
from evaluate_morse_correction_v3_independent import (  # noqa: E402
    decode_real_positions,
    encode_padded,
    validate_metadata_and_model,
)

REVERSE_MORSE = {code: char for char, code in MORSE.items()}


def exact_dictionary_decode(morse: str) -> tuple[str, bool]:
    """Decode valid A-Z/0-9 Morse exactly; return text and all-codes-valid flag."""
    words: list[str] = []
    all_valid = True
    for word in morse.strip().split(" / "):
        letters: list[str] = []
        for code in word.split(" "):
            if not code:
                continue
            char = REVERSE_MORSE.get(code)
            if char is None:
                all_valid = False
                char = "?"
            letters.append(char)
        if letters:
            words.append("".join(letters))
    return " ".join(words), all_valid


def _valid_payload_shape(payload: str) -> bool:
    return bool(payload.strip()) and all(char in ".- /" for char in payload.strip())


def _split_words_without_slash(payload: str) -> list[str]:
    """Split only exact word-boundary markers and reject malformed slash use."""
    if "/" not in payload:
        return [payload]
    words = payload.split(" / ")
    if any("/" in word or not word.strip() for word in words):
        raise ValueError("Word separator must be exactly ' / ' with nonempty words")
    return words


def infer_one_word(model: Any, morse_word: str, call_counter: list[int]) -> str:
    """Run one slash-free Morse word through the saved V3 model."""
    if "/" in morse_word:
        raise AssertionError("Slash must never be passed to V3 inference")
    if len(morse_word) > SEQUENCE_LENGTH:
        raise ValueError(f"Morse input has {len(morse_word)} characters; V3 limit is {SEQUENCE_LENGTH}")
    if not morse_word or any(char not in ".- " for char in morse_word):
        raise ValueError("Inference accepts only dot, dash, and letter-space characters")

    import numpy as np

    padded = np.asarray([encode_padded(morse_word, SEQUENCE_LENGTH)], dtype=np.int32)
    if padded.shape != (1, SEQUENCE_LENGTH):
        raise AssertionError(f"Unexpected encoded input shape: {padded.shape}")
    call_counter[0] += 1
    probabilities = model.predict(padded, verbose=0)
    if probabilities.shape != (1, SEQUENCE_LENGTH, len(VOCAB)):
        raise AssertionError(f"Unexpected model output shape: {probabilities.shape}")
    predicted_ids = probabilities[0].argmax(axis=-1)
    return decode_real_positions(morse_word, predicted_ids)


def process_payload(model: Any, payload: str, call_counter: list[int]) -> tuple[str, str]:
    """Validate before inference; split/rejoin words so slash is never tokenized."""
    if not _valid_payload_shape(payload):
        return "REJECTED: not a Morse-only payload", ""
    normalized = payload.strip()
    if len(normalized) > SEQUENCE_LENGTH:
        return f"REJECTED: {len(normalized)} characters exceeds V3 limit {SEQUENCE_LENGTH}", ""
    words = _split_words_without_slash(normalized)
    output_words = [infer_one_word(model, word, call_counter) for word in words]
    return "PROCESSED", " / ".join(output_words)


def _load_v3() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"V3 model not found: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"V3 metadata not found: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("Metadata model_file does not match explicit V3 model path")
    if metadata.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("Metadata sequence length mismatch")
    if metadata.get("vocabulary") != VOCAB:
        raise ValueError("Metadata vocabulary mismatch")
    if metadata.get("token_ids") != TOKEN_IDS or metadata.get("padding_id") != PAD_ID:
        raise ValueError("Metadata token ID or PAD mapping mismatch")

    import tensorflow as tf

    # Explicitly load the V3 filename. No backend, config, or MorseCorrector import.
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    validate_metadata_and_model(model, metadata)
    if tuple(model.input_shape) != (None, SEQUENCE_LENGTH):
        raise ValueError(f"Unexpected V3 input shape: {model.input_shape}")
    if tuple(model.output_shape) != (None, SEQUENCE_LENGTH, len(VOCAB)):
        raise ValueError(f"Unexpected V3 output shape: {model.output_shape}")
    return model, metadata


def _print_case(label: str, raw: str, corrected: str, expected_text: str | None = None) -> None:
    decoded_raw, raw_valid = exact_dictionary_decode(raw)
    decoded_corrected, corrected_valid = exact_dictionary_decode(corrected)
    print(f"\n{label}")
    print(f"  Raw Morse:       {raw!r}")
    print(f"  Corrected Morse: {corrected!r}")
    print(f"  Changed:         {raw != corrected}")
    print(f"  Raw exact decode:       {decoded_raw!r} (valid={raw_valid})")
    print(f"  Corrected exact decode: {decoded_corrected!r} (valid={corrected_valid})")
    if expected_text is not None:
        print(f"  Reference text:  {expected_text!r}")


def main() -> int:
    try:
        model, metadata = _load_v3()
        call_counter = [0]
        print(f"Loaded V3 model: {MODEL_PATH}")
        print(f"Metadata: sequence_length={metadata['sequence_length']}, token_ids={metadata['token_ids']}")

        clean_fixtures = [
            ("Known clean SOS", "... --- ...", "SOS"),
            ("Known clean HELLO", ".... . .-.. .-.. ---", "HELLO"),
            ("Known clean history Morse", "- . ... -", "TEST"),
        ]
        history_records = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        history_sample = next((row for row in history_records if row.get("morse_code") == "- . ... -"), None)
        if history_sample is None:
            raise AssertionError("Expected read-only history Morse sample was not found")

        for label, raw, expected in clean_fixtures:
            status, corrected = process_payload(model, raw, call_counter)
            if status != "PROCESSED":
                raise AssertionError(f"{label}: {status}")
            _print_case(label, raw, corrected, expected)
            if corrected != raw:
                raise AssertionError(f"Clean fixture changed: {label}: {raw!r} -> {corrected!r}")
            decoded, valid = exact_dictionary_decode(corrected)
            if not valid or decoded != expected:
                raise AssertionError(f"{label} decoded to {decoded!r}; expected exact {expected!r}")

        print("\nHistory reference note:")
        print(f"  Stored plain_text: {history_sample.get('plain_text')!r}")
        print("  Stored Morse decodes exactly to 'TEST'; its associated plain_text is not a trustworthy match.")

        multiword = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        status, corrected_multiword = process_payload(model, multiword, call_counter)
        if status != "PROCESSED" or "/" not in corrected_multiword:
            raise AssertionError("Multiword handling failed to split and rejoin the word boundary")
        _print_case("Multiword HELLO WORLD (words inferred separately)", multiword,
                    corrected_multiword, "HELLO WORLD")

        # These controls are rejected before process_payload can call model.predict.
        negative_controls = [
            ("Plaintext negative control", "HELLO WORLD"),
            ("AES-shaped history ciphertext negative control", history_sample.get("encrypted_data", "")),
        ]
        for label, raw in negative_controls:
            before = call_counter[0]
            status, corrected = process_payload(model, raw, call_counter)
            if not status.startswith("REJECTED") or corrected or call_counter[0] != before:
                raise AssertionError(f"Negative control reached V3 inference: {label}")
            print(f"\n{label}: rejected before inference; raw={raw!r}; model_calls=0")

        exact_120 = (". " * 59) + ".."  # 59 one-dot groups + one two-dot group + 59 spaces.
        exact_121 = (". " * 60) + "."  # 61 one-dot groups + 60 spaces.
        if len(exact_120) != 120 or len(exact_121) != 121:
            raise AssertionError("Length fixtures were not constructed as intended")
        calls_before_120 = call_counter[0]
        status120, output120 = process_payload(model, exact_120, call_counter)
        if status120 != "PROCESSED" or call_counter[0] != calls_before_120 + 1:
            raise AssertionError("Exactly 120 characters should be processed once")
        _print_case("Valid Morse at exactly 120 characters", exact_120, output120)

        calls_before_121 = call_counter[0]
        status121, output121 = process_payload(model, exact_121, call_counter)
        if not status121.startswith("REJECTED") or output121 or call_counter[0] != calls_before_121:
            raise AssertionError("121-character Morse must be rejected without inference or truncation")
        print(f"\nValid Morse at 121 characters: {status121}; inference calls unchanged; no truncation.")

        print(f"\nPASS: all offline compatibility assertions passed; V3 inference calls={call_counter[0]}.")
        print("No live backend, V2 corrector, AES/decryption, CRC packet path, serial port, or radio was used.")
        return 0
    except Exception as exc:
        print(f"FAIL: offline V3 compatibility test: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
