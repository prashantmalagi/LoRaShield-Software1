"""
LoRaShield – ML Correction Configuration
==========================================
Centralised flags for the V2 Morse error-correction model.

Change ``ML_CORRECTION_ENABLED`` to ``True`` only after integration
tests have passed in your environment.

Settings here can also be overridden at runtime by writing to these
attributes before loading the model::

    import correction.config as mlcfg
    mlcfg.ML_CORRECTION_ENABLED = True

Environment variable override (evaluated once at import)::

    LORASHIELD_ML_CORRECTION=1  # any non-empty string enables
"""

import os

# ── Primary on/off switch ─────────────────────────────────────────────────────
# Default: DISABLED.  Enable only after verifying model accuracy on
# independently generated test data in your environment.
ML_CORRECTION_ENABLED: bool = bool(
    os.environ.get("LORASHIELD_ML_CORRECTION", "").strip()
)

# ── Model file paths (relative to project root) ───────────────────────────────
# These must not be changed without retraining.
MODEL_FILENAME: str = "lorashield_morse_correction_v2.keras"
VOCAB_FILENAME: str = "lorashield_morse_correction_v2_vocab.json"

# ── Sequence constraints (must match training) ────────────────────────────────
# Overridden at load-time from the vocab JSON if the key is present.
MAX_SEQUENCE_LENGTH: int = 120

# ── Expected vocabulary (must match training exactly) ────────────────────────
# Order is critical: ID 0 → <PAD>, 1 → <UNK>, 2 → '.', 3 → '-', 4 → space.
EXPECTED_VOCAB: list[str] = ["<PAD>", "<UNK>", ".", "-", " "]
NUM_CLASSES: int = 5  # model output dimension
