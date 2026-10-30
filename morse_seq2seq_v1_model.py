"""Minimal standalone variable-length Morse encoder-decoder prototype.

Research code only. This module is disconnected from LoRaShield production and
does not load V3/V2 artifacts, real-capture data, or any hardware/backend code.
It provides architecture, teacher-forcing batch preparation, one-step training,
and reference-free bounded greedy inference. Tiny smoke tests do not establish
correction accuracy or radio-channel performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import morse_seq2seq_v1_data as data

PAD_ID = data.TOKEN_TO_ID["<PAD>"]
BOS_ID = data.TOKEN_TO_ID["<BOS>"]
EOS_ID = data.TOKEN_TO_ID["<EOS>"]
VOCAB_SIZE = len(data.TOKEN_TO_ID)
MAX_OUTPUT_TOKENS = data.MAX_SEQUENCE_TOKENS


@dataclass(frozen=True)
class TrainingBatch:
    """Teacher-forcing arrays; decoder_targets include EOS and PAD mask is explicit."""

    encoder_input_ids: Any
    decoder_input_ids: Any
    decoder_target_ids: Any
    decoder_target_weights: Any
    target_sequences: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class DecodeResult:
    """Bounded inference result; malformed/truncated output has no Morse string."""

    received_morse: str
    output_morse: str | None
    output_tokens: tuple[str, ...]
    terminated_by_eos: bool
    termination_reason: str


def _tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TensorFlow is required for the standalone seq2seq prototype; "
            "install it only after project approval."
        ) from exc
    return tf


def _pad_ids(sequences: Sequence[Sequence[str]]) -> list[list[int]]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    width = max(len(sequence) for sequence in sequences)
    if width > data.MAX_SEQUENCE_TOKENS:
        raise ValueError(f"sequence length {width} exceeds {data.MAX_SEQUENCE_TOKENS}")
    rows = []
    for sequence in sequences:
        if not sequence:
            raise ValueError("empty token sequence is not valid")
        row = []
        for token in sequence:
            if token not in data.TOKEN_TO_ID or token == "<PAD>":
                raise ValueError(f"invalid unpadded token: {token!r}")
            row.append(data.TOKEN_TO_ID[token])
        rows.append(row + [PAD_ID] * (width - len(row)))
    return rows


def prepare_training_batch(pairs: Sequence[data.TrainingPair]) -> TrainingBatch:
    """Prepare aligned encoder/decoder arrays from offline labeled pairs only."""
    import numpy as np

    if not pairs:
        raise ValueError("at least one training pair is required")
    encoder_sequences = []
    decoder_inputs = []
    decoder_targets = []
    target_sequences = []
    for pair in pairs:
        if pair.input_tokens[0] != "<BOS>" or pair.input_tokens[-1] != "<EOS>":
            raise ValueError("encoder input must include BOS and EOS")
        if pair.target_tokens[0] != "<BOS>" or pair.target_tokens[-1] != "<EOS>":
            raise ValueError("decoder target must include BOS and EOS")
        if data.decode_morse(pair.input_tokens) != pair.received_morse:
            raise ValueError("input tokens do not match received Morse")
        if data.decode_morse(pair.target_tokens) != pair.reference_morse:
            raise ValueError("target tokens do not match the offline reference")
        if len(pair.target_tokens) < 3:
            raise ValueError("target must contain Morse content and EOS")
        encoder_sequences.append(pair.input_tokens)
        decoder_inputs.append(pair.target_tokens[:-1])
        # Shifted labels include EOS, allowing the decoder to learn termination.
        decoder_targets.append(pair.target_tokens[1:])
        target_sequences.append(tuple(pair.target_tokens))

    encoder_ids = np.asarray(_pad_ids(encoder_sequences), dtype=np.int32)
    decoder_ids = np.asarray(_pad_ids(decoder_inputs), dtype=np.int32)
    target_ids = np.asarray(_pad_ids(decoder_targets), dtype=np.int32)
    target_weights = (target_ids != PAD_ID).astype(np.float32)
    return TrainingBatch(
        encoder_input_ids=encoder_ids,
        decoder_input_ids=decoder_ids,
        decoder_target_ids=target_ids,
        decoder_target_weights=target_weights,
        target_sequences=tuple(target_sequences),
    )


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Per-token sparse CE with PAD positions zeroed before Keras reduction."""
    tf = _tensorflow()
    y_true = tf.cast(y_true, tf.int32)
    per_token = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    valid = tf.cast(tf.not_equal(y_true, PAD_ID), per_token.dtype)
    return per_token * valid


def build_model(*, embedding_dim: int = 16, encoder_units: int = 24,
                learning_rate: float = 1e-3):
    """Build a variable-length BiGRU encoder + GRU decoder with additive attention."""
    tf = _tensorflow()
    if embedding_dim < 1 or encoder_units < 1 or learning_rate <= 0:
        raise ValueError("model dimensions and learning rate must be positive")

    encoder_tokens = tf.keras.Input(shape=(None,), dtype="int32", name="encoder_token_ids")
    decoder_tokens = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_token_ids")

    encoder_embedding = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE, output_dim=embedding_dim, mask_zero=True,
        name="encoder_embedding",
    )
    encoder_embedded = encoder_embedding(encoder_tokens)
    encoder_mask = encoder_embedding.compute_mask(encoder_tokens)
    encoder_sequence, forward_state, backward_state = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            encoder_units, return_sequences=True, return_state=True,
            zero_output_for_mask=True,
        ),
        name="bidirectional_gru_encoder",
    )(encoder_embedded)
    decoder_units = encoder_units * 2
    decoder_initial_state = tf.keras.layers.Dense(
        decoder_units, activation="tanh", name="decoder_initial_state",
    )(tf.keras.layers.Concatenate(name="encoder_final_state_concat")(
        [forward_state, backward_state]
    ))

    decoder_embedding = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE, output_dim=embedding_dim, mask_zero=True,
        name="decoder_embedding",
    )
    decoder_embedded = decoder_embedding(decoder_tokens)
    decoder_mask = decoder_embedding.compute_mask(decoder_tokens)
    decoder_sequence = tf.keras.layers.GRU(
        decoder_units, return_sequences=True, zero_output_for_mask=True,
        name="autoregressive_gru_decoder",
    )(decoder_embedded, initial_state=decoder_initial_state)
    context = tf.keras.layers.AdditiveAttention(name="encoder_decoder_attention")(
        [decoder_sequence, encoder_sequence], mask=[decoder_mask, encoder_mask]
    )
    fused = tf.keras.layers.Concatenate(name="decoder_context_concat")(
        [decoder_sequence, context]
    )
    token_probabilities = tf.keras.layers.Dense(
        VOCAB_SIZE, activation="softmax", name="next_token_probabilities",
    )(fused)
    model = tf.keras.Model(
        inputs=[encoder_tokens, decoder_tokens],
        outputs=token_probabilities,
        name="morse_seq2seq_v1_prototype",
    )
    # The custom loss masks PAD independently of the embedding/attention masks.
    tf.keras.utils.register_keras_serializable(
        package="LoRaShieldResearch",
    )(masked_sparse_categorical_crossentropy)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_sparse_categorical_crossentropy,
    )
    return model


def train_one_step(model: Any, batch: TrainingBatch) -> dict[str, float]:
    """Run exactly one optimization step; intended for smoke tests only."""
    result = model.train_on_batch(
        [batch.encoder_input_ids, batch.decoder_input_ids],
        batch.decoder_target_ids,
        return_dict=True,
    )
    return {key: float(value) for key, value in result.items()}


def save_model(model: Any, path: str | Path) -> None:
    """Save a candidate only to the explicitly supplied path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)


def load_model(path: str | Path):
    """Reload this prototype artifact; no V2/V3 model path is consulted."""
    tf = _tensorflow()
    tf.keras.utils.register_keras_serializable(
        package="LoRaShieldResearch",
    )(masked_sparse_categorical_crossentropy)
    return tf.keras.models.load_model(
        str(path), compile=False,
        custom_objects={
            "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy,
            "LoRaShieldResearch>masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy,
        },
    )


def preprocess_received(received_morse: str) -> data.RuntimeInput:
    """Runtime-only preprocessing. There is intentionally no reference argument."""
    return data.runtime_preprocess(received_morse)


def greedy_decode(model: Any, received_morse: str, *,
                  max_output_tokens: int = MAX_OUTPUT_TOKENS) -> DecodeResult:
    """Greedy decode, bounded by EOS or total output length including BOS/EOS.

    The signature accepts a model and received Morse only; reference labels and
    target sequences are never runtime inputs. If EOS is not reached, output
    Morse is None rather than silently returning a truncated message.
    """
    import numpy as np

    if type(max_output_tokens) is not int or not 2 <= max_output_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError(f"max_output_tokens must be in [2, {MAX_OUTPUT_TOKENS}]")
    runtime_input = preprocess_received(received_morse)
    encoder_ids = np.asarray([runtime_input.input_ids], dtype=np.int32)
    decoder_ids = [BOS_ID]
    emitted: list[str] = []
    # BOS is already present in decoder_ids; generated content/EOS gets the
    # remaining total-sequence budget, so BOS + predictions never exceeds cap.
    for _ in range(max_output_tokens - 1):
        decoder_batch = np.asarray([decoder_ids], dtype=np.int32)
        probabilities = model.predict([encoder_ids, decoder_batch], verbose=0)
        expected = (1, len(decoder_ids), VOCAB_SIZE)
        if tuple(probabilities.shape) != expected:
            raise ValueError(f"model output shape {probabilities.shape}; expected {expected}")
        token_id = int(np.asarray(probabilities)[0, -1].argmax())
        token = data.ID_TO_TOKEN[token_id]
        if token == "<EOS>":
            try:
                decoded = data.decode_morse(("<BOS>", *emitted, "<EOS>"))
            except ValueError:
                return DecodeResult(received_morse, None, tuple(emitted), True, "invalid_sequence")
            return DecodeResult(received_morse, decoded, tuple(emitted), True, "eos")
        if token in ("<PAD>", "<UNK>", "<BOS>"):
            return DecodeResult(received_morse, None, tuple(emitted), False, "invalid_output_token")
        emitted.append(token)
        decoder_ids.append(token_id)
    return DecodeResult(received_morse, None, tuple(emitted), False, "max_output_tokens")
