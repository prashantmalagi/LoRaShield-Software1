"""Small shared Transformer with separate symbol-edit and boundary heads."""
from __future__ import annotations

from .schema import MAX_SEQUENCE_TOKENS

MODEL_CONFIG = {
    "width": 128,
    "heads": 4,
    "feed_forward": 384,
    "layers": 4,
    "max_length": MAX_SEQUENCE_TOKENS,
    "max_insertions_per_gap": 2,
}


def _tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to build the V2 research model") from exc


def build_model(*, width: int = 128, heads: int = 4, feed_forward: int = 384,
                layers: int = 4, max_length: int = MAX_SEQUENCE_TOKENS,
                max_insertions_per_gap: int = 2):
    """Build inference model; construction does not train or save artifacts."""
    tf = _tf()
    if width <= 0 or heads <= 0 or width % heads or feed_forward <= 0 or layers <= 0:
        raise ValueError("invalid Transformer dimensions")
    if max_length < 4 or max_insertions_per_gap < 0:
        raise ValueError("invalid sequence or insertion limit")
    symbol_ids = tf.keras.Input(shape=(None,), dtype="int32", name="symbol_ids")
    input_gap_ids = tf.keras.Input(shape=(None,), dtype="int32", name="input_gap_ids")
    candidate_symbol_ids = tf.keras.Input(shape=(None,), dtype="int32", name="candidate_symbol_ids")
    candidate_alignment = tf.keras.Input(shape=(None,), dtype="int32", name="candidate_alignment")
    candidate_gap_ids = tf.keras.Input(shape=(None,), dtype="int32", name="candidate_gap_ids")
    candidate_boundary_ids = tf.keras.Input(shape=(None,), dtype="int32", name="candidate_boundary_ids")

    token_embedding = tf.keras.layers.Embedding(3, width, name="symbol_embedding")
    x = token_embedding(symbol_ids)
    positions = tf.keras.layers.Lambda(
        lambda ids: tf.tile(tf.range(tf.shape(ids)[1])[None, :], [tf.shape(ids)[0], 1]),
        name="position_ids")(symbol_ids)
    x = x + tf.keras.layers.Embedding(max_length, width, name="position_embedding")(positions)
    mask = tf.keras.layers.Lambda(lambda value: tf.not_equal(value, 0), name="symbol_mask")(symbol_ids)
    attention_mask = tf.keras.layers.Lambda(lambda value: value[:, None, :], name="attention_mask")(mask)
    for index in range(layers):
        normalized = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"encoder_norm_{index}")(x)
        attended = tf.keras.layers.MultiHeadAttention(
            num_heads=heads, key_dim=width // heads, name=f"encoder_attention_{index}")(
                normalized, normalized, attention_mask=attention_mask)
        x = tf.keras.layers.Add(name=f"attention_residual_{index}")([x, attended])
        normalized = tf.keras.layers.LayerNormalization(epsilon=1e-5, name=f"ffn_norm_{index}")(x)
        hidden = tf.keras.layers.Dense(feed_forward, activation="gelu", name=f"ffn_up_{index}")(normalized)
        hidden = tf.keras.layers.Dense(width, name=f"ffn_down_{index}")(hidden)
        x = tf.keras.layers.Add(name=f"ffn_residual_{index}")([x, hidden])
        mask_channel = tf.keras.layers.Lambda(lambda value: tf.cast(value[..., None], x.dtype),
                                              name=f"mask_channel_{index}")(mask)
        x = tf.keras.layers.Multiply(name=f"padding_zero_{index}")([x, mask_channel])

    symbol_logits = tf.keras.layers.Dense(3, name="symbol_operation_logits")(x)
    # Gap features use adjacent symbol states plus the observed separator tag.
    left = tf.keras.layers.Lambda(lambda value: tf.pad(value, [[0, 0], [1, 0], [0, 0]]),
                                  name="source_gap_left")(x)
    right = tf.keras.layers.Lambda(lambda value: tf.pad(value, [[0, 0], [0, 1], [0, 0]]),
                                   name="source_gap_right")(x)
    gap_embed = tf.keras.layers.Embedding(3, width, name="observed_gap_embedding")(input_gap_ids)
    source_gap = tf.keras.layers.Concatenate(name="source_gap_features")([left, right, gap_embed])
    source_gap = tf.keras.layers.Dense(width, activation="gelu", name="source_gap_projection")(source_gap)
    insertion_logits = tf.keras.layers.Dense(max_insertions_per_gap * 3,
                                             name="insertion_logits_flat")(source_gap)
    insertion_logits = tf.keras.layers.Lambda(
        lambda value: tf.reshape(value, [tf.shape(value)[0], tf.shape(value)[1],
                                         max_insertions_per_gap, 3]),
        output_shape=lambda shape: (shape[1], max_insertions_per_gap, 3),
        name="insertion_operation_logits")(insertion_logits)

    # Boundary stage consumes candidate symbols and their source alignment.
    candidate_emb = token_embedding(candidate_symbol_ids)
    clipped_alignment = tf.keras.layers.Lambda(
        lambda value: tf.maximum(value, 0), name="clipped_alignment")(candidate_alignment)
    gathered = tf.keras.layers.Lambda(
        lambda values: tf.gather(values[0], values[1], axis=1, batch_dims=1),
        output_shape=lambda shapes: (shapes[1][0], shapes[1][1], width),
        name="aligned_encoder_states")([x, clipped_alignment])
    inserted = tf.keras.layers.Lambda(
        lambda value: tf.equal(value, -1)[..., None], name="inserted_symbol_mask")(candidate_alignment)
    candidate_features = tf.keras.layers.Lambda(
        lambda values: tf.where(values[2], values[1], values[0]),
        output_shape=lambda shapes: (shapes[0][0], shapes[0][1], width),
        name="candidate_alignment_mix")([gathered, candidate_emb, inserted])
    candidate_features = tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                             name="candidate_feature_norm")(candidate_features)
    candidate_mask = tf.keras.layers.Lambda(
        lambda value: tf.not_equal(value, 0), name="candidate_symbol_mask")(candidate_symbol_ids)
    candidate_mask_channel = tf.keras.layers.Lambda(
        lambda value: tf.cast(value[..., None], candidate_features.dtype), name="candidate_mask_channel")(candidate_mask)
    candidate_features = tf.keras.layers.Multiply(name="candidate_padding_zero")(
        [candidate_features, candidate_mask_channel])
    c_left = tf.keras.layers.Lambda(lambda value: tf.pad(value, [[0, 0], [1, 0], [0, 0]]),
                                    name="candidate_gap_left")(candidate_features)
    c_right = tf.keras.layers.Lambda(lambda value: tf.pad(value, [[0, 0], [0, 1], [0, 0]]),
                                     name="candidate_gap_right")(candidate_features)
    candidate_prior = tf.keras.layers.Embedding(3, width, name="candidate_boundary_prior_embedding")(candidate_gap_ids)
    candidate_gap = tf.keras.layers.Concatenate(name="candidate_gap_features")(
        [c_left, c_right, candidate_prior])
    candidate_gap = tf.keras.layers.Dense(width, activation="gelu", name="boundary_gap_projection")(candidate_gap)
    boundary_logits = tf.keras.layers.Dense(3, name="boundary_logits")(candidate_gap)

    pooled_sum = tf.keras.layers.Lambda(
        lambda values: tf.reduce_sum(values[0] * tf.cast(values[1][..., None], values[0].dtype), axis=1),
        name="masked_encoder_sum")([x, mask])
    pooled_count = tf.keras.layers.Lambda(
        lambda value: tf.maximum(tf.reduce_sum(tf.cast(value, tf.float32), axis=1, keepdims=True), 1.0),
        name="masked_encoder_count")(mask)
    pooled = tf.keras.layers.Lambda(lambda values: values[0] / tf.cast(values[1], values[0].dtype),
                                    name="masked_mean_pool")([pooled_sum, pooled_count])
    clean_logits = tf.keras.layers.Dense(2, name="clean_logits")(pooled)
    # Risk is proposal-conditioned: combine the received-message context with
    # the actual candidate symbols, their alignment, and explicit edit counts.
    candidate_sum = tf.keras.layers.Lambda(
        lambda values: tf.reduce_sum(values[0] * tf.cast(values[1][..., None], values[0].dtype), axis=1),
        name="candidate_masked_sum")([candidate_features, candidate_mask])
    candidate_count = tf.keras.layers.Lambda(
        lambda value: tf.maximum(tf.reduce_sum(tf.cast(value, tf.float32), axis=1, keepdims=True), 1.0),
        name="candidate_masked_count")(candidate_mask)
    candidate_pool = tf.keras.layers.Lambda(
        lambda values: values[0] / tf.cast(values[1], values[0].dtype),
        name="candidate_mean_pool")([candidate_sum, candidate_count])
    edit_features = tf.keras.layers.Lambda(
        lambda values: _proposal_edit_features(tf, values[0], values[1], values[2], values[3], values[4], values[5]),
        output_shape=(8,),
        name="proposal_edit_features")([symbol_ids, candidate_symbol_ids, candidate_alignment,
                                         candidate_mask, input_gap_ids, candidate_boundary_ids])
    risk_features = tf.keras.layers.Concatenate(name="proposal_conditioned_risk_features")(
        [pooled, candidate_pool, edit_features])
    risk_features = tf.keras.layers.Dense(width, activation="gelu", name="risk_hidden")(risk_features)
    risk_logits = tf.keras.layers.Dense(3, name="risk_logits")(risk_features)
    return tf.keras.Model(
        inputs={"symbol_ids": symbol_ids, "input_gap_ids": input_gap_ids,
                "candidate_symbol_ids": candidate_symbol_ids,
                "candidate_alignment": candidate_alignment,
                "candidate_gap_ids": candidate_gap_ids,
                "candidate_boundary_ids": candidate_boundary_ids},
        outputs={"symbol_logits": symbol_logits, "insertion_logits": insertion_logits,
                 "boundary_logits": boundary_logits, "clean_logits": clean_logits,
                 "risk_logits": risk_logits},
        name="morse_correction_v2_research",
    )


def _proposal_edit_features(tf, source_ids, candidate_ids, alignment, candidate_mask,
                            source_gaps, candidate_gaps):
    """Differentiable normalized counts; -1 denotes a candidate insertion."""
    source_valid = tf.not_equal(source_ids, 0)
    candidate_valid = tf.cast(candidate_mask, tf.bool)
    aligned = tf.logical_and(candidate_valid, alignment >= 0)
    clipped = tf.maximum(alignment, 0)
    source_at_candidate = tf.gather(source_ids, clipped, axis=1, batch_dims=1)
    substitutions = tf.logical_and(aligned, tf.not_equal(candidate_ids, source_at_candidate))
    inserted = tf.logical_and(candidate_valid, alignment < 0)
    source_len = tf.reduce_sum(tf.cast(source_valid, tf.float32), axis=1)
    candidate_len = tf.reduce_sum(tf.cast(candidate_valid, tf.float32), axis=1)
    aligned_len = tf.reduce_sum(tf.cast(aligned, tf.float32), axis=1)
    values = [tf.reduce_sum(tf.cast(inserted, tf.float32), axis=1),
              tf.maximum(source_len - aligned_len, 0.0),
              tf.reduce_sum(tf.cast(substitutions, tf.float32), axis=1),
              candidate_len - source_len,
              tf.reduce_sum(tf.cast(tf.logical_and(candidate_gaps == 1, candidate_gaps != 0), tf.float32), axis=1),
              tf.reduce_sum(tf.cast(tf.equal(candidate_gaps, 2), tf.float32), axis=1),
              tf.reduce_sum(tf.cast(tf.equal(candidate_gaps, 1), tf.float32), axis=1) -
                  tf.reduce_sum(tf.cast(tf.equal(source_gaps, 1), tf.float32), axis=1),
              tf.reduce_sum(tf.cast(tf.equal(candidate_gaps, 2), tf.float32), axis=1) -
                  tf.reduce_sum(tf.cast(tf.equal(source_gaps, 2), tf.float32), axis=1)]
    return tf.stack(values, axis=1) / tf.maximum(source_len[:, None], 1.0)

