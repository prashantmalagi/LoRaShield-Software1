"""Masked multi-task losses for V2 training; no training is performed here."""
from __future__ import annotations


def compute_loss_components(outputs: dict, targets: dict, masks: dict,
                            *, weights: dict[str, float] | None = None) -> dict:
    """Return scalar loss components; risk loss is skipped when labels are -1."""
    import tensorflow as tf
    weights = weights or {"symbol": 1.0, "insertion": 0.5, "boundary": 1.0,
                          "clean": 0.25, "risk": 0.5, "noise": 0.0}

    def masked_ce(logits, labels, mask):
        values = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        mask = tf.cast(mask, values.dtype)
        return tf.reduce_sum(values * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

    symbol = masked_ce(outputs["symbol_logits"], targets["symbol_ops"], masks["symbol"])
    insertion = masked_ce(outputs["insertion_logits"], targets["insertion_ops"], masks["insertion"])
    boundary = masked_ce(outputs["boundary_logits"], targets["boundary_labels"], masks["boundary"])
    clean = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        targets["clean_labels"], outputs["clean_logits"], from_logits=True))
    risk_mask = tf.cast(targets["risk_labels"] >= 0, tf.float32)
    risk_labels = tf.maximum(targets["risk_labels"], 0)
    risk_values = tf.keras.losses.sparse_categorical_crossentropy(
        risk_labels, outputs["risk_logits"], from_logits=True)
    risk = tf.reduce_sum(risk_values * risk_mask) / tf.maximum(tf.reduce_sum(risk_mask), 1.0)
    total = (weights["symbol"] * symbol + weights["insertion"] * insertion
             + weights["boundary"] * boundary + weights["clean"] * clean
             + weights["risk"] * risk)
    return {"total": total, "symbol": symbol, "insertion": insertion,
            "boundary": boundary, "clean": clean, "risk": risk}

