"""Smoke tests for the isolated V1 encoder-decoder; all samples are synthetic fixtures."""

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import morse_seq2seq_v1_data as data
import morse_seq2seq_v1_model as prototype


class ScriptedDecoder:
    """Synthetic fake model for EOS and max-length behavior only."""

    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.calls = 0

    def predict(self, inputs, verbose=0):
        decoder_ids = inputs[1]
        result = np.full(
            (decoder_ids.shape[0], decoder_ids.shape[1], prototype.VOCAB_SIZE),
            1e-6, dtype=np.float32,
        )
        next_id = self.token_ids[min(self.calls, len(self.token_ids) - 1)]
        self.calls += 1
        result[:, -1, next_id] = 1.0
        return result


class MorseSeq2SeqPrototypeTests(unittest.TestCase):
    def test_runtime_preprocessing_is_deterministic_for_single_and_multiword_messages(self):
        for morse in (".", "-", "... --- ...", ".- -... / -.-."):
            with self.subTest(morse=morse):
                first = prototype.preprocess_received(morse)
                second = prototype.preprocess_received(morse)
                self.assertEqual(first, second)
                self.assertEqual(first.input_ids[0], data.TOKEN_TO_ID["<BOS>"])
                self.assertEqual(first.input_ids[-1], data.TOKEN_TO_ID["<EOS>"])

    def test_empty_unsupported_and_overlength_inputs_are_rejected(self):
        for value in ("", " .", "... X", "...  ---", ".../ ---"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    prototype.preprocess_received(value)
        maximum = "." * (data.MAX_SEQUENCE_TOKENS - 2)
        self.assertEqual(len(prototype.preprocess_received(maximum).input_ids),
                         data.MAX_SEQUENCE_TOKENS)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prototype.preprocess_received("." * (data.MAX_SEQUENCE_TOKENS - 1))

    def test_runtime_interface_has_no_reference_or_target_argument(self):
        parameters = set(inspect.signature(prototype.greedy_decode).parameters)
        self.assertEqual(parameters, {"model", "received_morse", "max_output_tokens"})
        self.assertNotIn("reference", parameters)
        self.assertNotIn("target", parameters)
        runtime_fields = set(prototype.DecodeResult.__dataclass_fields__)
        self.assertNotIn("reference_morse", runtime_fields)
        self.assertNotIn("target_tokens", runtime_fields)
        with self.assertRaises(TypeError):
            prototype.greedy_decode(
                ScriptedDecoder([prototype.EOS_ID]), ".", reference_morse="."
            )

    def test_eos_terminates_decode_and_output_limit_is_strict(self):
        scripted = ScriptedDecoder([
            data.TOKEN_TO_ID["<DOT>"], prototype.EOS_ID,
        ])
        result = prototype.greedy_decode(scripted, ".", max_output_tokens=4)
        self.assertTrue(result.terminated_by_eos)
        self.assertEqual(result.termination_reason, "eos")
        self.assertEqual(result.output_morse, ".")
        self.assertEqual(scripted.calls, 2)

        bounded = ScriptedDecoder([data.TOKEN_TO_ID["<DOT>"]])
        truncated = prototype.greedy_decode(bounded, ".", max_output_tokens=2)
        self.assertFalse(truncated.terminated_by_eos)
        self.assertEqual(truncated.termination_reason, "max_output_tokens")
        self.assertIsNone(truncated.output_morse)
        self.assertEqual(len(truncated.output_tokens), 1)
        with self.assertRaises(ValueError):
            prototype.greedy_decode(bounded, ".", max_output_tokens=0)

    def test_pad_positions_are_masked_and_eos_is_a_decoder_target(self):
        pairs = [
            data.make_training_pair("smoke-E", "train", "E", "clean", 1201),
            data.make_training_pair("smoke-SOS", "train", "SOS TEST", "clean", 1202),
        ]
        batch = prototype.prepare_training_batch(pairs)
        self.assertEqual(batch.encoder_input_ids.shape[0], 2)
        self.assertEqual(batch.decoder_input_ids.shape, batch.decoder_target_ids.shape)
        self.assertEqual(batch.decoder_target_ids.shape, batch.decoder_target_weights.shape)
        self.assertTrue(np.all(batch.decoder_target_weights[batch.decoder_target_ids == 0] == 0))
        self.assertTrue(np.all(batch.decoder_target_weights[batch.decoder_target_ids != 0] == 1))
        for row, target_sequence in enumerate(batch.target_sequences):
            self.assertEqual(target_sequence[0], "<BOS>")
            self.assertEqual(target_sequence[-1], "<EOS>")
            active_targets = batch.decoder_target_ids[row][batch.decoder_target_weights[row] == 1]
            self.assertEqual(int(active_targets[-1]), prototype.EOS_ID)

        tf = prototype._tensorflow()
        y_true = tf.constant([[data.TOKEN_TO_ID["<DOT>"], prototype.EOS_ID, prototype.PAD_ID]])
        probabilities = tf.constant([[[.01, .01, .90, .01, .02, .02, .02, .01],
                                      [.01, .01, .01, .90, .02, .02, .02, .01],
                                      [.01, .01, .01, .01, .01, .01, .01, .93]]], dtype=tf.float32)
        changed_pad = tf.tensor_scatter_nd_update(
            probabilities, indices=[[0, 2, data.TOKEN_TO_ID["<PAD>"]]], updates=[1.0]
        )
        loss_a = prototype.masked_sparse_categorical_crossentropy(y_true, probabilities)
        loss_b = prototype.masked_sparse_categorical_crossentropy(y_true, changed_pad)
        self.assertTrue(np.allclose(loss_a.numpy(), loss_b.numpy(), atol=1e-7))

    def test_tiny_model_train_step_shapes_and_save_reload(self):
        tf = prototype._tensorflow()
        tf.keras.utils.set_random_seed(1771)
        pairs = [
            data.make_training_pair("smoke-E", "train", "E", "clean", 101),
            data.make_training_pair(
                "smoke-SOS", "train", "SOS TEST", "dot_dash_substitution_05", 102
            ),
        ]
        batch = prototype.prepare_training_batch(pairs)
        model = prototype.build_model(embedding_dim=8, encoder_units=4)
        predictions = model(
            [batch.encoder_input_ids, batch.decoder_input_ids], training=False
        )
        self.assertEqual(tuple(predictions.shape), (
            2, batch.decoder_target_ids.shape[1], prototype.VOCAB_SIZE
        ))
        self.assertTrue(np.isfinite(
            prototype.masked_sparse_categorical_crossentropy(
                batch.decoder_target_ids, predictions
            ).numpy()
        ).all())
        step = prototype.train_one_step(model, batch)
        self.assertIn("loss", step)
        self.assertTrue(np.isfinite(step["loss"]))

        with tempfile.TemporaryDirectory(prefix="morse_seq2seq_v1_smoke_") as temp_dir:
            path = Path(temp_dir) / "prototype.keras"
            prototype.save_model(model, path)
            reloaded = prototype.load_model(path)
            after = reloaded(
                [batch.encoder_input_ids, batch.decoder_input_ids], training=False
            ).numpy()
            self.assertEqual(tuple(reloaded.input_shape[0]), (None, None))
            self.assertEqual(tuple(after.shape), tuple(predictions.shape))
            self.assertTrue(np.allclose(after, model(
                [batch.encoder_input_ids, batch.decoder_input_ids], training=False
            ).numpy(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
