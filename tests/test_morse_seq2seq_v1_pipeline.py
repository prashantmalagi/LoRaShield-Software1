"""Pipeline isolation and evaluation tests with tiny synthetic fixtures only."""

import inspect
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate_morse_seq2seq_v1 as evaluator
import morse_seq2seq_v1_data as data
import morse_seq2seq_v1_model as prototype
import train_morse_seq2seq_v1 as pipeline


class ScheduledBatchModel:
    def __init__(self, schedules):
        self.schedules = [list(schedule) for schedule in schedules]
        self.calls = 0

    def predict(self, inputs, verbose=0):
        decoder = inputs[1]
        result = np.full(
            (len(decoder), decoder.shape[1], prototype.VOCAB_SIZE),
            1e-6, dtype=np.float32,
        )
        for row, schedule in enumerate(self.schedules):
            token_id = schedule[min(self.calls, len(schedule) - 1)]
            result[row, -1, token_id] = 1.0
        self.calls += 1
        return result


def fixture_partitions():
    return {
        "train": ["TEST MESSAGE"],
        "validation": ["SOS SIGNAL"],
        "calibration": ["HELLO WORLD"],
        "locked_test": ["MORSE CODE"],
    }


class MorseSeq2SeqPipelineTests(unittest.TestCase):
    def test_training_pairs_are_isolated_to_train_and_validation(self):
        partitions = fixture_partitions()
        train_pairs, validation_pairs = pipeline.make_train_validation_pairs(
            partitions, seed=61021
        )
        pipeline.assert_pair_partition(train_pairs, "train")
        pipeline.assert_pair_partition(validation_pairs, "validation")
        self.assertTrue(all(pair.source_id.startswith("train-") for pair in train_pairs))
        self.assertTrue(all(pair.source_id.startswith("validation-") for pair in validation_pairs))
        self.assertFalse(any("calibration" in pair.source_id or "locked_test" in pair.source_id
                             for pair in train_pairs + validation_pairs))
        self.assertEqual(len(train_pairs), len(data.CORRUPTION_KINDS))
        self.assertEqual(len(validation_pairs), len(data.CORRUPTION_KINDS))

    def test_reference_labels_are_absent_from_runtime_interfaces(self):
        for function in (prototype.greedy_decode, evaluator.greedy_decode_batch):
            with self.subTest(function=function.__name__):
                names = set(inspect.signature(function).parameters)
                self.assertNotIn("reference", names)
                self.assertNotIn("reference_morse", names)
                self.assertNotIn("target", names)
                self.assertNotIn("target_tokens", names)
        self.assertEqual(set(data.RuntimeInput.__dataclass_fields__),
                         {"received_morse", "input_ids"})

    def test_checkpoint_selection_and_callbacks_use_validation_only(self):
        history = {
            "loss": [0.9, 0.5, 0.4],
            "val_loss": [0.8, 0.3, 0.35],
            "locked_test_loss": [0.01, 99.0, 0.0],
        }
        self.assertEqual(pipeline.best_epoch_from_validation(history), 2)
        changed_locked = dict(history, locked_test_loss=[500.0, -10.0, 500.0])
        self.assertEqual(pipeline.best_epoch_from_validation(changed_locked), 2)
        callbacks = pipeline.validation_callbacks(Path(tempfile.gettempdir()) / "unused.keras", 2)
        self.assertEqual([callback.monitor for callback in callbacks], ["val_loss", "val_loss"])
        self.assertTrue(callbacks[0].save_best_only)
        self.assertTrue(callbacks[1].restore_best_weights)

    def test_deterministic_setup_repeats_python_numpy_and_tensorflow_streams(self):
        tf = prototype._tensorflow()
        first_environment = pipeline.configure_determinism(932117)
        first = (random.random(), float(np.random.random()),
                 tf.random.uniform((3,), dtype=tf.float32).numpy())
        second_environment = pipeline.configure_determinism(932117)
        second = (random.random(), float(np.random.random()),
                  tf.random.uniform((3,), dtype=tf.float32).numpy())
        self.assertEqual(first_environment["seed"], second_environment["seed"])
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(np.array_equal(first[2], second[2]))

    def test_batch_decoder_terminates_at_eos_or_strict_token_cap(self):
        model = ScheduledBatchModel([
            [data.TOKEN_TO_ID["<DOT>"], prototype.EOS_ID],
            [data.TOKEN_TO_ID["<DASH>"]],
        ])
        results = evaluator.greedy_decode_batch(
            model, [".", "-"], max_output_tokens=3, batch_size=2
        )
        self.assertTrue(results[0].terminated_by_eos)
        self.assertEqual(results[0].output_morse, ".")
        self.assertFalse(results[1].terminated_by_eos)
        self.assertEqual(results[1].termination_reason, "max_output_tokens")
        self.assertIsNone(results[1].output_morse)
        self.assertLessEqual(len(results[1].output_tokens) + 1, 3)

    def test_batch_decoder_stops_predicting_when_all_items_finish(self):
        model = ScheduledBatchModel([
            [data.TOKEN_TO_ID["<DOT>"], prototype.EOS_ID],
            [data.TOKEN_TO_ID["<UNK>"]],
        ])
        results = evaluator.greedy_decode_batch(
            model, [".", "-"], max_output_tokens=8, batch_size=2
        )
        self.assertEqual(model.calls, 2)
        self.assertEqual(results[0].output_morse, ".")
        self.assertEqual(results[0].termination_reason, "eos")
        self.assertIsNone(results[1].output_morse)
        self.assertEqual(results[1].termination_reason, "invalid_output_token")

    def test_completed_manifest_has_actual_source_audit_and_no_locked_scores(self):
        path = Path(__file__).resolve().parents[1] / "morse_seq2seq_v1_completed_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["partition_audit"]["source_counts"],
                         data.PLANNED_PARTITION_COUNTS)
        results = manifest["partition_audit"]["overlap_check"]["results"]
        self.assertEqual(results["complete_message_overlap_count"], 0)
        self.assertEqual(results["word_overlap_count"], 0)
        self.assertEqual(results["cross_partition_message_collision_count"], 0)
        self.assertEqual(results["cross_partition_word_collision_count"], 0)
        self.assertNotIn("locked_test_metrics", manifest)
        self.assertTrue(manifest["partition_audit"]["source_fingerprints_sha256"]["locked_test"])

    def test_locked_metrics_compare_same_cases_and_group_by_corruption(self):
        ref = "... ---"
        received = "--- ..."
        pair = data.TrainingPair(
            source_id="locked-fixture", partition="locked_test",
            corruption_type="dot_dash_substitution_10", noise_percent=10,
            reference_morse=ref, received_morse=received,
            input_tokens=data.encode_morse(received), target_tokens=data.encode_morse(ref),
        )
        candidate = prototype.DecodeResult(
            received_morse=received, output_morse=ref,
            output_tokens=data.encode_morse(ref)[1:-1],
            terminated_by_eos=True, termination_reason="eos",
        )
        report = evaluator.score_outputs([pair], ["--- ---"], [candidate])
        self.assertIn("unchanged", report["systems"])
        self.assertIn("v3", report["systems"])
        self.assertIn("seq2seq_v1", report["systems"])
        self.assertEqual(report["systems"]["seq2seq_v1"]["improved_vs_unchanged"], 1)
        self.assertIn("dot_dash_substitution_10", report["by_corruption_type"])
        self.assertEqual(report["metric_definitions"]["harmful_severity"],
                         "Extra combined errors on worsened messages")


if __name__ == "__main__":
    unittest.main()
