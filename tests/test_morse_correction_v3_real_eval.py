"""Unit tests use explicitly synthetic in-memory fixtures, never receiver captures."""

import json
import ast
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

import evaluate_morse_correction_v3_real as evaluator


def synthetic_row(sample_id="synthetic-1", *, received="... --- ...", quality="clean",
                 noise="unknown", reference_text="SOS", reference_morse="... --- ...",
                 independent=True, reliable=True, reference_source="synthetic test oracle"):
    """A clearly labeled synthetic test fixture, not a real capture."""
    return {
        "sample_id": sample_id,
        "source_session_id": "synthetic-session",
        "captured_at_utc": None,
        "capture_source": "SYNTHETIC unit-test fixture; not receiver data",
        "received_raw_morse_payload": received,
        "quality_label": quality,
        "noise_type": noise,
        "reference_text": reference_text,
        "reference_morse": reference_morse,
        "reference_source": reference_source,
        "reference_independent": independent,
        "reference_reliable": reliable,
        "crc_status": "not_applicable",
        "notes": "SYNTHETIC unit-test fixture only",
    }


class IdentityModel:
    input_shape = (None, 120)
    output_shape = (None, 120, 5)

    def __init__(self):
        self.batches = []

    def predict(self, batch, verbose=0):
        self.batches.append(batch.copy())
        output = np.zeros((len(batch), 120, 5), dtype=np.float32)
        for row_index, row in enumerate(batch):
            for position, token_id in enumerate(row):
                output[row_index, position, token_id] = 1.0
        return output


class ControlledModel:
    """Returns deterministic synthetic Morse outputs, one per inference row."""
    input_shape = (None, 120)
    output_shape = (None, 120, 5)

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        output = np.zeros((len(batch), 120, 5), dtype=np.float32)
        for row_index in range(len(batch)):
            text = next(self.outputs)
            for position, token in enumerate(text):
                output[row_index, position, evaluator.TOKEN_IDS[token]] = 1.0
        return output


class RealEvaluationTests(unittest.TestCase):
    def test_valid_dataset_and_unscored_without_reference(self):
        row = synthetic_row(reference_text=None, reference_morse=None,
                            reference_source=None, independent=False, reliable=False,
                            quality="unknown")
        validated = evaluator._validate_row(row, 1)
        self.assertFalse(validated["_scored"])
        self.assertIn("no reference", validated["_unscored_reason"])

    def test_duplicate_ids_rejected(self):
        row = synthetic_row()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(evaluator.DatasetValidationError, "duplicate sample_id"):
                evaluator.validate_dataset(path)

    def test_unknown_dataset_keys_rejected(self):
        row = synthetic_row()
        row["unrecognized"] = "value"
        with self.assertRaisesRegex(evaluator.DatasetValidationError, "unknown fields"):
            evaluator._validate_row(row, 1)

    def test_malformed_morse_and_ciphertext_rejected(self):
        for payload in ("...  ---", "...//---", "... /  ---", "ABC", "AA:BB"):
            row = synthetic_row(received=payload)
            with self.subTest(payload=payload), self.assertRaises(evaluator.DatasetValidationError):
                evaluator._validate_row(row, 1)

    def test_valid_then_invalid_dataset_fails_before_model_or_inference(self):
        valid = synthetic_row(sample_id="synthetic-valid")
        invalid = synthetic_row(sample_id="synthetic-invalid", received="... X")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.jsonl"
            path.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8")
            with mock.patch.object(evaluator.sys, "argv", ["evaluate", "--dataset", str(path)]), \
                    mock.patch.object(evaluator, "_load_model") as load_model, \
                    redirect_stdout(StringIO()):
                self.assertEqual(evaluator.main(), 2)
            load_model.assert_not_called()

    def test_empty_dataset_and_validation_only_do_not_load_model(self):
        valid = synthetic_row()
        with tempfile.TemporaryDirectory() as directory:
            empty_path = Path(directory) / "empty.jsonl"
            empty_path.write_text("", encoding="utf-8")
            valid_path = Path(directory) / "valid.jsonl"
            valid_path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            for argv in (
                ["evaluate", "--dataset", str(empty_path)],
                ["evaluate", "--dataset", str(valid_path), "--validate-only"],
            ):
                with self.subTest(argv=argv), \
                        mock.patch.object(evaluator.sys, "argv", argv), \
                        mock.patch.object(evaluator, "_load_model") as load_model, \
                        redirect_stdout(StringIO()):
                    self.assertEqual(evaluator.main(), 0)
                    load_model.assert_not_called()

    def test_slash_splitting_and_model_never_receives_slash_or_unknown(self):
        self.assertEqual(evaluator.split_words("... --- / - ."), ["... ---", "- ."])
        model = IdentityModel()
        corrected = evaluator._infer_words(model, "... --- / - .")
        self.assertEqual(corrected, "... --- / - .")
        self.assertEqual(len(model.batches), 1)
        self.assertEqual(model.batches[0].shape, (2, 120))
        self.assertNotIn(evaluator.UNK_ID, model.batches[0])
        with self.assertRaises(ValueError):
            evaluator._infer_words(model, "... / ABC")
        self.assertEqual(len(model.batches), 1, "invalid input must fail before inference")

    def test_overlength_word_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "no truncation"):
            evaluator.split_words("." * 121)
        self.assertEqual(len(evaluator.split_words("." * 120)[0]), 120)

    def test_reference_validation_and_consistency(self):
        valid = synthetic_row(reference_text="SOS", reference_morse="... --- ...")
        self.assertTrue(evaluator._validate_row(valid, 1)["_scored"])
        inconsistent = synthetic_row(reference_text="SOS", reference_morse="... --- ..")
        with self.assertRaisesRegex(evaluator.DatasetValidationError, "disagree"):
            evaluator._validate_row(inconsistent, 1)
        unsupported = synthetic_row(reference_text="SOS!", reference_morse=None)
        with self.assertRaises(evaluator.DatasetValidationError):
            evaluator._validate_row(unsupported, 1)

    def test_aligned_metrics_handle_insertions_deletions_and_shift(self):
        # Synthetic Morse letter groups: insertion/deletion must align later groups.
        self.assertEqual(evaluator._letter_boundary_distance("... --- ...", "... .. --- ..."), 3)
        self.assertEqual(evaluator._letter_boundary_distance("... .. --- ...", "... --- ..."), 3)
        self.assertEqual(evaluator.boundary_edit_distance("... --- / - .", "... .. --- / - ."), 3)
        self.assertEqual(evaluator.symbol_edit_distance("... --- ...", "... -- ..."), 1)

    def test_word_boundary_positions_detect_correct_shift_insert_delete(self):
        reference = ".- / -... -.-."
        self.assertTrue(evaluator._word_boundary_info(reference, reference)["boundaries_preserved"])

        shifted = ".- -... / -.-."
        shifted_info = evaluator._word_boundary_info(reference, shifted)
        self.assertEqual(shifted_info["reference_boundary_positions_letter_groups"], [1])
        self.assertEqual(shifted_info["candidate_boundary_positions_letter_groups"], [2])
        self.assertFalse(shifted_info["boundaries_preserved"])

        inserted = ".- / -... / -.-."
        inserted_info = evaluator._word_boundary_info(".- / -... -.-.", inserted)
        self.assertFalse(inserted_info["boundaries_preserved"])
        self.assertEqual(inserted_info["candidate_boundary_count"], 2)

        deleted = ".- -... / -.-."
        deleted_info = evaluator._word_boundary_info(".- / -... / -.-.", deleted)
        self.assertFalse(deleted_info["boundaries_preserved"])
        self.assertEqual(deleted_info["candidate_boundary_count"], 1)
        self.assertEqual(evaluator._word_boundary_info(reference, reference)["candidate_word_count"], 2)

    def test_grouping_and_clean_preservation(self):
        model = IdentityModel()
        clean = evaluator._validate_row(synthetic_row(), 1)
        noisy = evaluator._validate_row(synthetic_row(
            sample_id="synthetic-2", received="... --- ..", quality="noisy",
            noise="dot_dash_substitution", reference_text="SOS", reference_morse="... --- ..."
        ), 2)
        report = evaluator.evaluate_rows([clean, noisy], model)
        self.assertEqual(report["clean_preservation"], {
            "scored_clean_samples": 1, "exactly_preserved": 1, "rate": 1.0
        })
        self.assertIn("synthetic-session", report["by_source_session"])
        self.assertIn("clean", report["by_quality_label"])
        self.assertIn("dot_dash_substitution", report["by_noise_type"])
        self.assertEqual(report["overall"]["scored_samples"], 2)

    def test_unscored_sample_is_reported_and_not_inferred(self):
        row = evaluator._validate_row(synthetic_row(
            reference_text=None, reference_morse=None, reference_source=None,
            independent=False, reliable=False
        ), 1)
        model = IdentityModel()
        report = evaluator.evaluate_rows([row], model)
        self.assertEqual(report["unscored"][0]["sample_id"], "synthetic-1")
        self.assertEqual(model.batches, [])

    def test_controlled_outcomes_and_unscored_rows_do_not_change_metrics(self):
        reference = ".- --- ..."  # Synthetic Morse for AOS, used only as a test oracle.
        rows = [
            evaluator._validate_row(synthetic_row(
                sample_id="synthetic-improved", received=".- --- ..-", quality="noisy",
                reference_text="AOS", reference_morse=reference), 1),
            evaluator._validate_row(synthetic_row(
                sample_id="synthetic-unchanged", received=reference, quality="noisy",
                reference_text="AOS", reference_morse=reference), 2),
            evaluator._validate_row(synthetic_row(
                sample_id="synthetic-worsened", received=reference, quality="noisy",
                reference_text="AOS", reference_morse=reference), 3),
        ]
        outputs = [reference, reference, ".- --- ..-"]
        baseline_report = evaluator.evaluate_rows(rows, ControlledModel(outputs))
        self.assertEqual([item["outcome"] for item in baseline_report["scored"]],
                         ["improved", "unchanged", "worsened"])
        self.assertEqual((baseline_report["overall"]["improved"],
                          baseline_report["overall"]["unchanged"],
                          baseline_report["overall"]["worsened"]), (1, 1, 1))

        unscored = evaluator._validate_row(synthetic_row(
            sample_id="synthetic-unscored", received="......", quality="clean",
            reference_text=None, reference_morse=None, reference_source=None,
            independent=False, reliable=False), 4)
        with_unscored = evaluator.evaluate_rows(rows + [unscored], ControlledModel(outputs))
        self.assertEqual(with_unscored["overall"], baseline_report["overall"])
        self.assertEqual(with_unscored["by_source_session"], baseline_report["by_source_session"])
        self.assertEqual(with_unscored["by_quality_label"], baseline_report["by_quality_label"])
        self.assertEqual(with_unscored["by_noise_type"], baseline_report["by_noise_type"])
        self.assertEqual(with_unscored["unscored"][0]["sample_id"], "synthetic-unscored")

    def test_no_live_or_sensitive_modules_imported(self):
        forbidden = ("backend.main", "utils.serial_comm", "utils.encryption", "utils.crc")
        # Import-state assertions must run in a clean interpreter: unittest's
        # process may already have imported encryption/CRC through other tests.
        check = "\n".join((
            "import importlib, sys",
            "importlib.import_module('evaluate_morse_correction_v3_real')",
            f"forbidden = {forbidden!r}",
            "for module in forbidden:",
            "    assert module not in sys.modules, f'{module} unexpectedly imported'",
        ))
        result = subprocess.run([sys.executable, "-c", check], capture_output=True,
                                text=True, cwd=Path(evaluator.__file__).parent)
        self.assertEqual(result.returncode, 0, result.stderr)
        tree = ast.parse(Path(evaluator.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for forbidden_prefix in ("backend", "utils.serial", "utils.encryption", "utils.crc"):
            self.assertFalse(any(name == forbidden_prefix or name.startswith(forbidden_prefix + ".")
                                 for name in imported))


if __name__ == "__main__":
    unittest.main()
