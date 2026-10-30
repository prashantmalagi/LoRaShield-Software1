import unittest
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from morse_correction_v2.boundary_augmentation import (
    make_boundary_candidate, validate_candidate_alignment,
)
from morse_correction_v2.calibration import (
    RiskCalibrationCase, calibrate_risk_and_select_threshold,
)
from morse_correction_v2.data import make_example
from morse_correction_v2.provenance import _content_fingerprint, verify_manifest_files
from morse_correction_v2.provenance import verify_partition_rows
from morse_correction_v2.risk_training import (
    CandidateProposal, build_verified_risk_batches, generate_oof_risk_examples,
    source_held_out_folds,
)


class V2RiskBoundaryTests(unittest.TestCase):
    def _rows(self):
        return [make_example(f"source-{i}", "train", f"SOS {chr(65+i)}", "clean", i)
                for i in range(5)]

    def _verified_manifest(self, calibration_rows):
        from morse_correction_v2.data import make_example
        partitions = {
            "train": [make_example("tr-manifest", "train", "SOS", "clean", 81)],
            "validation": [make_example("va-manifest", "validation", "TEST", "clean", 82)],
            "calibration": calibration_rows,
        }
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        entries = {}
        for partition, rows in partitions.items():
            raw = [asdict(row) for row in rows]
            (root / f"{partition}.jsonl").write_text("".join(
                json.dumps(row, sort_keys=True) + "\n" for row in raw), encoding="utf-8")
            entries[partition] = {"content_sha256": _content_fingerprint(raw),
                                  "source_ids": sorted({r.source_id for r in rows})}
        path = root / "manifest.json"
        path.write_text(json.dumps({"partition_content": entries}), encoding="utf-8")
        manifest = verify_manifest_files(path)
        directory.cleanup()
        return manifest

    def test_oof_folds_are_source_disjoint_and_train_only(self):
        rows = self._rows()
        for fit, held in source_held_out_folds(rows, folds=3):
            self.assertFalse({r.source_id for r in fit} & {r.source_id for r in held})
            self.assertTrue(all(r.partition == "train" for r in fit + held))
        def callback(fit, held):
            self.assertFalse({r.source_id for r in fit} & {r.source_id for r in held})
            self.assertTrue(all(not hasattr(r, "reference_morse") for r in held))
            return [CandidateProposal(r.received_morse,
                tuple(range(sum(c in ".-" for c in r.received_morse)))) for r in held]
        result = generate_oof_risk_examples(rows, callback, folds=3)
        self.assertEqual(len(result.examples), len(rows))
        self.assertFalse(result.rejected)
        self.assertTrue(all(row.partition == "train" and row.risk_label == 1 for row in result.examples))
        batches = build_verified_risk_batches(result.examples, batch_size=2)
        self.assertTrue(all(batch.is_verified for batch in batches))
        self.assertEqual(sum(len(batch.risk_labels) for batch in batches), len(rows))

    def test_oof_rejects_invalid_proposals_with_reasons(self):
        rows = self._rows()
        proposals = {
            "source-0": CandidateProposal(".X", (0, 1)),
            "source-1": CandidateProposal(".  -", (0, 1)),
            "source-2": CandidateProposal(".-", (200, 201)),
        }
        def callback(fit, held):
            return [proposals.get(row.source_id, CandidateProposal(row.received_morse,
                tuple(range(sum(c in ".-" for c in row.received_morse))))) for row in held]
        result = generate_oof_risk_examples(rows, callback, folds=3)
        reasons = " ".join(item.reason for item in result.rejected)
        self.assertEqual(len(result.rejected), 3)
        self.assertIn("unsupported Morse", reasons)
        self.assertIn("malformed Morse boundary", reasons)
        self.assertIn("outside source", reasons)

    def test_risk_training_rejects_caller_created_train_batch_without_update(self):
        from morse_correction_v2.model import build_model
        from morse_correction_v2.risk_training import train_risk_head
        class NeverUpdate:
            def apply_gradients(self, pairs):
                raise AssertionError("unverified batch reached optimizer")
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        flags = {layer.name: layer.trainable for layer in model.layers}
        with self.assertRaisesRegex(ValueError, "internally verified OOF"):
            train_risk_head(model, [{"partition": "train", "risk_labels": [1],
                "source_ids": ["pretend"], "inputs": {}}], NeverUpdate())
        self.assertEqual(flags, {layer.name: layer.trainable for layer in model.layers})

    def test_risk_training_optimizer_receives_only_risk_variables_and_restores_flags(self):
        import numpy as np
        from morse_correction_v2.model import build_model
        from morse_correction_v2.risk_training import train_risk_head
        rows = self._rows()[:2]
        def callback(fit, held):
            return [CandidateProposal(item.received_morse,
                tuple(range(sum(c in ".-" for c in item.received_morse)))) for item in held]
        result = generate_oof_risk_examples(rows, callback, folds=2)
        batch = build_verified_risk_batches(result.examples, batch_size=4)[0]
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        model(batch.inputs, training=False)
        before = [value.copy() for value in model.get_weights()]
        flags = {layer.name: layer.trainable for layer in model.layers}
        class CaptureOnly:
            variables = ()
            def apply_gradients(self, pairs):
                pairs = tuple(pairs)
                self.variables = tuple(variable for _, variable in pairs)
        optimizer = CaptureOnly()
        train_risk_head(model, batch if isinstance(batch, list) else [batch], optimizer)
        names = {getattr(variable, "path", variable.name).split("/")[0]
                 for variable in optimizer.variables}
        self.assertTrue(names)
        self.assertTrue(all(name in {"risk_hidden", "risk_logits"} for name in names))
        self.assertEqual(flags, {layer.name: layer.trainable for layer in model.layers})
        for old, new in zip(before, model.get_weights()):
            np.testing.assert_array_equal(old, new)

    def test_oof_rejects_non_train_rows(self):
        with self.assertRaises(ValueError):
            source_held_out_folds([make_example("v", "validation", "SOS", "clean", 1)], 2)

    def test_boundary_augmentation_alignment_and_labels_are_valid(self):
        row = make_example("boundary", "train", "SOS HI", "clean", 4)
        candidate = make_boundary_candidate(row.reference_morse, seed=91, corruption_rate=.4)
        validate_candidate_alignment(candidate, len("".join(c for c in row.received_morse if c in ".-")))
        self.assertEqual(len(candidate.gap_labels), len(candidate.symbols) + 1)
        self.assertEqual((candidate.gap_labels[0], candidate.gap_labels[-1]), (0, 0))
        with self.assertRaises(ValueError):
            validate_candidate_alignment(candidate, 0)

    def test_calibration_is_provenance_bound_and_abstains_when_unsafe(self):
        calibration_rows = [make_example("c1", "calibration", "HI", "clean", 3),
                            make_example("c2", "calibration", "YES", "clean", 4)]
        manifest = self._verified_manifest(calibration_rows)
        cases = [RiskCalibrationCase("c1", "calibration", (0.0, 0.0, 5.0), 2, True, True),
                 RiskCalibrationCase("c2", "calibration", (0.0, 5.0, 0.0), 1, False, True)]
        from morse_correction_v2.model import build_model
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        result = calibrate_risk_and_select_threshold(cases, manifest=manifest, model=model)
        self.assertTrue(result.selected_all_abstain)
        self.assertEqual(result.improve_threshold, .95)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            calibrate_risk_and_select_threshold(cases[:1], manifest=manifest, model=model)
        forbidden = [RiskCalibrationCase("c1", "locked_test", (0., 0., 1.), 2, False, True)]
        with self.assertRaises(ValueError):
            calibrate_risk_and_select_threshold(forbidden, manifest=manifest, model=model)
        with self.assertRaises(ValueError):
            calibrate_risk_and_select_threshold(cases, manifest=object(), model=model)
        fabricated = verify_partition_rows({
            "train": [make_example("t-fake", "train", "SOS", "clean", 5)],
            "validation": [make_example("v-fake", "validation", "TEST", "clean", 6)],
            "calibration": calibration_rows,
        })
        with self.assertRaisesRegex(ValueError, "partition files"):
            calibrate_risk_and_select_threshold(cases, manifest=fabricated, model=model)

    def test_calibration_accepts_logits_from_actual_model_forward_and_fails_closed(self):
        import numpy as np
        from morse_correction_v2.batching import prepare_batch
        from morse_correction_v2.calibration import derive_risk_label
        from morse_correction_v2.model import build_model
        rows = [
            make_example("cal-1", "calibration", "SOS HI", "substitution", 17),
            make_example("cal-2", "calibration", "SOS HI", "clean", 18),
            make_example("cal-3", "calibration", "TEST", "letter_boundary_shift", 19),
        ]
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        cases = []
        for row in rows:
            batch = prepare_batch([row])
            logits = model(batch.inputs, training=False)["risk_logits"].numpy()[0]
            candidate = row.reference_morse
            cases.append(RiskCalibrationCase(row.source_id, "calibration",
                tuple(float(x) for x in logits),
                derive_risk_label(row.reference_morse, row.received_morse, candidate),
                row.is_clean, candidate != row.received_morse))
        manifest = self._verified_manifest(rows)
        result = calibrate_risk_and_select_threshold(cases, manifest=manifest, model=model)
        self.assertEqual(len(result.calibrator.probabilities(cases[0].logits)), 3)
        self.assertTrue(result.selected_all_abstain)

    def test_risk_label_classes_follow_actual_candidate_error(self):
        from morse_correction_v2.calibration import derive_risk_label
        reference = ".- -..."
        worse = "-- -..."
        self.assertEqual(derive_risk_label(reference, worse, reference), 0)
        self.assertEqual(derive_risk_label(reference, reference, reference), 1)
        self.assertEqual(derive_risk_label(reference, reference, worse), 2)


if __name__ == "__main__":
    unittest.main()
