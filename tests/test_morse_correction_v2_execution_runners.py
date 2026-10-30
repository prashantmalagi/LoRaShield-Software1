import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from morse_correction_v2.data import make_example
from morse_correction_v2.inference import V2InferenceResult
from morse_correction_v2.oof_runner import run_oof_risk_training
from morse_correction_v2.provenance import _content_fingerprint, verify_manifest_files
from morse_correction_v2.risk_training import CandidateProposal


class V2ExecutionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"TensorFlow unavailable: {exc}")

    def _model(self):
        from morse_correction_v2.batching import prepare_batch
        from morse_correction_v2.model import build_model
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        model(prepare_batch([make_example("build", "train", "SOS HI", "clean", 8)]).inputs,
              training=False)
        return model

    def _rows(self):
        return [make_example(f"oof-{i}", "train", f"SOS {chr(65+i)}", "clean", i)
                for i in range(4)]

    def test_oof_runner_passes_only_disjoint_fit_and_heldout_sources(self):
        import tensorflow as tf
        rows = self._rows()
        fit_sets = []
        def fold_fit(fit_rows, **kwargs):
            fit_sets.append({row.source_id for row in fit_rows})
            return object()
        def fold_predict(model, heldout):
            held_ids = {row.source_id for row in heldout}
            self.assertFalse(fit_sets[-1] & held_ids)
            return [CandidateProposal(row.received_morse,
                tuple(range(sum(ch in ".-" for ch in row.received_morse)))) for row in heldout]
        risk_model = self._model()
        optimizer = tf.keras.optimizers.Adam()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("morse_correction_v2.oof_runner.train_risk_head",
                        return_value={"risk_loss": 0.0, "risk_head_layers": ["risk_hidden", "risk_logits"]}):
            root = Path(directory)
            manifest = self._write_manifest(root, [make_example("cal-oof", "calibration", "HI", "clean", 41)],
                                            train_rows=rows)
            output = root / "risk-run"
            generated, metadata = run_oof_risk_training(rows, risk_model, optimizer, output,
                verified_manifest=manifest,
                folds=2, fold_epochs=1, risk_epochs=1, batch_size=8,
                fold_fit=fold_fit, fold_predict=fold_predict)
            self.assertEqual(len(generated.examples), len(rows))
            self.assertTrue(metadata["risk_head_trained"])
            self.assertTrue((output / "CURRENT").is_file())

    def test_oof_interruption_leaves_no_output_and_retry_recovers(self):
        import tensorflow as tf
        rows = self._rows()
        calls = {"fold": 0}
        def fold_fit(fit_rows, **kwargs):
            calls["fold"] += 1
            if calls["fold"] == 1:
                raise RuntimeError("simulated fold interruption")
            return object()
        def fold_predict(model, heldout):
            return [CandidateProposal(row.received_morse,
                tuple(range(sum(ch in ".-" for ch in row.received_morse)))) for row in heldout]
        model, optimizer = self._model(), tf.keras.optimizers.Adam()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("morse_correction_v2.oof_runner.train_risk_head", return_value={"risk_loss": 0.0}):
            root = Path(directory)
            manifest = self._write_manifest(root, [make_example("cal-retry", "calibration", "HI", "clean", 42)],
                                            train_rows=rows)
            output = root / "risk-run"
            args = dict(folds=2, fold_epochs=1, risk_epochs=1, batch_size=8,
                        fold_fit=fold_fit, fold_predict=fold_predict)
            with self.assertRaisesRegex(RuntimeError, "fold interruption"):
                run_oof_risk_training(rows, model, optimizer, output, verified_manifest=manifest, **args)
            self.assertFalse(output.exists())
            generated, _ = run_oof_risk_training(rows, model, optimizer, output,
                                                  verified_manifest=manifest, **args)
            self.assertEqual(len(generated.examples), len(rows))

    def test_oof_checkpoint_interruption_leaves_no_partial_output_and_retry_recovers(self):
        import tensorflow as tf
        from morse_correction_v2.checkpointing import CheckpointStore
        rows = self._rows()
        fold_fit = lambda fit_rows, **kwargs: object()
        fold_predict = lambda model, held: [CandidateProposal(row.received_morse,
            tuple(range(sum(ch in ".-" for ch in row.received_morse)))) for row in held]
        model, optimizer = self._model(), tf.keras.optimizers.Adam()
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("morse_correction_v2.oof_runner.train_risk_head", return_value={"risk_loss": 0.0}):
            root = Path(directory)
            manifest = self._write_manifest(root, [make_example("cal-commit", "calibration", "HI", "clean", 43)],
                                            train_rows=rows)
            output = root / "risk-run"
            args = dict(folds=2, fold_epochs=1, risk_epochs=1, batch_size=8,
                        fold_fit=fold_fit, fold_predict=fold_predict)
            with mock.patch.object(CheckpointStore, "commit", side_effect=OSError("simulated checkpoint interruption")):
                with self.assertRaisesRegex(OSError, "checkpoint interruption"):
                    run_oof_risk_training(rows, model, optimizer, output, verified_manifest=manifest, **args)
            self.assertFalse(output.exists())
            run_oof_risk_training(rows, model, optimizer, output,
                                  verified_manifest=manifest, **args)
            self.assertTrue((output / "CURRENT").is_file())

    def _write_manifest(self, root, calibration_rows, train_rows=None):
        partitions = {
            "train": train_rows or [make_example("manifest-train", "train", "SOS", "clean", 51)],
            "validation": [make_example("manifest-val", "validation", "TEST", "clean", 52)],
            "calibration": calibration_rows,
        }
        entries = {}
        for name, rows in partitions.items():
            raw = [asdict(row) for row in rows]
            (root / f"{name}.jsonl").write_text("".join(
                json.dumps(row, sort_keys=True) + "\n" for row in raw), encoding="utf-8")
            entries[name] = {"content_sha256": _content_fingerprint(raw),
                             "source_ids": sorted({row.source_id for row in rows})}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps({"partition_content": entries}), encoding="utf-8")
        return verify_manifest_files(manifest_path)

    def test_calibration_runner_generates_cases_from_model_outputs_and_writes_report(self):
        from morse_correction_v2.calibration_runner import run_calibration
        rows = [make_example(f"cal-run-{i}", "calibration", "SOS HI", "clean", i)
                for i in range(2)]
        model = self._model()
        def proposer(model, received):
            return V2InferenceResult(received, received, received, False, "test_abstain", None,
                                     tuple(range(sum(ch in ".-" for ch in received))), (0.1, 0.2, 0.3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(root, rows)
            output = root / "calibration.json"
            report = run_calibration(model, rows, manifest, output,
                checkpoint_metadata={"risk_head_trained": True}, proposer=proposer)
            self.assertEqual(report["cases"], len(rows))
            self.assertTrue(report["selected_all_abstain"])
            self.assertEqual(report["model_weights_sha256"],
                             __import__("morse_correction_v2.calibration", fromlist=["fingerprint_model"]).fingerprint_model(model))
            with self.assertRaises(FileExistsError):
                run_calibration(model, rows, manifest, output,
                    checkpoint_metadata={"risk_head_trained": True}, proposer=proposer)

    def test_calibration_interruption_is_retryable_and_rejects_wrong_partition(self):
        from morse_correction_v2.calibration_runner import run_calibration
        calibration_rows = [make_example("cal-retry", "calibration", "SOS HI", "clean", 91)]
        model = self._model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(root, calibration_rows)
            output = root / "report.json"
            def interrupted(model, received):
                raise RuntimeError("simulated inference interruption")
            with self.assertRaisesRegex(RuntimeError, "inference interruption"):
                run_calibration(model, calibration_rows, manifest, output,
                    checkpoint_metadata={"risk_head_trained": True}, proposer=interrupted)
            self.assertFalse(output.exists())
            valid = lambda model, received: V2InferenceResult(
                received, received, received, False, "abstain", None, (), (0.2, 0.1, 0.0))
            run_calibration(model, calibration_rows, manifest, output,
                checkpoint_metadata={"risk_head_trained": True}, proposer=valid)
            wrong = [make_example("bad", "train", "HELLO", "clean", 4)]
            with self.assertRaisesRegex(ValueError, "calibration rows only"):
                run_calibration(model, wrong, manifest, root / "wrong.json",
                    checkpoint_metadata={"risk_head_trained": True}, proposer=valid)

    def test_calibration_rejects_malformed_candidate_without_report(self):
        from morse_correction_v2.calibration_runner import run_calibration
        rows = [make_example("cal-invalid", "calibration", "SOS HI", "clean", 92)]
        model = self._model()
        invalid = lambda model, received: V2InferenceResult(
            received, ".X", received, False, "proposal", None, (0, 1), (1.0, 0.0, 0.0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_manifest(root, rows)
            output = root / "report.json"
            with self.assertRaisesRegex(ValueError, "cover every verified calibration source"):
                run_calibration(model, rows, manifest, output,
                    checkpoint_metadata={"risk_head_trained": True}, proposer=invalid)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
