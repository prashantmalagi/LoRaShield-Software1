import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class V2CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"TensorFlow unavailable: {exc}")

    def _model_and_optimizer(self):
        import tensorflow as tf
        from morse_correction_v2.data import make_example
        from morse_correction_v2.batching import prepare_batch
        from morse_correction_v2.model import build_model
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        batch = prepare_batch([make_example("x", "train", "SOS HI", "clean", 4)])
        model(batch.inputs, training=False)
        optimizer = tf.keras.optimizers.Adam()
        optimizer.build(model.trainable_variables)
        return model, optimizer

    def test_checkpoint_restore_preserves_weights_and_refuses_overwrite(self):
        import numpy as np
        from morse_correction_v2.checkpointing import CheckpointStore
        model, optimizer = self._model_and_optimizer()
        before = [w.copy() for w in model.get_weights()]
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory) / "run")
            store.initialize(resume=False)
            store.commit(model=model, optimizer=optimizer, epoch=1, metadata={"tag": "test"})
            with self.assertRaises(FileExistsError):
                store.commit(model=model, optimizer=optimizer, epoch=1, metadata={})
            restored, restored_optimizer = self._model_and_optimizer()
            epoch, metadata = store.restore(model=restored, optimizer=restored_optimizer)
            self.assertEqual(epoch, 1)
            self.assertEqual(metadata["tag"], "test")
            for expected, actual in zip(before, restored.get_weights()):
                np.testing.assert_array_equal(expected, actual)
            with self.assertRaises(FileExistsError):
                store.initialize(resume=False)

    def test_interrupted_commit_keeps_previous_current_checkpoint_recoverable(self):
        import tensorflow as tf
        import os
        from morse_correction_v2.checkpointing import CheckpointStore
        model, optimizer = self._model_and_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory) / "run")
            store.initialize(resume=False)
            store.commit(model=model, optimizer=optimizer, epoch=1, metadata={"tag": "complete"})
            with mock.patch.object(tf.train.Checkpoint, "write", side_effect=OSError("simulated interruption")):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    store.commit(model=model, optimizer=optimizer, epoch=2, metadata={"tag": "partial"})
            self.assertEqual((store.root / "CURRENT").read_text(encoding="utf-8"), "epoch-000001")
            self.assertFalse((store.root / "epoch-000002").exists())
            restored, restored_optimizer = self._model_and_optimizer()
            epoch, metadata = store.restore(model=restored, optimizer=restored_optimizer)
            self.assertEqual((epoch, metadata["tag"]), (1, "complete"))

            real_replace = os.replace
            def fail_pointer_promotion(source, destination):
                if Path(source).name == ".CURRENT.tmp":
                    raise OSError("simulated pointer interruption")
                return real_replace(source, destination)
            with mock.patch("morse_correction_v2.checkpointing.os.replace",
                            side_effect=fail_pointer_promotion):
                with self.assertRaisesRegex(OSError, "pointer interruption"):
                    store.commit(model=model, optimizer=optimizer, epoch=2,
                                 metadata={"tag": "promoted-but-unpointed"})
            store.initialize(resume=True)
            self.assertEqual((store.root / "CURRENT").read_text(encoding="utf-8"), "epoch-000002")
            epoch, metadata = store.restore(model=restored, optimizer=restored_optimizer)
            self.assertEqual((epoch, metadata["tag"]), (2, "promoted-but-unpointed"))

    def test_restore_rejects_checkpoint_checksum_mismatch(self):
        from morse_correction_v2.checkpointing import CheckpointStore
        model, optimizer = self._model_and_optimizer()
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory) / "run")
            store.initialize(resume=False)
            checkpoint = store.commit(model=model, optimizer=optimizer, epoch=1, metadata={})
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data_name = next(name for name in manifest["files"] if name.endswith(".index"))
            (checkpoint / data_name).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                store.restore(model=model, optimizer=optimizer)


if __name__ == "__main__":
    unittest.main()
