import unittest

from morse_correction_v2.batching import prepare_batch
from morse_correction_v2.data import make_example
from morse_correction_v2.model import build_model
from morse_correction_v2.losses import compute_loss_components


class V2ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"TensorFlow unavailable: {exc}")

    def test_heads_have_separate_symbol_boundary_clean_and_risk_shapes(self):
        example = make_example("train-0", "train", "SOS HI", "clean", 2)
        batch = prepare_batch([example])
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        output = model(batch.inputs, training=False)
        self.assertEqual(tuple(output["symbol_logits"].shape), (1, len(example.symbol_operations), 3))
        self.assertEqual(tuple(output["insertion_logits"].shape),
                         (1, len(example.symbol_operations) + 1, 2, 3))
        self.assertEqual(tuple(output["boundary_logits"].shape), (1, len(example.boundary_labels), 3))
        self.assertEqual(tuple(output["clean_logits"].shape), (1, 2))
        self.assertEqual(tuple(output["risk_logits"].shape), (1, 3))
        self.assertGreater(model.count_params(), 0)
        losses = compute_loss_components(output, batch.targets, batch.masks)
        self.assertTrue(all(float(value.numpy()) >= 0 for value in losses.values()))

    def test_model_dimension_validation(self):
        with self.assertRaises(ValueError):
            build_model(width=15, heads=4, feed_forward=32, layers=1, max_length=64)

    def test_risk_logits_depend_on_proposed_symbols_and_alignment(self):
        import numpy as np
        example = make_example("train-risk", "train", "SOS HI", "clean", 2)
        batch = prepare_batch([example])
        model = build_model(width=16, heads=4, feed_forward=32, layers=1, max_length=64)
        base = model(batch.inputs, training=False)["risk_logits"].numpy()
        changed = {key: value.copy() for key, value in batch.inputs.items()}
        changed["candidate_symbol_ids"][:] = 2
        changed["candidate_alignment"][:] = -1
        proposal = model(changed, training=False)["risk_logits"].numpy()
        self.assertFalse(np.allclose(base, proposal), "risk head ignored candidate proposal/alignment")


if __name__ == "__main__":
    unittest.main()
