import unittest

import numpy as np

from morse_correction_v2.inference import propose
from morse_correction_v2.gate import GatePolicy


class FakeModel:
    def __init__(self, force_flip=False):
        self.force_flip = force_flip

    def __call__(self, inputs, training=False):
        symbols = inputs["symbol_ids"]
        batch, width = symbols.shape
        symbol_logits = np.full((batch, width, 3), -10.0, dtype=np.float32)
        for b in range(batch):
            for i, symbol_id in enumerate(symbols[b]):
                if symbol_id == 0:
                    symbol_logits[b, i, 0] = 10
                elif self.force_flip:
                    symbol_logits[b, i, 1 if symbol_id == 2 else 2] = 10
                else:
                    symbol_logits[b, i, int(symbol_id)] = 10
        gaps = inputs["input_gap_ids"].shape[1]
        insertions = np.full((batch, gaps, 2, 3), -10.0, dtype=np.float32)
        insertions[..., 0] = 10.0
        candidate_gaps = inputs["candidate_gap_ids"]
        boundary_logits = np.full((batch, candidate_gaps.shape[1], 3), -10.0, dtype=np.float32)
        for b in range(batch):
            for i, label in enumerate(candidate_gaps[b]):
                boundary_logits[b, i, int(label)] = 10.0
        return {"symbol_logits": symbol_logits, "insertion_logits": insertions,
                "boundary_logits": boundary_logits,
                "clean_logits": np.asarray([[0.0, 8.0]]),
                "risk_logits": np.asarray([[8.0, 0.0, 0.0]])}


class V2InferenceTests(unittest.TestCase):
    def test_clean_identity_proposal_is_exact_passthrough(self):
        received = "... --- / ..."
        result = propose(FakeModel(), received)
        self.assertEqual(result.output_morse, received)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "identity_pass_through")

    def test_changed_proposal_abstains_without_calibrators(self):
        received = ".- / -..."
        policy = GatePolicy(0.5, 0.2, 100, 100, 0.5, "calibration")
        result = propose(FakeModel(force_flip=True), received, policy=policy)
        self.assertIsNotNone(result.proposed_morse)
        self.assertNotEqual(result.proposed_morse, received)
        self.assertEqual(result.output_morse, received)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "uncalibrated_or_missing_risk")

    def test_malformed_received_input_is_returned_exactly(self):
        received = ".- / "
        result = propose(FakeModel(), received)
        self.assertEqual(result.output_morse, received)
        self.assertEqual(result.reason, "invalid_received_input")


if __name__ == "__main__":
    unittest.main()
