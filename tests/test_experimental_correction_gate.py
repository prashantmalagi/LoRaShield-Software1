import unittest

from correction.experimental_gate import GatePolicy, InferenceEvidence, decide


class ExperimentalGateTests(unittest.TestCase):
    def setUp(self):
        self.evidence = InferenceEvidence(True, "eos")
        self.policy = GatePolicy(max_symbol_edits=4, max_boundary_edits=4)

    def test_exact_pass_through_rejects_noop(self):
        result = decide(".-", ".-", self.evidence, self.policy)
        self.assertEqual(result.output_morse, ".-")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no_change")

    def test_invalid_proposals_return_original(self):
        for proposal in (None, "", ".-- ", ".-//.-", ".  -"):
            with self.subTest(proposal=proposal):
                result = decide(".-", proposal, self.evidence, self.policy)
                self.assertEqual(result.output_morse, ".-")
                self.assertFalse(result.accepted)

    def test_malformed_boundaries_are_rejected(self):
        for proposal in (" .-", ".- ", ".- / ", ".-  / ."):
            with self.subTest(proposal=proposal):
                self.assertEqual(decide(".-", proposal, self.evidence, self.policy).reason,
                                 "invalid_proposal")

    def test_abstention_returns_original(self):
        evidence = InferenceEvidence(False, "max_output_tokens")
        result = decide(".-", "--", evidence, self.policy)
        self.assertFalse(result.accepted)
        self.assertEqual(result.output_morse, ".-")

    def test_rejected_proposal_preserves_clean_message(self):
        clean = ".- / -..."
        result = decide(clean, "-- / -...", self.evidence,
                        GatePolicy(max_symbol_edits=0, max_boundary_edits=0))
        self.assertFalse(result.accepted)
        self.assertEqual(result.output_morse, clean)


if __name__ == "__main__":
    unittest.main()
