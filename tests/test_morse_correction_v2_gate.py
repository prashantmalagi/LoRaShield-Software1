import unittest

from morse_correction_v2.calibration import (
    RiskRecord, derive_risk_label, fit_temperature, select_gate_policy,
)
from morse_correction_v2.gate import GateEvidence, GatePolicy, decide


class V2GateTests(unittest.TestCase):
    def test_rejection_always_returns_exact_received_morse(self):
        original = ".- / -..."
        evidence = GateEvidence(True, True, 1, 0, 0.1, 0.8, 0.1, 0.1)
        decision = decide(original, "-- / -...", evidence, GatePolicy())
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.output_morse, original)

    def test_clean_or_uncertain_input_abstains_and_preserves_bytes(self):
        original = "... --- / ..."
        evidence = GateEvidence(True, True, 1, 0, 0.95, 0.99, 0.005, 0.005)
        policy = GatePolicy(0.9, 0.05, 2, 1, 0.2, "calibration")
        decision = decide(original, "... --- / -", evidence, policy)
        self.assertEqual(decision.reason, "clean_or_uncertain_input")
        self.assertEqual(decision.output_morse, original)

    def test_invalid_proposal_and_malformed_boundaries_abstain(self):
        original = ".-"
        evidence = GateEvidence(True, True, 1, 1, 0.1, 0.99, 0.005, 0.005)
        policy = GatePolicy(0.9, 0.05, 2, 2, 0.2, "calibration")
        for proposal in (None, ".- / ", " .-", ".-  / ..."):
            with self.subTest(proposal=proposal):
                decision = decide(original, proposal, evidence, policy)
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.output_morse, original)

    def test_missing_uncalibrated_evidence_abstains(self):
        evidence = GateEvidence(True, True, 1, 0, None, None, None, None)
        decision = decide(".-", "--", evidence, GatePolicy(0.5, 0.1, 2, 1, 0.5, "calibration"))
        self.assertEqual(decision.reason, "uncalibrated_or_missing_risk")
        self.assertEqual(decision.output_morse, ".-")

    def test_valid_policy_can_accept_only_with_calibrated_risk(self):
        evidence = GateEvidence(True, True, 1, 0, 0.01, 0.96, 0.02, 0.02)
        policy = GatePolicy(0.9, 0.05, 2, 1, 0.1, "calibration")
        result = decide(".-", "--", evidence, policy)
        self.assertTrue(result.accepted)
        self.assertEqual(result.output_morse, "--")

    def test_temperature_fit_is_calibration_only(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            fit_temperature([[2, 0]], [0], partition="locked_test")
        fitted = fit_temperature([[2, 0], [0, 2]], [0, 1], partition="calibration")
        self.assertEqual(fitted.fitted_on, "calibration")
        self.assertAlmostEqual(sum(fitted.probabilities([1, 2])), 1.0)

    def test_threshold_selection_rejects_noncalibration_records(self):
        evidence = GateEvidence(True, True, 1, 0, 0.01, 0.96, 0.02, 0.02)
        record = RiskRecord("locked_test", ".-", "--", ".-", evidence, "substitution")
        with self.assertRaisesRegex(ValueError, "calibration"):
            select_gate_policy([record], partition="locked_test")

    def test_calibration_with_only_unsafe_changes_selects_abstain_all(self):
        evidence = GateEvidence(True, True, 1, 0, 0.01, 0.96, 0.02, 0.02)
        record = RiskRecord("calibration", ".-", "--", ".-", evidence, "clean")
        policy, report = select_gate_policy([record], partition="calibration")
        self.assertEqual(report["selected"], "abstain_all")
        decision = decide(record.received_morse, record.proposed_morse,
                          record.evidence, policy)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.output_morse, record.received_morse)

    def test_calibration_policy_can_select_safe_nonzero_coverage(self):
        evidence = GateEvidence(True, True, 1, 0, 0.01, 0.96, 0.02, 0.02)
        record = RiskRecord("calibration", ".-", "--", "--", evidence, "substitution")
        policy, report = select_gate_policy([record], partition="calibration")
        self.assertEqual(report["selected"], "calibration_policy")
        decision = decide(record.received_morse, record.proposed_morse,
                          record.evidence, policy)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.output_morse, record.reference_morse)

    def test_risk_labels_compare_candidate_to_unchanged_input(self):
        self.assertEqual(derive_risk_label(".-", "--", ".-"), 0)
        self.assertEqual(derive_risk_label(".-", ".-", ".-"), 1)
        self.assertEqual(derive_risk_label(".-", ".-", "--"), 2)


if __name__ == "__main__":
    unittest.main()
