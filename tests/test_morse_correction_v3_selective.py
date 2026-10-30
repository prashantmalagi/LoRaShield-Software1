"""Tests for the offline V3 selective gate; fixtures and labels are synthetic."""

import inspect
import random
import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

import evaluate_morse_correction_v3_selective as evaluator


class EchoModel:
    input_shape = (None, evaluator.SEQUENCE_LENGTH)
    output_shape = (None, evaluator.SEQUENCE_LENGTH, len(evaluator.VOCABULARY))

    def __init__(self, substitutions=None):
        self.substitutions = substitutions or {}
        self.calls = 0
        self.last_batch = None

    def predict(self, batch, verbose=0):
        self.calls += 1
        self.last_batch = batch.copy()
        result = np.zeros((len(batch), evaluator.SEQUENCE_LENGTH,
                           len(evaluator.VOCABULARY)), dtype=np.float32)
        for row_index, row in enumerate(batch):
            for position, token_id in enumerate(row):
                original = {value: key for key, value in evaluator.TOKEN_IDS.items()}[int(token_id)]
                proposed = self.substitutions.get(original, original)
                proposed_id = evaluator.TOKEN_IDS[proposed]
                if proposed == original:
                    result[row_index, position, token_id] = 0.96
                    for other in range(len(evaluator.VOCABULARY)):
                        if other != token_id:
                            result[row_index, position, other] = 0.01
                else:
                    result[row_index, position, proposed_id] = 0.80
                    result[row_index, position, token_id] = 0.10
                    for other in range(len(evaluator.VOCABULARY)):
                        if other not in (token_id, proposed_id):
                            result[row_index, position, other] = 0.025
        return result

    def get_layer(self, _name):
        return SimpleNamespace(input_dim=len(evaluator.VOCABULARY), mask_zero=True)


def feature(received=".-.", proposed="...", *, margin=0.7,
            same_length=True, same_letter=True, same_word=True, consistent=True):
    change = evaluator.TokenChange(
        word_index=0, position_in_word=1, original_token="-", proposed_token=".",
        original_probability=0.1, proposed_probability=0.8,
        probability_margin=margin,
    )
    return evaluator.ProposalFeatures(
        received_morse=received,
        proposed_morse=proposed,
        changes=(change,),
        minimum_changed_token_margin=margin,
        mean_changed_token_margin=margin,
        maximum_changed_token_margin=margin,
        candidate_morse_consistent=consistent,
        same_character_length=same_length,
        same_letter_boundaries=same_letter,
        same_word_boundaries=same_word,
    )


class SelectiveEvaluationTests(unittest.TestCase):
    def test_source_partitions_are_deterministic_and_disjoint_before_noise(self):
        cal_a, test_a = evaluator.generate_partitions(31415, 12, 12)
        cal_b, test_b = evaluator.generate_partitions(31415, 12, 12)
        self.assertEqual((cal_a, test_a), (cal_b, test_b))
        self.assertEqual(len(set(cal_a)), len(cal_a))
        self.assertEqual(len(set(test_a)), len(test_a))
        self.assertFalse(set(cal_a) & set(test_a))
        self.assertTrue({w for msg in cal_a for w in msg.split()}.isdisjoint(
            {w for msg in test_a for w in msg.split()}))
        self.assertTrue(all(set(msg.split()).isdisjoint(evaluator.prior_source_words())
                            for msg in cal_a + test_a))

    def test_partition_before_variants_and_deterministic_substitution_levels(self):
        calibration, locked = evaluator.generate_partitions(2233, 8, 8)
        cal_cases = evaluator.generate_cases(calibration, 3322, "cal")
        lock_cases = evaluator.generate_cases(locked, 4433, "locked")
        self.assertEqual(len(cal_cases), 8 * len(evaluator.NOISE_LEVELS))
        self.assertEqual(len(lock_cases), 8 * len(evaluator.NOISE_LEVELS))
        repeated = evaluator.generate_cases(calibration, 3322, "cal")
        self.assertEqual(cal_cases, repeated)
        self.assertEqual({case.noise_percent for case in cal_cases}, set(evaluator.NOISE_LEVELS))
        for case in cal_cases:
            self.assertNotEqual(case.source_text, "")
            self.assertEqual(case.reference_morse,
                             evaluator.encode_alphanumeric(case.source_text))
            self.assertEqual(case.received_morse.count(" / "),
                             case.reference_morse.count(" / "))
        self.assertTrue({case.source_text for case in cal_cases}.isdisjoint(
            {case.source_text for case in lock_cases}))

    def test_probabilities_and_changed_positions_are_retained(self):
        model = EchoModel({"-": "."})
        result = evaluator.propose_one(model, ".-.")
        self.assertEqual(result.proposed_morse, "...")
        self.assertEqual(len(result.changes), 1)
        change = result.changes[0]
        self.assertEqual((change.position_in_word, change.original_token,
                          change.proposed_token), (1, "-", "."))
        self.assertAlmostEqual(change.proposed_probability, 0.8)
        self.assertAlmostEqual(change.original_probability, 0.1)
        self.assertAlmostEqual(change.probability_margin, 0.7)
        self.assertTrue(result.candidate_morse_consistent)

    def test_runtime_decision_has_no_reference_label_access(self):
        runtime_fields = {field.name for field in fields(evaluator.ProposalFeatures)}
        self.assertNotIn("reference_morse", runtime_fields)
        self.assertNotIn("reference_text", runtime_fields)
        self.assertEqual(set(inspect.signature(evaluator.decide).parameters),
                         {"features", "policy", "channel_profile"})
        decision = evaluator.decide(
            feature(), evaluator.GatePolicy(0.5, 0.5, 1), "dot_dash_substitution"
        )
        self.assertTrue(decision.accepted)

    def test_threshold_selection_accepts_calibration_only_not_locked_records(self):
        calibration_records = [
            evaluator.CalibrationRecord(feature(), "...", 5, "dot_dash_substitution"),
            evaluator.CalibrationRecord(
                evaluator.ProposalFeatures(
                    received_morse=".-.", proposed_morse=".-.", changes=(),
                    minimum_changed_token_margin=0.0, mean_changed_token_margin=0.0,
                    maximum_changed_token_margin=0.0, candidate_morse_consistent=True,
                    same_character_length=True, same_letter_boundaries=True,
                    same_word_boundaries=True,
                ), "...", 0, "clean"
            ),
        ]
        policy_a, report_a = evaluator.select_policy(calibration_records)
        policy_b, report_b = evaluator.select_policy(calibration_records)
        self.assertEqual((policy_a, report_a), (policy_b, report_b))
        self.assertEqual(tuple(inspect.signature(evaluator.select_policy).parameters),
                         ("calibration_records",))
        # A deliberately changed locked reference is not an input to selection.
        locked_a = evaluator.OfflineCase("lock-1", "R", ".-.", ".-.", 5,
                                         "dot_dash_substitution")
        locked_b = evaluator.OfflineCase("lock-1", "R", "...", ".-.", 5,
                                         "dot_dash_substitution")
        self.assertNotEqual(locked_a.reference_morse, locked_b.reference_morse)
        calibration_case = evaluator.OfflineCase(
            "cal-1", "C", "...", ".-.", 5, "dot_dash_substitution"
        )
        run_policy_a = evaluator.run_experiment(
            EchoModel(), [calibration_case], [locked_a]
        )[0]
        run_policy_b = evaluator.run_experiment(
            EchoModel(), [calibration_case], [locked_b]
        )[0]
        self.assertEqual(policy_a, evaluator.select_policy(calibration_records)[0])
        self.assertEqual(run_policy_a, run_policy_b)

    def test_zero_coverage_selection_is_reported_as_abstain_all(self):
        unchanged = evaluator.ProposalFeatures(
            received_morse=".-", proposed_morse=".-", changes=(),
            minimum_changed_token_margin=0.0, mean_changed_token_margin=0.0,
            maximum_changed_token_margin=0.0, candidate_morse_consistent=True,
            same_character_length=True, same_letter_boundaries=True,
            same_word_boundaries=True,
        )
        policy, report = evaluator.select_policy([
            evaluator.CalibrationRecord(unchanged, ".-", 0, "clean")
        ])
        self.assertEqual(report["selected_policy"], "abstain_all")
        self.assertEqual(report["accepted"], 0)
        self.assertEqual(policy.maximum_changes, 0)

    def test_structural_out_of_scope_profiles_abstain_without_inference(self):
        model = EchoModel({"-": "."})
        policy = evaluator.GatePolicy(0.0, 0.0, 4)
        for profile in ("dot_dash_insertion", "dot_dash_deletion", "letter_boundary_edit",
                        "word_boundary_edit", "unknown"):
            with self.subTest(profile=profile):
                decision = evaluator.runtime_gate(model, ".-.", policy, profile)
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.output_morse, ".-.")
                self.assertEqual(decision.reason, "out_of_scope_channel_profile")
        self.assertEqual(model.calls, 0)

    def test_unsupported_and_overlength_inputs_abstain_unchanged_before_inference(self):
        model = EchoModel({"-": "."})
        policy = evaluator.GatePolicy(0.0, 0.0, 4)
        for value in (".-X", ".- /  ...", "." * 121):
            with self.subTest(value=value):
                decision = evaluator.runtime_gate(
                    model, value, policy, "dot_dash_substitution"
                )
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.output_morse, value)
        self.assertEqual(model.calls, 0)

    def test_changed_letter_or_word_boundaries_and_inconsistent_candidate_abstain(self):
        policy = evaluator.GatePolicy(0.0, 0.0, 4)
        for proposal in (
            feature(received=".- ...", proposed="... ...", same_letter=False),
            feature(received=".- / ...", proposed="... / ...", same_word=False),
            feature(consistent=False),
            feature(same_length=False),
        ):
            decision = evaluator.decide(proposal, policy, "dot_dash_substitution")
            self.assertFalse(decision.accepted)
            self.assertEqual(decision.output_morse, proposal.received_morse)

    def test_abstentions_leave_input_unchanged_and_policy_reports_outcomes(self):
        policy = evaluator.GatePolicy(0.9, 0.9, 1)
        decision = evaluator.runtime_gate(
            EchoModel({"-": "."}), ".-.", policy, "dot_dash_substitution"
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.output_morse, decision.received_morse)

        case = evaluator.OfflineCase("locked-1", "R", ".-.", ".-.", 5,
                                     "dot_dash_substitution")
        report = evaluator.evaluate_locked([case], [feature()],
                                           evaluator.GatePolicy(0.5, 0.5, 1))
        self.assertEqual(report["accepted_corrections"], 1)
        self.assertEqual(report["accepted_outcomes"], {"worsened": 1})
        self.assertEqual(report["harmful_accepted"], 1)
        self.assertIsNotNone(report["harmful_accepted_wilson_95_percent"])
        self.assertEqual(report["metrics_by_corruption_type"]["dot_dash_substitution"][
            "raw_symbol_errors"], 0)
        self.assertEqual(report["metrics_by_corruption_type"]["dot_dash_substitution"][
            "selected_symbol_errors"], 1)

    def test_morse_length_and_word_boundaries_are_validated(self):
        self.assertEqual(evaluator.split_words(".- / -..."), [".-", "-..."])
        with self.assertRaisesRegex(ValueError, "no truncation"):
            evaluator.split_words("." * 121)
        reference = ".- / -..."
        self.assertNotEqual(evaluator.word_boundary_positions(reference),
                            evaluator.word_boundary_positions(".- -..."))


if __name__ == "__main__":
    unittest.main()
