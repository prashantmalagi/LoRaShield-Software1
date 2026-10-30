"""Unit tests for synthetic-only structured V3 corruption experiments."""

import random
import unittest
from types import SimpleNamespace

import numpy as np

import evaluate_morse_correction_v3_corruption as evaluator


class IdentityModel:
    input_shape = (None, evaluator.SEQUENCE_LENGTH)
    output_shape = (None, evaluator.SEQUENCE_LENGTH, len(evaluator.VOCABULARY))

    def __init__(self):
        self.calls = 0
        self.batch = None

    def predict(self, batch, verbose=0):
        self.calls += 1
        self.batch = batch.copy()
        output = np.zeros((len(batch), evaluator.SEQUENCE_LENGTH,
                           len(evaluator.VOCABULARY)), dtype=np.float32)
        for row_index, row in enumerate(batch):
            for position, token_id in enumerate(row):
                output[row_index, position, token_id] = 1.0
        return output

    def get_layer(self, _name):
        return SimpleNamespace(input_dim=len(evaluator.VOCABULARY), mask_zero=True)


class CorruptionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = evaluator.generate_source_corpus(evaluator.SEED, 30)[0]
        cls.reference = evaluator.encode_alphanumeric(cls.source)

    def test_corpus_and_every_condition_are_deterministic_and_disjoint(self):
        first = evaluator.generate_corpus(5001, 12)
        second = evaluator.generate_corpus(5001, 12)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(evaluator.CORRUPTION_TYPES))
        for kind in evaluator.CORRUPTION_TYPES:
            self.assertEqual(len(first[kind]), 12)
            self.assertTrue(all(case.corruption_type == kind for case in first[kind]))
        source_words = {word for case in first["clean"] for word in case.source_text.split()}
        self.assertTrue(source_words.isdisjoint(evaluator._excluded_source_words()))

    def test_dot_dash_substitution_changes_symbols_only(self):
        case = evaluator.make_corruption_case(
            self.source, "dot_dash_substitution", random.Random(30)
        )
        ref = case.reference_morse
        self.assertEqual(case.received_morse.count(" / "), ref.count(" / "))
        self.assertEqual(case.received_morse.replace(".", "").replace("-", ""),
                         ref.replace(".", "").replace("-", ""))
        self.assertGreater(evaluator.symbol_error_count(ref, case.received_morse), 0)

    def test_symbol_insertions_and_deletions_have_expected_length_effect(self):
        for kind, expected_delta in (("dot_dash_insertion", 1), ("dot_dash_deletion", -1)):
            with self.subTest(kind=kind):
                case = evaluator.make_corruption_case(self.source, kind, random.Random(50))
                self.assertEqual(len(evaluator.dot_dash_stream(case.received_morse))
                                 - len(evaluator.dot_dash_stream(case.reference_morse)),
                                 expected_delta)
                self.assertEqual(len(evaluator.split_words(case.received_morse)),
                                 len(evaluator.split_words(case.reference_morse)))

    def test_short_burst_is_two_or_three_consecutive_symbol_flips(self):
        case = evaluator.make_corruption_case(
            self.source, "short_burst_substitution", random.Random(61)
        )
        changed = [i for i, (left, right) in enumerate(
            zip(case.reference_morse, case.received_morse)
        ) if left != right]
        self.assertIn(len(changed), (2, 3))
        self.assertEqual(changed, list(range(changed[0], changed[0] + len(changed))))
        self.assertTrue(all(case.reference_morse[i] in ".-" for i in changed))

    def test_letter_boundary_edits_are_valid_and_keep_dot_dash_stream(self):
        for kind, expected_delta in (("letter_boundary_insertion", 1),
                                     ("letter_boundary_deletion", -1)):
            with self.subTest(kind=kind):
                case = evaluator.make_corruption_case(self.source, kind, random.Random(70))
                self.assertEqual(len(evaluator.dot_dash_stream(case.received_morse)),
                                 len(evaluator.dot_dash_stream(case.reference_morse)))
                self.assertEqual(len(case.received_morse) - len(case.reference_morse),
                                 expected_delta)
                evaluator.split_words(case.received_morse)

    def test_word_boundary_edits_are_valid_and_reported_as_unpreserved(self):
        for kind in ("word_boundary_insertion", "word_boundary_deletion", "word_boundary_shift"):
            with self.subTest(kind=kind):
                case = evaluator.make_corruption_case(self.source, kind, random.Random(88))
                evaluator.split_words(case.received_morse)
                self.assertFalse(evaluator.boundaries_preserved(
                    case.reference_morse, case.received_morse
                ))
                self.assertEqual(evaluator.dot_dash_stream(case.reference_morse),
                                 evaluator.dot_dash_stream(case.received_morse))

    def test_all_types_can_be_predicted_without_passing_slash(self):
        cases = evaluator.generate_corpus(91, 2)
        model = IdentityModel()
        for kind in evaluator.CORRUPTION_TYPES:
            outputs = evaluator.predict_messages(
                model, [case.received_morse for case in cases[kind]]
            )
            self.assertEqual(len(outputs), 2)
            self.assertTrue(np.isin(model.batch, list(evaluator.TOKEN_IDS.values())).all())
            expected_word_rows = sum(
                len(evaluator.split_words(case.received_morse)) for case in cases[kind]
            )
            self.assertEqual(model.batch.shape, (expected_word_rows, 120))
        self.assertNotIn("/", evaluator.TOKEN_IDS)

    def test_malformed_and_overlength_input_fails_before_model_call(self):
        model = IdentityModel()
        with self.assertRaises(ValueError):
            evaluator.predict_messages(model, ["... ---", "... /  ---"])
        self.assertEqual(model.calls, 0)
        with self.assertRaises(ValueError):
            evaluator.predict_messages(model, ["." * 121])
        self.assertEqual(model.calls, 0)
        with self.assertRaises(ValueError):
            evaluator.predict_messages(model, ["... / TEXT"])
        self.assertEqual(model.calls, 0)

    def test_metrics_outcomes_and_clean_baseline(self):
        ref = ".- / -..."
        cases = [
            evaluator.CorruptionCase("A B", ref, ".. / -...", "dot_dash_substitution"),
            evaluator.CorruptionCase("A B", ref, ref, "clean"),
            evaluator.CorruptionCase("A B", ref, ref, "clean"),
        ]
        outputs = [ref, ref, ".. / -..."]
        report = evaluator.score_condition(cases, outputs)
        self.assertEqual((report["improved"], report["unchanged"], report["worsened"]), (1, 1, 1))
        self.assertEqual(report["raw_symbol_errors"], 1)
        self.assertEqual(report["corrected_symbol_errors"], 1)
        self.assertEqual(report["raw_word_boundaries_preserved"], 3)
        self.assertEqual(report["corrected_word_boundaries_preserved"], 3)

        model = IdentityModel()
        generated = evaluator.generate_corpus(707, 10)
        results = evaluator.evaluate(model, generated)
        self.assertEqual(results["clean"]["exactly_preserved"], 10)
        self.assertEqual(results["clean"]["clean_preservation_rate"], 1.0)

    def test_fixed_output_contract_documents_length_correction_limit(self):
        for kind in ("dot_dash_insertion", "dot_dash_deletion",
                     "letter_boundary_insertion", "letter_boundary_deletion"):
            self.assertIn("cannot safely", evaluator.CAPABILITY_BY_TYPE[kind])
        self.assertIn("slash is not", evaluator.CAPABILITY_BY_TYPE["word_boundary_shift"])


if __name__ == "__main__":
    unittest.main()
