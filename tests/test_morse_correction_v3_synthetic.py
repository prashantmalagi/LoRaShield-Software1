"""Synthetic-only tests; no receiver captures, hardware, or live services used."""

import unittest
import random
from types import SimpleNamespace

import numpy as np

import evaluate_morse_correction_v3_synthetic as evaluator


class IdentityModel:
    input_shape = (None, evaluator.SEQUENCE_LENGTH)
    output_shape = (None, evaluator.SEQUENCE_LENGTH, len(evaluator.VOCABULARY))

    def __init__(self):
        self.calls = 0
        self.last_batch = None

    def predict(self, batch, verbose=0):
        self.calls += 1
        self.last_batch = batch.copy()
        output = np.zeros((len(batch), evaluator.SEQUENCE_LENGTH,
                           len(evaluator.VOCABULARY)), dtype=np.float32)
        for row_index, row in enumerate(batch):
            for position, token_id in enumerate(row):
                output[row_index, position, token_id] = 1.0
        return output

    def get_layer(self, name):
        self.asserted_layer_name = name
        return SimpleNamespace(input_dim=len(evaluator.VOCABULARY), mask_zero=True)


class SyntheticEvaluationTests(unittest.TestCase):
    def test_source_corpus_is_deterministic_unique_and_alphanumeric(self):
        first = evaluator.generate_source_corpus(12345, 40)
        second = evaluator.generate_source_corpus(12345, 40)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(all(" " in source for source in first))
        self.assertTrue(all(char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
                            for source in first for char in source))
        self.assertNotEqual(first, evaluator.generate_source_corpus(12346, 40))
        self.assertTrue(all(not (set(source.split()) & evaluator._prior_source_messages())
                            for source in first))

    def test_noise_levels_use_exact_rounded_dot_dash_flip_counts(self):
        morse = "." * 50 + " / " + "-" * 50
        for level in evaluator.NOISE_LEVELS:
            with self.subTest(level=level):
                received = evaluator.apply_substitution_noise(
                    morse, level, random.Random(9876 + level)
                )
                changed = sum(left != right for left, right in zip(morse, received))
                self.assertEqual(changed, round(100 * level / 100))
                self.assertEqual(received.count(" / "), morse.count(" / "))
                self.assertEqual(received.replace(".", "").replace("-", ""),
                                 morse.replace(".", "").replace("-", ""))
                if level == 0:
                    self.assertEqual(received, morse)

    def test_case_generation_keeps_reference_separate_from_corruption(self):
        cases = evaluator.generate_cases(77, 20)
        repeated = evaluator.generate_cases(77, 20)
        for level in evaluator.NOISE_LEVELS:
            self.assertEqual(len(cases[level]), 20)
            self.assertTrue(all(case.noise_percent == level for case in cases[level]))
            self.assertEqual(cases[level], repeated[level])
        for source_case, clean_case in zip(cases[20], cases[0]):
            self.assertEqual(source_case.source_text, clean_case.source_text)
            self.assertEqual(source_case.reference_morse, clean_case.reference_morse)
            self.assertNotEqual(source_case.received_morse, source_case.reference_morse)

    def test_edit_distance_and_rates(self):
        self.assertEqual(evaluator.levenshtein("...", "..."), 0)
        self.assertEqual(evaluator.symbol_error_count("...", "..-"), 1)
        self.assertEqual(evaluator.symbol_error_count("...", "...."), 1)
        self.assertEqual(evaluator.symbol_error_count("....", "..."), 1)
        cases = [evaluator.EvaluationCase("A B", ".- / -...", ".. / -...", 5)]
        score = evaluator.score_condition(cases, [".- / -..."])
        self.assertEqual(score["raw_symbol_errors"], 1)
        self.assertEqual(score["corrected_symbol_errors"], 0)
        self.assertEqual(score["improved"], 1)
        self.assertAlmostEqual(score["raw_symbol_error_rate"], 1 / 6)
        self.assertEqual(score["corrected_symbol_error_rate"], 0)

    def test_word_boundary_positions_cover_placed_shifted_inserted_deleted(self):
        reference = ".- / -... -.-."
        self.assertEqual(evaluator.word_boundary_positions(reference), (1,))
        self.assertTrue(evaluator.word_boundaries_preserved(reference, reference))
        self.assertFalse(evaluator.word_boundaries_preserved(
            reference, ".- -... / -.-."
        ))
        self.assertFalse(evaluator.word_boundaries_preserved(
            ".- / -... -.-.", ".- / -... / -.-."
        ))
        self.assertFalse(evaluator.word_boundaries_preserved(
            ".- / -... / -.-.", ".- -... / -.-."
        ))

    def test_morse_input_validation_and_word_length(self):
        self.assertEqual(evaluator.split_words("... --- / - ."), ["... ---", "- ."])
        for bad in ("", " ...", "...  ---", "...//---", "... /  ---", "TEXT", "AA:BB"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluator.split_words(bad)
        self.assertEqual(len(evaluator.split_words("." * 120)[0]), 120)
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            evaluator.split_words("." * 121)

    def test_slash_words_are_split_before_inference_and_bad_batch_never_infers(self):
        model = IdentityModel()
        corrected = evaluator.predict_messages(model, ["... --- / - ."])
        self.assertEqual(corrected, ["... --- / - ."])
        self.assertEqual(model.last_batch.shape, (2, 120))
        self.assertNotIn(evaluator.TOKEN_IDS.get("/"), evaluator.TOKEN_IDS)
        calls_before = model.calls
        with self.assertRaises(ValueError):
            evaluator.predict_messages(model, ["... ---", "... / TEXT"])
        self.assertEqual(model.calls, calls_before)

    def test_model_metadata_and_shape_contract(self):
        model = IdentityModel()
        metadata = {
            "model_file": evaluator.MODEL_PATH.name,
            "sequence_length": evaluator.SEQUENCE_LENGTH,
            "vocabulary": evaluator.VOCABULARY,
            "token_ids": evaluator.TOKEN_IDS,
            "padding_id": evaluator.PAD_ID,
        }
        evaluator.validate_model_metadata(model, metadata)
        metadata["padding_id"] = 2
        with self.assertRaisesRegex(ValueError, "PAD ID"):
            evaluator.validate_model_metadata(model, metadata)

    def test_clean_preservation_and_condition_counts(self):
        model = IdentityModel()
        cases = evaluator.generate_cases(444, 12)
        results = evaluator.evaluate(model, cases)
        self.assertEqual(results["clean_message_preservation"], {
            "messages": 12, "exactly_preserved": 12, "rate": 1.0
        })
        clean = results["noise_0_percent"]
        self.assertEqual(clean["raw_symbol_errors"], 0)
        self.assertEqual(clean["corrected_symbol_errors"], 0)
        self.assertEqual((clean["improved"], clean["unchanged"], clean["worsened"]), (0, 12, 0))
        self.assertEqual(len(results), len(evaluator.NOISE_LEVELS) + 1)


if __name__ == "__main__":
    unittest.main()
