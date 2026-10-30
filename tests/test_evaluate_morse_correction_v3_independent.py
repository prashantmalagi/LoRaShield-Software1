"""Read-only tests for the independent V3 evaluation pipeline."""

from __future__ import annotations

import unittest

from evaluate_morse_correction_v3_independent import (
    DASH_ID,
    DOT_ID,
    FRESH_SEED,
    MODEL_PATH,
    METADATA_PATH,
    PAD_ID,
    SEQUENCE_LENGTH,
    UNK_ID,
    _load_artifacts,
    _old_partition_messages,
    assert_disjoint,
    boundary_edit_distance,
    decode_real_positions,
    edit_errors,
    encode_padded,
    make_fresh_corpus,
    run_self_tests,
    validate_metadata_and_model,
)


class TestBoundaryEditDistance(unittest.TestCase):
    def test_identical_and_single_substitution(self):
        self.assertEqual(boundary_edit_distance("... ---", "... ---"), 0)
        self.assertEqual(boundary_edit_distance("... ---", "... --."), 1)

    def test_inserting_letter_group_aligns_following_groups(self):
        self.assertEqual(boundary_edit_distance("... --- ...", "... .... --- ..."), 5)
        self.assertEqual(edit_errors("... --- ...", "... .... --- ...")[0], 5)

    def test_deleting_letter_group_aligns_following_groups(self):
        self.assertEqual(boundary_edit_distance("... .... --- ...", "... --- ..."), 5)

    def test_empty_reference_and_candidate(self):
        self.assertEqual(boundary_edit_distance("", "..."), 3)
        self.assertEqual(boundary_edit_distance("...", ""), 3)


class TestVocabularyAndPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = _load_artifacts()

    def test_artifact_paths_and_model_metadata_compatibility(self):
        self.assertTrue(MODEL_PATH.is_file())
        self.assertTrue(METADATA_PATH.is_file())
        validate_metadata_and_model(self.model, self.metadata)
        self.assertEqual(self.model.input_shape, (None, SEQUENCE_LENGTH))
        self.assertEqual(self.model.output_shape, (None, SEQUENCE_LENGTH, 5))

    def test_padding_ids_and_decoding_ignore_padded_predictions(self):
        encoded = encode_padded(".-")
        self.assertEqual(encoded[:2], [DOT_ID, DASH_ID])
        self.assertEqual(encoded[2:], [PAD_ID] * (SEQUENCE_LENGTH - 2))
        predictions = [DOT_ID, DASH_ID, UNK_ID] + [PAD_ID] * (SEQUENCE_LENGTH - 3)
        self.assertEqual(decode_real_positions(".-", predictions), ".-")
        self.assertEqual(decode_real_positions(".-", [PAD_ID, UNK_ID] + predictions[2:]), ".-")


class TestIndependentSourcePartitions(unittest.TestCase):
    def test_fresh_corpus_is_unique_and_excludes_prior_sources(self):
        fresh = make_fresh_corpus(FRESH_SEED, 500)
        prior = _old_partition_messages()
        self.assertEqual(len(fresh), len(set(fresh)))
        self.assertTrue(set(fresh).isdisjoint(prior))
        assert_disjoint({"prior": prior, "new_test": fresh})

    def test_overlap_is_rejected(self):
        with self.assertRaises(AssertionError):
            assert_disjoint({"train": ["AB12"], "test": ["AB12"]})


class TestSelfTestBundle(unittest.TestCase):
    def test_full_read_only_self_test_bundle(self):
        run_self_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)
