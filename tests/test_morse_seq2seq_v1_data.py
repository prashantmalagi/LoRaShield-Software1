"""Small deterministic tests for the V1 seq2seq data foundation only."""

import inspect
import json
import random
import unittest
from pathlib import Path

import morse_seq2seq_v1_data as data


class MorseSeq2SeqDataTests(unittest.TestCase):
    def test_canonical_round_trip_for_letters_and_words(self):
        for text in ("SOS", "HELLO", "TEST", "A1 Z9"):
            with self.subTest(text=text):
                canonical = data.encode_text(text)
                encoded = data.encode_morse(canonical)
                self.assertEqual(data.decode_morse(encoded), canonical)
        self.assertEqual(data.decode_morse(data.encode_morse("... --- ...")),
                         "... --- ...")
        self.assertEqual(data.encode_text("SOS"), "... --- ...")

    def test_exhaustive_single_character_round_trip_for_a_to_z_and_digits(self):
        supported = tuple(data.MORSE)
        self.assertEqual(len(supported), 36)
        for char in supported:
            with self.subTest(char=char):
                reference = data.encode_text(char)
                self.assertEqual(data.decode_morse(data.encode_morse(reference)), reference)

    def test_empty_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            data.encode_morse("")
        with self.assertRaises(ValueError):
            data.decode_morse(())
        with self.assertRaises(ValueError):
            data.decode_morse(("<BOS>", "<EOS>"))

    def test_boundary_tokens_are_explicit_and_round_trip(self):
        tokens = data.encode_morse(".- -... / -.-.")
        self.assertEqual(tokens[0], "<BOS>")
        self.assertEqual(tokens[-1], "<EOS>")
        self.assertEqual(tokens.count("<LETTER>"), 1)
        self.assertEqual(tokens.count("<WORD>"), 1)
        self.assertEqual(data.decode_morse(tokens), ".- -... / -.-.")
        self.assertEqual(data.TOKEN_TO_ID["<PAD>"], 0)
        self.assertEqual(data.TOKEN_TO_ID["<UNK>"], 1)

    def test_padding_is_right_only_and_length_excludes_pad(self):
        short = data.encode_morse(".-")
        long = data.encode_morse("... --- ...")
        rows = data.pad_token_ids([short, long])
        self.assertEqual(len(rows[0]), len(long))
        self.assertEqual(rows[0][len(short):], [data.TOKEN_TO_ID["<PAD>"]] * (len(long) - len(short)))
        self.assertEqual(rows[0][:len(short)], [data.TOKEN_TO_ID[token] for token in short])

    def test_maximum_input_and_output_sequence_lengths(self):
        maximum = "." * (data.MAX_SEQUENCE_TOKENS - 2)
        encoded = data.encode_morse(maximum)
        self.assertEqual(len(encoded), data.MAX_SEQUENCE_TOKENS)
        self.assertEqual(data.decode_morse(encoded), maximum)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            data.encode_morse("." * (data.MAX_SEQUENCE_TOKENS - 1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            data.decode_morse(("<BOS>",) + ("<DOT>",) * (data.MAX_SEQUENCE_TOKENS - 1) + ("<EOS>",))

    def test_each_corruption_pairs_received_input_with_unchanged_clean_target(self):
        source = "TEST MESSAGE"
        reference = data.encode_text(source)
        for index, kind in enumerate(data.CORRUPTION_KINDS):
            with self.subTest(kind=kind):
                pair = data.make_training_pair(
                    "fixture-1", "train", source, kind, 12000 + index
                )
                self.assertEqual(pair.reference_morse, reference)
                self.assertEqual(data.decode_morse(pair.target_tokens), reference)
                self.assertEqual(data.decode_morse(pair.input_tokens), pair.received_morse)
                self.assertEqual(pair.partition, "train")
                self.assertLessEqual(len(pair.input_tokens), data.MAX_SEQUENCE_TOKENS)
                self.assertLessEqual(len(pair.target_tokens), data.MAX_SEQUENCE_TOKENS)
                if kind == "clean":
                    self.assertEqual(pair.received_morse, reference)
                else:
                    self.assertNotEqual(pair.received_morse, reference)

    def test_substitution_insertion_deletion_and_burst_are_reproducible(self):
        reference = data.encode_text("TEST MESSAGE")
        kinds = (
            "dot_dash_substitution_05", "dot_dash_substitution_10",
            "dot_dash_substitution_15", "dot_dash_substitution_20",
            "symbol_insertion", "symbol_deletion", "burst_substitution",
        )
        for kind in kinds:
            with self.subTest(kind=kind):
                first = data.corrupt_morse(reference, kind, random.Random(552211))
                second = data.corrupt_morse(reference, kind, random.Random(552211))
                self.assertEqual(first, second)
                self.assertNotEqual(first, reference)
                # Even if corruption makes a group an invalid Morse letter, its
                # raw dot/dash sequence remains tokenizable and target-aligned.
                self.assertEqual(data.decode_morse(data.encode_morse(first)), first)

    def test_letter_and_word_boundary_corruptions_are_structural(self):
        reference = data.encode_text("TEST MESSAGE")
        letter_kinds = (
            "letter_boundary_insertion", "letter_boundary_deletion",
            "letter_boundary_shift",
        )
        word_kinds = (
            "word_boundary_insertion", "word_boundary_deletion",
            "word_boundary_shift",
        )
        for index, kind in enumerate(letter_kinds + word_kinds):
            with self.subTest(kind=kind):
                corrupted = data.corrupt_morse(reference, kind, random.Random(88100 + index))
                self.assertNotEqual(corrupted, reference)
                self.assertEqual(data.decode_morse(data.encode_morse(corrupted)), corrupted)
                pair = data.make_training_pair(
                    "fixture-boundary", "validation", "TEST MESSAGE", kind, 88100 + index
                )
                self.assertEqual(pair.target_tokens, data.encode_morse(reference))

    def test_partition_generation_is_deterministic_and_disjoint(self):
        counts = {name: 3 for name in data.PARTITION_NAMES}
        exclusions = {"fixture_exclusions": ["AAAA BBBB", "CCCC DDDD"]}
        first = data.generate_source_partitions(331122, counts, exclusions)
        second = data.generate_source_partitions(331122, counts, exclusions)
        self.assertEqual(first, second)
        report = data.overlap_report(first, exclusions)
        self.assertEqual(report["complete_message_overlap_count"], 0)
        self.assertEqual(report["word_overlap_count"], 0)
        self.assertEqual(report["cross_partition_message_collision_count"], 0)
        self.assertEqual(report["cross_partition_word_collision_count"], 0)
        self.assertEqual({key: len(value) for key, value in first.items()}, counts)
        pairs = data.generate_partition_pairs(first, seed=331122)
        for partition in data.PARTITION_NAMES:
            self.assertEqual(len(pairs[partition]), counts[partition] * len(data.CORRUPTION_KINDS))
            self.assertTrue(all(pair.partition == partition for pair in pairs[partition]))
            for source_index in range(counts[partition]):
                variants = [pair for pair in pairs[partition]
                            if pair.source_id == f"{partition}-{source_index:06d}"]
                self.assertEqual(len(variants), len(data.CORRUPTION_KINDS))
                self.assertEqual(len({variant.reference_morse for variant in variants}), 1)

    def test_overlap_checker_detects_exact_and_word_collisions(self):
        report = data.overlap_report(
            {"train": ["ALPHA BETA"], "locked_test": ["GAMMA BETA", "ALPHA BETA"]},
            {"prior": ["ALPHA DELTA", "ALPHA BETA"]},
        )
        self.assertEqual(report["complete_message_overlap_count"], 2)
        self.assertEqual(report["word_overlap_count"], 5)
        self.assertGreater(report["cross_partition_message_collision_count"], 0)
        self.assertGreater(report["cross_partition_word_collision_count"], 0)

    def test_runtime_preprocessing_and_input_types_have_no_reference_field(self):
        parameters = inspect.signature(data.runtime_preprocess).parameters
        self.assertEqual(set(parameters), {"received_morse", "max_tokens"})
        self.assertEqual({field for field in data.RuntimeInput.__dataclass_fields__},
                         {"received_morse", "input_ids"})
        runtime_input = data.runtime_preprocess("... --- ...")
        self.assertEqual(runtime_input.received_morse, "... --- ...")
        self.assertEqual(runtime_input.input_ids[0], data.TOKEN_TO_ID["<BOS>"])
        with self.assertRaises(TypeError):
            data.runtime_preprocess("... --- ...", reference_morse="... --- ...")

    def test_existing_source_registry_and_manifest_are_reproducible(self):
        corpora = data.existing_source_corpora()
        required = {
            "v3_and_legacy_train_eval_sources", "v3_synthetic_eval",
            "v3_independent_eval", "v3_corruption_eval",
            "v3_selective_calibration", "v3_selective_locked_test",
        }
        self.assertEqual(set(corpora), required)
        small = data.generate_source_partitions(
            441122, {name: 1 for name in data.PARTITION_NAMES}, corpora
        )
        manifest = data.build_manifest(small, seed=441122, exclusions=corpora)
        self.assertEqual(manifest["partition_audit"]["source_counts"],
                         {name: 1 for name in data.PARTITION_NAMES})
        self.assertEqual(manifest["partition_audit"]["overlap_check"]["results"]["word_overlap_count"], 0)
        self.assertEqual(json.loads(data.manifest_json(manifest))["planned_settings"]["root_seed"], 441122)

        path = Path(__file__).resolve().parents[1] / "morse_seq2seq_v1_manifest.json"
        template = json.loads(path.read_text(encoding="utf-8"))
        def assert_schema(expected, actual, path="manifest"):
            if expected is None:  # Unmeasured template field; schema remains present.
                return
            if isinstance(expected, dict):
                self.assertIsInstance(actual, dict, path)
                self.assertEqual(set(expected), set(actual), path)
                for key in expected:
                    assert_schema(expected[key], actual[key], f"{path}.{key}")
            elif isinstance(expected, list):
                self.assertIsInstance(actual, list, path)
                self.assertEqual(len(expected), len(actual), path)
                for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
                    assert_schema(expected_item, actual_item, f"{path}[{index}]")
            else:
                self.assertNotIsInstance(actual, (dict, list), path)

        assert_schema(template, manifest)
        self.assertEqual(template["planned_settings"]["root_seed"], data.ROOT_SEED)
        self.assertEqual(template["partition_audit"]["measurement_scope"], "not_computed")
        self.assertIsNone(template["exclusion_audit"]["measured"]["unique_message_count"])
        self.assertIsNone(template["partition_audit"]["source_counts"]["locked_test"])
        expected_unique_messages = len({msg for rows in corpora.values() for msg in rows})
        self.assertEqual(manifest["exclusion_audit"]["measured"]["unique_message_count"],
                         expected_unique_messages)


if __name__ == "__main__":
    unittest.main()
