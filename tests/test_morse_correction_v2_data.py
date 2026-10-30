import unittest
from dataclasses import replace

from morse_correction_v2.batching import prepare_batch
from morse_correction_v2.data import (
    SourceGuard, corrupt_morse, derive_edit_labels, generate_sources,
    make_example, source_fingerprint,
)
from morse_correction_v2.schema import (
    CORRUPTION_KINDS, PARTITIONS, boundary_targets, encode_text, split_morse,
    preprocess_received, validate_morse,
)


class V2DataTests(unittest.TestCase):
    def setUp(self):
        self.guard = SourceGuard(frozenset(), frozenset())

    def test_canonical_parser_and_boundary_labels(self):
        self.assertEqual(boundary_targets(".- / ..."), (0, 0, 2, 0, 0, 0))
        self.assertEqual(len(boundary_targets("... ---")), 7)
        for malformed in ("", " .-", ".- ", ".-//...", ".-  / ...", ".- / "):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                validate_morse(malformed)

    def test_runtime_preprocessing_has_no_reference_field(self):
        prepared = preprocess_received(".- / ...")
        self.assertEqual(prepared.received_morse, ".- / ...")
        self.assertEqual(len(prepared.input_gap_ids), len(prepared.symbol_ids) + 1)
        self.assertNotIn("reference_morse", prepared.__dataclass_fields__)

    def test_all_corruption_families_produce_canonical_morse(self):
        reference = encode_text("HELLO WORLD AGAIN")
        for index, kind in enumerate(CORRUPTION_KINDS):
            with self.subTest(kind=kind):
                received = corrupt_morse(reference, kind, __import__("random").Random(index))
                validate_morse(received)
                split_morse(received)

    def test_edit_labels_align_clean_symbols_and_boundaries(self):
        example = make_example("train-0", "train", "SOS HI", "symbol_insertion", 18)
        target_symbols = sum(len(letter) for word in split_morse(example.reference_morse) for letter in word)
        self.assertEqual(len(example.output_alignment), target_symbols)
        self.assertEqual(len(example.symbol_operations), len("".join(c for c in example.received_morse if c in ".-")))
        self.assertEqual(len(example.boundary_labels), target_symbols + 1)
        self.assertTrue(any(-1 in example.output_alignment for _ in (0,)) or
                        len(example.output_alignment) <= len(example.symbol_operations))

    def test_insertions_and_deletions_are_explicit_in_alignment_labels(self):
        operations, insertions, alignment = derive_edit_labels(".-", ".--")
        self.assertEqual(len(alignment), 3)
        self.assertEqual(sum(value != 0 for gap in insertions for value in gap), 1)
        operations, insertions, alignment = derive_edit_labels(".--", ".-")
        self.assertEqual(sum(value == 0 for value in operations), 1)
        self.assertEqual(len(alignment), 2)

    def test_partitions_are_message_and_word_disjoint(self):
        counts = {name: 2 for name in PARTITIONS}
        seeds = {name: 100 + i for i, name in enumerate(PARTITIONS)}
        partitions = generate_sources(counts, seeds, self.guard)
        word_sets = {name: {word for source in sources for word in source.split()}
                     for name, sources in partitions.items()}
        for left_i, left in enumerate(PARTITIONS):
            for right in PARTITIONS[left_i + 1:]:
                self.assertFalse(word_sets[left] & word_sets[right])
                self.assertFalse(set(partitions[left]) & set(partitions[right]))
        self.assertEqual(len(source_fingerprint(partitions["train"])), 64)

    def test_hash_guard_excludes_messages_and_words_without_plaintext(self):
        import hashlib
        excluded = SourceGuard(
            frozenset({hashlib.sha256(b"ABC DEF").hexdigest()}),
            frozenset({hashlib.sha256(b"XYZ").hexdigest()}),
        )
        generated = generate_sources({name: 1 for name in PARTITIONS},
                                     {name: i + 10 for i, name in enumerate(PARTITIONS)}, excluded)
        for sources in generated.values():
            self.assertNotIn("ABC DEF", sources)
            self.assertNotIn("XYZ", " ".join(sources).split())

    def test_batch_masks_and_partition_isolation(self):
        example = make_example("train-0", "train", "SOS HI", "clean", 1)
        batch = prepare_batch([example])
        self.assertEqual(batch.targets["risk_labels"].tolist(), [-1])
        self.assertEqual(int(batch.masks["boundary"].sum()), len(example.boundary_labels))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_batch([example, replace(example, partition="validation")])
        with self.assertRaisesRegex(ValueError, "locked-test"):
            prepare_batch([replace(example, partition="locked_test")])


if __name__ == "__main__":
    unittest.main()
