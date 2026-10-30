import unittest

from morse_correction_v2.decoding import decode_boundaries, decode_symbol_edits, map_boundary_prior
from morse_correction_v2.schema import boundary_targets


def one_hot(index, count=3):
    return [10.0 if i == index else -10.0 for i in range(count)]


class V2DecodingTests(unittest.TestCase):
    def test_exact_symbol_pass_through_and_alignment(self):
        source = ".-.-"
        symbol_logits = [one_hot(1), one_hot(2), one_hot(1), one_hot(2)]
        insertions = [[one_hot(0), one_hot(0)] for _ in range(len(source) + 1)]
        result = decode_symbol_edits(source, symbol_logits, insertions)
        self.assertEqual(result.symbols, source)
        self.assertEqual(result.alignment, (0, 1, 2, 3))
        self.assertEqual((result.insertions, result.deletions, result.substitutions), (0, 0, 0))

    def test_insertion_and_deletion_are_bounded_and_aligned(self):
        source = ".-"
        symbols = [one_hot(0), one_hot(2)]  # delete dot, keep dash
        insertions = [[one_hot(0), one_hot(0)], [one_hot(1), one_hot(0)],
                      [one_hot(0), one_hot(0)]]
        result = decode_symbol_edits(source, symbols, insertions)
        self.assertEqual(result.symbols, ".-")
        self.assertEqual(result.alignment, (-1, 1))
        self.assertEqual((result.insertions, result.deletions), (1, 1))
        with self.assertRaisesRegex(ValueError, "output limit"):
            decode_symbol_edits(".-", [one_hot(1), one_hot(2)],
                                [[one_hot(1), one_hot(1)] for _ in range(3)],
                                max_output_symbols=3)

    def test_boundary_decode_restores_clean_message_exactly(self):
        original = "... --- ... / .... .."
        labels = boundary_targets(original)
        logits = [one_hot(label) for label in labels]
        self.assertEqual(decode_boundaries("...---....... ..".replace(" ", ""), logits), original)

    def test_edge_boundaries_are_forced_off(self):
        logits = [one_hot(2), one_hot(2), one_hot(2)]
        self.assertEqual(decode_boundaries(".-", logits), ". / -")

    def test_invalid_dimensions_and_nonfinite_scores_reject(self):
        with self.assertRaises(ValueError):
            decode_symbol_edits(".-", [one_hot(1)], [])
        with self.assertRaises(ValueError):
            decode_symbol_edits(".-", [[float("nan"), 0, 0], one_hot(2)],
                                [[one_hot(0), one_hot(0)] for _ in range(3)])

    def test_boundary_prior_follows_symbol_alignment(self):
        original = ".- / ..."
        mapped = map_boundary_prior(original, (0, -1, 1, 2, 3, 4))
        self.assertEqual(len(mapped), 7)
        self.assertIn(2, mapped)


if __name__ == "__main__":
    unittest.main()
