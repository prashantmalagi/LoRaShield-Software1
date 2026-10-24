"""
LoRaShield – Morse Prefix Suggestion Tests
============================================
Tests for utils/morse_correction.prefix_suggest_morse

These tests validate the real-time Morse character prefix suggestion feature
that provides candidates as the sender types dot/dash symbols.

Run from the project root:
    python -m pytest tests/test_morse_prefix_suggest.py -v
    # or without pytest:
    python tests/test_morse_prefix_suggest.py

Test categories
---------------
1.  Empty prefix → empty result.
2.  Single dot '.' → all characters whose Morse code starts with '.'.
3.  Single dash '-' → all characters whose Morse code starts with '-'.
4.  '.-' → characters whose codes start with '.-'.
5.  '...' → characters whose codes start with '...'.
6.  '.-.' → characters whose codes start with '.-.'.
7.  Filtering: typing additional symbol reduces candidate set.
8.  No invalid prefix matches (every returned morse starts with prefix).
9.  Exact match flagging (exact=True when prefix IS a complete Morse code).
10. Completed Morse character ('.' → E is exact, but other candidates exist).
11. Whitespace-only prefix → empty result.
12. Prefix with spaces → stripped and matched (or empty).
13. Invalid characters in prefix (non ./- ) → empty result.
14. Very long/nonexistent prefix → empty result.
15. All characters have at least one prefix level that returns them.
16. Sorted order: exact matches first, then by length, then alphabetically.
17. Regression: existing CRC and correction imports unaffected.
"""

import sys
import os
import unittest

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.morse_correction import (
    prefix_suggest_morse,
    MORSE_DICT,
    _MORSE_REVERSE,
)


# Helpers

def _chars(results):
    return [r['char'] for r in results]

def _morse_codes(results):
    return [r['morse'] for r in results]

def _exact_chars(results):
    return [r['char'] for r in results if r['exact']]


# 1. Empty prefix

class TestEmptyPrefix(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(prefix_suggest_morse(''), [])

    def test_whitespace_only(self):
        self.assertEqual(prefix_suggest_morse('   '), [])

    def test_tab_only(self):
        self.assertEqual(prefix_suggest_morse('\t'), [])


# 2. Single dot prefix

class TestSingleDotPrefix(unittest.TestCase):
    def setUp(self):
        self.results = prefix_suggest_morse('.')
        self.chars = _chars(self.results)
        self.codes = _morse_codes(self.results)

    def test_returns_results(self):
        self.assertGreater(len(self.results), 0)

    def test_all_codes_start_with_dot(self):
        for code in self.codes:
            self.assertTrue(code.startswith('.'), f"Code {code!r} does not start with '.'")

    def test_e_is_present(self):
        self.assertIn('E', self.chars)

    def test_i_is_present(self):
        self.assertIn('I', self.chars)

    def test_a_is_present(self):
        self.assertIn('A', self.chars)

    def test_s_is_present(self):
        self.assertIn('S', self.chars)

    def test_h_is_present(self):
        self.assertIn('H', self.chars)

    def test_no_t(self):
        self.assertNotIn('T', self.chars)

    def test_no_m(self):
        self.assertNotIn('M', self.chars)

    def test_e_is_exact(self):
        self.assertIn('E', _exact_chars(self.results))

    def test_no_invalid_morse_in_results(self):
        for code in self.codes:
            self.assertIn(code, _MORSE_REVERSE, f"Code {code!r} is not a valid ITU Morse code")


# 3. Single dash prefix

class TestSingleDashPrefix(unittest.TestCase):
    def setUp(self):
        self.results = prefix_suggest_morse('-')
        self.chars = _chars(self.results)
        self.codes = _morse_codes(self.results)

    def test_returns_results(self):
        self.assertGreater(len(self.results), 0)

    def test_all_codes_start_with_dash(self):
        for code in self.codes:
            self.assertTrue(code.startswith('-'), f"Code {code!r} does not start with '-'")

    def test_t_is_present(self):
        self.assertIn('T', self.chars)

    def test_n_is_present(self):
        self.assertIn('N', self.chars)

    def test_m_is_present(self):
        self.assertIn('M', self.chars)

    def test_d_is_present(self):
        self.assertIn('D', self.chars)

    def test_no_e(self):
        self.assertNotIn('E', self.chars)

    def test_no_s(self):
        self.assertNotIn('S', self.chars)

    def test_t_is_exact(self):
        self.assertIn('T', _exact_chars(self.results))

    def test_no_invalid_morse_in_results(self):
        for code in self.codes:
            self.assertIn(code, _MORSE_REVERSE, f"Code {code!r} is not a valid ITU Morse code")


# 4. '.-' prefix

class TestDotDashPrefix(unittest.TestCase):
    def setUp(self):
        self.results = prefix_suggest_morse('.-')
        self.chars = _chars(self.results)
        self.codes = _morse_codes(self.results)

    def test_returns_results(self):
        self.assertGreater(len(self.results), 0)

    def test_all_codes_start_with_prefix(self):
        for code in self.codes:
            self.assertTrue(code.startswith('.-'), f"Code {code!r} does not start with '.-'")

    def test_a_is_present(self):
        self.assertIn('A', self.chars)

    def test_r_is_present(self):
        self.assertIn('R', self.chars)

    def test_l_is_present(self):
        self.assertIn('L', self.chars)

    def test_w_is_present(self):
        self.assertIn('W', self.chars)

    def test_no_e(self):
        self.assertNotIn('E', self.chars)

    def test_no_s(self):
        self.assertNotIn('S', self.chars)

    def test_no_t(self):
        self.assertNotIn('T', self.chars)

    def test_a_is_exact(self):
        self.assertIn('A', _exact_chars(self.results))

    def test_r_is_not_exact(self):
        for r in self.results:
            if r['char'] == 'R':
                self.assertFalse(r['exact'], "R should not be exact for prefix '.-'")
                break

    def test_no_invalid_morse_in_results(self):
        for code in self.codes:
            self.assertIn(code, _MORSE_REVERSE, f"Code {code!r} is not a valid ITU Morse code")


# 5. '...' prefix

class TestTripleDotPrefix(unittest.TestCase):
    def setUp(self):
        self.results = prefix_suggest_morse('...')
        self.chars = _chars(self.results)
        self.codes = _morse_codes(self.results)

    def test_returns_results(self):
        self.assertGreater(len(self.results), 0)

    def test_all_codes_start_with_prefix(self):
        for code in self.codes:
            self.assertTrue(code.startswith('...'), f"Code {code!r} does not start with '...'")

    def test_s_is_present(self):
        self.assertIn('S', self.chars)

    def test_h_is_present(self):
        self.assertIn('H', self.chars)

    def test_v_is_present(self):
        self.assertIn('V', self.chars)

    def test_5_is_present(self):
        self.assertIn('5', self.chars)

    def test_no_e(self):
        self.assertNotIn('E', self.chars)

    def test_no_i(self):
        self.assertNotIn('I', self.chars)

    def test_s_is_exact(self):
        self.assertIn('S', _exact_chars(self.results))

    def test_no_invalid_morse_in_results(self):
        for code in self.codes:
            self.assertIn(code, _MORSE_REVERSE, f"Code {code!r} is not a valid ITU Morse code")


# 6. '.-.' prefix

class TestDotDashDotPrefix(unittest.TestCase):
    def setUp(self):
        self.results = prefix_suggest_morse('.-.')
        self.chars = _chars(self.results)
        self.codes = _morse_codes(self.results)

    def test_returns_results(self):
        self.assertGreater(len(self.results), 0)

    def test_all_codes_start_with_prefix(self):
        for code in self.codes:
            self.assertTrue(code.startswith('.-.'), f"Code {code!r} does not start with '.-.'")

    def test_r_is_present(self):
        self.assertIn('R', self.chars)

    def test_l_is_present(self):
        self.assertIn('L', self.chars)

    def test_a_not_present(self):
        # A = '.-' which is shorter than '.-.'; it does NOT start with '.-.'
        self.assertNotIn('A', self.chars)

    def test_r_is_exact(self):
        self.assertIn('R', _exact_chars(self.results))


# 7. Progressive filtering

class TestProgressiveFiltering(unittest.TestCase):
    def _count(self, prefix):
        return len(prefix_suggest_morse(prefix))

    def test_dot_to_dot_dot(self):
        self.assertLessEqual(self._count('..'), self._count('.'))

    def test_dot_dot_to_dot_dot_dot(self):
        self.assertLessEqual(self._count('...'), self._count('..'))

    def test_dash_to_dash_dash(self):
        self.assertLessEqual(self._count('--'), self._count('-'))

    def test_dot_dash_to_dot_dash_dot(self):
        self.assertLessEqual(self._count('.-.'), self._count('.-'))

    def test_longer_prefix_codes_correct(self):
        longer = prefix_suggest_morse('....')
        for r in longer:
            self.assertTrue(r['morse'].startswith('....'))


# 8. No invalid prefix matches

class TestNoInvalidPrefixMatches(unittest.TestCase):
    def _verify(self, prefix):
        for r in prefix_suggest_morse(prefix):
            self.assertTrue(
                r['morse'].startswith(prefix),
                f"Result {r['morse']!r} does not start with {prefix!r}"
            )

    def test_dot(self):         self._verify('.')
    def test_dash(self):        self._verify('-')
    def test_dot_dash(self):    self._verify('.-')
    def test_dash_dot(self):    self._verify('-.')
    def test_three_dots(self):  self._verify('...')
    def test_four_dashes(self): self._verify('----')
    def test_dot_dash_dot(self):self._verify('.-.')
    def test_dash_dash(self):   self._verify('--')


# 9 & 10. Exact match flagging

class TestExactMatchFlag(unittest.TestCase):
    def _check(self, prefix, expected_char):
        results = prefix_suggest_morse(prefix)
        exact = [r for r in results if r['exact']]
        self.assertEqual(len(exact), 1,
                         f"Expected 1 exact for {prefix!r}, got {len(exact)}: {exact}")
        self.assertEqual(exact[0]['char'], expected_char)
        self.assertEqual(exact[0]['morse'], prefix)

    def test_e_exact(self):    self._check('.', 'E')
    def test_t_exact(self):    self._check('-', 'T')
    def test_i_exact(self):    self._check('..', 'I')
    def test_a_exact(self):    self._check('.-', 'A')
    def test_m_exact(self):    self._check('--', 'M')
    def test_n_exact(self):    self._check('-.', 'N')
    def test_s_exact(self):    self._check('...', 'S')
    def test_r_exact(self):    self._check('.-.', 'R')
    def test_u_exact(self):
        results = prefix_suggest_morse('..-')
        exact = [r for r in results if r['exact']]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]['char'], 'U')


# 11-13. Invalid characters in prefix

class TestInvalidPrefixChars(unittest.TestCase):
    def test_letter_in_prefix(self):
        self.assertEqual(prefix_suggest_morse('a'), [])

    def test_digit_in_prefix(self):
        self.assertEqual(prefix_suggest_morse('1'), [])

    def test_slash_prefix(self):
        self.assertEqual(prefix_suggest_morse('/'), [])

    def test_space_with_dot(self):
        # '  .' strips to '.' which is valid
        results = prefix_suggest_morse('  .')
        self.assertGreater(len(results), 0)

    def test_mixed_valid_invalid(self):
        self.assertEqual(prefix_suggest_morse('.-a'), [])


# 14. Very long / nonexistent prefix

class TestLongPrefix(unittest.TestCase):
    def test_eight_dots_returns_empty(self):
        self.assertEqual(prefix_suggest_morse('........'), [])

    def test_seven_char_dollar_sign(self):
        results = prefix_suggest_morse('...-..-')
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['char'], '$')
        self.assertTrue(results[0]['exact'])

    def test_six_dashes_returns_empty(self):
        self.assertEqual(prefix_suggest_morse('------'), [])


# 15. Every character is reachable

class TestAllCharsReachable(unittest.TestCase):
    def test_all_characters_reachable_from_first_symbol(self):
        for char, morse in MORSE_DICT.items():
            first = morse[0]
            results = prefix_suggest_morse(first)
            self.assertIn(char, _chars(results),
                          f"{char!r} (code {morse!r}) not found under prefix {first!r}")

    def test_all_characters_exact_from_full_code(self):
        for char, morse in MORSE_DICT.items():
            results = prefix_suggest_morse(morse)
            self.assertIn(char, _exact_chars(results),
                          f"{char!r} not exact-matched for its own code {morse!r}")


# 16. Sort order

class TestSortOrder(unittest.TestCase):
    def test_exact_first_for_dot(self):
        results = prefix_suggest_morse('.')
        self.assertTrue(results[0]['exact'])
        self.assertEqual(results[0]['char'], 'E')

    def test_exact_first_for_dash(self):
        results = prefix_suggest_morse('-')
        self.assertTrue(results[0]['exact'])
        self.assertEqual(results[0]['char'], 'T')

    def test_a_before_r_for_dot_dash(self):
        results = prefix_suggest_morse('.-')
        idx_a = next((i for i, r in enumerate(results) if r['char'] == 'A'), None)
        idx_r = next((i for i, r in enumerate(results) if r['char'] == 'R'), None)
        self.assertIsNotNone(idx_a)
        self.assertIsNotNone(idx_r)
        self.assertLess(idx_a, idx_r)


# 17. CRC regression

class TestRegressionCRC(unittest.TestCase):
    def test_crc_importable(self):
        from utils.crc import calculate_crc, verify_crc, build_packet, parse_packet
        self.assertTrue(True)

    def test_crc_basic(self):
        from utils.crc import calculate_crc
        crc = calculate_crc("HELLO")
        self.assertIsInstance(crc, int)
        self.assertGreater(crc, 0)

    def test_morse_correction_all_exports(self):
        from utils.morse_correction import (
            correct_morse_character, suggest_morse_corrections,
            correct_morse_message, apply_correction,
            set_default_threshold, prefix_suggest_morse,
        )
        self.assertTrue(True)

    def test_existing_suggest_still_works(self):
        from utils.morse_correction import suggest_morse_corrections
        results = suggest_morse_corrections('....-', max_suggestions=3, threshold=0.6)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIn('char', r)
            self.assertIn('morse', r)
            self.assertIn('score', r)


# Model not required

class TestModelNotRequired(unittest.TestCase):
    def test_works_without_model(self):
        results = prefix_suggest_morse('.')
        self.assertGreater(len(results), 0)
        self.assertIn('E', _chars(results))

    def test_deterministic(self):
        r1 = prefix_suggest_morse('.-')
        r2 = prefix_suggest_morse('.-')
        self.assertEqual(r1, r2)


# No word completion

class TestNoWordCompletion(unittest.TestCase):
    def test_all_chars_are_single_characters(self):
        for prefix in ['.', '-', '.-', '...', '-.']:
            for r in prefix_suggest_morse(prefix):
                self.assertEqual(len(r['char']), 1,
                                 f"char {r['char']!r} for prefix {prefix!r} is not single character")


if __name__ == '__main__':
    unittest.main(verbosity=2)
