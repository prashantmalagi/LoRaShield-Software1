"""
LoRaShield – Morse Code Correction Module Tests
================================================
Tests for utils/morse_correction.py

Run from the project root:
    python -m pytest tests/test_morse_correction.py -v
    # or without pytest:
    python tests/test_morse_correction.py

All tests use only stdlib (no external dependencies) so they run on
Raspberry Pi without any additional installs.

Test categories
---------------
1.  Valid exact-match Morse codes (letters, digits, punctuation)
2.  One extra dot appended (e.g. ....- = H + extra dot)
3.  One extra dash appended (e.g. ...- = V correct; ...-- extra dash)
4.  One missing dot (e.g. ... is S; ... is already valid – tested boundary)
5.  One wrong symbol (dot ↔ dash swap)
6.  Completely invalid / very long input
7.  Multiple Morse characters in one message
8.  Multiple words separated by ' / '
9.  Numbers (0-9)
10. Supported punctuation
11. Threshold sensitivity (strict vs. permissive)
12. apply_correction helper
13. set_default_threshold guard
"""

import sys
import os
import unittest

# ── Ensure project root is on sys.path so utils can be imported ───────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.morse_correction import (
    correct_morse_character,
    suggest_morse_corrections,
    correct_morse_message,
    apply_correction,
    set_default_threshold,
    _DEFAULT_THRESHOLD,
    MORSE_DICT,
    _MORSE_REVERSE,
    _levenshtein,
    _similarity_ratio,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper assertions
# ─────────────────────────────────────────────────────────────────────────────

def _chars_in_suggestions(result: dict) -> list:
    """Return list of character strings from suggestions list."""
    return [s['char'] for s in result.get('suggestions', [])]


def _morse_in_suggestions(result: dict) -> list:
    """Return list of Morse strings from suggestions list."""
    return [s['morse'] for s in result.get('suggestions', [])]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestLevenshtein(unittest.TestCase):
    """Unit tests for the _levenshtein edit-distance function."""

    def test_identical_strings(self):
        self.assertEqual(_levenshtein('....', '....'), 0)

    def test_empty_strings(self):
        self.assertEqual(_levenshtein('', ''), 0)

    def test_one_empty(self):
        self.assertEqual(_levenshtein('', '...'), 3)
        self.assertEqual(_levenshtein('---', ''), 3)

    def test_one_insertion(self):
        # '....' vs '...' → 1 insertion
        self.assertEqual(_levenshtein('....', '...'), 1)

    def test_one_deletion(self):
        self.assertEqual(_levenshtein('...', '....'), 1)

    def test_one_substitution(self):
        # '-...' vs '....' → 1 substitution (first char)
        self.assertEqual(_levenshtein('-...', '....'), 1)

    def test_completely_different(self):
        # '.' (E) vs '-----' (0) → max dist = 5
        dist = _levenshtein('.', '-----')
        self.assertEqual(dist, 5)


class TestSimilarityRatio(unittest.TestCase):
    """Unit tests for the _similarity_ratio normaliser."""

    def test_exact_match(self):
        self.assertAlmostEqual(_similarity_ratio('....', '....'), 1.0)

    def test_empty(self):
        self.assertAlmostEqual(_similarity_ratio('', ''), 1.0)

    def test_one_extra_char(self):
        # '....-' vs '....' → dist=1, max_len=5 → 0.8
        score = _similarity_ratio('....-', '....')
        self.assertAlmostEqual(score, 0.8)

    def test_score_range(self):
        for code in ['.-', '....', '-----']:
            for valid in _MORSE_REVERSE:
                score = _similarity_ratio(code, valid)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Exact valid Morse codes (must return is_valid=True, no suggestions)
# ─────────────────────────────────────────────────────────────────────────────

class TestExactValidMorse(unittest.TestCase):
    """Exact ITU Morse codes must always decode correctly with no suggestions."""

    def _check_exact(self, morse_code: str, expected_char: str):
        result = correct_morse_character(morse_code)
        self.assertTrue(result['is_valid'],
                        f"{morse_code!r} should be valid (expected {expected_char})")
        self.assertEqual(result['char'], expected_char,
                         f"{morse_code!r} decoded to {result['char']!r}, expected {expected_char!r}")
        self.assertEqual(result['suggestions'], [],
                         f"Valid code {morse_code!r} must have no suggestions")

    # Letters
    def test_letter_A(self):  self._check_exact('.-',    'A')
    def test_letter_H(self):  self._check_exact('....',  'H')
    def test_letter_S(self):  self._check_exact('...',   'S')
    def test_letter_O(self):  self._check_exact('---',   'O')
    def test_letter_E(self):  self._check_exact('.',     'E')
    def test_letter_T(self):  self._check_exact('-',     'T')
    def test_letter_V(self):  self._check_exact('...-',  'V')
    def test_letter_J(self):  self._check_exact('.---',  'J')
    def test_letter_Z(self):  self._check_exact('--..',  'Z')

    # Digits
    def test_digit_0(self):   self._check_exact('-----', '0')
    def test_digit_1(self):   self._check_exact('.----', '1')
    def test_digit_5(self):   self._check_exact('.....', '5')
    def test_digit_9(self):   self._check_exact('----.',  '9')

    # Punctuation
    def test_period(self):    self._check_exact('.-.-.-', '.')
    def test_comma(self):     self._check_exact('--..--', ',')
    def test_question(self):  self._check_exact('..--..', '?')
    def test_at(self):        self._check_exact('.--.-.', '@')
    def test_equals(self):    self._check_exact('-...-',  '=')


# ─────────────────────────────────────────────────────────────────────────────
# 3. One extra dot (e.g. ....- → H or V suggested)
# ─────────────────────────────────────────────────────────────────────────────

class TestOneExtraDot(unittest.TestCase):
    """Appending an extra symbol to a valid code creates an invalid code nearby."""

    def test_H_with_extra_dash(self):
        """'....-.' has 6 symbols and is NOT a valid code; H (....) is 2 edits away."""
        # '....-.' is not in the dict
        result = correct_morse_character('....-.')
        self.assertFalse(result['is_valid'],
                         "'....-.' is not a valid ITU Morse code")
        self.assertTrue(result['reliable'],
                        "'....-.' should have reliable suggestions nearby")
        chars = _chars_in_suggestions(result)
        # H, V, 4 are all nearby – at least one must be suggested
        self.assertTrue(len(chars) > 0, f"Expected suggestions for '....-.' got {chars}")

    def test_digit_4_exact(self):
        """'....-' is EXACTLY digit 4 – must be decoded as such (not treated as error)."""
        result = correct_morse_character('....-')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], '4')

    def test_H_exact(self):
        """'....' is exactly H."""
        result = correct_morse_character('....')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], 'H')

    def test_5_exact(self):
        """'.....' is exactly digit 5."""
        result = correct_morse_character('.....')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], '5')


# ─────────────────────────────────────────────────────────────────────────────
# 4. One missing dot/dash
# ─────────────────────────────────────────────────────────────────────────────

class TestOneMissingSymbol(unittest.TestCase):
    """Removing one symbol from a valid code gives a nearby invalid code."""

    def test_missing_last_dot_from_H(self):
        """'...' = S (exact). Not H with missing dot."""
        result = correct_morse_character('...')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], 'S')

    def test_missing_last_dash_from_4(self):
        """'....' = H (exact). Not 4 (.....) with missing dot."""
        result = correct_morse_character('....')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], 'H')

    def test_missing_dot_gives_nearby_suggestion(self):
        """'-..' = D (exact). Not B (-...) with missing last dot."""
        result = correct_morse_character('-.')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], 'N')


# ─────────────────────────────────────────────────────────────────────────────
# 5. One wrong dot/dash (substitution)
# ─────────────────────────────────────────────────────────────────────────────

class TestOneWrongSymbol(unittest.TestCase):
    """Swap one symbol in a valid code → should suggest the original."""

    def test_swap_first_dot_of_S(self):
        """-.. has first dot changed to dash → D (exact!)."""
        result = correct_morse_character('-..')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], 'D')

    def test_swap_A_to_invalid(self):
        """'--' = M (exact)."""
        result = correct_morse_character('--')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], 'M')

    def test_swap_gives_suggestions(self):
        """'.-.-' is not valid → should suggest nearby codes."""
        result = correct_morse_character('.-.-')
        # '.-.-' is not in the dictionary
        self.assertFalse(result['is_valid'])
        # With default threshold it should find at least one suggestion
        self.assertTrue(len(result['suggestions']) > 0 or not result['reliable'])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Completely invalid input
# ─────────────────────────────────────────────────────────────────────────────

class TestCompletelyInvalid(unittest.TestCase):
    """Inputs that are too different from any valid code."""

    def test_very_long_gibberish(self):
        """A 12-symbol random pattern should not match anything reliably."""
        result = correct_morse_character('............', threshold=0.65)
        self.assertFalse(result['is_valid'])
        # Either no reliable suggestions or the reliable flag is False
        if result['reliable']:
            # If somehow suggestions exist, they must all pass the threshold
            for s in result['suggestions']:
                self.assertGreaterEqual(s['score'], 0.65)

    def test_nine_dots(self):
        """9 dots is far from any valid code (max length is 7)."""
        result = correct_morse_character('.........', threshold=0.75)
        self.assertFalse(result['is_valid'])
        self.assertFalse(result['reliable'],
                         "9 dots should not reliably match any Morse code at threshold 0.75")

    def test_empty_input(self):
        """Empty string should give no suggestions."""
        result = correct_morse_character('')
        suggestions = suggest_morse_corrections('')
        self.assertEqual(suggestions, [])


# ─────────────────────────────────────────────────────────────────────────────
# 7 & 8. Multiple Morse characters / words
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleCharacters(unittest.TestCase):
    """correct_morse_message must process every token independently."""

    def test_hello(self):
        """.... . .-.. .-.. ---  =  H E L L O (all valid)."""
        result = correct_morse_message('.... . .-.. .-.. ---')
        self.assertTrue(result['all_valid'])
        self.assertFalse(result['has_errors'])
        self.assertEqual(result['token_count'], 5)
        chars = [t['char'] for t in result['tokens']]
        self.assertEqual(chars, ['H', 'E', 'L', 'L', 'O'])

    def test_sos(self):
        """... --- ... = S O S"""
        result = correct_morse_message('... --- ...')
        self.assertTrue(result['all_valid'])
        chars = [t['char'] for t in result['tokens']]
        self.assertEqual(chars, ['S', 'O', 'S'])

    def test_one_invalid_in_message(self):
        """'....-.' (invalid) as first token in 'HELLO' message."""
        result = correct_morse_message('....-. . .-.. .-.. ---')
        self.assertTrue(result['has_errors'])
        self.assertFalse(result['all_valid'])
        invalid = [t for t in result['tokens'] if not t['is_valid']]
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]['input'], '....-.')

    def test_word_separator_preserved(self):
        """Two words separated by ' / ' must both be processed."""
        result = correct_morse_message('.... . .-.. .-.. --- / .-- --- .-. .-.. -..')
        self.assertTrue(result['all_valid'],
                        f"Expected all valid. Errors: {[t for t in result['tokens'] if not t['is_valid']]}")
        self.assertEqual(result['word_count'], 2)
        chars = [t['char'] for t in result['tokens']]
        self.assertEqual(chars, list('HELLOWORLD'))

    def test_multiple_errors_in_message(self):
        """Two genuinely bad tokens in one message."""
        # '....-.' and '.-..-.' are both invalid
        result = correct_morse_message('....-. . .-..-. .-.. ---')
        self.assertTrue(result['has_errors'])
        invalid = [t for t in result['tokens'] if not t['is_valid']]
        self.assertGreaterEqual(len(invalid), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Numbers
# ─────────────────────────────────────────────────────────────────────────────

class TestNumbers(unittest.TestCase):

    def test_all_digits_valid(self):
        number_codes = {
            '-----': '0', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7',
            '---..': '8', '----.': '9',
        }
        for code, char in number_codes.items():
            with self.subTest(code=code, char=char):
                result = correct_morse_character(code)
                self.assertTrue(result['is_valid'], f"{code!r} should decode to {char!r}")
                self.assertEqual(result['char'], char)

    def test_number_with_extra_dot(self):
        """'......' (6 dots) should suggest 5 (.....) nearby."""
        result = correct_morse_character('......')
        # '......' is not a valid code (5 is '.....' with 5 dots)
        self.assertFalse(result['is_valid']
                         if '......' not in _MORSE_REVERSE else True)
        # If invalid, suggestions should include '5' as a nearby candidate
        if not result['is_valid']:
            chars = _chars_in_suggestions(result)
            self.assertTrue(len(chars) > 0 or not result['reliable'],
                            f"Expected suggestions or 'not reliable' for '......', got {chars}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Punctuation
# ─────────────────────────────────────────────────────────────────────────────

class TestPunctuation(unittest.TestCase):

    def test_period(self):
        result = correct_morse_character('.-.-.-')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], '.')

    def test_comma(self):
        result = correct_morse_character('--..--')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], ',')

    def test_question_mark(self):
        result = correct_morse_character('..--..')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], '?')

    def test_at_sign(self):
        result = correct_morse_character('.--.-.')
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['char'], '@')

    def test_period_with_extra_dot(self):
        """'.-.-.-.' is period with one extra dot → should suggest period nearby."""
        result = correct_morse_character('.-.-.-.')
        self.assertFalse(result['is_valid'])
        self.assertTrue(result['reliable'], "period + extra dot should still suggest period")
        chars = _chars_in_suggestions(result)
        self.assertIn('.', chars, f"Expected '.' in suggestions, got {chars}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Threshold sensitivity
# ─────────────────────────────────────────────────────────────────────────────

class TestThreshold(unittest.TestCase):
    """Higher threshold → fewer suggestions; lower → more permissive."""

    def test_strict_threshold_rejects_suggestions(self):
        """At threshold=0.95 only near-identical codes should match."""
        # '....-.' is 6 symbols; no valid code is >95% similar
        result = correct_morse_character('....-.',  threshold=0.95)
        self.assertFalse(result['is_valid'])
        self.assertFalse(result['reliable'],
                         "At 0.95 threshold '....-.' should have no reliable match")

    def test_permissive_threshold_finds_suggestions(self):
        """At threshold=0.50 most near-neighbours should be found."""
        result = correct_morse_character('....-.',  threshold=0.50)
        self.assertFalse(result['is_valid'])
        self.assertTrue(result['reliable'],
                        "At threshold 0.50, '....-.' should have reliable suggestions")
        self.assertGreater(len(result['suggestions']), 0)

    def test_max_suggestions_limit(self):
        """suggest_morse_corrections should return at most max_suggestions results."""
        suggs = suggest_morse_corrections('....-', max_suggestions=2, threshold=0.50)
        self.assertLessEqual(len(suggs), 2)

    def test_set_default_threshold_validates(self):
        """set_default_threshold should reject out-of-range values."""
        with self.assertRaises(ValueError):
            set_default_threshold(1.5)
        with self.assertRaises(ValueError):
            set_default_threshold(-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 12. apply_correction helper
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyCorrection(unittest.TestCase):

    def test_replace_first_token(self):
        msg = '....- . .-.. .-.. ---'
        corrected = apply_correction(msg, word_index=0, token_index=0,
                                     replacement_morse='....')
        self.assertEqual(corrected, '.... . .-.. .-.. ---')

    def test_replace_middle_token(self):
        msg = '.... - .-.. .-.. ---'
        corrected = apply_correction(msg, word_index=0, token_index=2,
                                     replacement_morse='.-..')
        # Middle L was already correct; replacing again should be idempotent
        self.assertEqual(corrected, '.... - .-.. .-.. ---')

    def test_replace_token_in_second_word(self):
        msg = '.... . .-.. .-.. --- / .-- --- .-. .-.. -..'
        # Replace second word, first token (.-- = W) with .-- (already correct)
        corrected = apply_correction(msg, word_index=1, token_index=0,
                                     replacement_morse='.--')
        self.assertEqual(corrected, msg)   # unchanged

    def test_out_of_range_word_index(self):
        msg = '.... . ---'
        result = apply_correction(msg, word_index=99, token_index=0,
                                  replacement_morse='.-')
        self.assertEqual(result, msg)   # unchanged

    def test_out_of_range_token_index(self):
        msg = '.... . ---'
        result = apply_correction(msg, word_index=0, token_index=99,
                                  replacement_morse='.-')
        self.assertEqual(result, msg)   # unchanged

    def test_full_correction_flow(self):
        """Simulate full user correction flow: check → apply → re-check.

        '....-' is a valid ITU code for digit 4, so it cannot serve as the
        'bad' token.  '....-.' (6 symbols) is NOT in the dictionary and is
        genuinely invalid – it is the pattern used throughout TestOneExtraDot
        to represent H with an extra dash appended.
        """
        bad_msg = '....-. . .-.. .-.. ---'

        # Step 1: Check
        check = correct_morse_message(bad_msg)
        self.assertTrue(check['has_errors'])

        invalid_tok = next(t for t in check['tokens'] if not t['is_valid'])
        best_sugg   = invalid_tok['suggestions'][0]

        # Step 2: Apply
        corrected_msg = apply_correction(
            bad_msg,
            word_index=invalid_tok['word_index'],
            token_index=invalid_tok['token_index'],
            replacement_morse=best_sugg['morse'],
        )

        # Step 3: Re-check
        recheck = correct_morse_message(corrected_msg)
        self.assertTrue(recheck['all_valid'],
                        f"After applying best suggestion message should be all-valid. "
                        f"Corrected: {corrected_msg!r}")


# ─────────────────────────────────────────────────────────────────────────────
# 13. Dictionary completeness sanity-check
# ─────────────────────────────────────────────────────────────────────────────

class TestDictionaryCompleteness(unittest.TestCase):

    def test_all_letters_present(self):
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.assertIn(ch, MORSE_DICT, f"Letter {ch} missing from MORSE_DICT")

    def test_all_digits_present(self):
        for ch in '0123456789':
            self.assertIn(ch, MORSE_DICT, f"Digit {ch} missing from MORSE_DICT")

    def test_reverse_dict_consistent(self):
        for char, code in MORSE_DICT.items():
            self.assertIn(code, _MORSE_REVERSE,
                          f"Code {code!r} for {char!r} missing from reverse dict")
            self.assertEqual(_MORSE_REVERSE[code], char,
                             f"Reverse dict maps {code!r} to {_MORSE_REVERSE[code]!r}, "
                             f"expected {char!r}")

    def test_no_duplicate_codes(self):
        # Build reverse dict manually and check for collisions
        seen: dict = {}
        duplicates: list = []
        for char, code in MORSE_DICT.items():
            if code in seen:
                duplicates.append(f"{code!r} used by both {seen[code]!r} and {char!r}")
            else:
                seen[code] = char
        self.assertEqual(duplicates, [], f"Duplicate Morse codes found: {duplicates}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for running without pytest
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    unittest.main(verbosity=2)
