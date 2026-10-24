"""
LoRaShield – Morse Code Typing Error Correction Module
=======================================================
Provides interactive correction of manually-entered Morse code patterns.

This module is completely standalone and works offline (no external libraries,
no internet connection required).  It is safe to use on Raspberry Pi.

Public API
----------
    correct_morse_character(morse_input, threshold=None)
        Check a single Morse token and return validity + suggestions.

    suggest_morse_corrections(morse_input, max_suggestions=3, threshold=None)
        Return the top-N closest valid Morse patterns sorted by similarity.

    correct_morse_message(message, threshold=None)
        Process a full Morse message string, respecting letter/word separators.

    set_default_threshold(value)
        Globally adjust how aggressively corrections are suggested.

Algorithm Summary
-----------------
For every Morse token the correction engine:
  1. Strips surrounding whitespace.
  2. Does an exact-match lookup in the ITU Morse reverse dictionary.
     → If found: reports valid, returns the decoded character, no suggestions.
  3. Computes the Levenshtein edit distance from the input to every valid
     Morse code in the dictionary.
  4. Converts each distance to a normalized similarity ratio:
         similarity = 1.0 - (edit_distance / max(len_input, len_valid))
     This makes the score independent of symbol length.
  5. Discards candidates whose similarity < threshold.
  6. Returns the top-N remaining candidates sorted by similarity descending.
  7. If no candidate passes the threshold, reports "no reliable correction".

Configurable Threshold
----------------------
Default: 0.65  (up to ~35 % of the pattern can differ and still be suggested).
Raise to 0.80 for stricter matching; lower to 0.50 to be more permissive.
Call set_default_threshold() or pass `threshold=` per call.

Separator Conventions (identical to utils/ai.py)
-------------------------------------------------
  Letter separator : single space  ' '
  Word separator   : ' / '
"""

# ---------------------------------------------------------------------------
# Logging – reuse the project-wide rotating-file logger
# ---------------------------------------------------------------------------
from utils.logger import get_logger as _get_logger   # noqa: E402

_log = _get_logger(__name__)

# ---------------------------------------------------------------------------
# Complete ITU Morse dictionary  (A-Z, 0-9, common punctuation)
# Keep this self-contained so the module has no dependency on utils/ai.py.
# ---------------------------------------------------------------------------
MORSE_DICT: dict[str, str] = {
    # Letters
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    # Digits
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..",  "9": "----.",
    # Punctuation
    ".":  ".-.-.-",  ",": "--..--",  "?": "..--..",  "'": ".----.",
    "!":  "-.-.--",  "/": "-..-.",   "(": "-.--.",    ")": "-.--.-",
    "&":  ".-...",   ":": "---...",  ";": "-.-.-.",   "=": "-...-",
    "+":  ".-.-.",   "-": "-....-",  '"': ".-..-.",
    "$":  "...-..-", "@": ".--.-.",
}

# Reverse map: Morse pattern → character (built once at import time)
_MORSE_REVERSE: dict[str, str] = {v: k for k, v in MORSE_DICT.items()}

# ---------------------------------------------------------------------------
# Default similarity threshold (0.0 – 1.0).
# Raise for stricter; lower for more permissive suggestions.
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLD: float = 0.65


def set_default_threshold(value: float) -> None:
    """
    Globally set the default similarity threshold used when no per-call
    `threshold` argument is supplied.

    Args:
        value: Float in [0.0, 1.0].  0.65 is a sensible default.

    Raises:
        ValueError: if value is outside [0.0, 1.0].
    """
    global _DEFAULT_THRESHOLD
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {value}")
    _DEFAULT_THRESHOLD = value


# ---------------------------------------------------------------------------
# Core distance / similarity helpers
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Uses the single-row dynamic-programming optimisation so memory usage is
    O(min(|s1|, |s2|)) rather than O(|s1| × |s2|).

    Operations counted: insertion, deletion, substitution (each costs 1).

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Non-negative integer edit distance.
    """
    # Ensure s1 is the shorter string for the memory optimisation
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    m, n = len(s1), len(s2)

    # dp[j] = edit distance between s1[:i] and s2[:j]
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                # Characters match – no operation needed
                dp[j] = prev
            else:
                # Minimum of insert, delete, substitute
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def _similarity_ratio(input_code: str, valid_code: str) -> float:
    """
    Compute a normalized similarity score in [0.0, 1.0].

    Formula:
        similarity = 1.0 - (edit_distance / max(len_input, len_valid))

    A score of 1.0 means an exact match; 0.0 means completely unrelated
    (every character must change).  Normalising by the longer string makes
    the score comparable across Morse codes of different lengths (e.g.,
    'E' = '.' vs 'O' = '---').

    Args:
        input_code: The user-supplied Morse pattern.
        valid_code: A valid Morse pattern from the dictionary.

    Returns:
        Float in [0.0, 1.0].
    """
    max_len = max(len(input_code), len(valid_code))
    if max_len == 0:
        return 1.0  # Both empty → identical
    dist = _levenshtein(input_code, valid_code)
    return 1.0 - (dist / max_len)


# ---------------------------------------------------------------------------
# Public correction API
# ---------------------------------------------------------------------------

def suggest_morse_corrections(
    morse_input: str,
    max_suggestions: int = 3,
    threshold: float | None = None,
) -> list[dict]:
    """
    Find the closest valid Morse patterns for a given (possibly invalid) input.

    The function computes the normalized similarity ratio between `morse_input`
    and every entry in the Morse dictionary, then returns the top-N results
    whose similarity score meets the threshold.

    Args:
        morse_input:     A single Morse token (e.g. ``"....-"``).
        max_suggestions: Maximum number of suggestions to return (default 3).
        threshold:       Minimum similarity score (0.0–1.0) a candidate must
                         reach to be included.  Falls back to the module-level
                         default when ``None``.

    Returns:
        A list of dicts, sorted by ``score`` descending, each with keys:
            ``char``  – the decoded character (e.g. ``"H"``)
            ``morse`` – the valid Morse pattern (e.g. ``"...."``)
            ``score`` – normalized similarity float (e.g. ``0.89``)

        Returns an empty list when no candidate meets the threshold.
    """
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    code = morse_input.strip()
    if not code:
        return []

    candidates: list[dict] = []

    for valid_morse, char in _MORSE_REVERSE.items():
        score = _similarity_ratio(code, valid_morse)
        if score >= threshold:
            candidates.append({"char": char, "morse": valid_morse, "score": score})

    # Sort by score descending; break ties by preferring shorter Morse patterns
    candidates.sort(key=lambda c: (-c["score"], len(c["morse"])))

    top = candidates[:max_suggestions]
    if top:
        summary = ", ".join(f"{s['char']}({s['morse']}) {s['score']:.2f}" for s in top)
        _log.debug("[Correction] suggest %r  →  [%s]  (threshold=%.2f)", code, summary, threshold)
    else:
        _log.debug("[Correction] suggest %r  →  no matches above threshold %.2f", code, threshold)
    return top


def correct_morse_character(
    morse_input: str,
    threshold: float | None = None,
) -> dict:
    """
    Validate a single Morse token and, if invalid, suggest corrections.

    Decoding strategy:
      1. Exact ITU dictionary match → valid, no correction needed.
      2. Similarity search → up to 3 suggestions above the threshold.
      3. Below threshold → no reliable correction found.

    Args:
        morse_input: A single Morse token (dots and dashes only, e.g. ``"....-"``).
        threshold:   Minimum similarity threshold.  Uses module default if None.

    Returns:
        A dict with keys:
            ``input``       – the stripped input string
            ``is_valid``    – True when the input is a valid ITU Morse code
            ``char``        – decoded character when ``is_valid`` is True, else None
            ``suggestions`` – list of suggestion dicts (see suggest_morse_corrections)
            ``reliable``    – True when at least one suggestion exceeds the threshold
                              (always True when ``is_valid`` is True)
    """
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    code = morse_input.strip()

    # ── Step 1: Exact match ───────────────────────────────────────────────────
    if code in _MORSE_REVERSE:
        _log.debug("[Correction] %r  →  exact match '%s'", code, _MORSE_REVERSE[code])
        return {
            "input":       code,
            "is_valid":    True,
            "char":        _MORSE_REVERSE[code],
            "suggestions": [],
            "reliable":    True,
        }

    # ── Step 2: Similarity search ─────────────────────────────────────────────
    _log.info("[Correction] %r is not a valid Morse code – running similarity search.", code)
    suggestions = suggest_morse_corrections(code, max_suggestions=3, threshold=threshold)

    # ── Step 3: Determine reliability ─────────────────────────────────────────
    reliable = len(suggestions) > 0
    if not reliable:
        _log.info("[Correction] %r  →  no reliable correction found (threshold=%.2f).", code, threshold)

    return {
        "input":       code,
        "is_valid":    False,
        "char":        None,
        "suggestions": suggestions,
        "reliable":    reliable,
    }


def correct_morse_message(
    message: str,
    threshold: float | None = None,
) -> dict:
    """
    Validate and correct a full Morse message, preserving all separators.

    The message is split into words (separated by ``' / '``) and then into
    individual letter tokens (separated by single spaces).  Each token is
    validated independently.  Spaces are preserved in the output so that
    corrected tokens can be re-joined into a valid Morse string.

    Example
    -------
    Input : ``".... . .-.. .-.. ---"``
    Output: all tokens valid → ``H E L L O``

    Input : ``"....- . .-.. .-.. ---"``
    Output: first token invalid → suggests ``H (….)`` and ``V (…-)``

    Args:
        message:   Full Morse string, letters separated by spaces, words by
                   ``' / '``  (same convention as ``utils.ai.encode_text``).
        threshold: Minimum similarity threshold.  Uses module default if None.

    Returns:
        A dict with keys:
            ``tokens``     – list of per-token result dicts (see correct_morse_character).
                             Each dict additionally has ``word_index`` and ``token_index``
                             to identify position in the original message.
            ``has_errors`` – True when at least one token is invalid.
            ``all_valid``  – True when every token is valid (inverse of ``has_errors``).
            ``word_count`` – number of words (slash-separated groups).
            ``token_count``– total number of Morse tokens processed.
    """
    if threshold is None:
        threshold = _DEFAULT_THRESHOLD

    message = message.strip()
    if not message:
        return {
            "tokens":      [],
            "has_errors":  False,
            "all_valid":   True,
            "word_count":  0,
            "token_count": 0,
        }

    # Split into words on ' / ', then each word into letter tokens on ' '
    words = message.split(" / ")
    token_results: list[dict] = []
    has_errors = False

    for word_idx, word in enumerate(words):
        # Each word may contain multiple letter tokens separated by a space
        tokens = word.strip().split(" ")
        for tok_idx, tok in enumerate(tokens):
            tok = tok.strip()
            if not tok:
                # Skip empty tokens (e.g. double spaces)
                continue

            result = correct_morse_character(tok, threshold=threshold)

            # Annotate with positional information for the UI
            result["word_index"]  = word_idx
            result["token_index"] = tok_idx

            token_results.append(result)

            if not result["is_valid"]:
                has_errors = True

    error_count = sum(1 for t in token_results if not t["is_valid"])
    _log.info(
        "[Correction] message check: %d token(s), %d error(s). has_errors=%s",
        len(token_results), error_count, has_errors,
    )

    return {
        "tokens":      token_results,
        "has_errors":  has_errors,
        "all_valid":   not has_errors,
        "word_count":  len(words),
        "token_count": len(token_results),
    }



# ---------------------------------------------------------------------------
# Prefix-based character suggestion  (new – Morse typing assistance)
# ---------------------------------------------------------------------------

def prefix_suggest_morse(prefix: str) -> list[dict]:
    """
    Return all ITU Morse characters whose code starts with *prefix*.

    This is a pure deterministic prefix-filter over the full Morse dictionary.
    It is used to provide real-time character suggestions while the sender is
    typing a single Morse symbol (before they press space / complete the char).

    The BiLSTM model (v4) is **not** used here because it:
      • Expects a full tokenised Morse *string* (multiple characters).
      • Performs seq2seq decoding over complete message sequences.
      • Cannot meaningfully rank candidates for a single incomplete prefix —
        the model's input vocab has only 6 tokens {'<PAD>','<UNK>','.','-',' ','/'}
        and its output is a full text sequence, not a per-character probability
        over dot/dash candidates.
    To enable AI ranking of incomplete Morse prefixes, the model would need to
    be retrained as a character-level trie classifier (input: prefix tokens,
    output: softmax over all 36+ Morse characters matching that prefix), which
    is a fundamentally different architecture from the current seq2seq decoder.

    Args:
        prefix: Morse prefix being typed, e.g. ``"."`` or ``".-"`` or ``"..."``.
                Only ``"."`` and ``"-"`` characters are significant; leading/
                trailing whitespace is stripped.

    Returns:
        List of dicts sorted by Morse code length (ascending), then
        alphabetically, each with:
            ``char``  – the decoded character (e.g. ``"A"``)
            ``morse`` – the complete valid Morse code (e.g. ``".-"``)
            ``exact`` – True when *prefix* is itself a complete valid code
                        (i.e. this entry is an exact match, not just a
                        prefix match)

        Returns an empty list for an empty prefix or a prefix that matches
        no character.
    """
    prefix = prefix.strip()
    if not prefix:
        return []

    # Only dots and dashes are valid Morse symbols
    valid_chars = set(".-")
    if any(ch not in valid_chars for ch in prefix):
        return []

    candidates: list[dict] = []
    for morse_code, char in _MORSE_REVERSE.items():
        if morse_code.startswith(prefix):
            candidates.append({
                "char":  char,
                "morse": morse_code,
                "exact": morse_code == prefix,
            })

    # Sort: exact matches first, then by code length, then alphabetically
    candidates.sort(key=lambda c: (not c["exact"], len(c["morse"]), c["char"]))

    _log.debug(
        "[PrefixSuggest] prefix=%r  →  %d candidate(s): %s",
        prefix,
        len(candidates),
        ", ".join(f"{c['char']}({c['morse']})" for c in candidates[:8]),
    )
    return candidates


def apply_correction(
    message: str,
    word_index: int,
    token_index: int,
    replacement_morse: str,
) -> str:
    """
    Apply a correction to one token in a full Morse message and return the
    updated message string.

    This is a helper for GUI layers: after the user selects a suggestion, call
    this function to produce the corrected message string ready to be written
    back to the input field.

    Args:
        message:           The original Morse message string.
        word_index:        0-based index of the word (slash-separated group).
        token_index:       0-based index of the token within that word.
        replacement_morse: The valid Morse code to substitute in (e.g. ``"...."``)

    Returns:
        The corrected Morse message string with the same separator structure.
    """
    words = message.strip().split(" / ")

    if word_index >= len(words):
        # Word index out of range – return message unchanged
        return message

    tokens = words[word_index].strip().split(" ")

    if token_index >= len(tokens):
        # Token index out of range – return message unchanged
        return message

    tokens[token_index] = replacement_morse
    words[word_index] = " ".join(tokens)

    return " / ".join(words)
