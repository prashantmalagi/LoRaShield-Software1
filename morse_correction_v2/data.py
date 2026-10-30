"""Deterministic source-first synthetic data and edit/boundary labels.

This module never opens V1/V3 manifests or locked-test labels. Callers must
provide a custodian-produced hash guard when generating corpus sources.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Mapping, Sequence

from .schema import (
    CHARSET, CORRUPTION_KINDS, PARTITIONS,
    V2Example, boundary_targets, encode_text, split_morse, validate_morse,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceGuard:
    """Hashes only; plaintext from excluded corpora is never needed here."""
    excluded_message_sha256: frozenset[str]
    excluded_word_sha256: frozenset[str]

    @classmethod
    def from_json(cls, path: str) -> "SourceGuard":
        raw = json.loads(open(path, encoding="utf-8").read())
        messages = raw.get("excluded_message_sha256", [])
        words = raw.get("excluded_word_sha256", [])
        if any(not _is_sha256(x) for x in messages + words):
            raise ValueError("guard hashes must be lowercase SHA-256 hex strings")
        return cls(frozenset(messages), frozenset(words))


def extend_source_guard(guard: SourceGuard, sources: Sequence[str]) -> SourceGuard:
    """Return a hash-only guard extended with known nonlocked source messages."""
    messages = set(guard.excluded_message_sha256)
    words = set(guard.excluded_word_sha256)
    for source in sources:
        words_in_source = source.split(" ")
        messages.add(sha256_text(source))
        words.update(sha256_text(word) for word in words_in_source)
    return SourceGuard(frozenset(messages), frozenset(words))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def render_words(words: Sequence[Sequence[str]]) -> str:
    if not words or any(not word for word in words) or any(not letter for word in words for letter in word):
        raise ValueError("corruption may not create empty words or letters")
    return " / ".join(" ".join("".join(letter) for letter in word) for word in words)


def corrupt_morse(reference: str, corruption_type: str, rng: random.Random,
                  *, severity: float = 0.1) -> str:
    """Apply one tagged corruption; boundaries remain syntactically valid."""
    if corruption_type not in CORRUPTION_KINDS:
        raise ValueError(f"unknown corruption type: {corruption_type}")
    if not 0 <= severity <= 1:
        raise ValueError("severity must be between zero and one")
    if corruption_type == "clean":
        return reference
    if corruption_type == "combined":
        first = rng.choice(("substitution", "symbol_insertion", "symbol_deletion",
                            "letter_boundary_shift", "word_boundary_shift"))
        intermediate = corrupt_morse(reference, first, rng, severity=severity)
        second = rng.choice(("substitution", "burst_substitution", "symbol_insertion",
                             "symbol_deletion", "letter_boundary_shift"))
        try:
            return corrupt_morse(intermediate, second, rng, severity=severity)
        except ValueError:
            return intermediate
    words = split_morse(reference)
    symbols = [(wi, li, si) for wi, word in enumerate(words)
               for li, letter in enumerate(word) for si in range(len(letter))]
    if corruption_type == "substitution":
        count = max(1, round(len(symbols) * severity)) if symbols and severity else 0
        for wi, li, si in rng.sample(symbols, min(count, len(symbols))):
            words[wi][li][si] = "-" if words[wi][li][si] == "." else "."
    elif corruption_type == "burst_substitution":
        choices = [(wi, li, letter) for wi, word in enumerate(words)
                   for li, letter in enumerate(word) if len(letter) >= 2]
        if not choices:
            return corrupt_morse(reference, "substitution", rng, severity=severity)
        wi, li, letter = rng.choice(choices)
        count = min(len(letter), max(2, round(len(letter) * max(severity, 0.5))))
        start = rng.randrange(len(letter) - count + 1)
        for si in range(start, start + count):
            words[wi][li][si] = "-" if words[wi][li][si] == "." else "."
    elif corruption_type == "symbol_insertion":
        wi, li = rng.choice([(wi, li) for wi, word in enumerate(words) for li in range(len(word))])
        at = rng.randrange(len(words[wi][li]) + 1)
        words[wi][li].insert(at, rng.choice(".-"))
    elif corruption_type == "symbol_deletion":
        candidates = [(wi, li, si) for wi, word in enumerate(words) for li, letter in enumerate(word)
                      if len(letter) > 1 for si in range(len(letter))]
        if not candidates:
            return reference
        wi, li, si = rng.choice(candidates)
        del words[wi][li][si]
    elif corruption_type == "letter_boundary_insertion":
        candidates = [(wi, li, split) for wi, word in enumerate(words)
                      for li, letter in enumerate(word) for split in range(1, len(letter))]
        if not candidates:
            return reference
        wi, li, split = rng.choice(candidates)
        letter = words[wi][li]
        words[wi][li:li + 1] = [letter[:split], letter[split:]]
    elif corruption_type == "letter_boundary_deletion":
        candidates = [(wi, li) for wi, word in enumerate(words) for li in range(len(word) - 1)]
        if not candidates:
            return reference
        wi, li = rng.choice(candidates)
        words[wi][li:li + 2] = [words[wi][li] + words[wi][li + 1]]
    elif corruption_type == "letter_boundary_shift":
        candidates = [(wi, li) for wi, word in enumerate(words) for li in range(len(word) - 1)
                      if len(word[li]) > 1 or len(word[li + 1]) > 1]
        if not candidates:
            return reference
        wi, li = rng.choice(candidates)
        left, right = words[wi][li:li + 2]
        merged = left + right
        split = len(left) + rng.choice((-1, 1))
        split = min(max(1, split), len(merged) - 1)
        words[wi][li:li + 2] = [merged[:split], merged[split:]]
    elif corruption_type == "word_boundary_insertion":
        candidates = [(wi, split) for wi, word in enumerate(words) for split in range(1, len(word))]
        if not candidates:
            return reference
        wi, split = rng.choice(candidates)
        words[wi:wi + 1] = [words[wi][:split], words[wi][split:]]
    elif corruption_type == "word_boundary_deletion":
        if len(words) < 2:
            return reference
        wi = rng.randrange(len(words) - 1)
        words[wi:wi + 2] = [words[wi] + words[wi + 1]]
    elif corruption_type == "word_boundary_shift":
        choices = [(wi, side) for wi in range(len(words) - 1)
                   for side, can_move in (("right", len(words[wi]) > 1),
                                          ("left", len(words[wi + 1]) > 1)) if can_move]
        if not choices:
            return reference
        wi, side = rng.choice(choices)
        moved = words[wi].pop() if side == "right" else words[wi + 1].pop(0)
        (words[wi + 1].insert(0, moved) if side == "right" else words[wi].append(moved))
    result = render_words(words)
    validate_morse(result)
    return result


def derive_edit_labels(received: str, reference: str, *, max_insertions_per_gap: int = 2):
    """Deterministic Levenshtein alignment; ties prefer substitution, delete, insert."""
    source = "".join(ch for ch in received if ch in ".-")
    target = "".join(ch for ch in reference if ch in ".-")
    n, m = len(source), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][m] = n - i
    for j in range(m + 1):
        dp[n][j] = m - j
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = min((source[i] != target[j]) + dp[i + 1][j + 1],
                           1 + dp[i + 1][j], 1 + dp[i][j + 1])
    symbol_ops = [None] * n
    insertions = [[] for _ in range(n + 1)]
    alignment = []
    i = j = 0
    while i < n or j < m:
        if i < n and j < m and dp[i][j] == (source[i] != target[j]) + dp[i + 1][j + 1]:
            symbol_ops[i] = 1 if target[j] == "." else 2
            alignment.append(i)
            i += 1; j += 1
        elif i < n and dp[i][j] == 1 + dp[i + 1][j]:
            symbol_ops[i] = 0
            i += 1
        else:
            insertions[i].append(1 if target[j] == "." else 2)
            alignment.append(-1)
            j += 1
    if any(len(gap) > max_insertions_per_gap for gap in insertions):
        raise ValueError("target requires too many consecutive insertions for configured decoder")
    padded = tuple(tuple(gap + [0] * (max_insertions_per_gap - len(gap))) for gap in insertions)
    return tuple(symbol_ops), padded, tuple(alignment)


def make_example(source_id: str, partition: str, source_text: str, corruption_type: str,
                 seed: int, *, severity: float = 0.1,
                 max_insertions_per_gap: int = 2) -> V2Example:
    if partition not in PARTITIONS:
        raise ValueError("training API accepts train/validation/calibration only")
    reference = encode_text(source_text)
    received = corrupt_morse(reference, corruption_type, random.Random(seed), severity=severity)
    validate_morse(received)
    sym_ops, insertions, alignment = derive_edit_labels(
        received, reference, max_insertions_per_gap=max_insertions_per_gap)
    if len(alignment) != len("".join(c for c in reference if c in ".-")):
        raise AssertionError("edit alignment must account for each reference symbol")
    return V2Example(source_id, partition, corruption_type, severity, seed, source_text,
                     received, reference, sym_ops, insertions, alignment,
                     boundary_targets(reference), corruption_type == "clean")


def generate_sources(partition_counts: Mapping[str, int], seeds: Mapping[str, int],
                     guard: SourceGuard) -> dict[str, list[str]]:
    """Generate disjoint train/validation/calibration plaintext sources only."""
    if set(partition_counts) != set(PARTITIONS) or set(seeds) != set(PARTITIONS):
        raise ValueError(f"partitions must be exactly {PARTITIONS}; locked-test is separate")
    if not isinstance(guard, SourceGuard):
        raise ValueError("a source hash guard is required")
    if any(type(n) is not int or n < 1 for n in partition_counts.values()):
        raise ValueError("partition counts must be positive integers")
    seen_words = set(guard.excluded_word_sha256)
    seen_messages = set(guard.excluded_message_sha256)
    generated = {}
    for partition in PARTITIONS:
        rng = random.Random(seeds[partition])
        sources = []
        while len(sources) < partition_counts[partition]:
            words = ["".join(rng.choice(CHARSET) for _ in range(rng.randint(3, 6)))
                     for _ in range(rng.randint(2, 3))]
            message = " ".join(words)
            message_hash = sha256_text(message)
            word_hashes = {sha256_text(word) for word in words}
            if message_hash in seen_messages or seen_words.intersection(word_hashes):
                continue
            try:
                validate_morse(encode_text(message))
            except ValueError:
                continue
            sources.append(message)
            seen_messages.add(message_hash)
            seen_words.update(word_hashes)
        generated[partition] = sources
    return generated


def make_partition_examples(partition: str, sources: Sequence[str], seed: int,
                            *, severity: float = 0.1) -> list[V2Example]:
    if partition not in PARTITIONS:
        raise ValueError("locked-test examples are not available through this API")
    rows = []
    for si, source in enumerate(sources):
        source_id = f"{partition}-{si:07d}"
        for ci, corruption in enumerate(CORRUPTION_KINDS):
            pair_seed = seed + si * 10007 + ci * 1009
            rows.append(make_example(source_id, partition, source, corruption,
                                     pair_seed, severity=severity))
    return rows


def source_fingerprint(sources: Sequence[str]) -> str:
    canonical = "\n".join(sorted(sources))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_partition_manifest(partitions: Mapping[str, Sequence[str]], seeds: Mapping[str, int],
                             guard: SourceGuard) -> dict:
    if set(partitions) != set(PARTITIONS):
        raise ValueError("manifest only accepts train/validation/calibration partitions")
    return {
        "spec_version": "morse-correction-v2-data-1",
        "partition_counts": {name: len(partitions[name]) for name in PARTITIONS},
        "partition_seeds": dict(seeds),
        "source_fingerprints_sha256": {
            name: source_fingerprint(partitions[name]) for name in PARTITIONS
        },
        "guard_fingerprint_sha256": sha256_text("\n".join(sorted(
            guard.excluded_message_sha256 | guard.excluded_word_sha256))),
        "locked_test_labels_loaded": False,
    }

