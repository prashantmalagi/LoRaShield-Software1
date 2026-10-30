"""Phase 1 data foundation for a variable-length Morse seq2seq experiment.

No model is built or trained here. Functions produce small in-memory records
only when called. They never load a model, read the real-capture dataset, or
connect to application, hardware, serial, radio, AES, CRC, or backend code.

Canonical tokenization (version 1)
----------------------------------
Token IDs are fixed by TOKEN_TO_ID. A message is encoded as BOS, then Morse
symbols and explicit boundaries, then EOS. DOT and DASH represent individual
symbols; LETTER_BOUNDARY separates letter code groups; WORD_BOUNDARY separates
words. There is no leading, trailing, or repeated boundary. PAD is permitted
only as a right-padded batch suffix and is excluded from sequence lengths/loss.
UNK is reserved for future tokenizers; this strict Morse tokenizer rejects
unsupported characters rather than converting them to UNK. BOS/EOS/PAD/UNK are
never emitted by decode_morse as message content. Decoding requires exactly one
outer BOS/EOS pair and only well-formed interior symbol/boundary tokens.

The canonical text surface follows the current project convention: one ASCII
space between Morse letter groups and exactly `` / `` between words. A group
may be an invalid International Morse code during corruption; the tokenizer
preserves its dots/dashes because the clean target is still independently
known. Text-to-Morse references support uppercase A-Z and 0-9 only.

Partitions are generated from unique plaintext source messages before variants
are made. Both complete-message and constituent-word overlap are rejected
against prior corpora and across partitions. References live in offline
TrainingPair objects; runtime_preprocess accepts only received Morse and returns
token IDs, with no label/reference field in its input or output contract.

Recompute the source ledgers, counts, fingerprints, and overlap audit whenever
seeds, generator logic, or corpus sizes change. The checked-in JSON is a plan
template; measured audit values are supplied by build_manifest only after a
specific audit is actually performed.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evaluate_morse_correction_v3 import MORSE

SPEC_VERSION = "morse-seq2seq-data-v1"
ROOT_SEED = 73192741
MAX_SEQUENCE_TOKENS = 256  # Includes BOS and EOS; PAD excluded.
PLANNED_PARTITION_COUNTS = {
    "train": 5000,
    "validation": 600,
    "calibration": 500,
    "locked_test": 1000,
}
TOKEN_TO_ID = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "<DOT>": 4,
    "<DASH>": 5,
    "<LETTER>": 6,
    "<WORD>": 7,
}
ID_TO_TOKEN = {value: key for key, value in TOKEN_TO_ID.items()}
PAD, UNK, BOS, EOS, DOT, DASH, LETTER, WORD = TOKEN_TO_ID
SYMBOL_TOKENS = {".": "<DOT>", "-": "<DASH>"}
TOKEN_SYMBOLS = {value: key for key, value in SYMBOL_TOKENS.items()}
PARTITION_NAMES = ("train", "validation", "calibration", "locked_test")
PARTITION_SEED_OFFSETS = {
    "train": 11,
    "validation": 101,
    "calibration": 202,
    "locked_test": 303,
}
CORRUPTION_KINDS = (
    "clean",
    "dot_dash_substitution_05",
    "dot_dash_substitution_10",
    "dot_dash_substitution_15",
    "dot_dash_substitution_20",
    "symbol_insertion",
    "symbol_deletion",
    "burst_substitution",
    "letter_boundary_insertion",
    "letter_boundary_deletion",
    "letter_boundary_shift",
    "word_boundary_insertion",
    "word_boundary_deletion",
    "word_boundary_shift",
)


@dataclass(frozen=True)
class TrainingPair:
    """Offline supervised pair. Never pass this object to runtime inference."""

    source_id: str
    partition: str
    corruption_type: str
    noise_percent: int | None
    reference_morse: str
    received_morse: str
    input_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeInput:
    """Reference-free inference input. The dataclass has no target/reference."""

    received_morse: str
    input_ids: tuple[int, ...]


def _split_morse(morse: str) -> list[list[str]]:
    if not isinstance(morse, str) or not morse or morse != morse.strip():
        raise ValueError("Morse must be nonempty and have no edge whitespace")
    if any(char not in ".- /" for char in morse):
        raise ValueError("Morse contains unsupported characters")
    words = morse.split(" / ")
    if any(not word or "/" in word for word in words):
        raise ValueError("word boundaries must be exactly ' / ' with nonempty words")
    result = []
    for word in words:
        if word.startswith(" ") or word.endswith(" ") or "  " in word:
            raise ValueError("letter boundaries must be single spaces")
        groups = word.split(" ")
        if any(not group or any(char not in ".-" for char in group) for group in groups):
            raise ValueError("each Morse letter group must contain only dots or dashes")
        result.append(groups)
    return result


def encode_morse(morse: str, *, max_tokens: int = MAX_SEQUENCE_TOKENS) -> tuple[str, ...]:
    """Encode canonical Morse surface syntax to BOS...EOS token names."""
    if max_tokens < 2:
        raise ValueError("max_tokens must allow BOS and EOS")
    words = _split_morse(morse)
    tokens: list[str] = ["<BOS>"]
    for word_index, groups in enumerate(words):
        if word_index:
            tokens.append("<WORD>")
        for group_index, group in enumerate(groups):
            if group_index:
                tokens.append("<LETTER>")
            tokens.extend(SYMBOL_TOKENS[char] for char in group)
    tokens.append("<EOS>")
    if len(tokens) > max_tokens:
        raise ValueError(f"encoded sequence length {len(tokens)} exceeds {max_tokens}")
    return tuple(tokens)


def decode_morse(tokens: Sequence[str], *, max_tokens: int = MAX_SEQUENCE_TOKENS) -> str:
    """Decode one strict BOS...EOS sequence to canonical Morse surface syntax."""
    values = tuple(tokens)
    if len(values) > max_tokens:
        raise ValueError(f"encoded sequence length {len(values)} exceeds {max_tokens}")
    if len(values) < 3 or values[0] != "<BOS>" or values[-1] != "<EOS>":
        raise ValueError("sequence must have one outer BOS and EOS")
    interior = values[1:-1]
    if any(token not in (*TOKEN_SYMBOLS, "<LETTER>", "<WORD>") for token in interior):
        raise ValueError("PAD, UNK, BOS, EOS, or unknown token inside Morse sequence")
    if interior[0] not in TOKEN_SYMBOLS or interior[-1] not in TOKEN_SYMBOLS:
        raise ValueError("sequence cannot begin or end with a boundary")
    rendered: list[str] = []
    previous_boundary = False
    for token in interior:
        if token in TOKEN_SYMBOLS:
            rendered.append(TOKEN_SYMBOLS[token])
            previous_boundary = False
        else:
            if previous_boundary:
                raise ValueError("adjacent letter/word boundaries are invalid")
            rendered.append(" " if token == "<LETTER>" else " / ")
            previous_boundary = True
    return "".join(rendered)


def encode_text(text: str) -> str:
    """Encode uppercase alphanumeric plaintext with the project's Morse map."""
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError("text must be nonempty and have no edge whitespace")
    words = text.split(" ")
    if any(not word for word in words) or any(char not in MORSE for word in words for char in word):
        raise ValueError("text supports uppercase A-Z/0-9 and single word spaces only")
    return " / ".join(" ".join(MORSE[char] for char in word) for word in words)


def pad_token_ids(sequences: Sequence[Sequence[str]], *, max_tokens: int = MAX_SEQUENCE_TOKENS) -> list[list[int]]:
    """Right-pad tokenized sequences; PAD appears only after EOS."""
    if not sequences:
        return []
    actual_max = max(map(len, sequences))
    if actual_max > max_tokens:
        raise ValueError(f"sequence length {actual_max} exceeds {max_tokens}")
    rows = []
    for sequence in sequences:
        ids = []
        for token in sequence:
            if token not in TOKEN_TO_ID:
                raise ValueError(f"token outside vocabulary: {token!r}")
            if token == "<PAD>":
                raise ValueError("input sequences must not contain pre-existing PAD")
            ids.append(TOKEN_TO_ID[token])
        rows.append(ids + [TOKEN_TO_ID["<PAD>"]] * (actual_max - len(ids)))
    return rows


def runtime_preprocess(received_morse: str, *, max_tokens: int = MAX_SEQUENCE_TOKENS) -> RuntimeInput:
    """Reference-free preprocessing contract for future model inference."""
    tokens = encode_morse(received_morse, max_tokens=max_tokens)
    return RuntimeInput(received_morse=received_morse,
                        input_ids=tuple(TOKEN_TO_ID[token] for token in tokens))


def _validate_source_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text or text != text.strip():
        raise ValueError("source must be nonempty and have no edge whitespace")
    words = text.split(" ")
    if any(not word for word in words) or any(char not in MORSE for word in words for char in word):
        raise ValueError("source supports uppercase A-Z/0-9 with single spaces")
    return words


def existing_source_corpora() -> dict[str, list[str]]:
    """Recreate all known source corpora without loading models or doing inference."""
    from evaluate_morse_correction_v3_independent import (
        FRESH_SEED, DEFAULT_MESSAGES as FRESH_COUNT, make_fresh_corpus,
        _old_partition_messages,
    )
    from evaluate_morse_correction_v3_synthetic import (
        SEED as SYNTH_SEED, DEFAULT_MESSAGES as SYNTH_COUNT, generate_source_corpus,
    )
    from evaluate_morse_correction_v3_corruption import (
        SEED as CORRUPTION_SEED, DEFAULT_MESSAGES as CORRUPTION_COUNT,
        generate_source_corpus as generate_corruption_sources,
    )
    from evaluate_morse_correction_v3_selective import (
        SEED as SELECTIVE_SEED, CALIBRATION_MESSAGES, LOCKED_TEST_MESSAGES,
        generate_partitions,
    )

    old = sorted(_old_partition_messages())
    synthetic = generate_source_corpus(SYNTH_SEED, SYNTH_COUNT)
    independent = make_fresh_corpus(FRESH_SEED, FRESH_COUNT)
    corruption = generate_corruption_sources(CORRUPTION_SEED, CORRUPTION_COUNT)
    calibration, locked = generate_partitions(
        SELECTIVE_SEED, CALIBRATION_MESSAGES, LOCKED_TEST_MESSAGES
    )
    return {
        "v3_and_legacy_train_eval_sources": old,
        "v3_synthetic_eval": synthetic,
        "v3_independent_eval": independent,
        "v3_corruption_eval": corruption,
        "v3_selective_calibration": calibration,
        "v3_selective_locked_test": locked,
    }


def overlap_report(
    partitions: Mapping[str, Sequence[str]],
    exclusions: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Check exact message and constituent-word disjointness, returning counts."""
    known_messages: dict[str, str] = {}
    known_words: dict[str, str] = {}
    for corpus, messages in exclusions.items():
        for message in messages:
            canonical = " ".join(_validate_source_text(message))
            known_messages.setdefault(canonical, corpus)
            for word in canonical.split(" "):
                known_words.setdefault(word, corpus)

    partition_messages: set[str] = set()
    partition_words: set[str] = set()
    overlap_messages = []
    overlap_words = []
    exact_partition_collisions = []
    word_partition_collisions = []
    for partition, messages in partitions.items():
        local_messages: set[str] = set()
        local_words: set[str] = set()
        for message in messages:
            canonical = " ".join(_validate_source_text(message))
            if canonical in known_messages:
                overlap_messages.append({"partition": partition, "message": canonical,
                                         "excluded_corpus": known_messages[canonical]})
            for word in canonical.split(" "):
                if word in known_words:
                    overlap_words.append({"partition": partition, "word": word,
                                          "excluded_corpus": known_words[word]})
                if word in local_words:
                    word_partition_collisions.append({"partition": partition, "word": word})
                local_words.add(word)
            if canonical in local_messages:
                exact_partition_collisions.append({"partition": partition, "message": canonical})
            local_messages.add(canonical)
            if canonical in partition_messages:
                exact_partition_collisions.append({"partition": partition, "message": canonical})
            if any(word in partition_words for word in canonical.split(" ")):
                word_partition_collisions.extend(
                    {"partition": partition, "word": word}
                    for word in canonical.split(" ") if word in partition_words
                )
            partition_messages.add(canonical)
            partition_words.update(canonical.split(" "))

    return {
        "complete_message_overlap_count": len(overlap_messages),
        "word_overlap_count": len(overlap_words),
        "cross_partition_message_collision_count": len(exact_partition_collisions),
        "cross_partition_word_collision_count": len(word_partition_collisions),
        "complete_message_overlaps": overlap_messages,
        "word_overlaps": overlap_words,
        "partition_message_collisions": exact_partition_collisions,
        "partition_word_collisions": word_partition_collisions,
    }


def generate_source_partitions(
    seed: int,
    counts: Mapping[str, int],
    exclusions: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Generate unique word-disjoint sources; does not generate noisy variants."""
    if set(counts) != set(PARTITION_NAMES):
        raise ValueError(f"counts must define exactly {PARTITION_NAMES}")
    if any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError("each partition count must be a positive integer")
    excluded_messages = set()
    excluded_words = set()
    for messages in exclusions.values():
        for message in messages:
            words = _validate_source_text(message)
            excluded_messages.add(" ".join(words))
            excluded_words.update(words)

    partitions: dict[str, list[str]] = {}
    used_messages = set(excluded_messages)
    used_words = set(excluded_words)
    alphabet = tuple(MORSE)
    for partition in PARTITION_NAMES:
        rng = random.Random(seed + PARTITION_SEED_OFFSETS[partition])
        result = []
        while len(result) < counts[partition]:
            words = [
                "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 6)))
                for _ in range(rng.randint(2, 3))
            ]
            # Ensure generated sources exercise symbol deletion and short bursts.
            if not any(len(MORSE[char]) >= 3 for word in words for char in word):
                continue
            message = " ".join(words)
            if message in used_messages or set(words) & used_words:
                continue
            # Include BOS/EOS and reject any source outside the configured cap.
            reference = encode_text(message)
            encode_morse(reference)
            result.append(message)
            used_messages.add(message)
            used_words.update(words)
        partitions[partition] = result

    report = overlap_report(partitions, exclusions)
    if any(report[key] for key in (
        "complete_message_overlap_count", "word_overlap_count",
        "cross_partition_message_collision_count", "cross_partition_word_collision_count",
    )):
        raise AssertionError("source partition overlap check failed")
    return partitions


def _render_words(words: Sequence[Sequence[str]]) -> str:
    if not words or any(not word for word in words) or any(not group for word in words for group in word):
        raise ValueError("corruption produced an empty word or letter group")
    return " / ".join(" ".join(word) for word in words)


def corrupt_morse(reference_morse: str, corruption_type: str, rng: random.Random,
                  *, substitution_percent: int | None = None) -> str:
    """Apply one deterministic corruption operation to a canonical reference."""
    words = _split_morse(reference_morse)
    if corruption_type == "clean":
        return reference_morse
    if corruption_type.startswith("dot_dash_substitution_"):
        try:
            rate = int(corruption_type.rsplit("_", 1)[1])
        except ValueError as exc:
            raise ValueError("substitution kind must end with a percentage") from exc
    elif corruption_type == "dot_dash_substitution":
        if substitution_percent is None:
            raise ValueError("substitution_percent is required for generic substitution")
        rate = substitution_percent
    else:
        rate = None

    if rate is not None:
        if rate < 0 or rate > 100:
            raise ValueError("substitution percentage must be between 0 and 100")
        symbols = [(wi, gi, si) for wi, word in enumerate(words)
                   for gi, group in enumerate(word) for si, _ in enumerate(group)]
        flip_count = int(round(len(symbols) * rate / 100.0))
        for wi, gi, si in rng.sample(symbols, flip_count):
            group = list(words[wi][gi])
            group[si] = "-" if group[si] == "." else "."
            words[wi][gi] = "".join(group)
        return _render_words(words)

    if corruption_type == "symbol_insertion":
        wi = rng.randrange(len(words))
        gi = rng.randrange(len(words[wi]))
        group = words[wi][gi]
        at = rng.randrange(len(group) + 1)
        words[wi][gi] = group[:at] + rng.choice(".-") + group[at:]
    elif corruption_type == "symbol_deletion":
        choices = [(wi, gi, si) for wi, word in enumerate(words)
                   for gi, group in enumerate(word) if len(group) > 1
                   for si in range(len(group))]
        if not choices:
            raise ValueError("no symbol can be deleted without creating an empty letter group")
        wi, gi, si = rng.choice(choices)
        words[wi][gi] = words[wi][gi][:si] + words[wi][gi][si + 1:]
    elif corruption_type == "burst_substitution":
        choices = [(wi, gi, group) for wi, word in enumerate(words)
                   for gi, group in enumerate(word) if len(group) >= 2]
        if not choices:
            raise ValueError("no letter group is long enough for a burst")
        wi, gi, group = rng.choice(choices)
        burst = rng.randint(2, min(3, len(group)))
        start = rng.randint(0, len(group) - burst)
        chars = list(group)
        for pos in range(start, start + burst):
            chars[pos] = "-" if chars[pos] == "." else "."
        words[wi][gi] = "".join(chars)
    elif corruption_type == "letter_boundary_insertion":
        choices = [(wi, gi, pos) for wi, word in enumerate(words)
                   for gi, group in enumerate(word) for pos in range(1, len(group))]
        if not choices:
            raise ValueError("no letter group can be split")
        wi, gi, pos = rng.choice(choices)
        group = words[wi][gi]
        words[wi][gi:gi + 1] = [group[:pos], group[pos:]]
    elif corruption_type == "letter_boundary_deletion":
        choices = [(wi, gi) for wi, word in enumerate(words) for gi in range(len(word) - 1)]
        if not choices:
            raise ValueError("no letter boundary can be deleted")
        wi, gi = rng.choice(choices)
        words[wi][gi:gi + 2] = [words[wi][gi] + words[wi][gi + 1]]
    elif corruption_type == "letter_boundary_shift":
        choices = []
        for wi, word in enumerate(words):
            for gi in range(len(word) - 1):
                left, right = word[gi], word[gi + 1]
                if len(left) > 1:
                    choices.append((wi, gi, "left"))
                if len(right) > 1:
                    choices.append((wi, gi, "right"))
        if not choices:
            raise ValueError("no letter boundary can be shifted")
        wi, gi, direction = rng.choice(choices)
        left, right = words[wi][gi], words[wi][gi + 1]
        merged = left + right
        split = len(left) - 1 if direction == "left" else len(left) + 1
        words[wi][gi:gi + 2] = [merged[:split], merged[split:]]
    elif corruption_type == "word_boundary_insertion":
        choices = [(wi, pos) for wi, word in enumerate(words) for pos in range(1, len(word))]
        if not choices:
            raise ValueError("no word can be split at a letter boundary")
        wi, pos = rng.choice(choices)
        word = words[wi]
        words[wi:wi + 1] = [word[:pos], word[pos:]]
    elif corruption_type == "word_boundary_deletion":
        if len(words) < 2:
            raise ValueError("no word boundary can be deleted")
        wi = rng.randrange(len(words) - 1)
        words[wi:wi + 2] = [words[wi] + words[wi + 1]]
    elif corruption_type == "word_boundary_shift":
        choices = []
        for wi in range(len(words) - 1):
            if len(words[wi]) > 1:
                choices.append((wi, "right"))
            if len(words[wi + 1]) > 1:
                choices.append((wi, "left"))
        if not choices:
            raise ValueError("no word boundary can be shifted")
        wi, direction = rng.choice(choices)
        if direction == "right":
            moved = words[wi].pop()
            words[wi + 1].insert(0, moved)
        else:
            moved = words[wi + 1].pop(0)
            words[wi].append(moved)
    else:
        raise ValueError(f"unsupported corruption type: {corruption_type}")

    corrupted = _render_words(words)
    encode_morse(corrupted)
    return corrupted


def make_training_pair(source_id: str, partition: str, source_text: str,
                       corruption_type: str, seed: int,
                       *, substitution_percent: int | None = None) -> TrainingPair:
    """Create an offline noisy-input/clean-target pair for one source."""
    if partition not in PARTITION_NAMES:
        raise ValueError(f"unknown partition: {partition}")
    reference = encode_text(source_text)
    rate = substitution_percent
    if corruption_type.startswith("dot_dash_substitution_"):
        rate = int(corruption_type.rsplit("_", 1)[1])
    received = corrupt_morse(reference, corruption_type, random.Random(seed),
                             substitution_percent=rate)
    input_tokens = encode_morse(received)
    target_tokens = encode_morse(reference)
    noise_percent = rate if rate is not None else None
    return TrainingPair(
        source_id=source_id,
        partition=partition,
        corruption_type=corruption_type,
        noise_percent=noise_percent,
        reference_morse=reference,
        received_morse=received,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
    )


def make_variants_for_source(source_id: str, partition: str, source_text: str,
                             seed: int) -> list[TrainingPair]:
    """Create configured variants for one source; caller controls corpus size."""
    return [make_training_pair(
        source_id, partition, source_text, kind, seed + index * 1009
    ) for index, kind in enumerate(CORRUPTION_KINDS)]


def generate_partition_pairs(partitions: Mapping[str, Sequence[str]], seed: int) -> dict[str, list[TrainingPair]]:
    """Generate variants only from already-assigned source partitions."""
    if set(partitions) != set(PARTITION_NAMES):
        raise ValueError(f"partitions must define exactly {PARTITION_NAMES}")
    generated: dict[str, list[TrainingPair]] = {}
    for partition_index, partition in enumerate(PARTITION_NAMES):
        pairs = []
        for source_index, source_text in enumerate(partitions[partition]):
            source_id = f"{partition}-{source_index:06d}"
            pair_seed = seed + partition_index * 1_000_003 + source_index * 10_007
            pairs.extend(make_variants_for_source(source_id, partition, source_text, pair_seed))
        generated[partition] = pairs
    return generated


EXCLUSION_MANIFEST = [
    {"name": "V3 train/validation/test and legacy V2 sources", "seeds": [731941, 562813, 20261029, 999],
     "counts": {"train": 5000, "validation": 600, "v3_test": 200, "v2_eval": 200,
                "smoke_train": 40, "smoke_validation": 12, "smoke_test": 24}},
    {"name": "V3 synthetic evaluator", "seeds": [8631052], "counts": {"source_messages": 200}},
    {"name": "V3 independent evaluator", "seeds": [20261029141537], "counts": {"source_messages": 500}},
    {"name": "V3 corruption evaluator", "seeds": [2748193], "counts": {"source_messages": 100}},
    {"name": "V3 selective calibration and locked test", "seeds": [4819207],
     "counts": {"calibration_sources": 300, "locked_test_sources": 300}},
]


def _measured_exclusion_summary(
    exclusions: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    per_corpus = {}
    all_messages: list[str] = []
    all_words: set[str] = set()
    for name, messages in exclusions.items():
        canonical = [" ".join(_validate_source_text(message)) for message in messages]
        words = {word for message in canonical for word in message.split(" ")}
        per_corpus[name] = {
            "source_rows": len(canonical),
            "unique_message_count": len(set(canonical)),
            "unique_word_count": len(words),
            "source_fingerprint_sha256": source_fingerprint(canonical),
        }
        all_messages.extend(canonical)
        all_words.update(words)
    return {
        "source_rows_across_corpora": len(all_messages),
        "unique_message_count": len(set(all_messages)),
        "unique_word_count": len(all_words),
        "per_corpus": per_corpus,
    }


def build_manifest(partitions: Mapping[str, Sequence[str]], *, seed: int = ROOT_SEED,
                   exclusions: Mapping[str, Sequence[str]] | None = None) -> dict[str, object]:
    """Build measured counts/fingerprints/overlaps for supplied sources; write nothing."""
    if set(partitions) != set(PARTITION_NAMES):
        raise ValueError(f"partitions must define exactly {PARTITION_NAMES}")
    if any(not partitions[name] for name in PARTITION_NAMES):
        raise ValueError("cannot record measured counts or fingerprints for an empty partition")
    exclusions = exclusions if exclusions is not None else existing_source_corpora()
    report = overlap_report(partitions, exclusions)
    return {
        "spec_version": SPEC_VERSION,
        "planned_settings": {
            "root_seed": seed,
            "partition_seed_offsets": PARTITION_SEED_OFFSETS,
            "planned_partition_source_counts": PLANNED_PARTITION_COUNTS,
            "token_vocabulary_ids": TOKEN_TO_ID,
            "maximum_sequence_tokens_including_bos_eos": MAX_SEQUENCE_TOKENS,
            "corruption_settings": {
            "kinds": list(CORRUPTION_KINDS),
            "substitution_percentages": [5, 10, 15, 20],
            "symbol_insertion": "one dot or dash inserted within one Morse letter group",
            "symbol_deletion": "one dot/dash removed only when the letter group remains nonempty",
            "burst_substitution": "2-3 contiguous dot/dash flips within one letter group",
            "letter_boundary": ["insert", "delete", "shift by one symbol"],
            "word_boundary": ["insert", "delete", "shift by one whole letter group"],
            "pair_rule": "corrupted received Morse is input; canonical source Morse is target",
                "source_rng": "random.Random(root_seed + partition_seed_offsets[partition])",
                "pair_rng": "random.Random(pair_seed); pair_seed = corpus_seed + partition_index*1000003 + source_index*10007 + corruption_index*1009",
            },
        },
        "exclusion_audit": {
            "source_corpus_definitions": EXCLUSION_MANIFEST,
            "measured": _measured_exclusion_summary(exclusions),
        },
        "partition_audit": {
            "measurement_scope": "exactly the supplied source partitions; no corruption variants or model evaluation are implied",
            "source_counts": {name: len(partitions.get(name, ())) for name in PARTITION_NAMES},
            "source_fingerprints_sha256": {
                name: source_fingerprint(partitions.get(name, ())) for name in PARTITION_NAMES
            },
            "overlap_check": {
                "method": "uppercase canonical complete-message equality plus exact constituent-word set intersection; apply across exclusions and every assigned partition",
                "results": report,
            },
        },
        "reproducibility_policy": {
            "regenerate_ledgers_when": ["source seeds change", "generator logic changes", "corpus sizes change"],
            "fingerprints_are_measured_only": True,
        },
        "runtime_contract": {
            "preprocessor_parameters": ["received_morse", "max_tokens"],
            "runtime_input_fields": ["received_morse", "input_ids"],
            "reference_labels_available_at_runtime": False,
        },
    }


def manifest_json(manifest: Mapping[str, object]) -> str:
    """Stable JSON representation for an external manifest writer."""
    return json.dumps(manifest, sort_keys=True, indent=2)


def source_fingerprint(messages: Iterable[str]) -> str:
    """Stable digest for recording exact partition membership without plaintext."""
    payload = "\n".join(sorted(" ".join(_validate_source_text(message)) for message in messages))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
