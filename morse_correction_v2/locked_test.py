"""Fresh locked-test vault. Training/calibration code must not import this module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from typing import Sequence

from .data import SourceGuard, sha256_text
from .schema import CHARSET, CORRUPTION_KINDS, V2Example

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def generate_fresh_locked_sources(count: int, *, seed: int, guard: SourceGuard) -> list[str]:
    """Generate fresh sources using hash exclusions only; no old labels are read."""
    if type(count) is not int or count < 1:
        raise ValueError("locked source count must be positive")
    if (not isinstance(guard, SourceGuard) or not guard.excluded_message_sha256
            or not guard.excluded_word_sha256):
        raise ValueError("fresh locked generation requires prior and nonlocked source hashes")
    rng = random.Random(seed)
    used_words = set(guard.excluded_word_sha256)
    used_messages = set(guard.excluded_message_sha256)
    sources = []
    while len(sources) < count:
        words = ["".join(rng.choice(CHARSET) for _ in range(rng.randint(3, 6)))
                 for _ in range(rng.randint(2, 3))]
        message = " ".join(words)
        hashes = {sha256_text(word) for word in words}
        digest = sha256_text(message)
        if digest in used_messages or hashes & used_words:
            continue
        sources.append(message)
        used_messages.add(digest)
        used_words.update(hashes)
    return sources


def write_fresh_locked_test(sources: Sequence[str], path: str | Path, *, seed: int,
                            guard: SourceGuard) -> dict:
    """Create a new locked bundle at a new path; never called by preparation/training."""
    target = Path(path)
    if not sources:
        raise ValueError("fresh locked-test source list must be nonempty")
    if target.exists() or target.with_suffix(target.suffix + ".sealed.json").exists():
        raise FileExistsError(f"refusing to overwrite locked-test bundle: {target}")
    if not isinstance(guard, SourceGuard):
        raise ValueError("fresh locked sources require a hash guard for prior data and all nonlocked partitions")
    if not guard.excluded_message_sha256 or not guard.excluded_word_sha256:
        raise ValueError("fresh locked-test guard must include prior and nonlocked source hashes")
    used_messages = set()
    used_words = set()
    for source in sources:
        words = source.split(" ")
        if (sha256_text(source) in guard.excluded_message_sha256
                or any(sha256_text(word) in guard.excluded_word_sha256 for word in words)
                or source in used_messages or any(word in used_words for word in words)):
            raise ValueError("fresh locked sources overlap an excluded or repeated source")
        used_messages.add(source)
        used_words.update(words)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for si, source in enumerate(sources):
        for ci, corruption in enumerate(CORRUPTION_KINDS):
            seed_i = seed + si * 10007 + ci * 1009
            example = _locked_example(f"locked-{si:07d}", source, corruption, seed_i)
            rows.append(example.__dict__)
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = target.with_suffix(target.suffix + ".sealed.json")
    sidecar.write_text(json.dumps({"dataset_sha256": digest, "examples": len(rows),
                                   "purpose": "fresh_locked_test",
                                   "labels_available_to_training": False}, indent=2), encoding="utf-8")
    return {"dataset_sha256": digest, "examples": len(rows), "path": str(target)}


def _locked_example(source_id: str, source: str, corruption: str, seed: int) -> V2Example:
    # Kept private to the isolated vault; training constructors reject this partition.
    from .data import corrupt_morse, derive_edit_labels
    from .schema import boundary_targets, encode_text
    import random
    reference = encode_text(source)
    received = corrupt_morse(reference, corruption, random.Random(seed))
    ops, inserts, alignment = derive_edit_labels(received, reference)
    return V2Example(source_id, "locked_test", corruption, 0.1, seed, source,
                     received, reference, ops, inserts, alignment,
                     boundary_targets(reference), corruption == "clean")


def load_fresh_locked_once(path: str | Path, *, freeze_manifest_path: str | Path) -> list[dict]:
    """Read labels only with all frozen artifact digests explicitly supplied."""
    frozen = json.loads(Path(freeze_manifest_path).read_text(encoding="utf-8"))
    digests = (frozen.get("model_sha256"), frozen.get("policy_sha256"),
               frozen.get("report_schema_sha256"))
    if any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in digests):
        raise ValueError("freeze manifest must record model, policy, and report SHA-256 digests")
    target = Path(path)
    sidecar = target.with_suffix(target.suffix + ".sealed.json")
    consumed = target.with_suffix(target.suffix + ".evaluated.json")
    if consumed.exists():
        raise ValueError("fresh locked-test bundle has already been evaluated")
    sealed = json.loads(sidecar.read_text(encoding="utf-8"))
    if hashlib.sha256(target.read_bytes()).hexdigest() != sealed.get("dataset_sha256"):
        raise ValueError("locked dataset digest mismatch")
    if frozen.get("locked_dataset_sha256") != sealed.get("dataset_sha256"):
        raise ValueError("freeze manifest does not bind this locked dataset")
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
    consumed.write_text(json.dumps({"dataset_sha256": sealed["dataset_sha256"],
                                    "freeze_manifest_sha256": hashlib.sha256(
                                        Path(freeze_manifest_path).read_bytes()).hexdigest()}, indent=2),
                        encoding="utf-8")
    return rows

