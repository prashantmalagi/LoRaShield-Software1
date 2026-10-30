"""Partition manifest verification over actual V2 rows or prepared JSONL files."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .schema import PARTITIONS, V2Example

_VERIFICATION_TOKEN = object()
_FILE_VERIFICATION_TOKEN = object()


def _row_dict(row):
    return asdict(row) if is_dataclass(row) else dict(row)


def _content_fingerprint(rows) -> str:
    lines = [json.dumps(_row_dict(row), sort_keys=True, separators=(",", ":")) for row in rows]
    lines.sort()
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def partition_content_fingerprint(rows) -> str:
    """Hash canonical complete row content for one already-validated partition."""
    if not rows:
        raise ValueError("cannot fingerprint an empty partition")
    return _content_fingerprint(rows)


@dataclass(frozen=True)
class VerifiedPartitionManifest:
    source_ids: Mapping[str, tuple[str, ...]]
    content_fingerprints: Mapping[str, str]
    manifest_fingerprint: str
    _token: object

    @property
    def is_verified(self) -> bool:
        return self._token in (_VERIFICATION_TOKEN, _FILE_VERIFICATION_TOKEN)

    @property
    def is_file_verified(self) -> bool:
        return self._token is _FILE_VERIFICATION_TOKEN


def verify_partition_rows(partition_rows: Mapping[str, Sequence]) -> VerifiedPartitionManifest:
    """Mint verified provenance from the complete in-memory partition contents."""
    if set(partition_rows) != set(PARTITIONS):
        raise ValueError("manifest must contain train, validation, and calibration only")
    ids, fingerprints = {}, {}
    for partition in PARTITIONS:
        rows = tuple(partition_rows[partition])
        if not rows:
            raise ValueError(f"{partition} partition is empty")
        parsed = [_row_dict(row) for row in rows]
        if any(row.get("partition") != partition for row in parsed):
            raise ValueError(f"row partition mismatch in {partition}")
        source_ids = tuple(sorted({str(row["source_id"]) for row in parsed}))
        # A source may have many corruption variants but must use one source text.
        texts: dict[str, set[str]] = {}
        for row in parsed:
            texts.setdefault(str(row["source_id"]), set()).add(str(row["source_text"]))
        if any(len(values) != 1 for values in texts.values()):
            raise ValueError(f"source ID maps to multiple source texts in {partition}")
        ids[partition] = source_ids
        fingerprints[partition] = _content_fingerprint(parsed)
    for i, left in enumerate(PARTITIONS):
        for right in PARTITIONS[i + 1:]:
            if set(ids[left]) & set(ids[right]):
                raise ValueError(f"source IDs overlap between {left} and {right}")
    body = {"source_ids": ids, "content_fingerprints": fingerprints}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    return VerifiedPartitionManifest(ids, fingerprints, digest, _VERIFICATION_TOKEN)


def verify_manifest_files(manifest_path: str | Path) -> VerifiedPartitionManifest:
    """Read manifest and all three JSONL files, then verify content digests."""
    path = Path(manifest_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("partition_content")
    if not isinstance(entries, dict) or set(entries) != set(PARTITIONS):
        raise ValueError("manifest lacks complete partition content fingerprints")
    rows = {}
    for partition in PARTITIONS:
        source = path.parent / f"{partition}.jsonl"
        parsed = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
        entry = entries[partition]
        if entry.get("content_sha256") != _content_fingerprint(parsed):
            raise ValueError(f"{partition} content fingerprint mismatch")
        if tuple(sorted({row["source_id"] for row in parsed})) != tuple(entry.get("source_ids", ())):
            raise ValueError(f"{partition} source ID manifest mismatch")
        rows[partition] = parsed
    verified = verify_partition_rows(rows)
    declared = {p: entries[p]["content_sha256"] for p in PARTITIONS}
    if dict(verified.content_fingerprints) != declared:
        raise ValueError("partition fingerprints differ from verified contents")
    return VerifiedPartitionManifest(verified.source_ids, verified.content_fingerprints,
                                     verified.manifest_fingerprint, _FILE_VERIFICATION_TOKEN)
