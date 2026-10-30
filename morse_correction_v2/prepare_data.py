"""Prepare train/validation/calibration JSONL files; creates no locked data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .data import SourceGuard, build_partition_manifest, generate_sources, make_partition_examples
from .schema import PARTITIONS
from .provenance import _content_fingerprint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-guard", type=Path, required=True,
                        help="custodian-produced JSON of message/word SHA-256 exclusions")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=5000)
    parser.add_argument("--validation-sources", type=int, default=600)
    parser.add_argument("--calibration-sources", type=int, default=500)
    parser.add_argument("--seed", type=int, default=884201)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; refusing to overwrite prepared data")
    if not args.source_guard.is_file():
        parser.error("source guard file is required")
    counts = dict(zip(PARTITIONS, (args.train_sources, args.validation_sources,
                                   args.calibration_sources)))
    seeds = {name: args.seed + 1009 * (index + 1) for index, name in enumerate(PARTITIONS)}
    guard = SourceGuard.from_json(str(args.source_guard))
    if not guard.excluded_message_sha256 or not guard.excluded_word_sha256:
        parser.error("source guard must include hashes for prior messages and words, including the prior locked set")
    sources = generate_sources(counts, seeds, guard)
    args.output_dir.mkdir(parents=True)
    partition_rows = {}
    for index, partition in enumerate(PARTITIONS):
        examples = make_partition_examples(partition, sources[partition], args.seed + index * 100003)
        partition_rows[partition] = examples
        target = args.output_dir / f"{partition}.jsonl"
        target.write_text("".join(json.dumps(asdict(row), sort_keys=True) + "\n"
                                        for row in examples), encoding="utf-8")
    manifest = build_partition_manifest(sources, seeds, guard)
    manifest["partition_content"] = {
        partition: {
            "content_sha256": _content_fingerprint(partition_rows[partition]),
            "source_ids": sorted({row.source_id for row in partition_rows[partition]}),
        } for partition in PARTITIONS
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"prepared_partitions": list(PARTITIONS),
                      "source_counts": manifest["partition_counts"],
                      "example_counts": {name: counts[name] * 12 for name in PARTITIONS},
                      "locked_test_created": False,
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

