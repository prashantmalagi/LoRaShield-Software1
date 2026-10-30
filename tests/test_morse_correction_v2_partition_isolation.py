import unittest
from dataclasses import replace
from pathlib import Path
import json
import tempfile

from morse_correction_v2.data import make_example
from morse_correction_v2.train import balanced_batch_indices, load_partition, train_model
from morse_correction_v2.data import SourceGuard, sha256_text
from morse_correction_v2.locked_test import generate_fresh_locked_sources
from morse_correction_v2.provenance import (
    _content_fingerprint, verify_manifest_files, verify_partition_rows,
)


class V2PartitionIsolationTests(unittest.TestCase):
    def test_authoritative_manifest_fingerprints_content_and_rejects_partition_overlap(self):
        parts = {
            "train": [make_example("train-1", "train", "SOS", "clean", 1)],
            "validation": [make_example("validation-1", "validation", "TEST", "clean", 2)],
            "calibration": [make_example("calibration-1", "calibration", "HELLO", "clean", 3)],
        }
        verified = verify_partition_rows(parts)
        self.assertTrue(verified.is_verified)
        altered = dict(parts)
        altered["calibration"] = [make_example("calibration-1", "calibration", "WORLD", "clean", 4)]
        changed = verify_partition_rows(altered)
        self.assertNotEqual(verified.content_fingerprints["calibration"],
                            changed.content_fingerprints["calibration"])
        overlap = dict(parts)
        overlap["validation"] = [make_example("train-1", "validation", "TEST", "clean", 2)]
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_partition_rows(overlap)

    def test_manifest_file_verification_hashes_actual_jsonl_contents(self):
        from dataclasses import asdict
        parts = {
            "train": [make_example("train-1", "train", "SOS", "clean", 1)],
            "validation": [make_example("validation-1", "validation", "TEST", "clean", 2)],
            "calibration": [make_example("calibration-1", "calibration", "HELLO", "clean", 3)],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            for name, rows in parts.items():
                raw_rows = [asdict(row) for row in rows]
                (root / f"{name}.jsonl").write_text("".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in raw_rows), encoding="utf-8")
                entries[name] = {"content_sha256": _content_fingerprint(raw_rows),
                                 "source_ids": [rows[0].source_id]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"partition_content": entries}), encoding="utf-8")
            verified = verify_manifest_files(manifest_path)
            self.assertTrue(verified.is_verified)
            train_path = root / "train.jsonl"
            train_path.write_text(train_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                verify_manifest_files(manifest_path)

    def test_train_loader_rejects_locked_expected_partition_before_file_access(self):
        with self.assertRaisesRegex(ValueError, "train, validation, or calibration"):
            load_partition(Path("missing_locked_test_labels.jsonl"), "locked_test")

    def test_trainer_rejects_nontraining_rows_without_starting_training(self):
        row = make_example("locked-0", "train", "SOS HI", "clean", 1)
        target = Path(__file__).resolve().parents[1] / "morse_correction_v2" / "_must_not_exist.weights.h5"
        with self.assertRaisesRegex(ValueError, "train partition only"):
            train_model([replace(row, partition="locked_test")], [], target)
        self.assertFalse(target.exists())

    def test_calibration_rows_cannot_be_used_for_gradient_dataset(self):
        train = make_example("train-0", "train", "SOS HI", "clean", 1)
        calibration = make_example("cal-0", "calibration", "HELLO WORLD", "clean", 2)
        target = Path(__file__).resolve().parents[1] / "morse_correction_v2" / "_blocked.weights.h5"
        with self.assertRaisesRegex(ValueError, "train partition only"):
            train_model([train, calibration], [], target)
        self.assertFalse(target.exists())

    def test_training_batches_include_clean_identity_examples(self):
        rows = [make_example(f"train-{i}", "train", "SOS HI", "clean" if i < 4 else "substitution", i)
                for i in range(12)]
        batches = balanced_batch_indices(rows, 4, 17)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(rows))))
        self.assertTrue(all(any(rows[index].is_clean for index in batch) for batch in batches))

    def test_fresh_locked_source_generation_uses_hash_guard_and_is_disjoint(self):
        guard = SourceGuard(frozenset({sha256_text("PRIOR MESSAGE")}),
                            frozenset({sha256_text("KNOWN") }))
        sources = generate_fresh_locked_sources(3, seed=90210, guard=guard)
        self.assertEqual(len(sources), 3)
        self.assertTrue(all("KNOWN" not in source.split() for source in sources))
        self.assertEqual(len({word for source in sources for word in source.split()}),
                         sum(len(source.split()) for source in sources))


if __name__ == "__main__":
    unittest.main()
