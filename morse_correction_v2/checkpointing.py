"""Atomic, immutable TensorFlow training checkpoints for research runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CheckpointStore:
    """Each epoch is immutable; CURRENT is atomically advanced after commit."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def initialize(self, *, resume: bool) -> None:
        if self.root.exists():
            if not resume:
                raise FileExistsError("checkpoint directory exists; refusing overwrite")
            if not (self.root / "CURRENT").is_file():
                raise ValueError("checkpoint directory has no committed CURRENT checkpoint")
            self._recover_complete_orphan()
        else:
            if resume:
                raise FileNotFoundError("cannot resume: checkpoint directory does not exist")
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.root.mkdir()

    def commit(self, *, model, optimizer, epoch: int, metadata: dict) -> Path:
        import tensorflow as tf
        if epoch < 1:
            raise ValueError("epoch must be positive")
        final = self.root / f"epoch-{epoch:06d}"
        if final.exists():
            raise FileExistsError("epoch checkpoint already exists; refusing overwrite")
        temp = Path(tempfile.mkdtemp(prefix=".checkpoint-staging-", dir=self.root))
        try:
            epoch_var = tf.Variable(epoch, dtype=tf.int64, trainable=False, name="epoch")
            tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch_var).write(str(temp / "state"))
            files = sorted(p for p in temp.iterdir() if p.is_file())
            manifest = {"epoch": epoch, "metadata": metadata,
                        "files": {p.name: _sha256(p) for p in files}}
            (temp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
            os.replace(temp, final)
            pointer_tmp = self.root / ".CURRENT.tmp"
            pointer_tmp.write_text(final.name, encoding="utf-8")
            os.replace(pointer_tmp, self.root / "CURRENT")
            return final
        except BaseException:
            if temp.exists():
                shutil.rmtree(temp, ignore_errors=True)
            raise

    def restore(self, *, model, optimizer) -> tuple[int, dict]:
        import tensorflow as tf
        if not (self.root / "CURRENT").is_file():
            raise FileNotFoundError("no committed checkpoint to resume")
        name = (self.root / "CURRENT").read_text(encoding="utf-8").strip()
        if not name or Path(name).name != name:
            raise ValueError("invalid checkpoint pointer")
        checkpoint_dir = self.root / name
        manifest = self._validate_checkpoint(checkpoint_dir)
        epoch = tf.Variable(0, dtype=tf.int64, trainable=False, name="epoch")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch)
        status = checkpoint.restore(str(checkpoint_dir / "state"))
        # `write` intentionally omits Checkpoint.save_counter; all model,
        # optimizer, and epoch variables are verified through value checksums.
        status.assert_nontrivial_match()
        status.expect_partial()
        return int(epoch.numpy()), manifest["metadata"]

    def _validate_checkpoint(self, checkpoint_dir: Path) -> dict:
        manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
        for filename, digest in manifest["files"].items():
            path = checkpoint_dir / filename
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError("checkpoint file missing or fingerprint mismatch")
        return manifest

    def _recover_complete_orphan(self) -> None:
        """Advance CURRENT if a complete immutable epoch was promoted before a crash."""
        current_name = (self.root / "CURRENT").read_text(encoding="utf-8").strip()
        current_epoch = int(self._validate_checkpoint(self.root / current_name)["epoch"])
        candidates = []
        for path in self.root.glob("epoch-*"):
            if path.name == current_name:
                continue
            try:
                manifest = self._validate_checkpoint(path)
                epoch = int(manifest["epoch"])
                if epoch > current_epoch and path.name == f"epoch-{epoch:06d}":
                    candidates.append((epoch, path))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        if candidates:
            latest_epoch, latest_path = max(candidates, key=lambda item: item[0])
            if latest_epoch == current_epoch + 1:
                pointer_tmp = self.root / ".CURRENT.recovery.tmp"
                pointer_tmp.write_text(latest_path.name, encoding="utf-8")
                os.replace(pointer_tmp, self.root / "CURRENT")
