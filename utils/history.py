"""
LoRaShield – Message History Utility
---------------------------------------
Handles saving, loading, searching, and exporting message history records.

Records are stored as a JSON array at:
    <project_root>/data/message_history.json

Each record contains:
    time            – ISO timestamp  (set automatically if omitted)
    direction       – "TX" or "RX"
    plain_text      – Original human-readable text
    morse_code      – Morse representation of the text
    encrypted_data  – AES-256 CBC ciphertext (base64) or "" if not encrypted
    sender          – "LOCAL" / "REMOTE" / custom label
    receiver        – Destination label or "N/A"
    encryption_status – Human-readable status string, e.g. "AES-256 CBC"

All file-path resolution uses Path(__file__).resolve() – no hardcoded paths.
All events are logged via utils.logger.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

# ── Absolute path resolution ──────────────────────────────────────────────────
_BASE_DIR    = Path(__file__).resolve().parent.parent
_DATA_DIR    = _BASE_DIR / "data"
_HISTORY_FILE = _DATA_DIR / "message_history.json"

# Ordered fieldnames used for CSV export
_CSV_FIELDS = [
    "time",
    "direction",
    "plain_text",
    "morse_code",
    "encrypted_data",
    "sender",
    "receiver",
    "encryption_status",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_data_dir() -> None:
    """Create the data directory if it does not exist."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_file() -> list[dict[str, Any]]:
    """Load raw records from the JSON file; return [] on any error."""
    if not _HISTORY_FILE.exists():
        return []
    try:
        with _HISTORY_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
            log.warning("History file contained non-list data; resetting.")
            return []
    except (json.JSONDecodeError, IOError) as exc:
        log.error("Failed to read history file: %s", exc)
        return []


def _write_file(records: list[dict[str, Any]]) -> None:
    """Write records list back to the JSON file atomically."""
    _ensure_data_dir()
    tmp = _HISTORY_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    tmp.replace(_HISTORY_FILE)


# ── Public API ────────────────────────────────────────────────────────────────

def load_history() -> list[dict[str, Any]]:
    """
    Load all message history records from disk.

    Returns:
        List of record dicts (may be empty).
    """
    _ensure_data_dir()
    records = _read_file()
    log.debug("Loaded %d history records.", len(records))
    return records


def save_history(record: dict[str, Any]) -> tuple[bool, str]:
    """
    Append a new message record to the history file.

    Required / optional keys in ``record``:
        time            – auto-filled with current timestamp if missing
        direction       – "TX" or "RX"  (default "TX")
        plain_text      – original text  (maps from legacy ``original_text``)
        morse_code      – Morse string
        encrypted_data  – AES ciphertext or "" (maps from legacy field)
        sender          – "LOCAL" if omitted
        receiver        – "N/A" if omitted
        encryption_status – "None" if omitted

    Returns:
        ``(True, "Record saved successfully.")`` or ``(False, error_str)``.
    """
    try:
        _ensure_data_dir()
        records = _read_file()

        # ── Normalise / backfill fields ───────────────────────────────────────
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Support legacy key names from the encoder UI
        if "original_text" in record and "plain_text" not in record:
            record["plain_text"] = record.pop("original_text")

        normalised: dict[str, Any] = {
            "time":             record.get("time")             or now,
            "direction":        record.get("direction",        "TX"),
            "plain_text":       record.get("plain_text",       ""),
            "morse_code":       record.get("morse_code",       ""),
            "encrypted_data":   record.get("encrypted_data",   ""),
            "sender":           record.get("sender",           "LOCAL"),
            "receiver":         record.get("receiver",         "N/A"),
            "encryption_status": record.get("encryption_status", "None"),
        }

        records.append(normalised)
        _write_file(records)
        log.info("History record saved (direction=%s, plain_text=%.40s…).",
                 normalised["direction"], normalised["plain_text"])
        return (True, "Record saved successfully.")
    except Exception as exc:
        log.exception("Failed to save history record.")
        return (False, f"Failed to save history: {exc}")


def delete_history_record(index: int) -> tuple[bool, str]:
    """
    Delete a specific message record by its zero-based index.

    Args:
        index: Zero-based position of the record to delete.

    Returns:
        ``(True, "Record deleted.")`` or ``(False, error_str)``.
    """
    try:
        records = _read_file()
        if index < 0 or index >= len(records):
            return (False, f"Invalid record index: {index} (total: {len(records)}).")
        deleted = records.pop(index)
        _write_file(records)
        log.info("History record %d deleted (time=%s).", index, deleted.get("time"))
        return (True, "Record deleted.")
    except Exception as exc:
        log.exception("Failed to delete history record %d.", index)
        return (False, f"Delete failed: {exc}")


def export_history(output_path: str = "") -> tuple[bool, str]:
    """
    Export all message history records to a CSV file.

    Args:
        output_path: Destination file path.  Defaults to
                     ``<project_root>/data/history_export.csv``.

    Returns:
        ``(True, "History exported to: <path>")`` or ``(False, error_str)``.
    """
    try:
        records = _read_file()
        if not records:
            return (False, "No history records to export.")

        dest = Path(output_path) if output_path else (_DATA_DIR / "history_export.csv")
        dest.parent.mkdir(parents=True, exist_ok=True)

        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        msg = f"History exported to: {dest}"
        log.info(msg)
        return (True, msg)
    except Exception as exc:
        log.exception("Failed to export history.")
        return (False, f"Export failed: {exc}")


def clear_history() -> tuple[bool, str]:
    """Delete all history records (overwrites the JSON file with an empty list)."""
    try:
        _ensure_data_dir()
        _write_file([])
        log.info("History cleared.")
        return (True, "History cleared.")
    except Exception as exc:
        log.exception("Failed to clear history.")
        return (False, f"Clear failed: {exc}")


def search_history(query: str) -> list[dict[str, Any]]:
    """
    Search history records by matching a query string across all fields.

    The search is case-insensitive and matches any substring in any field
    value.

    Args:
        query: Search string.

    Returns:
        Filtered list of matching records.
    """
    if not query:
        return load_history()
    q = query.lower()
    results = [
        r for r in _read_file()
        if any(q in str(v).lower() for v in r.values())
    ]
    log.debug("History search '%s' → %d results.", query, len(results))
    return results


def record_count() -> int:
    """Return the total number of history records on disk."""
    return len(_read_file())
