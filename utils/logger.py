"""
LoRaShield – Application Logger
---------------------------------
Centralised logging setup for the entire application.

All modules import `get_logger()` to obtain a named Logger instance.
Log output goes to:
    • logs/app.log  (rotating file, max 2 MB, 3 backups)
    • stdout        (INFO and above, for development convenience)

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Connected to COM3")
    log.error("TensorFlow not found", exc_info=True)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Resolve paths relative to project root ────────────────────────────────────
_BASE_DIR  = Path(__file__).resolve().parent.parent
_LOG_DIR   = _BASE_DIR / "logs"
_LOG_FILE  = _LOG_DIR / "app.log"

_LOG_FORMAT      = "%(asctime)s  [%(levelname)-8s]  %(name)s  –  %(message)s"
_DATE_FORMAT     = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES       = 2 * 1024 * 1024   # 2 MB
_BACKUP_COUNT    = 3
_FILE_LEVEL      = logging.DEBUG
_CONSOLE_LEVEL   = logging.INFO

# ── Module-level flag: initialise only once ───────────────────────────────────
_initialised = False


def _initialise() -> None:
    """Configure root logger with file + console handlers (idempotent)."""
    global _initialised
    if _initialised:
        return
    _initialised = True

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Rotating file handler ─────────────────────────────────────────────────
    try:
        fh = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setLevel(_FILE_LEVEL)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except OSError as exc:
        # If we cannot write the log file, continue without it but warn on stderr.
        print(f"[LoRaShield] WARNING: Cannot open log file {_LOG_FILE}: {exc}",
              file=sys.stderr)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(_CONSOLE_LEVEL)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Suppress noisy third-party loggers
    for noisy in ("tensorflow", "absl", "h5py", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named Logger, initialising the logging system on first call.

    Args:
        name: Module name, conventionally ``__name__``.

    Returns:
        logging.Logger instance.
    """
    _initialise()
    return logging.getLogger(name)


def log_path() -> Path:
    """Return the absolute path to the current log file."""
    return _LOG_FILE
