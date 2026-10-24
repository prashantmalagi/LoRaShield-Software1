"""
LoRaShield – Application Launcher
=====================================
Starts the FastAPI backend and opens the React frontend in the browser.

Usage:
    python run.py
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_PORT  = 8000
FRONTEND_PORT = 5173

BACKEND_URL  = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"


def _python() -> str:
    """Return the venv python (or sys.executable as fallback)."""
    return str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def start_backend() -> subprocess.Popen:
    print("[LoRaShield] Starting FastAPI backend …")
    cmd = [
        _python(), "-m", "uvicorn",
        "backend.main:app",
        "--host", "127.0.0.1",
        "--port", str(BACKEND_PORT),
    ]
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    return proc


def start_frontend() -> subprocess.Popen:
    print("[LoRaShield] Starting React frontend …")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(FRONTEND_DIR),
    )
    return proc


def main():
    backend_proc  = start_backend()
    frontend_proc = start_frontend()

    print("[LoRaShield] Waiting for servers to start …")
    time.sleep(4)

    print(f"[LoRaShield] Opening browser at {FRONTEND_URL}")
    webbrowser.open(FRONTEND_URL)

    print("[LoRaShield] Running. Press Ctrl+C to stop.")
    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\n[LoRaShield] Shutting down …")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
