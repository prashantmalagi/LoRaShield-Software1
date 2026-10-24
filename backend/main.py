"""
LoRaShield – FastAPI Backend
==============================
Exposes all LoRaShield Python logic as a REST + WebSocket API
consumed by the React frontend.

Run with:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import serial_comm
from utils.logger import get_logger

log = get_logger(__name__)
from utils.encryption import encrypt_message, decrypt_message, generate_key, is_crypto_available
from utils.crc import build_packet, parse_packet, calculate_crc, verify_crc
from utils.history import (
    load_history, save_history, delete_history_record,
    clear_history, export_history, search_history, record_count,
)
from utils.ai import encode_text, decode_morse, load_ai_model, is_model_loaded, get_model_info
from utils.morse_correction import (
    correct_morse_character,
    correct_morse_message,
    suggest_morse_corrections,
    apply_correction,
    set_default_threshold,
    prefix_suggest_morse,
    _DEFAULT_THRESHOLD as CORRECTION_DEFAULT_THRESHOLD,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="LoRaShield API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared app state (mirrors Tkinter app_state) ──────────────────────────────
_app_state: dict[str, Any] = {
    "com_port":     "",
    "enc_key":      "LoRaShield2026",
    "ai_loaded":    False,
    "connected":    False,
    "baud_rate":    9600,
    "auto_connect": False,
}

_SETTINGS_FILE = ROOT / "data" / "settings.json"

# Load settings on startup
def _load_settings_on_boot():
    if _SETTINGS_FILE.exists():
        try:
            with _SETTINGS_FILE.open() as fh:
                data = json.load(fh)
            _app_state["com_port"]     = data.get("com_port", "")
            _app_state["enc_key"]      = data.get("enc_key", "LoRaShield2026")
            _app_state["baud_rate"]    = int(data.get("baud_rate", 9600))
            _app_state["auto_connect"] = bool(data.get("auto_connect", False))
        except Exception:
            pass

_load_settings_on_boot()

# ── Active WebSocket clients ───────────────────────────────────────────────────
_ws_clients: list[WebSocket] = []
_event_loop: asyncio.AbstractEventLoop | None = None


def _broadcast_from_thread(payload: dict) -> None:
    """Safely schedule a WebSocket broadcast from a worker thread."""
    loop = _event_loop
    if loop is None or not loop.is_running():
        log.warning("Cannot broadcast serial event: application event loop is unavailable.")
        return
    try:
        asyncio.run_coroutine_threadsafe(_broadcast(payload), loop)
    except RuntimeError:
        log.exception("Could not schedule serial event broadcast.")


async def _broadcast(payload: dict):
    """Send a JSON message to all connected WebSocket clients."""
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


def _serial_rx_callback(data: str):
    """Called on background thread when serial data arrives; schedule WS broadcast."""
    # ── CRC verification ────────────────────────────────────────────────
    parsed = parse_packet(data)
    if not parsed["valid"]:
        # Corrupted packet – broadcast error, do NOT decrypt
        crc_payload = {
            "type": "crc_error",
            "error": parsed["error"],
            "crc_rx": parsed["crc_rx"],
            "crc_calc": parsed["crc_calc"],
            "message": "CRC: FAILED - PACKET CORRUPTED",
        }
        _broadcast_from_thread(crc_payload)
        return

    crc_status = "LEGACY (no CRC)" if parsed["is_legacy"] else f"VALID 0x{parsed['crc_rx']:04X}"
    payload = parsed["payload"]

    key = _app_state.get("enc_key", "")
    if key and ":" in payload:
        plaintext, err = decrypt_message(payload, key)
        if err:
            rx_payload = {"type": "rx", "cipher": payload, "plain": payload,
                          "enc_status": "Decrypt Error", "crc_status": crc_status}
        else:
            rx_payload = {"type": "rx", "cipher": payload, "plain": plaintext,
                          "enc_status": "AES-256 CBC", "crc_status": crc_status}
    else:
        rx_payload = {"type": "rx", "cipher": "", "plain": payload,
                      "enc_status": "None", "crc_status": crc_status}

    # Schedule the coroutine-based broadcast safely from the serial thread.
    _broadcast_from_thread(rx_payload)


# ═══════════════════════════════════════════════════════════════════════════════
# Serial Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/serial/ports")
def get_ports():
    return {"ports": serial_comm.list_com_ports()}


class ConnectRequest(BaseModel):
    port: str
    baud_rate: int = 9600


@app.on_event("startup")
async def _capture_event_loop():
    global _event_loop
    _event_loop = asyncio.get_running_loop()


@app.on_event("shutdown")
async def _shutdown_serial():
    global _event_loop
    serial_comm.stop_receive_loop()
    if serial_comm.is_connected():
        serial_comm.disconnect_esp32()
    _event_loop = None


@app.post("/api/serial/connect")
def connect_serial(req: ConnectRequest):
    ok, msg = serial_comm.connect_esp32(req.port, req.baud_rate)
    if ok:
        _app_state["com_port"] = req.port
        _app_state["baud_rate"] = req.baud_rate
        _app_state["connected"] = True
        serial_comm.start_receive_loop(
            callback=_serial_rx_callback,
            error_callback=lambda err: _broadcast_from_thread(
                {"type": "error", "message": err}
            )
        )
    return {"ok": ok, "message": msg}


@app.post("/api/serial/disconnect")
def disconnect_serial():
    serial_comm.stop_receive_loop()
    ok, msg = serial_comm.disconnect_esp32()
    if ok:
        _app_state["connected"] = False
    return {"ok": ok, "message": msg}


@app.get("/api/serial/status")
def serial_status():
    return {
        "connected": serial_comm.is_connected(),
        "port": _app_state.get("com_port", ""),
        **serial_comm.get_stats(),
    }


class SendRequest(BaseModel):
    message: str
    use_encryption: bool = True


@app.post("/api/serial/send")
async def send_message(req: SendRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "Message cannot be empty")

    key = _app_state.get("enc_key", "") if req.use_encryption else ""

    if key:
        encrypted, err = encrypt_message(msg, key)
        if err:
            raise HTTPException(500, f"Encryption failed: {err}")
        payload = encrypted
        enc_status = "AES-256 CBC"
    else:
        payload = msg
        encrypted = ""
        enc_status = "None"

    # ── Wrap payload in CRC packet ────────────────────────────────────
    crc_val = calculate_crc(payload)
    packet = build_packet(payload)

    ok, result = serial_comm.send_lora(packet)

    if ok:
        # Broadcast TX to terminal
        await _broadcast({
            "type": "tx",
            "plain": msg,
            "cipher": encrypted,
            "enc_status": enc_status,
            "crc_status": f"GENERATED 0x{crc_val:04X}",
        })
        # Save to history
        save_history({
            "direction": "TX",
            "plain_text": msg,
            "morse_code": "",
            "encrypted_data": encrypted,
            "sender": "LOCAL",
            "receiver": _app_state.get("com_port", "Remote"),
            "encryption_status": enc_status,
        })

    return {"ok": ok, "message": result, "crc": crc_val}


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket – Live Serial Terminal
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/serial")
async def ws_serial(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    # Send current connection state
    await websocket.send_json({
        "type": "status",
        "connected": serial_comm.is_connected(),
        "port": _app_state.get("com_port", ""),
    })
    try:
        while True:
            # Keep the socket alive; actual data is pushed via _serial_rx_callback
            await asyncio.sleep(2)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ═══════════════════════════════════════════════════════════════════════════════
# Morse Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class EncodeRequest(BaseModel):
    text: str


class DecodeRequest(BaseModel):
    morse: str


@app.post("/api/morse/encode")
def morse_encode(req: EncodeRequest):
    if not req.text:
        raise HTTPException(400, "Text cannot be empty")
    result = encode_text(req.text)
    return {"morse": result, "input_length": len(req.text)}


@app.post("/api/morse/decode")
def morse_decode(req: DecodeRequest):
    if not req.morse:
        raise HTTPException(400, "Morse input cannot be empty")
    text, confidence, elapsed, method = decode_morse(req.morse)
    return {
        "text": text,
        "confidence": round(confidence * 100, 2),
        "elapsed_ms": round(elapsed * 1000, 1),
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Morse Correction Endpoints  (manual typing error correction)
# These endpoints are completely separate from the AI decoder.
# They use offline Levenshtein edit-distance matching against the ITU
# Morse dictionary and require no internet connection or loaded model.
# ═══════════════════════════════════════════════════════════════════════════════

class CorrectCharRequest(BaseModel):
    morse: str
    max_suggestions: int = 3
    threshold: float = CORRECTION_DEFAULT_THRESHOLD


class CorrectMessageRequest(BaseModel):
    message: str
    threshold: float = CORRECTION_DEFAULT_THRESHOLD


class ApplyCorrectionRequest(BaseModel):
    message: str
    word_index: int
    token_index: int
    replacement_morse: str


@app.post("/api/morse/correct-char")
def morse_correct_char(req: CorrectCharRequest):
    """
    Validate a single Morse token and return correction suggestions.

    Returns:
        is_valid:    True when the input is an exact ITU Morse code.
        char:        Decoded character for a valid input, else null.
        suggestions: Up to max_suggestions closest valid patterns.
        reliable:    False when no suggestions meet the similarity threshold.
    """
    if not req.morse:
        raise HTTPException(400, "Morse input cannot be empty")
    result = correct_morse_character(req.morse, threshold=req.threshold)
    return result


@app.post("/api/morse/correct-message")
def morse_correct_message(req: CorrectMessageRequest):
    """
    Validate every token in a full Morse message and return per-token results.

    Word separator is ' / '; letter separator is a single space.
    Each token is checked independently; valid tokens are returned as-is.

    Returns:
        tokens:      List of per-token result objects.
        has_errors:  True when at least one token is not a valid Morse code.
        all_valid:   Inverse of has_errors.
        word_count:  Number of slash-separated word groups.
        token_count: Total number of Morse tokens processed.
    """
    if not req.message:
        raise HTTPException(400, "Message cannot be empty")
    result = correct_morse_message(req.message, threshold=req.threshold)
    return result


@app.post("/api/morse/apply-correction")
def morse_apply_correction(req: ApplyCorrectionRequest):
    """
    Replace one Morse token in a full message and return the corrected string.

    Called after the user clicks a suggestion chip in the GUI.
    """
    if not req.message:
        raise HTTPException(400, "Message cannot be empty")
    corrected = apply_correction(
        req.message,
        req.word_index,
        req.token_index,
        req.replacement_morse,
    )
    return {"corrected_message": corrected}


@app.get("/api/morse/correction-threshold")
def get_correction_threshold():
    """Return the current default similarity threshold for the correction engine."""
    return {"threshold": CORRECTION_DEFAULT_THRESHOLD}


class PrefixSuggestRequest(BaseModel):
    prefix: str


@app.post("/api/morse/prefix-suggest")
def morse_prefix_suggest(req: PrefixSuggestRequest):
    """
    Return all ITU Morse characters whose code begins with the given prefix.

    This endpoint provides real-time character suggestions as the sender types
    a single Morse symbol (dots and dashes before the character is completed
    with a space).

    Suggestions are determined by pure deterministic prefix filtering over the
    full ITU Morse dictionary.  The BiLSTM model is intentionally not involved:
    the v4 model is a seq2seq decoder that expects a complete tokenised Morse
    message string; it cannot meaningfully rank candidates for an incomplete
    single-character prefix.

    Args:
        prefix: The dot/dash prefix typed so far, e.g. '.' or '.-' or '...'.
                Spaces and slashes are invalid here and will produce an empty
                result.

    Returns:
        suggestions: List of matching characters, each with:
                       char  – the decoded character (e.g. 'A')
                       morse – the full Morse code (e.g. '.-')
                       exact – True when the prefix is itself a complete code
        count:        Total number of suggestions returned.
        prefix:       The stripped prefix that was evaluated.
    """
    candidates = prefix_suggest_morse(req.prefix)
    return {
        "suggestions": candidates,
        "count": len(candidates),
        "prefix": req.prefix.strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ai/load")
def ai_load():
    ok, msg = load_ai_model()
    _app_state["ai_loaded"] = ok
    return {"ok": ok, "message": msg}


@app.get("/api/ai/status")
def ai_status():
    return {
        "loaded": is_model_loaded(),
        **get_model_info(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Encryption Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class EncryptRequest(BaseModel):
    message: str
    key: str


class DecryptRequest(BaseModel):
    encrypted: str
    key: str


@app.post("/api/encrypt/encrypt")
def api_encrypt(req: EncryptRequest):
    if not is_crypto_available():
        raise HTTPException(503, "pycryptodome not installed")
    enc, err = encrypt_message(req.message, req.key)
    if err:
        raise HTTPException(400, err)
    return {"encrypted": enc}


@app.post("/api/encrypt/decrypt")
def api_decrypt(req: DecryptRequest):
    if not is_crypto_available():
        raise HTTPException(503, "pycryptodome not installed")
    plain, err = decrypt_message(req.encrypted, req.key)
    if err:
        raise HTTPException(400, err)
    return {"plaintext": plain}


@app.post("/api/encrypt/generate-key")
def api_generate_key():
    key = generate_key(32)
    return {"key": key}


@app.get("/api/encrypt/status")
def api_crypto_status():
    return {"available": is_crypto_available()}


# ═══════════════════════════════════════════════════════════════════════════════
# CRC Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

class CrcCalculateRequest(BaseModel):
    data: str


class CrcVerifyRequest(BaseModel):
    data: str
    received_crc: int


class CrcBuildRequest(BaseModel):
    payload: str


class CrcParseRequest(BaseModel):
    raw: str


@app.post("/api/crc/calculate")
def api_crc_calculate(req: CrcCalculateRequest):
    """Calculate CRC-16-CCITT for a given data string."""
    if not req.data and req.data != "":
        raise HTTPException(400, "data field required")
    crc = calculate_crc(req.data)
    return {"data_length": len(req.data), "crc": crc, "crc_hex": f"0x{crc:04X}"}


@app.post("/api/crc/verify")
def api_crc_verify(req: CrcVerifyRequest):
    """Verify that a data string matches a given CRC value."""
    ok = verify_crc(req.data, req.received_crc)
    computed = calculate_crc(req.data)
    return {
        "valid": ok,
        "computed_crc": computed,
        "computed_crc_hex": f"0x{computed:04X}",
        "received_crc": req.received_crc,
        "received_crc_hex": f"0x{req.received_crc:04X}",
    }


@app.post("/api/crc/build-packet")
def api_crc_build(req: CrcBuildRequest):
    """Wrap a payload in a JSON CRC packet (same as the TX flow)."""
    if not req.payload and req.payload != "":
        raise HTTPException(400, "payload field required")
    packet = build_packet(req.payload)
    crc = calculate_crc(req.payload)
    return {"packet": packet, "crc": crc, "crc_hex": f"0x{crc:04X}"}


@app.post("/api/crc/parse-packet")
def api_crc_parse(req: CrcParseRequest):
    """Parse and CRC-verify a raw packet string (same as the RX flow)."""
    if not req.raw:
        raise HTTPException(400, "raw field required")
    result = parse_packet(req.raw)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# History Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/history")
def api_get_history(q: str = ""):
    if q:
        return {"records": search_history(q)}
    return {"records": load_history()}


@app.post("/api/history")
def api_save_history(record: dict):
    ok, msg = save_history(record)
    if not ok:
        raise HTTPException(500, msg)
    return {"ok": True, "message": msg}


@app.delete("/api/history/clear")
def api_clear_history():
    ok, msg = clear_history()
    return {"ok": ok, "message": msg}


@app.get("/api/history/export")
def api_export_history():
    ok, msg = export_history()
    return {"ok": ok, "message": msg}


@app.delete("/api/history/{index}")
def api_delete_history(index: int):
    ok, msg = delete_history_record(index)
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


# ═══════════════════════════════════════════════════════════════════════════════
# Settings Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/settings")
def api_get_settings():
    if _SETTINGS_FILE.exists():
        try:
            with _SETTINGS_FILE.open() as fh:
                return json.load(fh)
        except Exception:
            pass
    return _app_state


class SettingsPayload(BaseModel):
    com_port: str = ""
    baud_rate: str = "9600"
    enc_key: str = ""
    auto_connect: bool = False
    ai_threshold: float = 0.75
    theme: str = "Dark"


@app.post("/api/settings")
def api_save_settings(settings: SettingsPayload):
    data = settings.model_dump()
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_FILE.open("w") as fh:
        json.dump(data, fh, indent=2)
    # Update app state
    _app_state["com_port"]     = data.get("com_port", "")
    _app_state["enc_key"]      = data.get("enc_key", "")
    _app_state["baud_rate"]    = int(data.get("baud_rate", 9600))
    _app_state["auto_connect"] = bool(data.get("auto_connect", False))
    return {"ok": True, "message": "Settings saved."}


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard / Stats Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/dashboard")
def api_dashboard():
    stats = serial_comm.get_stats()
    return {
        "connected":       serial_comm.is_connected(),
        "port":            _app_state.get("com_port", ""),
        "msgs_sent":       stats.get("msgs_sent", 0),
        "msgs_received":   stats.get("msgs_received", 0),
        "last_comm_time":  stats.get("last_comm_time", "Never"),
        "ai_loaded":       is_model_loaded(),
        "crypto_available": is_crypto_available(),
        "record_count":    record_count(),
    }
