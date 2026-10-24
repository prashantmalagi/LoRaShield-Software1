"""
LoRaShield – CRC-16 Packet Integrity Module
---------------------------------------------
Provides software CRC-16-CCITT (polynomial 0x1021, initial value 0xFFFF)
for LoRa packet integrity checking.

NOTE on hardware CRC
---------------------
The SX1278 / Ra-02 LoRa radio has a built-in hardware CRC feature that
is enabled or disabled at the firmware (ESP32 / Arduino) level.  When
hardware CRC is active the radio silently drops packets with a bad CRC
before they ever reach the serial UART, so the Python host-side software
never sees corrupted packets.

However, because:
  1. The current serial protocol is raw UTF-8 lines (no framing that
     exposes the hardware CRC result to the host), and
  2. The firmware configuration is unknown / out of scope,

this module implements a **software CRC-16-CCITT** that is appended to
every packet as a JSON field.  This gives end-to-end integrity checking
that works regardless of the firmware CRC setting.

Packet format (JSON, one line on the wire):
    {
        "payload": "<AES-256-CBC ciphertext or plaintext>",
        "crc":     12345
    }

Usage:
    from utils.crc import build_packet, parse_packet, calculate_crc, verify_crc

    # Sender
    packet_json = build_packet(encrypted_payload)

    # Receiver
    result = parse_packet(raw_line)
    if result["valid"]:
        plaintext = decrypt(result["payload"])
    else:
        log("Packet corrupted")
"""

import json
from utils.logger import get_logger

log = get_logger(__name__)

# ── CRC-16-CCITT parameters ───────────────────────────────────────────────────
_POLY: int = 0x1021   # CRC-16-CCITT polynomial
_INIT: int = 0xFFFF   # Initial value
_MASK: int = 0xFFFF


# ── Pre-computed look-up table for speed ──────────────────────────────────────

def _build_lut() -> list:
    """Build a 256-entry CRC-16-CCITT look-up table."""
    table = []
    for byte in range(256):
        crc = 0
        for _ in range(8):
            if (byte ^ (crc >> 8)) & 0x80:
                crc = ((crc << 1) ^ _POLY) & _MASK
            else:
                crc = (crc << 1) & _MASK
            byte = (byte << 1) & 0xFF
        table.append(crc)
    return table


_LUT: list = _build_lut()


# ── Public API ────────────────────────────────────────────────────────────────

def calculate_crc(data) -> int:
    """
    Calculate the CRC-16-CCITT checksum for *data*.

    Args:
        data: The payload string (UTF-8 encoded internally) or raw bytes.

    Returns:
        An integer in the range [0, 65535].

    Raises:
        TypeError: If *data* is neither ``str`` nor ``bytes``.
    """
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, bytes):
        raw = data
    else:
        raise TypeError(f"data must be str or bytes, got {type(data).__name__}")

    crc = _INIT
    for byte in raw:
        pos = (byte ^ (crc >> 8)) & 0xFF
        crc = (_LUT[pos] ^ ((crc << 8) & _MASK)) & _MASK

    log.debug("CRC-16 calculated: 0x%04X (%d) for %d byte(s).", crc, crc, len(raw))
    return crc


def verify_crc(data, received_crc: int) -> bool:
    """
    Verify that the CRC of *data* matches *received_crc*.

    Args:
        data:         The payload (same value passed to :func:`calculate_crc`).
        received_crc: The CRC value extracted from the received packet.

    Returns:
        ``True`` when computed CRC matches *received_crc*, ``False`` otherwise.
    """
    computed = calculate_crc(data)
    match = (computed == received_crc)
    if match:
        log.debug("CRC verification PASSED (0x%04X).", computed)
    else:
        log.warning(
            "CRC verification FAILED: computed=0x%04X, received=0x%04X.",
            computed, received_crc,
        )
    return match


# ── Packet helpers ────────────────────────────────────────────────────────────

def build_packet(payload: str) -> str:
    """
    Wrap *payload* in a JSON packet that includes its CRC.

    The returned string is a single-line JSON object ready to be sent over
    the serial link::

        {"payload": "<payload>", "crc": <int>}

    Args:
        payload: Plaintext or encrypted payload string.

    Returns:
        JSON string (no trailing newline; ``send_lora`` adds ``\\n``).
    """
    crc = calculate_crc(payload)
    packet = json.dumps({"payload": payload, "crc": crc}, separators=(",", ":"))
    log.info("Packet built – payload_len=%d, CRC=0x%04X (%d).", len(payload), crc, crc)
    return packet


def parse_packet(raw: str) -> dict:
    """
    Parse a received JSON packet and verify its CRC.

    Args:
        raw: Raw line received from the serial port.

    Returns:
        A dict with keys:

        * ``valid``    – ``True`` when the packet is intact.
        * ``payload``  – The payload string (do NOT decrypt when ``valid`` is False).
        * ``crc_rx``   – CRC embedded in the packet (``-1`` if unparseable).
        * ``crc_calc`` – CRC computed from payload (``-1`` if unparseable).
        * ``error``    – Human-readable error string, or ``""`` on success.
        * ``is_legacy``– ``True`` when the raw line is not a JSON-wrapped packet.
    """
    raw = raw.strip()

    # ── Legacy / non-JSON line ─────────────────────────────────────────────────
    if not raw.startswith(("{", "[")):
        log.debug("Legacy (non-JSON) packet received: no CRC field present.")
        return {
            "valid":     True,
            "payload":   raw,
            "crc_rx":    -1,
            "crc_calc":  -1,
            "error":     "",
            "is_legacy": True,
        }

    # ── JSON packet ────────────────────────────────────────────────────────────
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Malformed JSON packet: %s – raw='%s'", exc, raw[:80])
        return {
            "valid":     False,
            "payload":   raw,
            "crc_rx":    -1,
            "crc_calc":  -1,
            "error":     f"Malformed JSON: {exc}",
            "is_legacy": False,
        }

    if not isinstance(obj, dict):
        log.error("JSON packet must be an object.")
        return {
            "valid": False,
            "payload": "",
            "crc_rx": -1,
            "crc_calc": -1,
            "error": "JSON packet must be an object.",
            "is_legacy": False,
        }

    if "payload" not in obj or "crc" not in obj:
        log.error("JSON packet missing 'payload' or 'crc' field.")
        return {
            "valid":     False,
            "payload":   obj.get("payload", ""),
            "crc_rx":    -1,
            "crc_calc":  -1,
            "error":     "Missing 'payload' or 'crc' in JSON packet.",
            "is_legacy": False,
        }

    if not isinstance(obj["payload"], str):
        return {
            "valid": False,
            "payload": "",
            "crc_rx": -1,
            "crc_calc": -1,
            "error": "Invalid 'payload' field type; expected a string.",
            "is_legacy": False,
        }

    crc_value = obj["crc"]
    if isinstance(crc_value, bool) or not isinstance(crc_value, int) or not 0 <= crc_value <= _MASK:
        log.error("Invalid CRC field type or range: %r", crc_value)
        return {
            "valid": False,
            "payload": obj["payload"],
            "crc_rx": -1,
            "crc_calc": calculate_crc(obj["payload"]),
            "error": "Invalid 'crc' field; expected an integer from 0 to 65535.",
            "is_legacy": False,
        }

    payload = obj["payload"]
    crc_rx = crc_value
    crc_calc = calculate_crc(payload)
    valid    = (crc_rx == crc_calc)

    if valid:
        log.info(
            "Packet received – CRC VALID (0x%04X). payload_len=%d.",
            crc_calc, len(payload),
        )
    else:
        log.error(
            "Packet CORRUPTED – CRC mismatch: received=0x%04X, computed=0x%04X. "
            "Packet rejected.",
            crc_rx, crc_calc,
        )

    return {
        "valid":     valid,
        "payload":   payload,
        "crc_rx":    crc_rx,
        "crc_calc":  crc_calc,
        "error":     "" if valid else (
            f"CRC mismatch: received=0x{crc_rx:04X}, computed=0x{crc_calc:04X}"
        ),
        "is_legacy": False,
    }
