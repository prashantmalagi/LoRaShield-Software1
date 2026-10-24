"""
LoRaShield – Serial Communication Utility
-------------------------------------------
Handles ESP32 + SX1278 LoRa serial communication via pyserial.

Protocol:
    The ESP32 firmware is expected to speak a simple newline-delimited
    protocol.  Every outbound message is sent as a raw UTF-8 line.
    Received lines from the ESP32 are returned as-is for the UI to parse.

    If the ESP32 uses AT-command mode (e.g., REYAX RYLR998), you can
    set USE_AT_FRAMING = True and the send function will wrap messages
    in:  AT+SEND=0,<length>,<data>\r\n
    and wait for "+OK" acknowledgment.

    Set USE_AT_FRAMING = False (default) for raw-line firmware.

All events are logged via utils.logger.
"""

import threading
import time
from typing import Callable

from utils.logger import get_logger

log = get_logger(__name__)

# ── AT-command framing toggle ─────────────────────────────────────────────────
# Set to True if your ESP32 firmware uses AT+SEND=0,<len>,<data> syntax.
USE_AT_FRAMING: bool = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
    log.debug("pyserial loaded successfully.")
except ImportError:
    SERIAL_AVAILABLE = False
    log.warning("pyserial not installed. Serial communication unavailable.")

# ── Module-level state ────────────────────────────────────────────────────────
_connection:      "serial.Serial | None" = None
_receive_thread:  threading.Thread | None = None
_running:         bool = False
_msgs_sent:       int = 0
_msgs_received:   int = 0
_last_comm_time:  str = "Never"


# ── Statistics ────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return a snapshot of transmission statistics."""
    return {
        "msgs_sent":      _msgs_sent,
        "msgs_received":  _msgs_received,
        "last_comm_time": _last_comm_time,
        "connected":      is_connected(),
        "port":           _connection.port if is_connected() else "",
    }


# ── Port Discovery ────────────────────────────────────────────────────────────

def list_com_ports() -> list[str]:
    """
    List all available COM ports on the system.

    Returns:
        List of port name strings, e.g. ``["COM3", "COM5"]``.
        Returns ``["No ports found"]`` when none are detected.
        Returns ``["pyserial not installed"]`` when the library is missing.
    """
    if not SERIAL_AVAILABLE:
        return ["pyserial not installed"]
    ports = serial.tools.list_ports.comports()
    found = [p.device for p in sorted(ports, key=lambda p: p.device)]
    log.debug("COM port scan found: %s", found)
    return found if found else ["No ports found"]


# ── Connection Management ─────────────────────────────────────────────────────

def connect_esp32(
    port: str,
    baud_rate: int = 9600,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Open a serial connection to the ESP32 + LoRa module.

    After opening the port the function sends a blank line to flush the
    ESP32 UART buffer and waits briefly for any startup banner.

    Args:
        port:         COM port string, e.g. ``"COM3"``.
        baud_rate:    Baud rate matching the ESP32 firmware (default 9600).
        log_callback: Optional callable receiving plain-text log messages.

    Returns:
        ``(True, message)`` on success, ``(False, error)`` on failure.
    """
    global _connection, _running

    if not SERIAL_AVAILABLE:
        return (False, "pyserial not installed. Run: pip install pyserial")

    if not port or port in ("No ports found", "pyserial not installed"):
        return (False, "Invalid port selected.")

    if _connection and _connection.is_open:
        return (False, f"Already connected to {_connection.port}.")

    try:
        _connection = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
        )
        # Flush any stale data in the UART buffer
        _connection.reset_input_buffer()
        _connection.reset_output_buffer()
        # Send a blank line to wake the ESP32
        _connection.write(b"\r\n")
        time.sleep(0.3)

        _running = True
        msg = f"Connected to {port} at {baud_rate} baud."
        log.info(msg)
        if log_callback:
            log_callback(msg)
        return (True, msg)
    except serial.SerialException as exc:
        log.error("Serial connection failed: %s", exc)
        _connection = None
        return (False, f"Connection failed: {exc}")
    except Exception as exc:
        log.exception("Unexpected error during connection.")
        _connection = None
        return (False, f"Unexpected error: {exc}")


def disconnect_esp32(
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Close the active serial connection to the ESP32.

    The background receive thread is signalled to stop before the port
    is closed, preventing read errors in the thread.

    Returns:
        ``(True, message)`` on success, ``(False, error)`` on failure.
    """
    global _connection, _running

    _running = False

    if _connection and _connection.is_open:
        port = _connection.port
        try:
            _connection.close()
            _connection = None
            msg = f"Disconnected from {port}."
            log.info(msg)
            if log_callback:
                log_callback(msg)
            return (True, msg)
        except Exception as exc:
            log.exception("Error while disconnecting.")
            return (False, f"Disconnect error: {exc}")

    return (False, "No active connection to disconnect.")


def is_connected() -> bool:
    """Return ``True`` when a serial connection is currently open."""
    return _connection is not None and _connection.is_open


# ── Transmission ──────────────────────────────────────────────────────────────

def send_lora(
    message: str,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Transmit a message string over the LoRa module via serial.

    When ``USE_AT_FRAMING`` is ``True`` the message is wrapped in:
    ``AT+SEND=0,<length>,<data>\\r\\n``
    and the function waits up to 2 s for a ``+OK`` acknowledgment.

    When ``USE_AT_FRAMING`` is ``False`` (default) the message is sent
    as a plain UTF-8 line terminated by ``\\n``.

    Args:
        message:      Text or encrypted payload to transmit.
        log_callback: Optional callable receiving plain-text log messages.

    Returns:
        ``(True, "[TX] message")`` on success, ``(False, error)`` on failure.
    """
    global _msgs_sent, _last_comm_time

    if not is_connected():
        return (False, "Not connected. Please connect to a COM port first.")
    if not message:
        return (False, "Cannot send empty message.")

    try:
        if USE_AT_FRAMING:
            payload = f"AT+SEND=0,{len(message)},{message}\r\n".encode("utf-8")
            _connection.write(payload)

            # Wait for +OK acknowledgment (up to 2 s)
            deadline = time.time() + 2.0
            ack = ""
            while time.time() < deadline:
                line = _connection.readline()
                if line:
                    ack = line.decode("utf-8", errors="replace").strip()
                    if "+OK" in ack:
                        break
            if "+OK" not in ack:
                return (False, f"No +OK from ESP32 (got: '{ack}').")
        else:
            payload = (message + "\n").encode("utf-8")
            _connection.write(payload)

        _msgs_sent += 1
        _last_comm_time = time.strftime("%Y-%m-%d %H:%M:%S")
        result = f"[TX] {message}"
        log.info("LoRa TX: %s", message)
        if log_callback:
            log_callback(result)
        return (True, result)

    except Exception as exc:
        log.exception("Send error.")
        return (False, f"Send error: {exc}")


def receive_lora(timeout: float = 2.0) -> tuple[str, str]:
    """
    Read a single newline-terminated response from the serial buffer.

    Args:
        timeout: Maximum seconds to wait for incoming data.

    Returns:
        ``(data: str, "")`` on success,
        ``("", error: str)`` when nothing arrives or on error.
    """
    global _msgs_received, _last_comm_time

    if not is_connected():
        return ("", "Not connected.")

    try:
        _connection.timeout = timeout
        raw = _connection.readline()
        if raw:
            data = raw.decode("utf-8", errors="replace").strip()
            if data:
                _msgs_received += 1
                _last_comm_time = time.strftime("%Y-%m-%d %H:%M:%S")
                log.info("LoRa RX: %s", data)
                return (data, "")
        return ("", "No data received within timeout.")
    except Exception as exc:
        log.exception("Receive error.")
        return ("", f"Receive error: {exc}")


def start_receive_loop(
    callback:       Callable[[str], None],
    error_callback: Callable[[str], None] | None = None,
) -> None:
    """
    Start a background daemon thread that continuously reads from the
    serial port and calls ``callback`` with each received line.

    The loop runs until :func:`stop_receive_loop` is called or the
    connection is lost.

    Args:
        callback:       Called on the background thread with each received line.
        error_callback: Called on the background thread if an error occurs.
    """
    global _receive_thread, _running

    def _loop() -> None:
        log.debug("Receive loop started.")
        while _running and is_connected():
            data, err = receive_lora(timeout=1.0)
            if data:
                callback(data)
            elif err and "No data" not in err:
                log.warning("Receive loop error: %s", err)
                if error_callback:
                    error_callback(err)
            time.sleep(0.05)
        log.debug("Receive loop stopped.")

    _running = True
    _receive_thread = threading.Thread(target=_loop, daemon=True, name="LoRa-RX")
    _receive_thread.start()


def stop_receive_loop() -> None:
    """Signal the background receive thread to stop."""
    global _running
    _running = False
    log.debug("Receive loop stop requested.")
