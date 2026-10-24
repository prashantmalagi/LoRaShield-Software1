"""
LoRaShield – AES Encryption Utility
--------------------------------------
Implements AES-256 CBC encryption and decryption using pycryptodome.
The secret key is hashed with SHA-256 to produce a 32-byte AES key.

All errors are logged via utils.logger and returned as (result, error_string)
tuples so the UI can display them without try/except boilerplate.
"""

import base64
import hashlib
import os
import secrets
import string

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad, unpad
    CRYPTO_AVAILABLE = True
    log.debug("pycryptodome loaded successfully.")
except ImportError:
    CRYPTO_AVAILABLE = False
    log.warning("pycryptodome not installed. Encryption/decryption unavailable.")


# ── Key utilities ─────────────────────────────────────────────────────────────

def _derive_key(key: str) -> bytes:
    """Derive a 32-byte AES key from an arbitrary-length string using SHA-256."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def generate_key(length: int = 32) -> str:
    """
    Generate a cryptographically strong random key string.

    Args:
        length: Number of characters in the generated key (default 32).

    Returns:
        A random alphanumeric+symbol key string of the requested length.
    """
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+[]{}|"
    key = "".join(secrets.choice(alphabet) for _ in range(length))
    log.info("New encryption key generated (length=%d).", length)
    return key


# ── Encryption ────────────────────────────────────────────────────────────────

def encrypt_message(message: str, key: str) -> tuple[str, str]:
    """
    Encrypt a plaintext message using AES-256 CBC.

    Args:
        message: Plaintext string to encrypt.
        key:     Secret key (any length; SHA-256 hashed internally).

    Returns:
        (encrypted_b64: str, error: str)
        On success: (base64-encoded ``IV:CIPHERTEXT``, "")
        On failure: ("", error_message)
    """
    if not CRYPTO_AVAILABLE:
        err = "pycryptodome not installed. Run: pip install pycryptodome"
        log.error(err)
        return ("", err)
    if not message:
        return ("", "Message cannot be empty.")
    if not key:
        return ("", "Encryption key cannot be empty.")

    try:
        key_bytes = _derive_key(key)
        cipher    = AES.new(key_bytes, AES.MODE_CBC)
        ct_bytes  = cipher.encrypt(pad(message.encode("utf-8"), AES.block_size))
        iv_b64    = base64.b64encode(cipher.iv).decode("ascii")
        ct_b64    = base64.b64encode(ct_bytes).decode("ascii")
        result    = f"{iv_b64}:{ct_b64}"
        log.info("Message encrypted successfully (%d bytes plaintext).", len(message))
        return (result, "")
    except Exception as exc:
        log.exception("Encryption error.")
        return ("", f"Encryption error: {exc}")


# ── Decryption ────────────────────────────────────────────────────────────────

def decrypt_message(encrypted: str, key: str) -> tuple[str, str]:
    """
    Decrypt a base64-encoded AES-256 CBC ciphertext.

    Args:
        encrypted: Base64 string in ``IV:CIPHERTEXT`` format
                   (as produced by :func:`encrypt_message`).
        key:       Secret key used during encryption.

    Returns:
        (plaintext: str, error: str)
        On success: (decrypted plaintext, "")
        On failure: ("", error_message)
    """
    if not CRYPTO_AVAILABLE:
        err = "pycryptodome not installed. Run: pip install pycryptodome"
        log.error(err)
        return ("", err)
    if not encrypted:
        return ("", "Encrypted text cannot be empty.")
    if not key:
        return ("", "Decryption key cannot be empty.")

    try:
        parts = encrypted.strip().split(":")
        if len(parts) != 2:
            return ("", "Invalid encrypted format. Expected 'IV:CIPHERTEXT'.")
        iv       = base64.b64decode(parts[0])
        ct       = base64.b64decode(parts[1])
        key_bytes = _derive_key(key)
        cipher   = AES.new(key_bytes, AES.MODE_CBC, iv)
        pt       = unpad(cipher.decrypt(ct), AES.block_size)
        plaintext = pt.decode("utf-8")
        log.info("Message decrypted successfully (%d bytes plaintext).", len(plaintext))
        return (plaintext, "")
    except Exception as exc:
        log.exception("Decryption error.")
        return ("", f"Decryption error: {exc}")


# ── Status ────────────────────────────────────────────────────────────────────

def is_crypto_available() -> bool:
    """Return True when pycryptodome is installed and importable."""
    return CRYPTO_AVAILABLE
