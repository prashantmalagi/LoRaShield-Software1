"""
LoRaShield – CRC Packet Integrity Tests
=========================================
Tests for utils/crc.py

Run from the project root:
    python -m pytest tests/test_crc.py -v
    # or without pytest:
    python tests/test_crc.py

Test categories
---------------
1.  Normal packet → CRC matches (VALID).
2.  Modified payload → CRC mismatch (CORRUPTED).
3.  Empty payload.
4.  Short message.
5.  Long message.
6.  Multiple packets in sequence.
7.  AES encryption/decryption still works with CRC flow.
8.  Morse correction still works (import check).
9.  Legacy / non-JSON packet handling.
10. Malformed JSON packet handling.
11. build_packet() output is valid JSON.
12. verify_crc() direct API.
13. CRC-16 known vectors.
14. Bytes input to calculate_crc().
15. Invalid type to calculate_crc().
"""

import sys
import os
import json
import unittest

# ── Ensure project root is on sys.path ────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.crc import calculate_crc, verify_crc, build_packet, parse_packet


# ═════════════════════════════════════════════════════════════════════════════
# Helper
# ═════════════════════════════════════════════════════════════════════════════

def _tamper(packet_json: str, replacement: str = "TAMPERED") -> str:
    """Replace the payload inside a built packet JSON with *replacement*."""
    obj = json.loads(packet_json)
    obj["payload"] = replacement
    return json.dumps(obj, separators=(",", ":"))


def _flip_crc(packet_json: str) -> str:
    """Flip a single bit in the CRC value to simulate corruption."""
    obj = json.loads(packet_json)
    obj["crc"] = (obj["crc"] ^ 0x0001) & 0xFFFF
    return json.dumps(obj, separators=(",", ":"))


# ═════════════════════════════════════════════════════════════════════════════
# 1. Normal packet – CRC VALID
# ═════════════════════════════════════════════════════════════════════════════

class TestNormalPacket(unittest.TestCase):
    """Test 1: A normally built packet verifies successfully."""

    def test_valid_short(self):
        payload = "Hello LoRa"
        packet  = build_packet(payload)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"], "Normal packet should be VALID")
        self.assertEqual(result["payload"], payload)
        self.assertFalse(result["is_legacy"])
        self.assertEqual(result["error"], "")

    def test_valid_with_encryption_format(self):
        """Simulate an AES-256 CBC ciphertext (IV:CIPHERTEXT format)."""
        payload = "abc123==:XYZciphertext+base64=="
        packet  = build_packet(payload)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], payload)

    def test_crc_values_match(self):
        payload = "Test CRC round-trip"
        packet  = build_packet(payload)
        obj     = json.loads(packet)
        self.assertEqual(obj["crc"], calculate_crc(payload))


# ═════════════════════════════════════════════════════════════════════════════
# 2. Modified payload → CRC MISMATCH
# ═════════════════════════════════════════════════════════════════════════════

class TestCorruptedPacket(unittest.TestCase):
    """Test 2: Tampering with the payload must produce CRC FAILED."""

    def test_tampered_payload(self):
        payload = "Hello LoRa"
        packet  = build_packet(payload)
        bad     = _tamper(packet, "EVIL_PAYLOAD")
        result  = parse_packet(bad)
        self.assertFalse(result["valid"], "Tampered payload must be INVALID")
        self.assertNotEqual(result["error"], "")

    def test_flipped_crc_bit(self):
        payload = "Hello LoRa"
        packet  = build_packet(payload)
        bad     = _flip_crc(packet)
        result  = parse_packet(bad)
        self.assertFalse(result["valid"])

    def test_appended_byte(self):
        payload = "Sensitive data"
        packet  = build_packet(payload)
        obj     = json.loads(packet)
        obj["payload"] += "X"           # append a single character
        bad    = json.dumps(obj, separators=(",", ":"))
        result = parse_packet(bad)
        self.assertFalse(result["valid"])

    def test_missing_last_char(self):
        payload = "Sensitive data"
        packet  = build_packet(payload)
        obj     = json.loads(packet)
        obj["payload"] = obj["payload"][:-1]   # drop last character
        bad    = json.dumps(obj, separators=(",", ":"))
        result = parse_packet(bad)
        self.assertFalse(result["valid"])


# ═════════════════════════════════════════════════════════════════════════════
# 3. Empty payload
# ═════════════════════════════════════════════════════════════════════════════

class TestEmptyPayload(unittest.TestCase):
    """Test 3: An empty string payload is legal and must round-trip."""

    def test_empty_string(self):
        payload = ""
        crc     = calculate_crc(payload)
        self.assertIsInstance(crc, int)
        self.assertGreaterEqual(crc, 0)
        self.assertLessEqual(crc, 0xFFFF)

        packet = build_packet(payload)
        result = parse_packet(packet)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], "")

    def test_verify_crc_empty(self):
        crc = calculate_crc("")
        self.assertTrue(verify_crc("", crc))
        self.assertFalse(verify_crc("a", crc))


# ═════════════════════════════════════════════════════════════════════════════
# 4. Short messages
# ═════════════════════════════════════════════════════════════════════════════

class TestShortMessages(unittest.TestCase):
    """Test 4: Single-character and two-character payloads."""

    CASES = ["A", "Z", "0", "!", "a", " ", "\t", "AB", "Hi"]

    def test_all_short(self):
        for payload in self.CASES:
            with self.subTest(payload=payload):
                packet = build_packet(payload)
                result = parse_packet(packet)
                self.assertTrue(result["valid"], f"Short payload '{payload}' should be VALID")
                self.assertEqual(result["payload"], payload)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Long messages
# ═════════════════════════════════════════════════════════════════════════════

class TestLongMessages(unittest.TestCase):
    """Test 5: Payloads up to and exceeding typical LoRa MTU (255 bytes)."""

    def test_255_bytes(self):
        payload = "A" * 255
        packet  = build_packet(payload)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], payload)

    def test_512_bytes(self):
        """Longer than LoRa MTU – software CRC still works."""
        payload = "B" * 512
        packet  = build_packet(payload)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])

    def test_1kb_payload(self):
        import string
        payload = (string.ascii_lowercase * 40)[:1000]
        packet  = build_packet(payload)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])

    def test_unicode_payload(self):
        payload = "こんにちは LoRa 📡 CRC test"
        packet  = build_packet(payload)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], payload)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Multiple packets in sequence
# ═════════════════════════════════════════════════════════════════════════════

class TestMultiplePackets(unittest.TestCase):
    """Test 6: CRC must be independent per packet."""

    PAYLOADS = [
        "Packet 1",
        "Second packet with more data",
        "abc123==:ENCRYPTED_BLOB==",
        "X" * 100,
        "",
        "Final packet",
    ]

    def test_sequence(self):
        for i, payload in enumerate(self.PAYLOADS):
            with self.subTest(packet_index=i):
                packet = build_packet(payload)
                result = parse_packet(packet)
                self.assertTrue(result["valid"], f"Packet {i} should be valid")
                self.assertEqual(result["payload"], payload)

    def test_crc_differs_across_packets(self):
        """Different payloads must produce different CRCs (collision check)."""
        payloads = ["Hello", "World", "LoRa", "CRC", "Test"]
        crcs = [calculate_crc(p) for p in payloads]
        # All CRCs should be unique for these distinct inputs
        self.assertEqual(len(crcs), len(set(crcs)), "CRC collision detected among test payloads")


# ═════════════════════════════════════════════════════════════════════════════
# 7. AES encryption/decryption still works with CRC flow
# ═════════════════════════════════════════════════════════════════════════════

class TestAesWithCrc(unittest.TestCase):
    """Test 7: End-to-end: encrypt → build_packet → parse_packet → decrypt."""

    def setUp(self):
        try:
            from utils.encryption import encrypt_message, decrypt_message, is_crypto_available
            self._encrypt = encrypt_message
            self._decrypt = decrypt_message
            self._available = is_crypto_available()
        except ImportError:
            self._available = False

    def test_aes_encrypt_decrypt_still_works(self):
        """AES encrypt/decrypt must succeed independently of CRC."""
        if not self._available:
            self.skipTest("pycryptodome not installed")
        msg = "LoRaShield AES test"
        key = "TestKey1234"
        enc, err = self._encrypt(msg, key)
        self.assertEqual(err, "", f"Encrypt error: {err}")
        self.assertIn(":", enc, "Encrypted should be IV:CIPHERTEXT format")

        dec, err2 = self._decrypt(enc, key)
        self.assertEqual(err2, "", f"Decrypt error: {err2}")
        self.assertEqual(dec, msg)

    def test_aes_plus_crc_round_trip(self):
        """Full TX→RX pipeline: encrypt, wrap in CRC packet, verify, decrypt."""
        if not self._available:
            self.skipTest("pycryptodome not installed")
        msg = "Secure LoRa message via CRC"
        key = "MySecretKey2026"

        # TX side
        enc, _ = self._encrypt(msg, key)
        packet = build_packet(enc)

        # RX side
        result = parse_packet(packet)
        self.assertTrue(result["valid"], "CRC must be VALID after encrypt+wrap")
        self.assertEqual(result["payload"], enc)

        dec, err = self._decrypt(result["payload"], key)
        self.assertEqual(err, "")
        self.assertEqual(dec, msg)

    def test_corrupted_packet_not_decrypted(self):
        """A tampered AES packet must fail CRC; decryption must NOT be attempted."""
        if not self._available:
            self.skipTest("pycryptodome not installed")
        msg = "Private data"
        key = "MySecretKey2026"
        enc, _ = self._encrypt(msg, key)
        packet  = build_packet(enc)
        bad     = _tamper(packet, "CORRUPTED_CIPHER")

        result  = parse_packet(bad)
        self.assertFalse(result["valid"])
        # Caller contract: do NOT attempt decryption when valid=False
        # (just verify we correctly signal the corruption)
        self.assertIn("CRC mismatch", result["error"])

    def test_wrong_key_produces_decrypt_error_not_crc_error(self):
        """A valid packet with the wrong key should pass CRC but fail decryption."""
        if not self._available:
            self.skipTest("pycryptodome not installed")
        msg = "Hello"
        enc, _ = self._encrypt(msg, "correct_key")
        packet  = build_packet(enc)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])   # CRC is fine
        _, err = self._decrypt(result["payload"], "wrong_key")
        self.assertNotEqual(err, "", "Decryption with wrong key must return an error")


# ═════════════════════════════════════════════════════════════════════════════
# 8. Morse correction still works (import check)
# ═════════════════════════════════════════════════════════════════════════════

class TestMorseCorrectionUnchanged(unittest.TestCase):
    """Test 8: Verify Morse correction module is untouched and importable."""

    def test_morse_correction_importable(self):
        from utils.morse_correction import (
            correct_morse_character,
            correct_morse_message,
            apply_correction,
        )
        # Basic sanity: '.-' should decode to 'A'
        result = correct_morse_character(".-")
        self.assertTrue(result.get("is_valid"), ".- must be a valid Morse code")
        self.assertEqual(result.get("char"), "A")

    def test_morse_correction_message(self):
        from utils.morse_correction import correct_morse_message
        result = correct_morse_message(".- -... -.-.")
        self.assertFalse(result.get("has_errors"), "Valid Morse must have no errors")

    def test_crc_module_does_not_interfere(self):
        """Importing CRC module must not break Morse correction."""
        from utils.crc import calculate_crc
        from utils.morse_correction import correct_morse_character
        crc = calculate_crc(".-")
        result = correct_morse_character(".-")
        self.assertTrue(result["is_valid"])
        self.assertIsInstance(crc, int)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Legacy / non-JSON packet handling
# ═════════════════════════════════════════════════════════════════════════════

class TestLegacyPackets(unittest.TestCase):
    """Test 9: Raw string lines (no JSON/CRC) must be marked as legacy=True."""

    def test_plain_string(self):
        result = parse_packet("Hello plain text")
        self.assertTrue(result["valid"])
        self.assertTrue(result["is_legacy"])
        self.assertEqual(result["payload"], "Hello plain text")
        self.assertEqual(result["crc_rx"], -1)

    def test_aes_cipher_plain_string(self):
        result = parse_packet("abc123:ENCRYPTEDBLOB==")
        self.assertTrue(result["valid"])
        self.assertTrue(result["is_legacy"])

    def test_empty_string_legacy(self):
        result = parse_packet("")
        self.assertTrue(result["valid"])
        self.assertTrue(result["is_legacy"])

    def test_whitespace_only(self):
        result = parse_packet("   ")
        self.assertTrue(result["valid"])
        self.assertTrue(result["is_legacy"])


# ═════════════════════════════════════════════════════════════════════════════
# 10. Malformed JSON packet handling
# ═════════════════════════════════════════════════════════════════════════════

class TestMalformedPackets(unittest.TestCase):
    """Test 10: Malformed JSON must be marked invalid."""

    def test_broken_json(self):
        result = parse_packet("{bad json")
        self.assertFalse(result["valid"])
        self.assertIn("Malformed JSON", result["error"])

    def test_missing_crc_field(self):
        raw = json.dumps({"payload": "hello"}, separators=(",", ":"))
        result = parse_packet(raw)
        self.assertFalse(result["valid"])
        self.assertIn("Missing", result["error"])

    def test_missing_payload_field(self):
        raw = json.dumps({"crc": 12345}, separators=(",", ":"))
        result = parse_packet(raw)
        self.assertFalse(result["valid"])

    def test_crc_string_is_rejected_without_raising(self):
        raw = json.dumps({"payload": "hello", "crc": "bad"})
        result = parse_packet(raw)
        self.assertFalse(result["valid"])
        self.assertIn("Invalid 'crc'", result["error"])

    def test_crc_out_of_range_is_rejected(self):
        raw = json.dumps({"payload": "hello", "crc": 65536})
        result = parse_packet(raw)
        self.assertFalse(result["valid"])

    def test_non_object_json_is_rejected(self):
        result = parse_packet('["payload", 123]')
        self.assertFalse(result["valid"])
        self.assertIn("must be an object", result["error"])

    def test_non_string_payload_is_rejected(self):
        result = parse_packet(json.dumps({"payload": 123, "crc": 0}))
        self.assertFalse(result["valid"])
        self.assertIn("payload", result["error"])

    def test_extra_fields_still_valid(self):
        """Extra JSON fields must not break parsing."""
        payload = "Extra field test"
        crc     = calculate_crc(payload)
        raw     = json.dumps({"payload": payload, "crc": crc, "extra": "ignored"},
                             separators=(",", ":"))
        result = parse_packet(raw)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], payload)


# ═════════════════════════════════════════════════════════════════════════════
# 11. build_packet() output is valid JSON
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildPacketFormat(unittest.TestCase):
    """Test 11: Verify packet structure."""

    def test_is_valid_json(self):
        packet = build_packet("test payload")
        obj = json.loads(packet)
        self.assertIn("payload", obj)
        self.assertIn("crc", obj)

    def test_crc_is_integer(self):
        packet = build_packet("test payload")
        obj = json.loads(packet)
        self.assertIsInstance(obj["crc"], int)

    def test_crc_in_range(self):
        for payload in ["", "A", "short", "X" * 300]:
            obj = json.loads(build_packet(payload))
            self.assertGreaterEqual(obj["crc"], 0)
            self.assertLessEqual(obj["crc"], 0xFFFF)

    def test_payload_preserved(self):
        special = '{"nested": "json"} and more\n\ttabs'
        packet  = build_packet(special)
        result  = parse_packet(packet)
        self.assertTrue(result["valid"])
        self.assertEqual(result["payload"], special)


# ═════════════════════════════════════════════════════════════════════════════
# 12. verify_crc() direct API
# ═════════════════════════════════════════════════════════════════════════════

class TestVerifyCrcDirect(unittest.TestCase):
    """Test 12: verify_crc() standalone tests."""

    def test_correct_crc(self):
        data = "Direct verify test"
        crc  = calculate_crc(data)
        self.assertTrue(verify_crc(data, crc))

    def test_wrong_crc(self):
        self.assertFalse(verify_crc("Hello", 0xDEAD))

    def test_crc_zero(self):
        data = "some data"
        crc  = calculate_crc(data)
        if crc == 0:
            self.assertTrue(verify_crc(data, 0))
        else:
            self.assertFalse(verify_crc(data, 0))

    def test_max_crc(self):
        self.assertFalse(verify_crc("anything", 0xFFFF))  # very unlikely to match


# ═════════════════════════════════════════════════════════════════════════════
# 13. CRC-16 known test vectors
# ═════════════════════════════════════════════════════════════════════════════

class TestKnownVectors(unittest.TestCase):
    """Test 13: CRC-16-CCITT (0x1021, init=0xFFFF) known vectors.

    Reference: https://crccalc.com/ → CRC-16/CCITT-FALSE
    Input "123456789" → 0x29B1
    """

    def test_standard_vector(self):
        """CRC-16-CCITT-FALSE: '123456789' → 0x29B1."""
        crc = calculate_crc("123456789")
        self.assertEqual(crc, 0x29B1,
                         f"Expected 0x29B1, got 0x{crc:04X}. CRC algorithm mismatch.")

    def test_single_byte_0x00(self):
        crc = calculate_crc(b"\x00")
        self.assertIsInstance(crc, int)
        self.assertGreaterEqual(crc, 0)
        self.assertLessEqual(crc, 0xFFFF)

    def test_all_zeros_8_bytes(self):
        crc = calculate_crc(b"\x00" * 8)
        self.assertIsInstance(crc, int)

    def test_deterministic(self):
        """Same input must always produce the same CRC."""
        data = "Reproducible CRC"
        self.assertEqual(calculate_crc(data), calculate_crc(data))

    def test_bytes_vs_string_equivalent(self):
        """str and its UTF-8 bytes encoding must give the same CRC."""
        data = "LoRa CRC test"
        self.assertEqual(calculate_crc(data), calculate_crc(data.encode("utf-8")))


# ═════════════════════════════════════════════════════════════════════════════
# 14. Bytes input to calculate_crc()
# ═════════════════════════════════════════════════════════════════════════════

class TestBytesInput(unittest.TestCase):
    """Test 14: calculate_crc() accepts bytes."""

    def test_bytes_accepted(self):
        raw = b"\x01\x02\x03\xFF"
        crc = calculate_crc(raw)
        self.assertIsInstance(crc, int)

    def test_bytes_round_trip(self):
        payload = b"binary \x00 data \xFF"
        crc     = calculate_crc(payload)
        self.assertTrue(verify_crc(payload, crc))


# ═════════════════════════════════════════════════════════════════════════════
# 15. Invalid type to calculate_crc()
# ═════════════════════════════════════════════════════════════════════════════

class TestInvalidInput(unittest.TestCase):
    """Test 15: TypeError is raised for unsupported input types."""

    def test_int_raises(self):
        with self.assertRaises(TypeError):
            calculate_crc(12345)

    def test_list_raises(self):
        with self.assertRaises(TypeError):
            calculate_crc([1, 2, 3])

    def test_none_raises(self):
        with self.assertRaises(TypeError):
            calculate_crc(None)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
