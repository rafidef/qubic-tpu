"""
Qubic Key Utilities — Python port of keyUtils.cpp
Identity encoding/decoding and key derivation for Qubic.
"""
import struct
from qubic_k12 import kangaroo_twelve


def get_public_key_from_identity(identity: str) -> bytes:
    """
    Decode a 60-char Qubic identity (uppercase A-Z) to 32-byte public key.
    Port of getPublicKeyFromIdentity from keyUtils.cpp.
    """
    public_key = bytearray(32)
    for i in range(4):
        val = 0
        for j in range(13, -1, -1):
            ch = identity[i * 14 + j]
            if ch < 'A' or ch > 'Z':
                return bytes(32)
            val = val * 26 + (ord(ch) - ord('A'))
        struct.pack_into('<Q', public_key, i * 8, val)
    return bytes(public_key)


def get_identity_from_public_key(public_key: bytes, lower_case: bool = False) -> str:
    """
    Encode a 32-byte public key to 60-char Qubic identity.
    Port of getIdentityFromPublicKey from keyUtils.cpp.
    """
    base = ord('a') if lower_case else ord('A')
    identity = [''] * 60

    for i in range(4):
        fragment = struct.unpack_from('<Q', public_key, i * 8)[0]
        for j in range(14):
            identity[i * 14 + j] = chr(fragment % 26 + base)
            fragment //= 26

    # Checksum (last 4 chars)
    checksum_bytes = kangaroo_twelve(public_key[:32], 3)
    checksum = struct.unpack_from('<I', checksum_bytes + b'\x00', 0)[0]
    checksum &= 0x3FFFF
    for i in range(4):
        identity[56 + i] = chr(checksum % 26 + base)
        checksum //= 26

    return ''.join(identity)


def get_subseed_from_seed(seed: str) -> bytes:
    """
    Derive subseed from a 55-char lowercase seed.
    Port of getSubseedFromSeed from keyUtils.cpp.
    """
    if len(seed) < 55:
        seed = seed.ljust(55, 'a')
    seed_bytes = bytes([ord(c) - ord('a') for c in seed[:55]])
    return kangaroo_twelve(seed_bytes, 32)


def get_private_key_from_subseed(subseed: bytes) -> bytes:
    """Derive private key from subseed via K12."""
    return kangaroo_twelve(subseed, 32)


def check_sum_identity(identity: str) -> bool:
    """Verify the checksum of a Qubic identity."""
    public_key = get_public_key_from_identity(identity)
    checksum_bytes = kangaroo_twelve(public_key, 3)
    checksum = struct.unpack_from('<I', checksum_bytes + b'\x00', 0)[0]
    checksum &= 0x3FFFF
    for i in range(4):
        expected = chr(checksum % 26 + ord('A'))
        if expected != identity[56 + i]:
            return False
        checksum //= 26
    return True
