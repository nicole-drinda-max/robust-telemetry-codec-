# Robust Telemetry Codec (PHI/Fibonacci) v1.2
# Original author: Alexandra-Nicole Anna Drinda (2026)
# License: MIT (see LICENSE)
# Status: Public release – no ongoing maintenance or support guaranteed.
# Provided "AS IS", without warranty of any kind.

"""
Robust Telemetry Codec (PHI/Fibonacci) v1.2 (Engineering-safe, self-contained)
=============================================================================

What this gives you (complete stack):
- UTF-8 message -> frame (MAGIC, VERSION, FLAGS, SEQ, LEN, CRC32, PAYLOAD)
- Integer stream -> Fibonacci/Zeckendorf self-delimiting coding (bitstream)
- Optional PHI-inspired interleaving with coprime-safe dynamic step
- Optional FEC: Extended Hamming (13,8) per byte (SEC-ish: 1-bit correct, detects many multi-bit errors)
- Bit-pack to bytes (+pad bits)
- Decode reverses everything + CRC verification

Quick run:
    python robust_telemetry_codec.py

Quick use:
    from robust_telemetry_codec import encode_message, decode_message
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import zlib

# -----------------------------
# 0) Constants / Header
# -----------------------------

PHI = (1 + 5**0.5) / 2

MAGIC = int.from_bytes(b"PHI1", "big")   # 0x50484931
VERSION = 1

FLAG_INTERLEAVE = 1 << 0
FLAG_FEC_HAMMING13_8 = 1 << 1

# -----------------------------
# 1) Fibonacci / Zeckendorf coding (self-delimiting ints)
# -----------------------------

def _fib_sequence_upto(n: int) -> List[int]:
    """Fibonacci numbers for coding: 1,2,3,5,8,..."""
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def fib_encode_int(x: int) -> str:
    """
    Fibonacci coding for x >= 1 (positive integer).
    Produces a bitstring that ends with '11' (via terminator rule).
    """
    if x < 1:
        raise ValueError("fib_encode_int expects x >= 1")

    fibs = _fib_sequence_upto(x)
    remaining = x

    bits = []
    used_prev = False
    for f in reversed(fibs[:-1]):  # skip last which is > x
        if f <= remaining and not used_prev:
            bits.append("1")
            remaining -= f
            used_prev = True
        else:
            bits.append("0")
            used_prev = False

    s = "".join(bits).lstrip("0")
    if not s:
        s = "0"

    return s + "1"

def fib_decode_stream(bitstream: str, max_items: Optional[int] = None) -> List[int]:
    """Decode concatenated Fibonacci-coded integers from a bitstream."""
    out = []
    i = 0

    while i < len(bitstream):
        if max_items is not None and len(out) >= max_items:
            break

        # Read until we hit the '11' pattern that terminates a codeword
        code_bits = []
        prev = "0"
        while i < len(bitstream):
            b = bitstream[i]
            code_bits.append(b)
            i += 1
            if prev == "1" and b == "1":
                break
            prev = b

        if len(code_bits) < 2 or not (code_bits[-2] == "1" and code_bits[-1] == "1"):
            raise ValueError("Invalid/incomplete Fibonacci codeword at end of stream.")

        # Drop final terminator '1'
        rep = code_bits[:-1]

        # Map rep bits to descending fibs of same length
        fibs = [1, 2]
        while len(fibs) < len(rep):
            fibs.append(fibs[-1] + fibs[-2])
        fibs = list(reversed(fibs[:len(rep)]))

        val = 0
        for bit, f in zip(rep, fibs):
            if bit == "1":
                val += f
        out.append(val)

    return out

# -----------------------------
# 2) Bit packing
# -----------------------------

def bits_to_bytes(bits: str) -> Tuple[bytes, int]:
    """Pack '0'/'1' bits into bytes MSB-first. Pads with '0's."""
    pad = (-len(bits)) % 8
    bits_padded = bits + ("0" * pad)
    out = bytearray()
    for j in range(0, len(bits_padded), 8):
        out.append(int(bits_padded[j:j + 8], 2))
    return bytes(out), pad

def bytes_to_bits(data: bytes, pad_bits: int = 0) -> str:
    bits = "".join(f"{b:08b}" for b in data)
    return bits[:-pad_bits] if pad_bits else bits

# -----------------------------
# 3) PHI-inspired interleaving (coprime-safe permutation)
# -----------------------------

def get_phi_step(n: int) -> int:
    """
    Choose a step ~ n / phi^2 (≈ n/2.618) and force gcd(step,n)=1
    so it is a full-cycle permutation (no collisions / no lost positions).
    """
    if n <= 1:
        return 1

    step = max(1, int(round(n / (PHI**2))))
    step = step % n
    if step == 0:
        step = 1

    while math.gcd(step, n) != 1:
        step += 1
        if step >= n:
            step = 1
    return step

def phi_interleave(bits: str) -> str:
    n = len(bits)
    if n == 0:
        return bits
    step = get_phi_step(n)

    out = ["0"] * n
    j = 0
    for b in bits:
        out[j] = b
        j = (j + step) % n
    return "".join(out)

def phi_deinterleave(bits: str) -> str:
    """Inverse of phi_interleave because we replay the same index walk."""
    n = len(bits)
    if n == 0:
        return bits
    step = get_phi_step(n)

    out = ["0"] * n
    j = 0
    for i in range(n):
        out[i] = bits[j]
        j = (j + step) % n
    return "".join(out)

# -----------------------------
# 4) Optional FEC: Extended Hamming (13,8)
# -----------------------------

HAMMING_PARITY_POS = (1, 2, 4, 8)          # 1-indexed
HAMMING_TOTAL_BITS = 13                     # positions 1..13
HAMMING_DATA_POS = [3, 5, 6, 7, 9, 10, 11, 12]
HAMMING_OVERALL_PARITY_POS = 13

def _hamming13_8_encode_byte(byte: int) -> str:
    """Encode a byte into a 13-bit Hamming codeword."""
    if not (0 <= byte <= 255):
        raise ValueError("Byte out of range")

    bits = ["0"] * (HAMMING_TOTAL_BITS + 1)  # 1-indexed
    data = f"{byte:08b}"

    for pos, db in zip(HAMMING_DATA_POS, data):
        bits[pos] = db

    # parity bits (even parity), excluding overall parity bit
    for p in HAMMING_PARITY_POS:
        parity = 0
        for i in range(1, HAMMING_TOTAL_BITS):
            if i & p and i != p:
                parity ^= int(bits[i])
        bits[p] = str(parity)

    # overall parity across positions 1..12
    overall = 0
    for i in range(1, HAMMING_OVERALL_PARITY_POS):
        overall ^= int(bits[i])
    bits[HAMMING_OVERALL_PARITY_POS] = str(overall)

    return "".join(bits[1:])

def _hamming13_8_decode_bits(codeword: str) -> Tuple[int, bool, bool]:
    """
    Decode 13-bit codeword.
    Returns (decoded_byte, corrected, detected_uncorrectable).
    """
    if len(codeword) != 13 or any(c not in "01" for c in codeword):
        raise ValueError("Invalid 13-bit codeword")

    bits = ["0"] + list(codeword)  # 1-indexed

    syndrome = 0
    for p in HAMMING_PARITY_POS:
        parity = 0
        for i in range(1, HAMMING_OVERALL_PARITY_POS):
            if i & p:
                parity ^= int(bits[i])
        if parity != 0:
            syndrome |= p

    overall = 0
    for i in range(1, HAMMING_TOTAL_BITS + 1):
        overall ^= int(bits[i])

    corrected = False
    uncorrectable = False

    if syndrome == 0 and overall == 0:
        pass
    elif syndrome != 0 and overall == 1:
        if 1 <= syndrome <= 12:
            bits[syndrome] = "1" if bits[syndrome] == "0" else "0"
            corrected = True
        else:
            uncorrectable = True
    elif syndrome == 0 and overall == 1:
        bits[HAMMING_OVERALL_PARITY_POS] = "1" if bits[HAMMING_OVERALL_PARITY_POS] == "0" else "0"
        corrected = True
    else:
        uncorrectable = True

    data_bits = "".join(bits[pos] for pos in HAMMING_DATA_POS)
    decoded_byte = int(data_bits, 2)
    return decoded_byte, corrected, uncorrectable

def fec_hamming13_8_encode(payload: bytes) -> str:
    return "".join(_hamming13_8_encode_byte(b) for b in payload)

def fec_hamming13_8_decode(bitstream: str) -> Tuple[bytes, int, int]:
    if len(bitstream) % 13 != 0:
        raise ValueError("Hamming13/8 bitstream length must be multiple of 13")

    out = bytearray()
    corrected = 0
    uncorrectable = 0
    for i in range(0, len(bitstream), 13):
        cw = bitstream[i:i+13]
        b, c, u = _hamming13_8_decode_bits(cw)
        out.append(b)
        corrected += 1 if c else 0
        uncorrectable += 1 if u else 0
    return bytes(out), corrected, uncorrectable

# -----------------------------
# 5) Frame model
# -----------------------------

@dataclass
class Frame:
    seq: int
    flags: int
    payload: bytes
    crc32: int

def _u32(x: int) -> int:
    return x & 0xFFFFFFFF

def make_frame(payload: bytes, seq: int, flags: int) -> Frame:
    crc = _u32(zlib.crc32(payload))
    return Frame(seq=seq, flags=flags, payload=payload, crc32=crc)

def frame_to_ints(frame: Frame) -> List[int]:
    ints = [
        MAGIC + 1,
        VERSION + 1,
        frame.flags + 1,
        frame.seq + 1,
        len(frame.payload) + 1,
        frame.crc32 + 1,
    ]
    ints.extend([b + 1 for b in frame.payload])
    return ints

def ints_to_frame(ints: List[int]) -> Frame:
    if len(ints) < 6:
        raise ValueError("Not enough ints for header")

    magic = ints[0] - 1
    if magic != MAGIC:
        raise ValueError("Bad MAGIC (not a PHI1 frame)")

    version = ints[1] - 1
    if version != VERSION:
        raise ValueError(f"Unsupported VERSION={version}")

    flags = ints[2] - 1
    seq = ints[3] - 1
    length = ints[4] - 1
    crc32 = ints[5] - 1

    data_ints = ints[6:6+length]
    if len(data_ints) != length:
        raise ValueError("Truncated payload")

    payload = bytes([x - 1 for x in data_ints])

    if _u32(zlib.crc32(payload)) != _u32(crc32):
        raise ValueError("CRC mismatch (frame corrupted)")

    return Frame(seq=seq, flags=flags, payload=payload, crc32=crc32)

# -----------------------------
# 6) High-level API
# -----------------------------

def encode_message(
    text: str,
    seq: int = 0,
    use_interleave: bool = True,
    use_fec_hamming: bool = True,
) -> Tuple[bytes, int]:
    payload = text.encode("utf-8")

    flags = 0
    if use_interleave:
        flags |= FLAG_INTERLEAVE
    if use_fec_hamming:
        flags |= FLAG_FEC_HAMMING13_8

    transport_payload = payload
    if use_fec_hamming:
        fec_bits = fec_hamming13_8_encode(payload)
        transport_payload, fec_pad = bits_to_bytes(fec_bits)
        if fec_pad > 7:
            raise ValueError("Internal error: pad too large")
        transport_payload = bytes([fec_pad]) + transport_payload

    frame = make_frame(payload=transport_payload, seq=seq, flags=flags)
    ints = frame_to_ints(frame)

    bitstream = "".join(fib_encode_int(x) for x in ints)

    if use_interleave:
        bitstream = phi_interleave(bitstream)

    data, pad_bits = bits_to_bytes(bitstream)
    return data, pad_bits

def decode_message(data: bytes, pad_bits: int) -> Tuple[str, Frame, dict]:
    diagnostics = {
        "corrected_codewords": 0,
        "uncorrectable_codewords": 0,
        "used_interleave": False,
        "used_fec_hamming": False,
    }

    bits = bytes_to_bits(data, pad_bits=pad_bits)

    def _try_decode(bitstream: str) -> Optional[Frame]:
        try:
            ints = fib_decode_stream(bitstream)
            return ints_to_frame(ints)
        except Exception:
            return None

    frame = _try_decode(bits)
    if frame is None:
        bits2 = phi_deinterleave(bits)
        frame = _try_decode(bits2)
        if frame is None:
            raise ValueError("Failed to decode frame (bad data or wrong pad_bits).")
        diagnostics["used_interleave"] = True
    else:
        bits2 = bits  # keep for consistency

    if frame.flags & FLAG_INTERLEAVE:
        diagnostics["used_interleave"] = True
    if frame.flags & FLAG_FEC_HAMMING13_8:
        diagnostics["used_fec_hamming"] = True

    transport_payload = frame.payload

    if frame.flags & FLAG_FEC_HAMMING13_8:
        if len(transport_payload) < 1:
            raise ValueError("Missing FEC pad header")
        fec_pad = transport_payload[0]
        fec_bytes = transport_payload[1:]
        fec_bits = bytes_to_bits(fec_bytes, pad_bits=fec_pad)
        decoded_payload, corrected, uncorrectable = fec_hamming13_8_decode(fec_bits)
        diagnostics["corrected_codewords"] = corrected
        diagnostics["uncorrectable_codewords"] = uncorrectable

        text = decoded_payload.decode("utf-8", errors="strict")
        clean_frame = Frame(
            seq=frame.seq,
            flags=frame.flags,
            payload=decoded_payload,
            crc32=_u32(zlib.crc32(decoded_payload)),
        )
        return text, clean_frame, diagnostics

    text = transport_payload.decode("utf-8", errors="strict")
    clean_frame = Frame(
        seq=frame.seq,
        flags=frame.flags,
        payload=transport_payload,
        crc32=_u32(zlib.crc32(transport_payload)),
    )
    return text, clean_frame, diagnostics

# -----------------------------
# 7) Demo / Self-test
# -----------------------------

def _demo():
    msg = "Nicole: φ/Fibo-Codec v1.2 — MAGIC+CRC+PHI-interleave+Hamming(13,8)."
    enc, pad = encode_message(msg, seq=42, use_interleave=True, use_fec_hamming=True)
    dec, frame, diag = decode_message(enc, pad)

    print("Encoded bytes:", len(enc), "pad_bits:", pad)
    print("Decoded text:", dec)
    print("Seq:", frame.seq, "Flags:", frame.flags)
    print("Diagnostics:", diag)

    # Deterministic 1-bit flip to demonstrate correction/detection behavior
    corrupt = bytearray(enc)
    if len(corrupt) > 10:
        corrupt[10] ^= 0b00010000  # flip one bit
        try:
            dec2, frame2, diag2 = decode_message(bytes(corrupt), pad)
            print("\nAfter 1-bit flip:")
            print("Decoded text:", dec2)
            print("Diagnostics:", diag2)
        except Exception as e:
            print("\nAfter 1-bit flip: decode failed:", e)

if __name__ == "__main__":
    _demo()
