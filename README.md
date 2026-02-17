# Robust Telemetry Codec

**Author (original):** Alexandra-Nicole Anna Drinda

**Pure-Python telemetry framing codec**  
Fibonacci/Zeckendorf self-delimiting + phi-inspired interleaving + optional Hamming(13,8) FEC + CRC32

Noise-resilient, self-delimiting telemetry framing written in pure Python.  
Designed for harsh or noisy channels (space telemetry, fusion diagnostics, embedded systems, IoT, industrial sensors).

**License:** MIT  
**Status:** Public release – no ongoing maintenance or support guaranteed.

## Features
- Self-delimiting integer encoding (Fibonacci/Zeckendorf, no length prefix needed)
- Phi-inspired interleaving with dynamic coprime step for burst-error spreading
- Optional extended Hamming(13,8) FEC: corrects 1-bit errors per codeword, detects many 2-bit errors
- Framing: MAGIC, version, flags, sequence number, length, CRC32 + payload
- Pure Python – zero external dependencies

## Notes
- v1.x prioritizes robustness and framing clarity over minimal bandwidth overhead.
- Interleaving + FEC improve burst-error tolerance; CRC32 provides a fast integrity check.

## Quick Example

```python
from robust_telemetry_codec import encode_message, decode_message

msg = "Sensor data: temp=42.5°C, pressure=1013 hPa"
encoded, pad = encode_message(msg, seq=123, use_interleave=True, use_fec_hamming=True)
print(f"Encoded bytes: {len(encoded)} (pad: {pad})")

decoded, frame, diag = decode_message(encoded, pad)
print("Decoded:", decoded)
print("Diagnostics:", diag)
