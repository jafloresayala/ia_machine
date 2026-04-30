"""Fix mojibake: file was saved as cp1252 bytes interpreted as UTF-8."""
import sys

# cp1252 has 5 undefined bytes that Python maps to control chars U+0081 etc.
# We round-trip them via their raw byte value.
_CP1252_UNDEFINED = {0x0081: b'\x81', 0x008D: b'\x8D', 0x008F: b'\x8F',
                      0x0090: b'\x90', 0x009D: b'\x9D'}

def encode_back(garbled: str) -> bytes:
    """Encode garbled cp1252-as-unicode back to original UTF-8 bytes."""
    out = bytearray()
    for ch in garbled:
        code = ord(ch)
        if code < 0x80:
            out.append(code)
        elif code in _CP1252_UNDEFINED:
            out += _CP1252_UNDEFINED[code]
        else:
            # Try cp1252 for all non-ASCII chars (including > U+00FF like €, etc.)
            try:
                out += ch.encode('cp1252')
            except UnicodeEncodeError:
                # Fallback: encode as UTF-8 (shouldn't happen if all content was corrupted)
                out += ch.encode('utf-8')ad()

# Strip UTF-8 BOM if present
if raw.startswith(b"\xef\xbb\xbf"):
    raw = raw[3:]

garbled = raw.decode("utf-8")
original_bytes = encode_back(garbled)

# Verify result is valid UTF-8
try:
    original_bytes.decode("utf-8")
except Exception as e:
    print(f"Result is not valid UTF-8: {e}", file=sys.stderr)
    sys.exit(1)

with open(path, "wb") as f:
    f.write(original_bytes)
print(f"Fixed! {len(raw)} -> {len(original_bytes)} bytes")
