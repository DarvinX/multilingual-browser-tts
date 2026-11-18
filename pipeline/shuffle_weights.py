"""Lossless byte grouping for float data inside an INT8 browser asset."""
import gzip
import json
import struct

import numpy as np


def shuffle_weights(source, target):
    raw = gzip.decompress(source.read_bytes())
    header_bytes, = struct.unpack_from("<I", raw)
    offset = 4 + header_bytes
    header = json.loads(raw[4:offset])
    header["shuffled"] = True
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    result = bytearray(struct.pack("<I", len(encoded)))
    result.extend(encoded)
    for format, _, decoded_bytes, length in header["entries"]:
        payload = memoryview(raw)[offset:offset+length]
        offset += length
        if format in ("int8", "int4"):
            group_size = header["groupSize"]
            groups = (decoded_bytes//4 + group_size - 1)//group_size
            codes = groups*group_size//(2 if format == "int4" else 1)
            result.extend(payload[:codes])
            payload = payload[codes:]
        width = 4 if format == "raw" else 2
        ordered = np.frombuffer(payload, dtype=np.uint8).reshape(-1, width).T.copy()
        result.extend(ordered.tobytes())
    if offset != len(raw):
        raise ValueError("unexpected browser asset length")
    target.write_bytes(gzip.compress(result, mtime=0))
