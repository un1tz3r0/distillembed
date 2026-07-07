"""Binary export/read for .dem model files and .demc corpus files.

Format spec lives in DESIGN.md. Everything is little-endian; sections are
padded so f32 arrays land on 4-byte boundaries (the C++ loader is zero-copy).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .quantize import dequantize_int4, dequantize_int8, quantize_int4, quantize_int8

MAGIC_MODEL = b"DEMB"
MAGIC_CORPUS = b"DEMC"
MODEL_VERSION = 2  # v2: surface-sorted search index for zero-RAM C++ lookup
CORPUS_VERSION = 1

QTYPES = {"f32": 0, "int8": 1, "int4": 2, "binary": 3, "rescore": 4}
MODEL_QUANTS = ("f32", "int8", "int4")
# binary: 1 bit/dim signs, Hamming scoring.
# rescore: int8 table + binary prefilter section — two-stage search.
CORPUS_QUANTS = ("f32", "int8", "binary", "rescore")

TYPE_NORMAL, TYPE_UNK, TYPE_CONTROL, TYPE_BYTE = 0, 1, 2, 3


def piece_type(sp, i: int) -> int:
    if sp.is_unknown(i):
        return TYPE_UNK
    if sp.is_control(i) or sp.is_unused(i):
        return TYPE_CONTROL
    if sp.is_byte(i):
        return TYPE_BYTE
    return TYPE_NORMAL


def _pad4(buf: bytearray) -> bytearray:
    buf += b"\0" * ((-len(buf)) % 4)
    return buf


def _pack_vectors(vectors: np.ndarray, quant: str) -> bytes:
    if quant == "int8":
        codes, scales = quantize_int8(vectors)
        return scales.tobytes() + codes.tobytes()
    if quant == "int4":
        packed, scales = quantize_int4(vectors)
        return scales.tobytes() + packed.tobytes()
    if quant == "f32":
        return np.ascontiguousarray(vectors, dtype=np.float32).tobytes()
    if quant == "binary":
        # Sign bits, little bit-order — mirrored by demb::binarize in C++.
        return np.packbits(vectors > 0, axis=1, bitorder="little").tobytes()
    if quant == "rescore":
        # int8 section (row_bytes are whole bytes, so the binary section that
        # follows stays byte-aligned) then the binary prefilter section.
        return _pack_vectors(vectors, "int8") + _pack_vectors(vectors, "binary")
    raise ValueError(f"unsupported quant type: {quant!r}")


def write_dem(out_path: str | Path, sp, table: np.ndarray, quant: str = "int8") -> Path:
    """Write tokenizer + embedding table as a single .dem blob."""
    if quant not in MODEL_QUANTS:
        raise ValueError(f"model quant must be one of {MODEL_QUANTS}, got {quant!r}")
    vocab = sp.vocab_size()
    if table.shape[0] != vocab:
        raise ValueError(f"table rows {table.shape[0]} != vocab size {vocab}")
    dim = table.shape[1]
    qtype = QTYPES[quant]

    # Records in id order (v1 layout); normal pieces additionally indexed by a
    # surface-sorted (rec_offset, piece_id) table so the C++ tokenizer can
    # binary-search the section in place — no RAM structures at load time.
    records = bytearray()
    search_entries: list[tuple[bytes, int, int]] = []
    max_len = 1
    min_score = 0.0
    for i in range(vocab):
        surface = sp.id_to_piece(i).encode("utf-8")
        ptype = piece_type(sp, i)
        score = float(sp.get_score(i))
        if ptype == TYPE_NORMAL:
            max_len = max(max_len, len(surface))
            min_score = min(min_score, score)
            search_entries.append((surface, len(records), i))
        records += struct.pack("<fHBB", score, len(surface), ptype, 0)
        records += surface
    search_entries.sort(key=lambda entry: entry[0])

    tok = bytearray(struct.pack("<I", len(search_entries)))
    for _surface, rec_offset, piece_id in search_entries:
        tok += struct.pack("<II", rec_offset, piece_id)
    tok += records
    _pad4(tok)

    header = struct.pack(
        "<4sHBBIIIIfI",
        MAGIC_MODEL, MODEL_VERSION, qtype, 0, dim, vocab, sp.unk_id(), max_len,
        min_score, len(tok),
    )
    out_path = Path(out_path)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(tok)
        f.write(_pack_vectors(table, quant))
    return out_path


def read_dem(path: str | Path) -> dict:
    """Parse a .dem file back into arrays (test/debug mirror of the C++ loader)."""
    data = Path(path).read_bytes()
    magic, version, qtype, _res, dim, vocab, unk_id, max_len, min_score, tok_size = (
        struct.unpack_from("<4sHBBIIIIfI", data, 0)
    )
    if magic != MAGIC_MODEL or version != MODEL_VERSION:
        raise ValueError(f"not a v{MODEL_VERSION} .dem file: {path}")
    off = 32
    tok_end = off + tok_size
    (n_search,) = struct.unpack_from("<I", data, off)
    off += 4
    search = [struct.unpack_from("<II", data, off + 8 * j) for j in range(n_search)]
    off += 8 * n_search
    pieces, scores, types = [], [], []
    for _ in range(vocab):
        score, length, ptype, _pad = struct.unpack_from("<fHBB", data, off)
        off += 8
        pieces.append(data[off : off + length].decode("utf-8"))
        off += length
        scores.append(score)
        types.append(ptype)
    off = tok_end

    scales = codes = None
    if qtype == QTYPES["int8"]:
        scales = np.frombuffer(data, dtype=np.float32, count=vocab, offset=off)
        off += 4 * vocab
        codes = np.frombuffer(data, dtype=np.int8, count=vocab * dim, offset=off).reshape(vocab, dim)
        table = dequantize_int8(codes, scales)
    elif qtype == QTYPES["int4"]:
        scales = np.frombuffer(data, dtype=np.float32, count=vocab, offset=off)
        off += 4 * vocab
        row_bytes = (dim + 1) // 2
        codes = np.frombuffer(data, dtype=np.uint8, count=vocab * row_bytes, offset=off).reshape(
            vocab, row_bytes
        )
        table = dequantize_int4(codes, scales, dim)
    elif qtype == QTYPES["f32"]:
        table = np.frombuffer(data, dtype=np.float32, count=vocab * dim, offset=off).reshape(vocab, dim)
    else:
        raise ValueError(f"unknown qtype {qtype}")

    return {
        "qtype": qtype, "dim": dim, "vocab": vocab, "unk_id": unk_id,
        "max_piece_len": max_len, "min_score": min_score, "pieces": pieces,
        "scores": np.array(scores, dtype=np.float32),
        "types": np.array(types, dtype=np.uint8),
        "search": search,  # (rec_offset, piece_id), surface-sorted
        "table": table, "scales": scales, "codes": codes,
    }


def write_corpus(
    out_path: str | Path,
    vectors: np.ndarray,
    texts: list[str],
    quant: str = "int8",
) -> Path:
    """Write chunk vectors + their source texts as a .demc blob.

    Vectors are L2-normalized before quantization so int8 dot ≈ cosine.
    """
    if quant not in CORPUS_QUANTS:
        raise ValueError(f"corpus quant must be one of {CORPUS_QUANTS}, got {quant!r}")
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != len(texts):
        raise ValueError("vectors/texts length mismatch")
    count, dim = vectors.shape
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = vectors / norms

    vec = bytearray(_pack_vectors(vectors, quant))
    _pad4(vec)
    header = struct.pack(
        "<4sHBBIII", MAGIC_CORPUS, CORPUS_VERSION, QTYPES[quant], 0, dim, count, len(vec)
    )

    out_path = Path(out_path)
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(vec)
        for text in texts:
            raw = text.encode("utf-8")
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
    return out_path


def write_c_array(src_path: str | Path, out_path: str | Path, symbol: str) -> Path:
    """Render any blob (.dem/.demc) as a C++ header for flash-resident linking.

    Usage on-target: demb::Model::load(g_model, g_model_len) — zero-copy, the
    tables are read straight from flash.
    """
    data = Path(src_path).read_bytes()
    rows = (
        "  " + ", ".join(f"0x{b:02x}" for b in data[i : i + 16]) + ","
        for i in range(0, len(data), 16)
    )
    body = "\n".join(rows)
    (out := Path(out_path)).write_text(
        f"// Generated by distillembed from {Path(src_path).name} — do not edit.\n"
        f"#pragma once\n"
        f"#include <cstddef>\n\n"
        f"alignas(4) inline constexpr unsigned char {symbol}[] = {{\n{body}\n}};\n"
        f"inline constexpr std::size_t {symbol}_len = sizeof({symbol});\n"
    )
    return out
