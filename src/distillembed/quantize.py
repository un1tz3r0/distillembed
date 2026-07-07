"""Per-row symmetric quantization of embedding tables."""

from __future__ import annotations

import numpy as np


def quantize_int8(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize (rows, dim) float table to int8 codes + per-row f32 scales.

    Reconstruction: row ≈ codes.astype(f32) * scale.
    """
    table = np.asarray(table, dtype=np.float32)
    scales = np.abs(table).max(axis=1) / 127.0
    scales = np.where(scales == 0.0, 1.0, scales)
    codes = np.clip(np.round(table / scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales.astype(np.float32)


def dequantize_int8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scales[:, None].astype(np.float32)


def quantize_int4(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int4: codes in [-7, 7], two per byte + per-row f32 scale.

    Packing: even dim -> low nibble, odd dim -> high nibble (matches the C++
    unpack in quant.hpp). Odd dims are zero-padded to a whole byte.
    """
    table = np.asarray(table, dtype=np.float32)
    scales = np.abs(table).max(axis=1) / 7.0
    scales = np.where(scales == 0.0, 1.0, scales)
    codes = np.clip(np.round(table / scales[:, None]), -7, 7).astype(np.int8)
    if codes.shape[1] % 2:
        codes = np.pad(codes, ((0, 0), (0, 1)))
    lo = codes[:, 0::2].astype(np.uint8) & 0xF
    hi = codes[:, 1::2].astype(np.uint8) & 0xF
    return (lo | (hi << 4)).astype(np.uint8), scales.astype(np.float32)


def dequantize_int4(packed: np.ndarray, scales: np.ndarray, dim: int) -> np.ndarray:
    def sign_extend(nibbles: np.ndarray) -> np.ndarray:
        return ((nibbles ^ 8).astype(np.int8) - 8).astype(np.float32)

    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
    codes[:, 0::2] = sign_extend(packed & 0xF)
    codes[:, 1::2] = sign_extend((packed >> 4) & 0xF)
    return codes[:, :dim] * scales[:, None].astype(np.float32)
