"""Python↔C++ parity: the C++ engine must tokenize and embed identically to the
numpy reference encoder. Requires the demo binary (`make -C cpp`)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from distillembed.export import write_dem
from distillembed.model import StaticModel

BIN = Path(__file__).resolve().parents[1] / "cpp" / "build" / "demo_embed"

needs_bin = pytest.mark.skipif(not BIN.exists(), reason="C++ demo not built (make -C cpp)")

TEXTS = [
    "the quick brown fox",
    "sensor reading temperature  humidity",
    "  battery level low ",
    "firmware update restart module",
    "zzz unseen-word 123 !!",  # forces byte fallback (digits/punct not in corpus)
]


def run_demo(dem_path, text):
    out = subprocess.run(
        [str(BIN), str(dem_path), text, "--full"],
        capture_output=True, text=True, check=True,
    ).stdout
    lines = {line.split(":", 1)[0]: line.split(":", 1)[1] for line in out.splitlines()}
    ids = [int(x) for x in lines["ids"].split()]
    vec = np.array([float(x) for x in lines["vec"].split()], dtype=np.float32)
    return ids, vec


@needs_bin
def test_parity_f32(tiny_model_dir, tiny_sp, tiny_table, tmp_path):
    dem = write_dem(tmp_path / "m_f32.dem", tiny_sp, tiny_table, quant="f32")
    sm = StaticModel(tiny_model_dir)
    for text in TEXTS:
        ids, vec = run_demo(dem, text)
        assert ids == sm.tokenize(text), f"tokenizer mismatch on {text!r}"
        np.testing.assert_allclose(vec, sm.encode(text), atol=1e-4)


@needs_bin
@pytest.mark.parametrize("quant,min_cos", [("int8", 0.99), ("int4", 0.95)])
def test_parity_quantized(tiny_model_dir, tiny_sp, tiny_table, tmp_path, quant, min_cos):
    dem = write_dem(tmp_path / f"m_{quant}.dem", tiny_sp, tiny_table, quant=quant)
    sm = StaticModel(tiny_model_dir)
    for text in TEXTS:
        _, vec = run_demo(dem, text)
        ref = sm.encode(text)
        cos = float(np.dot(vec, ref))
        assert cos > min_cos, f"{quant} drift on {text!r}: cosine {cos:.4f}"


@needs_bin
def test_empty_and_whitespace(tiny_sp, tiny_table, tmp_path):
    dem = write_dem(tmp_path / "m.dem", tiny_sp, tiny_table, quant="f32")
    for text in ["", "   "]:
        ids, vec = run_demo(dem, text)
        assert np.all(vec == 0.0) or len(ids) > 0  # no crash; empty input -> zero vector
