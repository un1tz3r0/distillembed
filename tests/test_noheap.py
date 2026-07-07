"""The no-heap encode/embed path must agree exactly with the heap path.
Runs the demb_selftest binary (static buffers only) against every model quant."""

import subprocess
from pathlib import Path

import pytest

from distillembed.export import write_dem

BIN = Path(__file__).resolve().parents[1] / "cpp" / "build" / "demb_selftest"

needs_bin = pytest.mark.skipif(not BIN.exists(), reason="C++ selftest not built (make -C cpp)")


@needs_bin
@pytest.mark.parametrize("quant", ["f32", "int8", "int4"])
def test_noheap_matches_heap(tiny_sp, tiny_table, tmp_path, quant):
    dem = write_dem(tmp_path / f"m_{quant}.dem", tiny_sp, tiny_table, quant=quant)
    result = subprocess.run([str(BIN), str(dem)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("ok ")
