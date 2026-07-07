import numpy as np

from distillembed.export import TYPE_NORMAL, read_dem, write_dem
from distillembed.model import StaticModel


def test_roundtrip_f32(tiny_model_dir, tiny_sp, tiny_table, tmp_path):
    path = write_dem(tmp_path / "m.dem", tiny_sp, tiny_table, quant="f32")
    m = read_dem(path)
    assert m["vocab"] == tiny_sp.vocab_size()
    assert m["dim"] == tiny_table.shape[1]
    assert m["unk_id"] == tiny_sp.unk_id()
    np.testing.assert_array_equal(m["table"], tiny_table)
    for i in (0, 1, tiny_sp.vocab_size() - 1):
        assert m["pieces"][i] == tiny_sp.id_to_piece(i)
        assert m["scores"][i] == np.float32(tiny_sp.get_score(i))


def test_roundtrip_int8(tiny_sp, tiny_table, tmp_path):
    path = write_dem(tmp_path / "m.dem", tiny_sp, tiny_table, quant="int8")
    m = read_dem(path)
    # Per-row symmetric int8: error bounded by half a quantization step per cell.
    step = np.abs(tiny_table).max(axis=1, keepdims=True) / 127.0
    assert np.all(np.abs(m["table"] - tiny_table) <= step / 2 + 1e-6)


def test_roundtrip_int4(tiny_sp, tiny_table, tmp_path):
    path = write_dem(tmp_path / "m.dem", tiny_sp, tiny_table, quant="int4")
    m = read_dem(path)
    step = np.abs(tiny_table).max(axis=1, keepdims=True) / 7.0
    assert np.all(np.abs(m["table"] - tiny_table) <= step / 2 + 1e-6)
    # int4 blob must be smaller than int8 by ~dim/2 bytes per row.
    int8_path = write_dem(tmp_path / "m8.dem", tiny_sp, tiny_table, quant="int8")
    assert path.stat().st_size < int8_path.stat().st_size


def test_normal_pieces_have_reasonable_surface(tiny_sp, tiny_table, tmp_path):
    m = read_dem(write_dem(tmp_path / "m.dem", tiny_sp, tiny_table, quant="f32"))
    normal_lens = [
        len(p.encode("utf-8")) for p, t in zip(m["pieces"], m["types"]) if t == TYPE_NORMAL
    ]
    assert normal_lens and max(normal_lens) == m["max_piece_len"]


def test_static_model_encode(tiny_model_dir):
    sm = StaticModel(tiny_model_dir)
    vec = sm.encode("the quick brown fox")
    assert vec.shape == (sm.dim,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
    batch = sm.encode(["sensor reading", "battery level low"])
    assert batch.shape == (2, sm.dim)
    # Encoding is deterministic and text-sensitive.
    assert not np.allclose(batch[0], batch[1])
