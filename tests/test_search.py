"""End-to-end search through demo_search: int8, binary, and hybrid paths.
Requires the demo binaries (`make -C cpp`)."""

import subprocess
from pathlib import Path

import pytest

from distillembed.export import write_corpus, write_dem
from distillembed.model import StaticModel

BIN = Path(__file__).resolve().parents[1] / "cpp" / "build" / "demo_search"

needs_bin = pytest.mark.skipif(not BIN.exists(), reason="C++ demos not built (make -C cpp)")

DOCS = [
    "battery level low alarm threshold",
    "firmware update over the network",
    "temperature sensor reading data",
    "device restart after error status",
    "sleep mode power timer clock",
]


@pytest.fixture(scope="module")
def artifacts(tiny_model_dir, tiny_sp, tiny_table, tmp_path_factory):
    root = tmp_path_factory.mktemp("search")
    dem = write_dem(root / "m.dem", tiny_sp, tiny_table, quant="int8")
    student = StaticModel(tiny_model_dir)
    vectors = student.encode(DOCS)
    corpora = {
        quant: write_corpus(root / f"c_{quant}.demc", vectors, DOCS, quant=quant)
        for quant in ("f32", "int8", "binary", "rescore")
    }
    return dem, corpora


def top1(dem, demc, query, *extra):
    out = subprocess.run(
        [str(BIN), str(dem), str(demc), query, "3", *extra],
        capture_output=True, text=True, check=True,
    ).stdout
    first = out.splitlines()[0]
    return int(first.split("[", 1)[1].split("]", 1)[0])


@needs_bin
@pytest.mark.parametrize("quant", ["f32", "int8", "binary", "rescore"])
def test_search_finds_matching_doc(artifacts, quant):
    dem, corpora = artifacts
    # Queries reuse each doc's tokens: with a random table, matching rows
    # dominate the score, so the right doc must win under every quant scheme.
    for i, doc in enumerate(DOCS):
        assert top1(dem, corpora[quant], doc) == i


@needs_bin
def test_hybrid_search(artifacts):
    dem, corpora = artifacts
    for i, doc in enumerate(DOCS):
        assert top1(dem, corpora["int8"], doc, "--hybrid") == i


@needs_bin
def test_rescore_matches_int8_scores(artifacts):
    """Rescored top-1 must carry the exact int8 score, not the Hamming proxy."""
    dem, corpora = artifacts

    def top1_line(demc, query):
        out = subprocess.run(
            [str(BIN), str(dem), str(demc), query, "1"],
            capture_output=True, text=True, check=True,
        ).stdout
        return out.splitlines()[0]

    for doc in DOCS:
        assert top1_line(corpora["rescore"], doc) == top1_line(corpora["int8"], doc)
