"""CLSR-format manifest tasks.

Synthetic manifests test the logic. The release itself is tested too when
``LSR_CLSR_ROOT`` points at a clone of github.com/193746/CLSR; those tests pin
the split sizes and the known defects, so a different checkout fails loudly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lsr.tasks import ManifestTask, check_task, read_manifest


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{k} {v}\n" for k, v in rows), encoding="utf-8")


@pytest.fixture
def slue_split(tmp_path):
    """Three questions, two of them about the same context."""
    d = tmp_path / "test"
    d.mkdir()
    write(d / "text_que.txt", [("S0", "who won"), ("S1", "where was it"), ("S2", "how many")])
    write(d / "wav_que.scp", [("S0", "./q/0.wav"), ("S1", "./q/1.wav"), ("S2", "./q/2.wav")])
    write(d / "text_con.txt", [("S0", "the game"), ("S1", "the game"), ("S2", "a city")])
    write(d / "wav_con.scp", [("S0", "./c/A.wav"), ("S1", "./c/A.wav"), ("S2", "./c/B.wav")])
    return d


def test_read_manifest_keeps_order_and_rejects_duplicate_ids(tmp_path):
    write(tmp_path / "m", [("b", "two words"), ("a", "one")])
    assert read_manifest(tmp_path / "m") == {"b": "two words", "a": "one"}
    write(tmp_path / "m", [("a", "x"), ("a", "y")])
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(tmp_path / "m")


def test_unique_pool_dedupes_contexts(slue_split):
    task = ManifestTask("slue_sqa5", slue_split, "audio", "audio")
    check_task(task)
    assert task.name == "slue_sqa5_aq2ac_test"
    docs = list(task.documents())
    assert [d.audio_path for d in docs] == ["c/A.wav", "c/B.wav"]
    qrels = task.qrels()
    assert qrels["S0"] == qrels["S1"] != qrels["S2"]


def test_row_pool_matches_release_protocol(slue_split):
    task = ManifestTask("slue_sqa5", slue_split, "audio", "audio", pool="rows")
    check_task(task)
    assert task.name == "slue_sqa5_aq2ac_test_rowpool"
    assert [d.id for d in task.documents()] == ["S0", "S1", "S2"]
    assert task.qrels() == {"S0": {"S0"}, "S1": {"S1"}, "S2": {"S2"}}


def test_direction_is_in_the_name_and_the_fields(slue_split):
    task = ManifestTask("slue_sqa5", slue_split, "text", "text")
    check_task(task)
    assert task.name == "slue_sqa5_tq2tc_test"
    assert next(task.queries()).text == "who won"


def test_audio_root_resolves_relative_paths(slue_split, tmp_path):
    task = ManifestTask("slue_sqa5", slue_split, "audio", "audio", audio_root=tmp_path)
    assert next(task.queries()).audio_path == str(tmp_path / "q/0.wav")


def test_librisqa_has_no_question_audio(slue_split):
    with pytest.raises(ValueError, match="no audio on the query side"):
        ManifestTask("librisqa", slue_split, "audio", "text")


def test_misaligned_manifests_rejected(slue_split):
    write(slue_split / "wav_con.scp", [("S0", "a"), ("S2", "b"), ("S1", "c")])
    with pytest.raises(ValueError, match="different rows"):
        ManifestTask("slue_sqa5", slue_split, "audio", "audio")


# --- the release itself -------------------------------------------------------

CLSR_ROOT = os.environ.get("LSR_CLSR_ROOT")
release = pytest.mark.skipif(not CLSR_ROOT, reason="LSR_CLSR_ROOT not set")


def librisqa(split, doc="audio", **kw):
    d = Path(CLSR_ROOT) / "CLSR_aqtc/dataset/LibriSQA/train_data" / split
    return ManifestTask("librisqa", d, "text", doc, **kw)


def librisqa_text(split, **kw):
    return librisqa(split, doc="text", **kw)


def slue(split, **kw):
    d = Path(CLSR_ROOT) / "CLSR_aqac/dataset/SLUE-SQA-5/train_data" / split
    return ManifestTask("slue_sqa5", d, "audio", "audio", **kw)


@release
@pytest.mark.parametrize(
    ("make", "split", "n_queries", "n_unique", "n_rows"),
    [
        (librisqa, "test", 2620, 2620, 2620),
        # Two distinct recordings share a transcript, so the text pool is one smaller.
        (librisqa_text, "test", 2620, 2619, 2620),
        (slue, "test", 2382, 1969, 2382),
        (slue, "validation", 1939, 1624, 1939),
    ],
)
def test_release_split_sizes(make, split, n_queries, n_unique, n_rows):
    unique, rows = make(split), make(split, pool="rows")
    check_task(unique)
    check_task(rows)
    assert len(list(unique.queries())) == n_queries
    assert len(list(unique.documents())) == n_unique
    assert len(list(rows.documents())) == n_rows


def _contexts(task):
    return {d.audio_path for d in task.documents()}


@release
def test_release_librisqa_validation_is_test():
    """Known defect: no held-out validation split for LibriSQA."""
    assert _contexts(librisqa("validation")) == _contexts(librisqa("test"))


@release
@pytest.mark.parametrize("make", [librisqa, slue])
def test_release_train_test_contexts_disjoint(make):
    assert not _contexts(make("train")) & _contexts(make("test"))
