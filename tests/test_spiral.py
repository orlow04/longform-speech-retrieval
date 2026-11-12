"""SPIRAL builder, on synthetic WAVs and annotations in SPIRAL's own layout."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from lsr.longaudio.grouping import group_containing, group_units
from lsr.tasks.longform import read_jsonl
from lsr.tasks.spiral import build_spiral, read_spiral

RATE = 100


def make_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(b"\x00\x00" * int(seconds * RATE))


def entry(i: int) -> dict:
    sentences = [f"Opening {i}.", f"Fact  number {i} is here.", f"Closing {i}."]
    return {
        "transcript": [{"speaker": "spk_1", "text": sentences}],
        "original_key_sentence": sentences[1],
        "test_question": {"question": f"What is fact {i}?", "choices": ["A. x", "B. y"],
                          "correct_answer": "A. x"},
        "metadata": {"main_topic": "T", "subtopic": f"S{i % 2}", "transcript_type": "lecture",
                     "id": f"lecture_{i}"},
        "audio_data": {
            "key_sentences": {"speaker": "spk_1", "text": sentences[1], "indices": [1, 2],
                              "audio_file": f"lecture_{i}_spk_1_key.wav",
                              "audio_dir": "spiral/key_wavs"},
            "combined": {"audio_files": [f"lecture_{i}.wav"], "audio_folder": "spiral/wavs",
                         "sentences": sentences},
        },
    }


@pytest.fixture
def raw(tmp_path):
    """Seven recordings of 10 s each."""
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(entry(i)) + "\n" for i in range(7)))
    for i in range(7):
        make_wav(tmp_path / "wavs" / f"lecture_{i}.wav", 10.0)
    return tmp_path


def test_read_spiral_flattens_and_normalises_whitespace(raw):
    rows = read_spiral(raw / "data.jsonl")
    assert [r["id"] for r in rows] == [f"lecture_{i}" for i in range(7)]
    assert rows[3]["key_sentence"] == "Fact number 3 is here."
    assert rows[3]["sentences"][1] == "Fact number 3 is here."
    assert rows[3]["key_sentence_index"] == [1, 2]


def test_build_packs_sessions_and_keeps_every_recording_whole(raw):
    out = raw / "build"
    counts = build_spiral(raw / "data.jsonl", raw / "wavs", out, session_s=30.0, seed=0)
    assert counts == {"recordings": 3, "source_recordings": 7, "questions": 7,
                      "duplicate_question_texts": 0, "total_s": 70}
    recordings = read_jsonl(out / "recordings.jsonl")
    assert [r["duration_s"] for r in recordings] == [30.0, 30.0, 10.0]
    assert all(r["kind"] == "concatenated" and r["source"] == "spiral" for r in recordings)
    questions = read_jsonl(out / "questions.jsonl")
    assert sorted(q["query_id"] for q in questions) == sorted(f"lecture_{i}" for i in range(7))
    for q in questions:
        units = read_jsonl(out / "transcripts" / f"{q['recording_id']}.jsonl")
        unit = next(u for u in units if u["piece_id"] == q["query_id"])
        # gold is the whole source recording, and the question belongs to that unit
        assert (unit["start_s"], unit["end_s"]) == (q["gold_start_s"], q["gold_end_s"])
        assert q["key_sentence"] in unit["sentences"]
        # so no grouping at unit boundaries can split a gold span
        assert group_containing(group_units(q["recording_id"], units, 20.0),
                                q["gold_start_s"], q["gold_end_s"]) is not None
    with wave.open(str(out / recordings[0]["audio"]), "rb") as f:
        assert f.getnframes() == 30 * RATE


def test_order_is_shuffled_but_reproducible(raw):
    a = build_spiral(raw / "data.jsonl", raw / "wavs", raw / "a", session_s=1000.0, seed=0)
    build_spiral(raw / "data.jsonl", raw / "wavs", raw / "b", session_s=1000.0, seed=0)
    build_spiral(raw / "data.jsonl", raw / "wavs", raw / "c", session_s=1000.0, seed=1)
    order = {k: [u["piece_id"] for u in read_jsonl(raw / k / "transcripts" / "s00.jsonl")]
             for k in "abc"}
    assert a["recordings"] == 1
    assert order["a"] == order["b"] != order["c"]
    assert order["a"] != [f"lecture_{i}" for i in range(7)]


def test_limit_and_missing_audio(raw):
    assert build_spiral(raw / "data.jsonl", raw / "wavs", raw / "small",
                        limit=3)["questions"] == 3
    (raw / "wavs" / "lecture_5.wav").unlink()
    with pytest.raises(FileNotFoundError, match="1 WAVs missing"):
        build_spiral(raw / "data.jsonl", raw / "wavs", raw / "broken")
