"""Spoken-SQuAD builder and long-audio primitives, on synthetic WAVs."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from lsr.longaudio import (
    FixedWindowSegmenter,
    Segment,
    as_documents,
    localization_accuracy,
    wav_duration_s,
)
from lsr.tasks import ManifestTask, check_task
from lsr.tasks.spoken_squad import build_spoken_squad, collect_paragraphs, concat_wavs

RATE = 100  # frames per second; tiny files, exact arithmetic


def make_wav(path: Path, seconds: float, rate: int = RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * int(seconds * rate))


@pytest.fixture
def raw(tmp_path):
    """Two articles; article 0 has two paragraphs, paragraph 1_1 has no audio."""
    squad = {"data": [
        {"title": "A", "paragraphs": [
            {"context": "First para.  Two sentences.",
             "qas": [{"id": "qa", "question": "what first?",
                      "answers": [{"text": "Two", "answer_start": 13}]},
                     {"id": "qb", "question": "what  else?"}]},
            {"context": "Second para.", "qas": [{"id": "qc", "question": "second?"}]},
        ]},
        {"title": "B", "paragraphs": [
            {"context": "Only para.", "qas": [{"id": "qd", "question": "only?"}]},
            {"context": "Silent para.", "qas": [{"id": "qe", "question": "silent?"}]},
        ]},
    ]}
    (tmp_path / "squad.json").write_text(json.dumps(squad))
    wavs = tmp_path / "wav"
    # Written out of order on purpose: chunk order must come from the name.
    make_wav(wavs / "0_0_1.wav", 2.0)
    make_wav(wavs / "0_0_0.wav", 1.0)
    make_wav(wavs / "0_1_0.wav", 3.0)
    make_wav(wavs / "1_0_0.wav", 4.0)
    return tmp_path


def test_collect_orders_chunks_and_counts_missing(raw):
    paras, missing, gapped = collect_paragraphs(raw / "squad.json", raw / "wav")
    assert [p.id for p in paras] == ["0_0", "0_1", "1_0"]
    assert [w.name for w in paras[0].wav_paths] == ["0_0_0.wav", "0_0_1.wav"]
    assert missing == 1 and gapped == []


def test_collect_accepts_one_voice_rejects_mixed(raw):
    voiced = raw / "voiced"
    make_wav(voiced / "0_0_0__alice.wav", 1.0)
    assert len(collect_paragraphs(raw / "squad.json", voiced)[0]) == 1
    make_wav(voiced / "0_1_0__bob.wav", 1.0)
    with pytest.raises(ValueError, match="mixes voices"):
        collect_paragraphs(raw / "squad.json", voiced)


def test_collect_skips_and_reports_paragraphs_with_missing_clips(raw):
    make_wav(raw / "wav" / "1_0_2.wav", 1.0)  # clip 1 absent: audio incomplete
    paras, _, gapped = collect_paragraphs(raw / "squad.json", raw / "wav")
    assert gapped == ["1_0"]
    assert "1_0" not in [p.id for p in paras]


def test_concat_spans_and_format_check(tmp_path):
    make_wav(tmp_path / "a.wav", 1.0)
    make_wav(tmp_path / "b.wav", 2.5)
    spans = concat_wavs([tmp_path / "a.wav", tmp_path / "b.wav"], tmp_path / "out.wav")
    assert spans == [(0.0, 1.0), (1.0, 3.5)]
    assert wav_duration_s(str(tmp_path / "out.wav")) == 3.5
    make_wav(tmp_path / "c.wav", 1.0, rate=200)
    with pytest.raises(ValueError, match="format"):
        concat_wavs([tmp_path / "a.wav", tmp_path / "c.wav"], tmp_path / "bad.wav")


def test_build_end_to_end(raw):
    from lsr.tasks.longform import read_jsonl

    out = raw / "out"
    counts = build_spoken_squad(raw / "squad.json", raw / "wav", out)
    assert counts == {"articles": 2, "paragraphs": 3, "paragraphs_without_audio": 1,
                      "paragraphs_skipped_missing_clips": [],
                      "questions": 4, "sentence_aligned_paragraphs": 3,
                      "paragraph_aligned_paragraphs": 0, "questions_with_sentence_gold": 1,
                      "text_source": "spoken_squad_asr"}

    task = ManifestTask("spoken_squad", out, "text", "audio", audio_root=out)
    check_task(task)
    assert task.name == "spoken_squad_tq2ac_out"
    assert len(list(task.documents())) == 3  # qa and qb share a paragraph
    assert next(task.queries()).text == "what first?"
    qrels = task.qrels()
    assert qrels["qa"] == qrels["qb"] != qrels["qc"]
    for doc in task.documents():
        assert Path(doc.audio_path).exists()

    text_task = ManifestTask("spoken_squad", out, "text", "text")
    assert next(text_task.documents()).text == "First para. Two sentences."

    (r0, r1) = read_jsonl(out / "recordings.jsonl")
    assert (r0["recording_id"], r0["kind"], r0["duration_s"]) == ("0", "concatenated", 6.0)
    assert [(p["piece_id"], p["start_s"], p["end_s"]) for p in r0["pieces"]] == [
        ("0_0", 0.0, 3.0), ("0_1", 3.0, 6.0)]
    assert wav_duration_s(str(out / r0["audio"])) == 6.0

    q = {row["query_id"]: row for row in read_jsonl(out / "questions.jsonl")}
    # qa's answer is in the second sentence, the 1-3 s clip of paragraph 0_0.
    assert (q["qa"]["gold_unit"], q["qa"]["gold_start_s"], q["qa"]["gold_end_s"]) == (
        "sentence", 1.0, 3.0)
    assert q["qa"]["paragraph_span"] == [0.0, 3.0]
    assert (q["qb"]["gold_unit"], q["qb"]["gold_start_s"]) == ("paragraph", 0.0)
    assert (q["qc"]["gold_start_s"], q["qc"]["gold_end_s"]) == (3.0, 6.0)

    transcript = read_jsonl(out / "transcripts" / "0.jsonl")
    assert [(u["start_s"], u["end_s"], u["text"]) for u in transcript[:2]] == [
        (0.0, 1.0, "First para."), (1.0, 3.0, "Two sentences.")]


# --- long audio -----------------------------------------------------------------


def spans(segments):
    return [(s.start_s, s.end_s) for s in segments]


def test_fixed_windows_cover_without_gaps():
    seg = FixedWindowSegmenter(window_s=40)
    assert spans(seg.segment("x.wav", "rec", duration_s=100)) == [(0, 40), (40, 80), (80, 100)]
    overlap = FixedWindowSegmenter(window_s=40, hop_s=30)
    assert spans(overlap.segment("x.wav", "rec", duration_s=100)) == [
        (0, 40), (30, 70), (60, 100)]
    assert spans(seg.segment("x.wav", "rec", duration_s=10)) == [(0, 10)]
    assert overlap.name == "fixed_w40_h30"


def test_fixed_windows_read_duration_from_file(tmp_path):
    make_wav(tmp_path / "r.wav", 5.0)
    segs = FixedWindowSegmenter(window_s=2).segment(str(tmp_path / "r.wav"), "r")
    assert spans(segs) == [(0, 2), (2, 4), (4, 5)]
    assert [s.id for s in segs] == ["r#0", "r#1", "r#2"]


@pytest.mark.parametrize("kw", [{"window_s": 0}, {"window_s": 10, "hop_s": 11},
                                {"window_s": 10, "hop_s": 0}])
def test_fixed_windows_reject_bad_params(kw):
    with pytest.raises(ValueError):
        FixedWindowSegmenter(**kw).segment("x.wav", "r", duration_s=30)


def test_localization():
    seg = [Segment(id="s", parent_id="r", audio_path="x", start_s=10, end_s=20)]
    assert localization_accuracy(seg, 15, 25)
    assert not localization_accuracy(seg, 20, 30)  # touching is not overlapping
    assert not localization_accuracy(seg, 15, 25, min_overlap_s=5)
    assert localization_accuracy(seg, 12, 30, min_overlap_s=5)


def test_segments_as_documents_keep_timing():
    seg = Segment(id="r#1", parent_id="r", audio_path="r.wav", start_s=40, end_s=80)
    (doc,) = as_documents([seg])
    assert (doc.id, doc.parent_id, doc.audio_path, doc.start_s, doc.end_s) == (
        "r#1", "r", "r.wav", 40, 80)


# --- article filter, long-form task, within-recording ranks ---------------------


def test_build_restricts_to_articles(raw):
    counts = build_spoken_squad(raw / "squad.json", raw / "wav", raw / "out1", articles=[1])
    assert (counts["articles"], counts["questions"]) == (1, 1)
    with pytest.raises(ValueError, match=r"no audio for articles \[7\]"):
        build_spoken_squad(raw / "squad.json", raw / "wav", raw / "out2", articles=[1, 7])


def test_longaudio_task_over_spoken_squad(raw):
    from lsr.tasks.longform import LongAudioTask

    out = raw / "out"
    build_spoken_squad(raw / "squad.json", raw / "wav", out)
    # Article 0 is 6 s; article 1 is 4 s.
    task = LongAudioTask(out, FixedWindowSegmenter(window_s=2), min_overlap_s=1.0)
    check_task(task)
    assert task.name == "longform_tq2ac_out_fixed_w2_h2"
    assert [d.id for d in task.documents()] == ["0#0", "0#1", "0#2", "1#0", "1#1"]
    qrels = task.qrels()
    assert qrels["qa"] == {"0#0", "0#1"}  # sentence gold 1-3 s: 1 s in each window
    assert qrels["qb"] == {"0#0", "0#1"}  # paragraph gold 0-3 s: 2-3 s overlaps 1 s
    assert qrels["qc"] == {"0#1", "0#2"}
    assert task.query_group["qd"] == task.document_group["1#1"] == "1"
    assert task.archive_valid and task.has_transcripts()
    assert task.transcript_in_span("0", 0, 2) == "First para."  # midpoint rule
    assert task.transcript_in_span("0", 0, 3) == "First para. Two sentences."

    only = LongAudioTask(out, FixedWindowSegmenter(window_s=2), gold_units=("sentence",))
    assert [q.id for q in only.queries()] == ["qa"]
    assert only.name.endswith("_gold-sentence")


def test_ranks_within_groups():
    from lsr.eval import ranks_from_scores, ranks_within_groups

    qids, dids = ["q0", "q1"], ["a0", "a1", "b0"]
    qrels = {"q0": {"a1"}, "q1": {"b0"}}
    scores = np.array([[0.9, 0.5, 0.95],   # q0: b0 outranks gold only across groups
                       [0.9, 0.8, 0.1]])   # q1: alone in group b
    groups_q, groups_d = {"q0": "a", "q1": "b"}, {"a0": "a", "a1": "a", "b0": "b"}
    assert ranks_from_scores(scores, qrels, qids, dids).tolist() == [3, 3]
    assert ranks_within_groups(scores, qrels, qids, dids, groups_q, groups_d).tolist() == [2, 1]


def test_text_in_span_and_wer_normalisation():
    from lsr.backends.asr import Chunk, normalize_for_wer, text_in_span

    chunks = [Chunk(0, 4, "Hello there."), Chunk(4, 9, "General Kenobi!"), Chunk(9, 12, "x")]
    assert text_in_span(chunks, 0, 6) == "Hello there."  # second chunk's midpoint is 6.5
    assert text_in_span(chunks, 6, 12) == "General Kenobi! x"
    assert normalize_for_wer("It's  the Super-Bowl, 50!") == "it's the super bowl 50"


def test_original_text_replaces_asr_context(raw):
    """Spoken-SQuAD's contexts are ASR output; the read-aloud text is SQuAD's."""
    from lsr.tasks.longform import read_jsonl

    original = json.loads((raw / "squad.json").read_text())
    asr = json.loads((raw / "squad.json").read_text())
    para = asr["data"][0]["paragraphs"][0]
    para["context"] = "first para two sentences"  # what Spoken-SQuAD ships
    para["qas"][0]["answers"] = [{"text": "two", "answer_start": 11}]
    (raw / "asr.json").write_text(json.dumps(asr))
    (raw / "orig.json").write_text(json.dumps(original))

    counts = build_spoken_squad(raw / "asr.json", raw / "wav", raw / "o",
                                text_json=raw / "orig.json")
    assert counts["text_source"] == "squad_v1.1"
    q = {r["query_id"]: r for r in read_jsonl(raw / "o" / "questions.jsonl")}
    assert (q["qa"]["gold_unit"], q["qa"]["gold_start_s"]) == ("sentence", 1.0)
    assert read_jsonl(raw / "o" / "transcripts" / "0.jsonl")[1]["text"] == "Two sentences."

    counts = build_spoken_squad(raw / "asr.json", raw / "wav", raw / "a")
    assert counts["text_source"] == "spoken_squad_asr"
    assert counts["sentence_aligned_paragraphs"] == 2  # unpunctuated 0_0 no longer splits

    original["data"][0]["paragraphs"][0]["qas"] = []
    (raw / "bad.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="not in"):
        build_spoken_squad(raw / "asr.json", raw / "wav", raw / "b", text_json=raw / "bad.json")
