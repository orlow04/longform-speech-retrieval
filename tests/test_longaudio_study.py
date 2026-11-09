"""Native-vs-concatenated study: CORAAL-QA builder, variants, diagnostics, BM25.

All on synthetic audio whose structure is known exactly: tones for speech,
zeros for silence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lsr.audio import read_wav, resample, write_wav
from lsr.longaudio import FixedWindowSegmenter
from lsr.longaudio.diagnostics import boundary_cues, frame_db, pauses, window_cue_strength
from lsr.retrievers.bm25 import bm25_scores
from lsr.tasks import check_task, variants
from lsr.tasks.coraal_qa import build_coraal_qa, parse_coraal_transcript, parse_qa_file
from lsr.tasks.longform import LongAudioTask, read_jsonl, write_jsonl

RATE = 1000


def tone(seconds: float, amp: float = 0.5, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 50 * t)).astype(np.float32)


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), np.float32)


# --- audio ----------------------------------------------------------------------


def test_wav_round_trip_and_slice(tmp_path):
    x = tone(2.0)
    write_wav(tmp_path / "a.wav", x, RATE)
    y, rate = read_wav(tmp_path / "a.wav")
    assert rate == RATE and np.allclose(x, y, atol=1e-4)
    part, _ = read_wav(tmp_path / "a.wav", 0.5, 1.0)
    assert len(part) == 500 and np.allclose(part, x[500:1000], atol=1e-4)
    assert len(resample(x, RATE, 2 * RATE)) == 4000


# --- CORAAL-QA --------------------------------------------------------------------


@pytest.fixture
def coraal(tmp_path):
    qa = tmp_path / "coraal_qa" / "LES"
    qa.mkdir(parents=True)
    (qa / "LES_a_1.txt").write_text(
        "Where did the speaker’s mother work? : 2.0, 4.0 \n"
        "When was the speaker born? : 6.5, 8.0\n", encoding="utf-8")
    (qa / "LES_b_1.txt").write_text("When was the speaker born? : 1.0, 2.0\n")
    (qa / "LES_gone_1.txt").write_text("Anything? : 1.0, 2.0\n")
    audio = tmp_path / "coraal" / "LES"
    write_wav(audio / "LES_a_1.wav", tone(10.0), RATE)
    write_wav(audio / "LES_b_1.wav", tone(5.0), RATE)
    (audio / "LES_a_1.txt").write_text(
        "Line\tSpkr\tStTime\tContent\tEnTime\n"
        "1\tLES_int_01\t0.0\tSo, tell me.\t1.5\n"
        "2\tLES_int_01\t1.5\t(pause 0.50)\t2.0\n"
        "3\tLES_se_01\t2.0\tMy mother worked at the mill.\t4.0\n")
    return tmp_path


def test_parse_qa_and_transcript(coraal):
    rows = parse_qa_file(coraal / "coraal_qa/LES/LES_a_1.txt")
    assert rows[0] == ("Where did the speaker's mother work?", 2.0, 4.0)
    utterances = parse_coraal_transcript(coraal / "coraal/LES/LES_a_1.txt")
    assert [u["text"] for u in utterances] == ["So, tell me.", "My mother worked at the mill."]


def test_parse_qa_rejects_malformed(tmp_path):
    (tmp_path / "x.txt").write_text("no span here\n")
    with pytest.raises(ValueError):
        parse_qa_file(tmp_path / "x.txt")


def test_build_coraal_qa(coraal):
    out = coraal / "build"
    counts = build_coraal_qa(coraal / "coraal_qa", coraal / "coraal", out)
    assert counts == {"recordings": 2, "questions": 3, "qa_files_without_audio": 1,
                      "recordings_without_transcript": 1, "questions_past_end_of_audio": 0}
    task = LongAudioTask(out, FixedWindowSegmenter(window_s=5), min_overlap_s=1.0)
    check_task(task)
    assert all(r["kind"] == "native" for r in task.recordings.values())
    assert (out / task.recordings["LES_a_1"]["audio"]).resolve() == \
        (coraal / "coraal/LES/LES_a_1.wav").resolve()
    # "When was the speaker born?" is asked of both recordings.
    assert task.n_ambiguous_questions == 2 and not task.archive_valid
    assert not task.has_transcripts()  # LES_b_1 has none
    assert task.transcript_in_span("LES_a_1", 0, 5) == "So, tell me. My mother worked at the mill."


def test_build_coraal_qa_refuses_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_coraal_qa(tmp_path, tmp_path, tmp_path / "out")


# --- variants ---------------------------------------------------------------------


@pytest.fixture
def concat_build(tmp_path):
    """One recording of three 2 s pieces separated by 1 s of silence, as a build.

    Piece k spans [3k, 3k + 3): 2 s of tone then 1 s of silence. Each piece's
    question has gold [3k, 3k + 2) — the tone.
    """
    out = tmp_path / "concat"
    amps = [0.5, 0.1, 0.3]
    x = np.concatenate([np.concatenate([tone(2.0, a), silence(1.0)]) for a in amps])
    write_wav(out / "audio" / "r.wav", x, RATE)
    write_jsonl(out / "recordings.jsonl", [{
        "recording_id": "r", "audio": "audio/r.wav", "duration_s": 9.0,
        "kind": "concatenated", "source": "toy", "transcript": "transcripts/r.jsonl",
        "pieces": [{"piece_id": f"p{k}", "start_s": 3.0 * k, "end_s": 3.0 * k + 3}
                   for k in range(3)]}])
    write_jsonl(out / "questions.jsonl", [
        {"query_id": f"q{k}", "recording_id": "r", "text": f"piece {k}?",
         "gold_start_s": 3.0 * k, "gold_end_s": 3.0 * k + 2, "gold_unit": "sentence"}
        for k in range(3)])
    write_jsonl(out / "transcripts" / "r.jsonl", [
        {"start_s": 3.0 * k, "end_s": 3.0 * k + 2, "text": f"words of piece {k}"}
        for k in range(3)])
    return out


def _gold(build):
    return {q["query_id"]: (round(q["gold_start_s"], 3), round(q["gold_end_s"], 3))
            for q in read_jsonl(build / "questions.jsonl")}


def test_shuffled_moves_gold_with_its_piece(concat_build, tmp_path):
    out = tmp_path / "shuf"
    variants.shuffled(concat_build, out, seed=1)
    (rec,) = read_jsonl(out / "recordings.jsonl")
    assert rec["duration_s"] == pytest.approx(9.0)
    order = [int(float(p["piece_id"].split("@")[1]) // 3) for p in rec["pieces"]]  # r@3.00 -> 1
    assert sorted(order) == [0, 1, 2] and order != [0, 1, 2]
    gold = _gold(out)
    for slot, k in enumerate(order):
        assert gold[f"q{k}"] == (3.0 * slot, 3.0 * slot + 2)
    # The audio moved with it: piece k's loudness is now at its slot.
    x, _ = read_wav(out / rec["audio"])
    slot0 = order.index(0)
    assert np.abs(x[slot0 * 3000: slot0 * 3000 + 2000]).max() == pytest.approx(0.5, abs=1e-3)
    task = LongAudioTask(out, FixedWindowSegmenter(window_s=3))
    check_task(task)


def test_smoothed_trims_silence_and_levels(concat_build, tmp_path):
    out = tmp_path / "smooth"
    variants.smoothed(concat_build, out, target_db=-20.0)
    (rec,) = read_jsonl(out / "recordings.jsonl")
    assert rec["duration_s"] == pytest.approx(6.0, abs=0.05)  # 1 s gaps removed
    assert _gold(out) == {"q0": (0.0, 2.0), "q1": (2.0, 4.0), "q2": (4.0, 6.0)}
    x, _ = read_wav(out / rec["audio"])
    levels = [20 * np.log10(np.sqrt(np.mean(x[i * 2000:(i + 1) * 2000] ** 2)))
              for i in range(3)]
    assert levels == pytest.approx([-20.0] * 3, abs=0.5)


def test_spliced_inserts_pieces_and_keeps_host_questions(concat_build, tmp_path):
    host = tmp_path / "host"
    write_wav(host / "audio" / "h.wav", tone(20.0, 0.2, rate=2 * RATE), 2 * RATE)
    write_jsonl(host / "recordings.jsonl", [{
        "recording_id": "h", "audio": "audio/h.wav", "duration_s": 20.0, "kind": "native",
        "source": "toyhost", "transcript": "transcripts/h.jsonl"}])
    write_jsonl(host / "transcripts" / "h.jsonl", [
        {"start_s": float(t), "end_s": float(t + 5), "text": f"host {t}"} for t in (0, 5, 10, 15)])
    write_jsonl(host / "questions.jsonl", [
        {"query_id": "hq", "recording_id": "h", "text": "host?", "gold_start_s": 16.0,
         "gold_end_s": 18.0}])

    out = tmp_path / "spliced"
    variants.spliced(concat_build, host, out, pieces_per_host=2, seed=0)
    (rec,) = read_jsonl(out / "recordings.jsonl")
    assert rec["duration_s"] == pytest.approx(26.0)  # 20 s host + two 3 s pieces
    _, rate = read_wav(out / rec["audio"], 0, 0.01)
    assert rate == 2 * RATE  # pieces resampled to the host

    qs = read_jsonl(out / "questions.jsonl")
    origins = {q["query_id"]: q["origin"] for q in qs}
    assert origins["hq"] == "host" and sorted(origins.values()).count("spliced") == 2
    host_q = next(q for q in qs if q["query_id"] == "hq")
    assert host_q["gold_start_s"] == pytest.approx(16.0 + 6.0)  # both pieces land before 16 s
    # Insertions happen only at host utterance ends.
    piece_starts = [p["start_s"] for p in rec["pieces"] if p["piece_id"].startswith("r@")]
    assert all(round(s - 3 * i) in (5, 10, 15) for i, s in enumerate(piece_starts))
    task = LongAudioTask(out, FixedWindowSegmenter(window_s=5))
    check_task(task)


# --- diagnostics -------------------------------------------------------------------


def test_pauses_found_relative_to_floor():
    x = np.concatenate([tone(1.0), silence(0.5), tone(1.0), silence(0.1), tone(1.0)])
    found = pauses(frame_db(x, RATE))
    assert len(found) == 1  # the 0.1 s gap is under min_pause_s
    (s, e), = found
    assert s == pytest.approx(1.0, abs=0.03) and e == pytest.approx(1.5, abs=0.03)


def test_boundary_cues_separate_concatenated_from_native():
    # "Concatenated": gold edges where a clip's speech meets the join's silence.
    concat = np.concatenate([tone(5, 0.5), silence(1.0), tone(5, 0.05), silence(1.0),
                             tone(5, 0.5)])
    cues = boundary_cues(concat, RATE, edges=[5.0, 12.0])
    assert cues["edge_at_pause"] == 1.0 and cues["jump_percentile"] > 0.9
    # "Native": edges in the middle of continuous speech with short ordinary pauses.
    native = np.concatenate([np.concatenate([tone(2.0), silence(0.3)]) for _ in range(8)])
    cues = boundary_cues(native, RATE, edges=[3.0, 7.5])  # both inside speech
    assert cues["edge_at_pause"] == 0.0


def test_window_cue_strength_peaks_at_the_join():
    x = np.concatenate([tone(4.0), silence(1.0), tone(4.0)])
    scores = window_cue_strength(x, RATE, [(0, 3), (3, 6), (6, 9)])
    assert scores.argmax() == 1 and scores[0] == scores[2] == 0


# --- BM25 ---------------------------------------------------------------------------


def test_bm25_prefers_overlap_and_ignores_stopwords():
    docs = ["my mother worked at the mill", "we played basketball every day", ""]
    s = bm25_scores(["where did the mother work", "what is the of"], docs)
    assert s[0].argmax() == 0 and s[0, 2] == 0
    assert np.all(s[1] == 0)  # only stopwords: no evidence, everything ties


def test_question_rows_are_kept(concat_build):
    task = LongAudioTask(concat_build, FixedWindowSegmenter(window_s=3))
    assert task.question_rows["q1"]["gold_unit"] == "sentence"
    json.dumps(task.question_rows)
