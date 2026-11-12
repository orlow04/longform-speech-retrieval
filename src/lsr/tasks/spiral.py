"""SPIRAL -> long-audio builds: its ~90 s recordings joined into long sessions.

SPIRAL (Lin et al., SpeechPrune, ICME 2025; ``scripts/download_spiral.py``) has
1,012 synthetic recordings (GPT-4o scripts, StyleTTS2 voices), each with one
multiple-choice question about one "key sentence" spoken inside it. A recording
is a self-contained passage, so "concatenating passages simply" here means
joining whole recordings: they are shuffled (seeded) and packed end to end into
sessions of about ``session_s``, which ``lsr.longaudio.grouping`` then cuts
into documents of any target duration at recording boundaries.

Unlike Spoken-SQuAD's articles, a session is topically mixed: neighbouring
recordings have nothing to do with each other.

The annotations carry no timestamps, only the key sentence's index among the
recording's sentences, so the gold span is the whole source recording
(``gold_unit`` "recording"). Transcript units are source recordings too, with
their sentences kept under ``sentences``.
"""

from __future__ import annotations

import json
import random
import wave
from pathlib import Path

from lsr.tasks.longform import write_jsonl
from lsr.tasks.spoken_squad import concat_wavs


def read_spiral(jsonl: Path) -> list[dict]:
    """One row per SPIRAL sample, flattened to the fields used here."""
    rows = []
    for line in Path(jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        combined, key = r["audio_data"]["combined"], r["audio_data"]["key_sentences"]
        if len(combined["audio_files"]) != 1:
            raise ValueError(f"{r['metadata']['id']}: expected one audio file, "
                             f"got {combined['audio_files']}")
        rows.append({
            "id": r["metadata"]["id"],
            "wav": combined["audio_files"][0],
            "sentences": [" ".join(s.split()) for s in combined["sentences"]],
            "question": " ".join(r["test_question"]["question"].split()),
            "key_sentence": " ".join(key["text"].split()),
            "key_sentence_index": list(key["indices"]),
            "transcript_type": r["metadata"]["transcript_type"],
            "main_topic": r["metadata"]["main_topic"],
            "subtopic": r["metadata"]["subtopic"],
        })
    return rows


def _duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def build_spiral(jsonl: Path, wav_dir: Path, out_dir: Path, session_s: float = 3600.0,
                 seed: int = 0, limit: int | None = None) -> dict[str, int]:
    """Build the long-audio format (``lsr.tasks.longform``) under ``out_dir``.

    ``limit`` keeps only the first samples of the file, for smoke runs. A sample
    whose WAV is absent raises rather than shrinking the set.
    """
    out_dir, wav_dir = Path(out_dir), Path(wav_dir)
    rows = read_spiral(jsonl)[:limit]
    absent = [r["wav"] for r in rows if not (wav_dir / r["wav"]).exists()]
    if absent:
        raise FileNotFoundError(f"{wav_dir}: {len(absent)} WAVs missing, e.g. {absent[:3]}")
    random.Random(seed).shuffle(rows)

    sessions: list[list[dict]] = [[]]
    filled = 0.0
    for r in rows:
        if sessions[-1] and filled >= session_s:
            sessions.append([])
            filled = 0.0
        sessions[-1].append(r)
        filled += _duration_s(wav_dir / r["wav"])

    recordings, questions = [], []
    for k, members in enumerate(sessions):
        rid = f"s{k:02d}"
        rel = Path("audio/sessions") / f"{rid}.wav"
        spans = concat_wavs([wav_dir / r["wav"] for r in members], out_dir / rel)
        write_jsonl(out_dir / "transcripts" / f"{rid}.jsonl", [
            {"start_s": s, "end_s": e, "text": " ".join(r["sentences"]),
             "sentences": r["sentences"], "piece_id": r["id"]}
            for r, (s, e) in zip(members, spans)])
        recordings.append({
            "recording_id": rid, "audio": str(rel), "duration_s": spans[-1][1],
            "kind": "concatenated", "source": "spiral",
            "transcript": f"transcripts/{rid}.jsonl",
            "pieces": [{"piece_id": r["id"], "start_s": s, "end_s": e}
                       for r, (s, e) in zip(members, spans)],
        })
        questions += [
            {"query_id": r["id"], "recording_id": rid, "text": r["question"],
             "key_sentence": r["key_sentence"], "key_sentence_index": r["key_sentence_index"],
             "gold_start_s": s, "gold_end_s": e, "gold_unit": "recording",
             "transcript_type": r["transcript_type"], "main_topic": r["main_topic"],
             "subtopic": r["subtopic"]}
            for r, (s, e) in zip(members, spans)]
    write_jsonl(out_dir / "recordings.jsonl", recordings)
    write_jsonl(out_dir / "questions.jsonl", questions)
    texts = [q["text"].lower() for q in questions]
    return {
        "recordings": len(recordings),
        "source_recordings": len(rows),
        "questions": len(questions),
        "duplicate_question_texts": len(texts) - len(set(texts)),
        "total_s": round(sum(r["duration_s"] for r in recordings)),
    }
