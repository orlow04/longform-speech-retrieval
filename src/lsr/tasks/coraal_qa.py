"""Build CORAAL-QA into the long-audio format: native, spontaneous, long.

CORAAL-QA (Balaji Shankar et al.; github.com/christinachance/CORAAL-QA) labels
questions over interviews from the Corpus of Regional African American Language
(CORAAL, CC BY-NC-SA 4.0). Each question is answered exactly once in its
recording, within a span of at most 45 s, given in seconds:

    When was the speaker born? : 17.0216, 19.5291

One QA file per CORAAL recording, same stem as the CORAAL ``.wav`` and the
CORAAL time-aligned transcript (``Line  Spkr  StTime  Content  EnTime``, tab
separated). Recordings are referenced in place, not copied.

The questions are interview-generic ("Where did the speaker grow up?" recurs
across interviews), so the archive pool is ill-posed here: score
within-recording. ``LongAudioTask.archive_valid`` reports this from the data.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from lsr.longaudio.base import wav_duration_s
from lsr.tasks.longform import write_jsonl

_PAUSE = re.compile(r"^\(pause [\d.]+\)$")


def parse_qa_file(path: Path) -> list[tuple[str, float, float]]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        question, sep, span = line.rpartition(":")
        try:
            start, end = (float(x) for x in span.split(","))
        except ValueError:
            raise ValueError(f"{path}:{n}: cannot parse span in {line!r}") from None
        if not sep or not question.strip() or end <= start:
            raise ValueError(f"{path}:{n}: malformed line {line!r}")
        out.append((" ".join(question.replace("’", "'").split()), start, end))
    return out


def parse_coraal_transcript(path: Path) -> list[dict]:
    """CORAAL text file -> time-aligned utterances, pauses dropped."""
    rows = []
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        _, speaker, start, content, end = cols[:5]
        content = content.strip()
        if not content or _PAUSE.match(content):
            continue
        rows.append({"start_s": float(start), "end_s": float(end), "speaker": speaker,
                     "text": content})
    return rows


def build_coraal_qa(qa_dir: Path, coraal_dir: Path, out_dir: Path) -> dict[str, int]:
    """Write recordings, questions and transcripts for every QA file with audio.

    ``coraal_dir`` is searched recursively for ``<stem>.wav`` (audio) and
    ``<stem>.txt`` (transcript) — the layout of the extracted CORAAL tarballs.
    QA files whose recording is absent are counted, not silently dropped.
    """
    qa_dir, coraal_dir, out_dir = Path(qa_dir), Path(coraal_dir), Path(out_dir)
    wavs = {p.stem: p for p in coraal_dir.rglob("*.wav")}
    texts = {p.stem: p for p in coraal_dir.rglob("*.txt")
             if not p.name.startswith(("README", "readme"))}

    qa_files = sorted(qa_dir.glob("*/*.txt"))
    if not qa_files:
        raise FileNotFoundError(f"no CORAAL-QA files under {qa_dir}/<COMPONENT>/")
    if not wavs:
        raise FileNotFoundError(f"no CORAAL .wav files under {coraal_dir}")
    recordings, questions = [], []
    missing_audio = missing_transcript = out_of_range = 0
    for qa_file in qa_files:
        rid = qa_file.stem
        if rid not in wavs:
            missing_audio += 1
            continue
        duration = wav_duration_s(str(wavs[rid]))
        rec = {"recording_id": rid,
               "audio": os.path.relpath(wavs[rid], out_dir),
               "duration_s": duration, "kind": "native", "source": "coraal_qa",
               "component": rid.split("_")[0]}
        if rid in texts:
            write_jsonl(out_dir / "transcripts" / f"{rid}.jsonl",
                        parse_coraal_transcript(texts[rid]))
            rec["transcript"] = f"transcripts/{rid}.jsonl"
        else:
            missing_transcript += 1
        recordings.append(rec)
        for i, (text, start, end) in enumerate(parse_qa_file(qa_file)):
            if end > duration + 1.0:
                out_of_range += 1
                continue
            questions.append({"query_id": f"{rid}#{i}", "recording_id": rid, "text": text,
                              "gold_start_s": start, "gold_end_s": min(end, duration)})

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "recordings.jsonl", recordings)
    write_jsonl(out_dir / "questions.jsonl", questions)
    return {"recordings": len(recordings), "questions": len(questions),
            "qa_files_without_audio": missing_audio,
            "recordings_without_transcript": missing_transcript,
            "questions_past_end_of_audio": out_of_range}
