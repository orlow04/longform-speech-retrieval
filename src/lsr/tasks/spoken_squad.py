"""Build Spoken-SQuAD tasks from the raw release.

Spoken-SQuAD is SQuAD with every context read aloud by TTS, one WAV per
sentence, named ``{article}_{paragraph}_{chunk}.wav`` (indices into the SQuAD
JSON). Seed-VC voice-converted copies append the voice: ``..._{chunk}__{voice}.wav``.
Questions exist only as text.

Two granularities come out of one pass:

- **Paragraph** — one recording per paragraph (its chunks concatenated in chunk
  order), written as CLSR-format manifests (``que.txt`` / ``con.txt`` /
  ``wav.scp``, one row per question) so ``ManifestTask("spoken_squad", ...)``
  loads it. This is CLSR's direction: text question -> spoken context. CLASP's
  task over the same audio is the reverse self-retrieval, a different task.
- **Article** — one recording per article (its paragraphs concatenated in
  order), in the long-audio format of ``lsr.tasks.longform``, ``kind``
  "concatenated". A question's gold span is its paragraph's span. Dev articles
  run 43 minutes at the median — long, but assembled from short clips, which
  is exactly what the native-vs-concatenated comparison is about.

Audio I/O is the standard library's ``wave``, so this runs in the core
environment. It handles PCM WAV only, and refuses to concatenate files whose
format differs rather than resampling silently.
"""

from __future__ import annotations

import json
import re
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paragraph:
    article: int
    paragraph: int
    title: str
    context: str
    questions: list[tuple[str, str]]  # (squad qa id, question text)
    answer_starts: dict[str, int] = field(default_factory=dict)  # qa id -> char offset
    wav_paths: list[Path] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.article}_{self.paragraph}"


def collect_paragraphs(json_path: Path, wav_dir: Path
                       ) -> tuple[list[Paragraph], int, list[str]]:
    """Pair each SQuAD paragraph with its sentence WAVs, in chunk order.

    Returns the paragraphs that have audio, the count of those that do not
    (Spoken-SQuAD drops some), and the ids of paragraphs skipped because
    sentence clips are missing from the middle: their audio is incomplete and
    clip i would no longer be sentence i. The official dev release has three
    (20_10, 28_20, 33_0). Both belong in the build log, not in a silent filter.
    """
    chunks: dict[tuple[int, int], list[tuple[int, Path]]] = defaultdict(list)
    voices: set[str] = set()
    for wav in sorted(Path(wav_dir).glob("*.wav")):
        stem, _, voice = wav.stem.partition("__")
        try:
            a, p, c = (int(x) for x in stem.split("_"))
        except ValueError:
            continue
        voices.add(voice)
        chunks[(a, p)].append((c, wav))
    if len(voices) > 1:
        raise ValueError(f"{wav_dir} mixes voices {sorted(voices)}; build one voice at a time")

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))["data"]
    out, missing, gapped = [], 0, []
    for a, article in enumerate(data):
        for p, para in enumerate(article["paragraphs"]):
            found = sorted(chunks.get((a, p), []))
            if not found:
                missing += 1
                continue
            idx = [c for c, _ in found]
            if idx != list(range(len(idx))):
                gapped.append(f"{a}_{p}")
                continue
            out.append(Paragraph(
                article=a, paragraph=p, title=article.get("title", ""),
                context=para["context"],
                questions=[(qa["id"], qa["question"]) for qa in para["qas"]],
                answer_starts={qa["id"]: qa["answers"][0]["answer_start"]
                               for qa in para["qas"] if qa.get("answers")},
                wav_paths=[w for _, w in found],
            ))
    return out, missing, gapped


def concat_wavs(paths: list[Path], out_path: Path) -> list[tuple[float, float]]:
    """Concatenate PCM WAVs into ``out_path``; return each input's (start_s, end_s)."""
    if not paths:
        raise ValueError("nothing to concatenate")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spans, n_frames, params = [], 0, None
    with wave.open(str(out_path), "wb") as out:
        for path in paths:
            with wave.open(str(path), "rb") as src:
                fmt = (src.getnchannels(), src.getsampwidth(), src.getframerate())
                if params is None:
                    params = fmt
                    out.setnchannels(fmt[0])
                    out.setsampwidth(fmt[1])
                    out.setframerate(fmt[2])
                elif fmt != params:
                    raise ValueError(f"{path} has format {fmt}, expected {params}")
                frames = src.readframes(src.getnframes())
                count = src.getnframes()
            out.writeframes(frames)
            spans.append((n_frames / params[2], (n_frames + count) / params[2]))
            n_frames += count
    return spans


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split a SQuAD context on sentence ends. Approximate on purpose: it is only
    trusted when its count matches the number of TTS clips for the paragraph."""
    return [s for s in _SENTENCE_END.split(" ".join(text.split())) if s]


def _sentence_index(context: str, sentences: list[str], answer_start: int | None) -> int | None:
    """Index of the sentence holding the answer, in whitespace-normalised text."""
    if answer_start is None:
        return None
    offset = len(" ".join(context[:answer_start].split()))
    if answer_start > 0 and context[answer_start - 1].isspace():
        offset += 1  # the separating space survives normalisation
    end = 0
    for k, sentence in enumerate(sentences):
        end += len(sentence) + 1
        if offset < end:
            return k
    return None


def use_original_text(paragraphs: list[Paragraph], text_json: Path) -> None:
    """Replace each paragraph's context and answer offsets with original SQuAD's.

    Spoken-SQuAD's JSON carries the *ASR transcript* of the TTS audio as its
    context — lowercase, unpunctuated, ~22% WER — with answer offsets into that
    transcript. The text that was read aloud is SQuAD v1.1's. Article and
    paragraph indices coincide (48 / 2,067 in dev), and every Spoken-SQuAD
    question id is found at the same position; anything else raises.
    """
    data = json.loads(Path(text_json).read_text(encoding="utf-8"))["data"]
    for para in paragraphs:
        orig = data[para.article]["paragraphs"][para.paragraph]
        ids = {qa["id"] for qa in orig["qas"]}
        answers = {qa["id"]: qa["answers"][0]["answer_start"]
                   for qa in orig["qas"] if qa.get("answers")}
        missing = [qid for qid, _ in para.questions if qid not in ids]
        if missing:
            raise ValueError(f"paragraph {para.id}: questions {missing[:3]} not in {text_json} "
                             "at the same position")
        para.context = orig["context"]
        para.answer_starts = {qid: answers[qid] for qid, _ in para.questions if qid in answers}


def build_spoken_squad(
    json_path: Path, wav_dir: Path, out_dir: Path, articles: list[int] | None = None,
    text_json: Path | None = None,
) -> dict[str, int]:
    """Build one split into paragraph manifests and the long-audio format.

    Layout under ``out_dir``::

        que.txt  con.txt  wav.scp          CLSR format, paragraph level, one row per question
        recordings.jsonl  questions.jsonl  long-audio format (lsr.tasks.longform),
        transcripts/{a}.jsonl              one recording per article, kind "concatenated"
        audio/paragraphs/{a}_{p}.wav
        audio/articles/{a}.wav

    A question's gold span is the span of the sentence holding its answer
    (``gold_unit`` "sentence") where the sentence split matches the paragraph's
    clip count, and its paragraph's span otherwise (``gold_unit`` "paragraph");
    ``paragraph_span`` is always kept. Sentence-level gold is what makes this
    comparable with CORAAL-QA's short answer spans. The transcript is aligned
    at the same granularity; the counts are returned.

    ``text_json`` (SQuAD v1.1, same split) supplies the text that was actually
    read aloud; without it, contexts and transcripts are Spoken-SQuAD's own ASR
    output, which is not a clean-text ceiling and has no punctuation to split
    sentences on. Always pass it; ``text_source`` in the counts records which.

    ``articles`` restricts the build to those article indices — for smoke runs,
    and so that conditions built from different voice folders cover the same
    articles. An index with no audio raises rather than shrinking the set.
    Audio paths are relative to ``out_dir``; pass ``audio_root=out_dir`` to
    ``ManifestTask``.
    """
    from lsr.tasks.longform import write_jsonl

    out_dir = Path(out_dir)
    paragraphs, missing, gapped = collect_paragraphs(json_path, wav_dir)
    if articles is not None:
        absent = set(articles) - {p.article for p in paragraphs}
        if absent:
            raise ValueError(f"{wav_dir} has no audio for articles {sorted(absent)}")
        paragraphs = [p for p in paragraphs if p.article in set(articles)]
    if text_json is not None:
        use_original_text(paragraphs, text_json)

    by_article: dict[int, list[Paragraph]] = defaultdict(list)
    sentence_spans: dict[str, list[tuple[float, float]]] = {}
    que, con, scp = [], [], []
    for para in paragraphs:
        rel = Path("audio/paragraphs") / f"{para.id}.wav"
        sentence_spans[para.id] = concat_wavs(para.wav_paths, out_dir / rel)
        by_article[para.article].append(para)
        context = " ".join(para.context.split())
        for qid, question in para.questions:
            que.append(f"{qid} {' '.join(question.split())}\n")
            con.append(f"{qid} {context}\n")
            scp.append(f"{qid} {rel}\n")

    recordings, questions = [], []
    aligned = {"sentence": 0, "paragraph": 0}
    for a, paras in sorted(by_article.items()):
        rid = str(a)
        rel = Path("audio/articles") / f"{a}.wav"
        spans = concat_wavs([out_dir / "audio/paragraphs" / f"{p.id}.wav" for p in paras],
                            out_dir / rel)
        transcript = []
        for p, (start, end) in zip(paras, spans):
            sentences = split_sentences(p.context)
            sentence_aligned = len(sentences) == len(sentence_spans[p.id])
            if sentence_aligned:
                aligned["sentence"] += 1
                transcript += [{"start_s": start + s, "end_s": start + e, "text": t}
                               for t, (s, e) in zip(sentences, sentence_spans[p.id])]
            else:
                aligned["paragraph"] += 1
                transcript.append({"start_s": start, "end_s": end,
                                   "text": " ".join(p.context.split())})
            for qid, q in p.questions:
                row = {"query_id": qid, "recording_id": rid, "text": " ".join(q.split()),
                       "gold_start_s": start, "gold_end_s": end, "gold_unit": "paragraph",
                       "paragraph_span": [start, end]}
                k = _sentence_index(p.context, sentences, p.answer_starts.get(qid))
                if sentence_aligned and k is not None:
                    s_start, s_end = sentence_spans[p.id][k]
                    row.update(gold_start_s=start + s_start, gold_end_s=start + s_end,
                               gold_unit="sentence")
                questions.append(row)
        write_jsonl(out_dir / "transcripts" / f"{rid}.jsonl", transcript)
        recordings.append({
            "recording_id": rid, "audio": str(rel), "duration_s": spans[-1][1],
            "kind": "concatenated", "source": "spoken_squad", "title": paras[0].title,
            "transcript": f"transcripts/{rid}.jsonl",
            "pieces": [{"piece_id": p.id, "audio": f"audio/paragraphs/{p.id}.wav",
                        "start_s": s, "end_s": e} for p, (s, e) in zip(paras, spans)],
        })
    write_jsonl(out_dir / "recordings.jsonl", recordings)
    write_jsonl(out_dir / "questions.jsonl", questions)

    for name, lines in (("que.txt", que), ("con.txt", con), ("wav.scp", scp)):
        (out_dir / name).write_text("".join(lines), encoding="utf-8")
    return {
        "articles": len(by_article),
        "paragraphs": len(paragraphs),
        "paragraphs_without_audio": missing,
        "paragraphs_skipped_missing_clips": gapped,
        "questions": len(que),
        "sentence_aligned_paragraphs": aligned["sentence"],
        "paragraph_aligned_paragraphs": aligned["paragraph"],
        "questions_with_sentence_gold": sum(q["gold_unit"] == "sentence" for q in questions),
        "text_source": "squad_v1.1" if text_json else "spoken_squad_asr",
    }
