# Plan

Set 2026-11-03. Supersedes the ordering in [PLAN_CLSR.md](PLAN_CLSR.md): CLSR
is now a reference system and its benchmarks are reference tasks, not the
goal. DRCD is dropped.

Three pillars, sharing one harness (`lsr.tasks`, `lsr.eval`, `lsr.store`):

1. **MSEB** — an external, standard benchmark for how well speech-text
   retrieval works, so our numbers sit next to published ones.
2. **Long-form audio** — retrieval directly from audio vs. an ASR cascade,
   when the thing being searched is minutes long rather than one utterance.
3. **Degradations** — ParaSpeechRAG's perturbation axes, carried over to both
   of the above.

## 1. MSEB

The Massive Sound Embedding Benchmark (`github.com/google-research/mseb`)
evaluates any `mseb.encoder.MultiModalEncoder` on SVQ (Simple Voice Questions:
short spoken queries, 26 locales). CLASP already has an adapter
(`clasp/src/clasp/mseb_adapter/clasp_encoder.py`) and a write-up
(`clasp/docs/MSEB.md`) that record the environment traps; read both first.

- **Adapter, not fork.** One `lsr` module wraps any `lsr` retriever as a
  `MultiModalEncoder` (Sound -> query embedding, Text -> document embedding,
  both L2-normalised so MSEB's `dot_product` is our cosine). A new backend gets
  MSEB for free.
- **Environment.** MSEB needs Python ≥ 3.12 and `scann` (x86-Linux wheels only);
  the SVQ loader segfaults on Apple Silicon. So it is an `[mseb]` extra in its
  own 3.12 venv, run on `lab150`. Install `mseb` alone first — co-resolving it
  backtracks apache-beam to a version that does not build.
- **Tasks, in order.** `SVQEnUs{Passage,Document}InLangRetrieval` (the
  retrieval number), `SVQEnUsQueryReranking` (runs on macOS, a cheap
  shakedown), then the `stabilities` tasks, which measure embedding stability
  under augmentation and so connect directly to pillar 3.
- **Also import SVQ as an `lsr` task**, so our metric core, bootstrap CIs and
  perturbations apply to it. Check that both paths agree on one backend before
  trusting either. SVQ's recordings carry an environment label (clean,
  background speech, media noise, traffic): real-world degradation to set
  against our synthetic ones.

## 2. Long-form audio: retrieval vs. ASR

**Data, in order of cost:**

- **L1 — Spoken-SQuAD articles.** Built now by `lsr.tasks.spoken_squad`: every
  article's paragraphs joined into one recording, with each paragraph's exact
  time span. A question's gold span is its paragraph. Several minutes per
  article, ground truth exact, no new data. TTS speech, so it is the easy case.
- **L2 — Spoken Wikipedia Corpus**, ~30-minute real recordings, with SLUE-SQA-5
  questions spliced in as CLSR's Table 5 does (PLAN_CLSR.md, Track C). Real
  speech, real length.

**Two pools, two questions — both reported, never mixed:**

- *within-recording*: rank only the segments of the question's own article.
  "Find the answer in this recording."
- *archive*: rank all segments of all articles. "Find the answer in a
  collection." Fits the current `Task` protocol as is; within-recording needs
  per-query candidate restriction by `parent_id` in the metric path.

**Arms**, with segmentation (`FixedWindowSegmenter`, window 30 / 40 / 60 s, with
and without overlap) and the text encoder held fixed:

1. **Direct audio retrieval** — embed each window's audio (CLASP-style pooled
   dual encoder first; a CLSR-style bridge if Track B of PLAN_CLSR.md lands).
2. **ASR cascade** — transcribe the whole recording once with word timestamps
   (Parakeet TDT or Whisper), cut the transcript on the same windows, embed it
   as text.
3. **Clean-text ceiling** — the gold text cut on the same windows.

**Measured per arm:** Hits@k / MRR over segments (relevant = overlaps the gold
span), localization accuracy, WER for the cascade, wall-clock and **seconds of
audio processed**. The 30-second encoder windows of most audio models are why
window length is a parameter, not a constant.

## 3. Degradations

Port ParaSpeechRAG's axes into `lsr.perturb`, keeping its pinned parameters
(`ParaSpeechRAG/configs/perturb/*.yaml`) and fixing the known defects listed in
LANDSCAPE.md: noise (white, ESC-50 ambient) with a named SNR definition, speaking
rate, codec, reverberation with a knob that is what it says, and speaker /
emotion via Seed-VC run offline on the box.

- **Query side** on MSEB/SVQ: comparable to ParaSpeechRAG, and checkable
  against SVQ's real noise environments.
- **Document side** on long-form: new. Here the audio is the document, so the
  question becomes whether a degraded archive is still searchable, and whether
  the cascade's text bottleneck protects it as it protected short queries.

## Order

1. Build Spoken-SQuAD dev on `lab150` with `scripts/build_spoken_squad.py`
   (paths in `configs/spoken_squad_dev.yaml` still to confirm).
2. Long-form task over the article recordings, within-recording pool in the
   metric path, and a `scripts/score_run.py` that scores a run directory.
3. Cascade arm, then the clean-text ceiling — both cheap, and together they
   bracket what direct retrieval has to beat.
4. Direct audio arm.
5. MSEB adapter and the SVQ task.
6. Perturbations, query side on SVQ first (comparable to prior work), then
   document side on long-form.
