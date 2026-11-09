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

## 2. Long spoken context: does the field need a native long-audio dataset?

**Set 2026-11-03.** Retrieval targets long spoken context: the question is
text, the document is a long recording, the answer is a span inside it.

**Research question.** Long-audio retrieval benchmarks are mostly *assembled*:
Spoken-SQuAD articles are TTS sentences joined end to end; CLSR's Table 5
splices a SLUE context into a Spoken Wikipedia recording. Do such benchmarks
measure what natively long recordings demand, or do they reward artifacts of
the assembly?

**Hypothesised artifacts,** each with the measurement that tests it:

| Artifact | Why concatenation has it | Measured by |
|---|---|---|
| boundary cues | gold edges coincide with joins: pauses, loudness jumps | gold-edge pause / jump percentiles; `cue_only` baseline |
| topical outlier | a spliced piece is off-topic in its host | `spliced` vs. host questions in the same audio |
| self-containment | SQuAD paragraphs were written to stand alone; native speech refers back | `shuffled` vs. original order |
| lexical overlap | SQuAD questions were written while reading the passage | question–gold token overlap; BM25 vs. dense gap |
| read / TTS style | uniform, fluent, no disfluency | ASR WER and the ASR–clean gap, native vs. concatenated |

If retrieval on assembled data moves with these knobs while native data does
not, or if system rankings differ between them, assembled benchmarks are not
a substitute and a native dataset is needed.

**Datasets** (`lsr.tasks.longform` format; one format, so nothing downstream
can tell them apart except by content):

- **Native — CORAAL-QA**: 143 sociolinguistic interviews (CORAAL, CC BY-NC-SA
  4.0), mostly > 45 min, spontaneous; 1,946 hand-written questions, each
  answered once, with gold spans in seconds (median 8.4 s). Questions are
  interview-generic, so only the within-recording pool is well-posed.
- **Concatenated — Spoken-SQuAD dev articles**: 48 articles, 43 min median.
  Gold at **sentence** level (the sentence holding the SQuAD answer, ~CORAAL's
  span length) and at paragraph level (~62 s, the usual setup).
- **Variants** (`lsr.tasks.variants`): `shuffled` (paragraph order),
  `smoothed` (edge silence trimmed, loudness levelled), `spliced` (Spoken-SQuAD
  paragraphs inside CORAAL interviews, host questions kept as control).
- **Later**: Spoken Wikipedia Corpus — native *read* speech; if its articles
  overlap SQuAD's, the same questions on human-read whole articles isolate
  native-vs-assembled from read-vs-spontaneous. Then the male / female voice
  conversions as a further condition on the concatenated side.

**Arms** (`scripts/run_longaudio.py`), windows 30 s / 15 s hop and 60 s / 20 s
hop: `random`, `cue_only` (content-free), `bm25_clean` / `dense_clean` (gold
transcript), `bm25_asr` / `dense_asr` (Whisper). A direct audio retriever
(CLASP-style, then a CLSR-style bridge) is the next arm.

**First results (2026-11-03, CPU, gold-text arms only; no ASR yet).** CORAAL
LES (15 interviews, 215 questions) vs. Spoken-SQuAD dev articles 0, 10, 11
(3 x ~45 min, 617 questions with sentence-level gold), text from SQuAD v1.1.
30 s windows / 15 s hop, within-recording Hits@1 with 95% bootstrap CI
(`runs/20261103-125524_*`):

| Build | cue_only | BM25 clean | dense clean |
|---|---|---|---|
| CORAAL native | 0.005 | 0.744 [0.68, 0.80] | 0.763 [0.71, 0.82] |
| SQuAD same-article concatenation | 0.003 | 0.663 [0.63, 0.70] | 0.661 [0.62, 0.70] |
| — shuffled | 0.005 | 0.648 | 0.653 |
| — smoothed | 0.003 | 0.661 | 0.666 |
| — paragraph-level gold | 0.019 | 0.720 | 0.725 |
| SQuAD paragraphs spliced into CORAAL: spliced questions | **0.136** [0.09, 0.18] | **0.843** [0.79, 0.89] | **0.843** [0.79, 0.90] |
| — same recordings, host questions | 0.005 | 0.712 [0.65, 0.77] | 0.749 [0.69, 0.81] |

What it says so far:

1. **The construction decides the difficulty.** The same SQuAD questions, with
   the same gold audio, go from 0.66 Hits@1 among their own article's
   paragraphs to 0.84 when spliced into an unrelated interview (both arms,
   intervals disjoint). Same-article distractors are on-topic (54 paragraphs
   on Super Bowl 50); a splice is an outlier. CLSR Table 5's construction is
   the easy one.
2. **Splicing leaks acoustically; same-article concatenation does not, here.**
   The content-free `cue_only` finds spliced paragraphs at 0.136 Hits@1, ~5x
   chance and 27x the host questions in the same audio. In same-article
   concatenation every *sentence* is a joint (99% of gold edges at a pause vs.
   25% of random points), so joints do not single out the gold sentence:
   `cue_only` is at chance. The leak depends on construction, not merely on
   concatenation.
3. **Order and smoothing do not move text retrieval** (shuffled, smoothed within
   CI of the original), as expected for window-independent text arms. These
   variants become informative with the audio arm (smoothing) and with
   context-using models (shuffling).
4. **Native is not "harder" on clean text.** CORAAL's gold text is as findable as
   SQuAD's. The native/concatenated difference expected to matter most —
   spontaneous speech under ASR — needs the ASR arms, on the GPU.

Caveats: one CORAAL component and three SQuAD articles; host and spliced
questions differ in style as well as in placement (the same-questions
comparison in point 1 is the controlled one).

### Pilot: does CLASP degrade with document duration? (Notion card "Validar")

`scripts/run_duration_pilot.py`, `configs/duration_pilot.yaml`. Spoken-SQuAD
sentence clips joined in article order into documents of ~30 s, ~1 min, ~5 min
(`lsr.longaudio.grouping`, sentences never split), plus the original paragraphs
(CLASP's training unit). A text question retrieves the document holding its
answer sentence from all documents of that duration. CLASP is reproduced in
`lsr.backends.clasp` (parity with CLASP's code: cosine 1.000000 on HuBERT,
image and fused outputs; `scripts/check_clasp_parity.py`). Controls: the same
629 questions in every condition; a fixed pool of 27 (exact expectation over
random negatives, `lsr.eval.sampled_pool_values`) besides the full pool; a
text arm with the *same* mean-pooling rule (LaBSE per sentence, averaged).

**First run (2026-11-09, 3 articles, Mac/MPS, ~20 min of feature extraction).**
Fixed-pool Hits@1 (chance 0.037), 95% CI; `runs/20261109-152120_*`:

| Duration | CLASP mean | LaBSE text mean | BM25 text | CLASP pairwise cos | CLASP eff. rank @27 |
|---|---|---|---|---|---|
| paragraph (~53 s) | 0.183 [0.16, 0.21] | 0.683 | 0.858 | 0.892 | 21.9 |
| 30 s | 0.202 [0.18, 0.23] | 0.679 | 0.841 | 0.876 | 22.9 |
| 1 min | 0.154 [0.13, 0.18] | 0.614 | 0.830 | 0.909 | 22.0 |
| 5 min | 0.110 [0.09, 0.14] | 0.417 | 0.731 | 0.963 | 19.1 |

Readings, provisional (27 five-minute documents in total):

1. **R@k falls with duration for every arm** (H2.1 not falsified): CLASP
   0.20 → 0.11 from 30 s to 5 min (−45%), LaBSE-mean −39%, BM25 −13%.
2. **Most of the fall is the pooling rule, not the audio encoder.** Mean-pooled
   gold text loses nearly as much as mean-pooled audio; BM25 (no averaging)
   loses a third as much. Averaging a document into one vector is what fails.
3. **CLASP is weak on this task even at its training unit** (0.18 at
   paragraph, 5x chance): it was trained to match audio to its own transcript,
   not to a question. Query-gold cosine is ~0.04 against ~0.45 for LaBSE text.
4. **Latent space: CLASP audio embeddings are already near-collapsed** (mean
   pairwise cosine 0.89 at paragraph level) and tighten further with duration
   (0.96 at 5 min; first principal direction 27% → 45% of variance; effective
   rank at equal n 22 → 19). The text arm shows the same direction, from a
   far more spread-out start (0.38 → 0.62).

Next: all 48 dev articles on the 4090 (`bash scripts/run_pilot_4090.sh`;
`configs/duration_pilot_4090.yaml`), official CLASP checkpoint. Then max-sim over chunks as the second pooling arm, a native
long-audio control (CORAAL cut to the same durations), SPIRAL.

## 3. Degradations

Port ParaSpeechRAG's axes into `lsr.perturb`, keeping its pinned parameters
(`ParaSpeechRAG/configs/perturb/*.yaml`) and fixing the known defects listed in
LANDSCAPE.md: noise (white, ESC-50 ambient) with a named SNR definition, speaking
rate, codec, reverberation with a knob that is what it says, and speaker /
emotion via Seed-VC run offline on the box.

- **Query side** on MSEB/SVQ: comparable to ParaSpeechRAG, and checkable
  against SVQ's real noise environments.
- **Document side** on long spoken context: new. Here the audio is the document, so the
  question becomes whether a degraded archive is still searchable, and whether
  the cascade's text bottleneck protects it as it protected short queries.

## Order

1. Build CORAAL-QA (all five components) and Spoken-SQuAD dev with its
   variants: `scripts/build_longaudio.py --config configs/longaudio.yaml`.
2. Run every arm on every build: `scripts/run_longaudio.py`. The CPU arms run
   anywhere; the ASR arms want the GPU.
3. Direct audio arm (CLASP-style pooled encoder over windows).
4. Spoken Wikipedia as native read speech.
5. MSEB adapter and the SVQ task.
6. Perturbations, document side on the long recordings first, then query side
   on SVQ.
