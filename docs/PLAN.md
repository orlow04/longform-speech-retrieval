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
(CLASP's training unit); ~30 min and SPIRAL were added on 2026-11-11 (end of
this section). A text question retrieves the document holding its
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

**Full run (2026-11-09, all 48 dev articles, 4090, official checkpoint
sha256 9ad6954a…, parity OK on cuda, 112 min).** 5,342 questions, none dropped;
documents per condition 2,064 / 4,047 / 2,091 / 423. Fixed pool = 423 (every
5-min document; chance 0.0024). A random pool of 423 holds ~9 same-article
distractors in every condition, so the fixed pool compares like with like.
`runs/20261109-204337_*`, `results/pilot_4090_20261109-204322.tar.gz`:

| Duration | CLASP mean | LaBSE text mean | BM25 text | CLASP pairwise cos | CLASP eff. rank @423 | CLASP top-1 var |
|---|---|---|---|---|---|---|
| paragraph | 0.084 [0.078, 0.091] | 0.641 [0.63, 0.65] | 0.859 | 0.892 | 139 | 0.17 |
| 30 s | 0.086 [0.080, 0.092] | 0.662 [0.65, 0.67] | 0.840 | 0.863 | 147 | 0.14 |
| 1 min | 0.076 [0.070, 0.082] | 0.581 [0.57, 0.59] | 0.830 | 0.900 | 138 | 0.17 |
| 5 min | 0.081 [0.073, 0.088] | 0.386 [0.37, 0.40] | 0.752 | 0.954 | 112 | 0.20 |

Top-1 split into "right article" vs "right document given right article"
(full pool, from the saved embeddings):

| Duration | CLASP: top-1 in gold article | CLASP: correct given article | LaBSE-mean: in article | LaBSE-mean: correct given article | docs / article |
|---|---|---|---|---|---|
| paragraph | 0.21 | 0.15 | 0.81 | 0.56 | 43 |
| 30 s | 0.19 | 0.10 | 0.82 | 0.50 | 84 |
| 1 min | 0.22 | 0.13 | 0.79 | 0.48 | 44 |
| 5 min | 0.28 | 0.29 | 0.76 | 0.51 | 9 |

What changes with 48 articles:

1. **The text arms confirm the 3-article result.** Mean-pooled LaBSE falls
   0.66 → 0.39 from 30 s to 5 min (−42%), BM25 0.84 → 0.75 (−10%). Averaging
   is still what fails. Mean-pooled LaBSE barely loses the article (0.82 →
   0.76) but cannot pick the document inside it: 0.51 correct among 9
   candidates at 5 min (chance 0.11) against 0.50 among 84 at 30 s (chance
   0.012).
2. **CLASP is flat with duration** (0.08–0.09, CIs overlapping), not falling
   as on 3 articles. This is a floor effect, not robustness. CLASP finds the
   right article for only ~1 question in 5 (chance 0.02); its query-gold
   cosine (0.05–0.07) is below the best negative (0.07–0.15) on average. With
   47 off-topic articles in the pool, cross-topic failure dominates, and it
   does not get worse with length. Longer audio even helps CLASP find the
   article (0.19 → 0.28) while hurting localisation within it. The two effects
   cancel. The 3-article run, whose pool was mostly same-topic, saw the
   localisation loss alone.
3. **The latent space tightens with duration on the full data too:** CLASP
   pairwise cosine 0.86 → 0.95, effective rank at 423 147 → 112, first
   principal direction 14% → 20% of variance. 30 s is the least collapsed
   condition for both arms, as on 3 articles. LaBSE-mean moves the same way
   (0.27 → 0.50).
4. **Paragraph ≈ 30 s for every arm.** Matching CLASP's training unit buys
   nothing over a duration-matched grouping.

Implication: the released CLASP is too weak at text-question → audio for
duration effects to show through it. Duration effects need either an audio
encoder that clears the topic floor, or the within-article pool (CLSR-style,
as in §2) where CLASP's topical weakness is held fixed.

**Added 2026-11-11, to run on the 4090 (`scripts/run_pilot_4090.sh`): the
30 min condition and SPIRAL**, the two things the card names that the runs
above lack.

- *Spoken-SQuAD at ~30 min* (`durations: [paragraph, 30, 60, 300, 1800]`).
  Articles have a median of 43 min of audio, so a 30 min document is half an
  article or all of it, and there are only one or two per article (5 documents
  on the 3-article build, 909–2,633 s). The task becomes "find the article".
  The fixed pool shrinks to that document count, so every condition's
  fixed-pool number is recomputed at the smaller pool. The table at pool 423
  above stays the reference for the four shorter conditions.
- *SPIRAL* (`configs/spiral_pilot.yaml`, `lsr.tasks.spiral`,
  `scripts/download_spiral.py`; Lin et al., SpeechPrune, ICME 2025). 1,012
  synthetic recordings (GPT-4o scripts, StyleTTS2 voices; 350 lectures, 338
  conversations, 324 meetings; 61–134 s, median 87 s, 24.7 h in all, 24 kHz),
  each with one question about one "key sentence" inside it. The passage is
  the whole recording. Recordings are shuffled (seed 0) and joined into 25
  sessions of ~1 h, then cut at recording boundaries: 1,012 documents of one
  recording, 304 of ~5 min (median 283 s), 49 of ~30 min (median 1,818 s).
  Fixed pool 49. Two differences from Spoken-SQuAD, both wanted: a document is
  topically mixed (neighbouring recordings are unrelated), so there is no
  same-article localisation problem and averaging has to dilute the gold
  passage with unrelated speech; and the 30 min condition keeps 49 documents
  instead of collapsing to one per article.
- *Two queries on SPIRAL*: the question, and the key sentence itself (a
  transcript sentence of the gold recording, i.e. the audio-to-transcript
  matching CLASP was trained on, and the query CLASP's own SPIRAL evaluation
  uses). If CLASP clears the floor with the key sentence, the duration effect
  can show through it. If not, the floor is not about question style.
- The annotations give the key sentence's index, not its time, so gold is the
  whole source recording. Parity with CLASP's code holds on 24 kHz SPIRAL audio
  too (cosine 1.000000, Mac).
- Smoke run on 40 recordings (Mac, `runs/20261111-210210_*`, numbers not
  meaningful at this size): the pipeline runs end to end for both queries.
  CLASP full-pool Hits@1 at one recording per document was 0.10 for the
  question and 0.05 for the key sentence among 40 documents (chance 0.025),
  against 0.85 and 0.88 for mean-pooled LaBSE: the floor may well persist.

Each run now also writes `runs/<stamp>_duration_pilot_gold.json` (the gold
document of every query in every condition), so within-recording pools can be
scored afterwards from the saved embeddings.

Next, after these two runs: within-article pool results alongside the global
pool; max-sim over chunks as the second pooling arm (per-chunk features are
cached on the box); a native long-audio control (CORAAL cut to the same
durations).

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
