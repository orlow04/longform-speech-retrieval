# Plan: reproduce CLSR, and test its claim against CLASP

Drafted 2026-10-05, rehomed into `lsr` on 2026-10-19. Background in
[PAPERS.md](PAPERS.md) and [LANDSCAPE.md](LANDSCAPE.md).

## Why

CLSR argues that the dual-encoder design is wrong and that a text-like
intermediate representation beats it at matched parameter count (49.82 vs
17.79 R@1). CLASP *is* that dual-encoder design. Two questions follow, and they
are different questions:

1. **Does CLSR replicate** from the authors' release?
2. **Does the text-like bridge beat CLASP** head-to-head on identical splits
   and metrics — the comparison neither paper makes?

The scaffolding in this repo already settles the machinery those questions need
(`lsr.retrievers` for the protocol, `lsr.eval` for metrics, `lsr.store` for
the cached-embedding boundary and provenance), so this plan covers only the
work specific to CLSR.

## Track 0 — Settle the task definition. Blocking.

Nothing below is meaningful until this is done.

1. Establish from `build_spoken_squad_pkl.py` and the WAV layout on `lab150`
   (`/timevoz/CLASP/data/datasets/`) exactly what the audio is: a reading of
   the paragraph context, or a synthesised question. See the finding in
   [LANDSCAPE.md](LANDSCAPE.md).
2. If it is context readings, the ParaSpeechRAG paper's "spoken queries"
   framing needs correcting. That is a paper-level issue — raise it, don't work
   around it.
3. Define the comparison task once, as a config: **spoken question → text
   passage**, question audio on the query side, the full 2,067-paragraph
   validation pool as candidates. Build it from SQuAD questions; where question
   audio does not exist, synthesise it and record the TTS system and voice as
   part of the task definition. Name it so it can never be confused with the
   self-retrieval variant.

## Track A — Reproduce the authors' release

Runs on `lab150`. Multi-day.

1. **Preflight, read-only.** Record CUDA driver and toolkit, free VRAM, free
   disk, and whether a Python 3.8 interpreter is obtainable; tarball the
   output. Confirm the box can host `torch==2.0.1+cu118` on cpython-38 before
   anything else. `ParaSpeechRAG/scripts/sanity/recon_gpu_box.sh` is the
   pattern to copy.
2. **Clone and audit before running.** Clone to its own directory outside this
   repo, pin the commit SHA, and read `finetune.py`, `evaluate.py`,
   `prepare_data.py`, `collate_fn.py`, `build_model.py` and
   `my_vector_quantizer.py` to establish:
   - which direction each variant actually retrieves (paper and code disagree);
   - what the recall metric counts, and **what the candidate pool is per
     query** — if the pool is the batch rather than the split, their R@k is not
     comparable to ours and everything downstream must be rescored;
   - whether `aqtc` is really missing `build_model_from_file.py`;
   - **how the VQ adaptor bridges vocabularies** (see [LANDSCAPE.md](LANDSCAPE.md)).
     This gates Track B's implementation, so do it early.
3. **Build an environment and record every deviation.** The pin set cannot
   resolve; resolve it deliberately in a container, and commit the lockfile.
   `funasr==0.7.8` will not coexist with anything else here, which is exactly
   what the cached-embedding boundary is for.
4. **Reproduce, in this order:**
   - **SLUE-SQA-5 (`aqac`)** — smallest, manifests shipped. Target 30.65 /
     29.89, WER 16.69. Do this one first as the end-to-end shakedown.
   - **LibriSQA (`aqtc`)** — manifests shipped. Target 85.04 / 85.53, WER 4.09.
     Report on the release's split and note that it is also its validation split.
   - **Spoken-SQuAD** — manifests not shipped, so write the builder in the
     release's `wav.scp` / `que.txt` / `con.txt` format. Build **both** the
     filtered and unfiltered variants, so Table 2's 70.03 and Tables 3–4's
     49.82 each get their own reference.
   - Skip DRCD\* unless the above replicate.
5. **Reproduce the architectural claim.** Train ParaBGE on Spoken-SQuAD and
   LibriSQA. It is the paper's load-bearing evidence and CLASP's direct
   analogue, so it must be our number, not theirs.
6. **Ablate.** At minimum **no-VQ** on Spoken-SQuAD; then no-sampler and the
   pretrain/post-train grid if budget allows.
7. **Seeds.** Three on the headline configs, one on the rest, reporting
   mean ± sd with bootstrap CIs over the query pool.

## Track B — Head-to-head against CLASP

Three retrievers, one harness, one split, one metric, three seeds. Everything
except the bridge is held fixed: same frozen text encoder, same speech
frontend, same contrastive loss, same pool. A comparison that changes two
things at once answers nothing.

1. **Arm 1 — dual encoder (CLASP-style)**: frozen HuBERT → MLP, contrastive.
2. **Arm 2 — text-like bridge (CLSR-style)**: CIF → VQ adaptor → frozen text
   encoder. The new module, and the piece most likely to consume the schedule;
   it needs a waveform dataset with variable-length collation, which does not
   exist in any of the source repos. Resolve the vocabulary question from
   Track A step 2 first — if the bridge requires the ASR vocabulary and the
   text encoder's vocabulary to coincide, the honest options are to adopt a
   matching pair or to state that the idea does not transfer, and say which.
3. **Arm 3 — ASR cascade**: Parakeet TDT transcript → the same text encoder.
   Lift `codigos_matheus/asr/transcribe_from_config.py`, de-hardcode its paths.
   This also closes the ~30 unbacked numbers in the ParaSpeechRAG paper's
   cascade column, so it earns its keep twice.
4. Report WER alongside retrieval for arms 2 and 3 — CLSR's thesis is that
   joint training improves both.
5. **If the bridge wins**, run it through the perturbation axes. A retriever
   routing through a text bottleneck should inherit the cascade's robustness,
   which would be a genuinely new result rather than a reproduction.

## Track C — Long-form SQA

No released code; built from scratch. The 30-second encoder window governs the
design — see the trap in [PAPERS.md](PAPERS.md).

1. Acquire the Spoken Wikipedia Corpus; select documents averaging ~30 minutes.
2. Build the modified SLUE-SQA-5 test set: 500 instances (seed recorded) whose
   context audio is replaced by a full document. The gold answer must stay
   findable; the paper never says how it ensured that, so splice the original
   context audio in and record its offset, documented as our assumption.
3. Implement both arms, with segment length a parameter — run **40 s** (the
   paper) and **30 s** (the encoder window).
4. Score EM / macro-F1 and wall-clock per instance, stating for each arm how
   many seconds of audio actually reached the model. Measure segment
   localization separately from answer correctness: a pipeline that answers
   correctly from the wrong segment is not working.

## Verification

1. **Unit-test CIF** against the paper's worked example: α = {0.8, 0.3, 0.4,
   0.4, 0.1}, β = 1 → expected firing positions and overflow carry.
2. **`gradcheck` the straight-through estimator**: gradients reach `D`, and the
   forward output equals a plain embedding lookup of the argmax.
3. **Overfit one batch** per arm — in-batch contrastive loss to ~0 on 32 pairs.
   Catches dataset and collate wiring without a real run.
4. **Smoke the full path** on ~100 paragraphs with split-disjointness asserted
   and provenance stamped.
5. **Positive control for every retriever** before trusting a low score. CLAP's
   published R@1 ≈ 0.000 was indistinguishable from a broken pipeline because
   no control existed.
6. **On the box**: one seed end-to-end on SLUE-SQA-5 against the paper's row
   before committing to the full grid.

## Budget

A single RTX 4090 on `lab150`. Track A alone is CLSR + ParaBGE across three
datasets and three seeds plus an ablation grid; Track B adds three trained
arms; Track C adds Qwen-Audio inference over 500 long documents. This does not
fit a short schedule.

Order if it has to be cut: Track 0 → Track A's audit (cheap, and may invalidate
later steps) → Track A on SLUE-SQA-5 → Track B arms 1 and 2 → Track C. Drop
seeds before dropping an arm; drop DRCD and the ablation grid before dropping
ParaBGE, which is the whole argument.
