# Landscape

The four codebases this work grew out of: what each one is, what is worth
taking, and what is broken in it.

All file paths below point into **frozen repositories** and were accurate on
2026-10-19. Verify against the tree before relying on any of them — these notes
will outlive the code they describe.

## The axis that makes these comparable

Every system here differs in essentially one thing: how frame-level speech
features become something that lives in a text embedding space.

| System | Bridge | Granularity | Objective |
|---|---|---|---|
| CLASP | mean-pool → MLP (+ spectrogram-image branch) | one vector per utterance | contrastive, in-batch InfoNCE |
| speech-rag | mean-pool → linear | one vector per utterance | distillation to a text embedding |
| CLSR | CIF → VQ → frozen text encoder | one embedding per token | contrastive + ASR losses |

Holding the task, the pool and the metric fixed while swapping that column is
the comparison none of the source papers makes, and the reason `lsr` exists.

---

## `~/dev/paper-audio/clasp/CLASP` — read-only source of truth

The original CLASP implementation, plus an MSEB/SVQ adapter and an SVQ RAG
pipeline added later.

**Architecture.** The only trained parameters are a fusion MLP over *frozen,
precomputed* features: `models/fusion.py::HubertLabseConcat` takes HuBERT-large
(1024-d) and EfficientNet-B7 classifier logits over a rendered spectrogram
(1000-d), projects both to 768, concatenates and mixes to 768, and is trained
contrastively against frozen LaBSE sentence embeddings.

**Why `lsr` does not build on it: there is no gradient path from a waveform.**
Everything upstream of the MLP is run offline and baked into PKL files. Any
trainable speech encoder — which CLSR needs, and which long-audio work needs —
cannot reuse that stack.

**The finding that matters most.** `scripts/build_spoken_squad_pkl.py` pairs
`article["paragraphs"][p]["context"]` with a WAV reading *of that same
context*, and `evaluate_model_on_paragraph_groups`
(`src/clasp/evaluation/metrics.py:259-263`) uses the LaBSE context embedding as
the **query** and the fused audio chunks as candidates — its own docstring says
so. That is passage → reading-of-itself self-retrieval, not question → passage
retrieval. Matching a passage to a recording of itself is a far easier task, so
CLASP's 0.954 and CLSR's 0.700 are not comparable numbers. This is why a task's
name in `lsr` states its direction.

**Worth imitating, not importing.** `evaluation/ranking_metrics.py::
compute_ranking_metrics`: 1-based ranks, ties broken optimistically (count only
scores *strictly* greater than gold), Hits@k / MRR / MAP / mean / median rank.
Hits@k equals Recall@k there only because every query has exactly one relevant
document — a property of the task, not of the metric.

**Defects not to re-inherit:**

- No seeding anywhere in `scripts/train.py` or `src/clasp/train/`, and
  `build_test_metadata` samples candidates with an unseeded `random.sample`, so
  evaluations are not reproducible run to run.
- `trainer.train_the_model`'s default `temperature=np.log(0.07)` *shrinks*
  logits by ~200×; it is harmless only because `train.py` always passes
  `log(1/0.07)`. A latent footgun for anyone calling the trainer directly.
- `BatchNorm1d` in the fusion MLP with `drop_last=False` crashes on a trailing
  batch of size 1.
- `retrieval/search.py::build_similarity_matrix` is a Python double loop with a
  `.item()` per cell — one GPU sync per pair, unusable past a few thousand rows.
- Checkpoints are pickled whole `nn.Module`s, not state dicts, so loading needs
  the defining classes importable under their original names.
- The ColBERT max-sim scorer exists twice, byte-identically, in
  `evaluation/metrics.py` and `evaluation/spiral_max_sim.py`; the former does
  not import the latter.

**Untracked `codigos_matheus/`** (never in git history, never ran in-repo):
Parakeet TDT transcription (`nvidia/parakeet-tdt-0.6b-v2`) with
`compute_wer.py`, plus codec, speaking-rate and GenVC perturbation scripts.
Hardcodes absolute `/home/dollis/...` paths — the reason `lsr` forbids absolute
paths in committed code.

---

## `~/dev/paper-audio/ParaSpeechRAG` — frozen conference artifact

The port of CLASP plus `codigos_matheus/` that backs the ParaSpeechRAG
submission. **Do not edit it.** `docs/GAPS.md` audits a specific draft, and a
tree that keeps moving invalidates that audit.

`docs/GAPS.md` (1,144 lines) is the authoritative inventory. Headlines:

- **Reproduced** on the GPU box: CLASP-mean `Hits@1 0.9540396710208031`,
  `MRR 0.9718183247238209`; all seven emotion conditions to three decimals on
  verified-disjoint splits.
- **Not reproduced**: CLASP-chunked measures `0.4799225931301403 /
  0.6174332843147826` against the paper's `0.547 / 0.671`. Chunk granularity,
  checkpoint, scorer and pool were each excluded as the cause. Two PKLs built
  four months apart give bit-identical metrics, so the scorer is at least
  deterministic.
- **Unbacked by any code**: CLAP, GLAP, the codec axis, the rate axis, and the
  entire ASR + LaBSE retrieval column (~30 published numbers).
- The reverb axis is inverted and near-inert, and the paper's §5.3 interprets
  the artifact as a finding.

**Useful prior art:** `configs/perturb/*.yaml` pin every perturbation
parameter, and `scripts/sanity/recon_gpu_box.sh` is a good pattern for a
read-only inventory of a remote machine that tarballs its output.

---

## `marcospaulo429/speech-rag` — a collaborator's repo

Implements *Speech RAG without Automatic Speech Recognition* (arXiv
2412.16500): frozen HuBERT-large → trainable adapter → a text embedding space
(E5-Mistral-7B at 4096-d, or Qwen3-Embedding at 1024/2560-d), trained by
distillation (MSE / cosine against the text embedding), FAISS retrieval, plus a
generator and a RAG pipeline.

The only one of the four with real tests — roughly 3,200 lines across 11 files.

**Defect worth knowing:** `SpeechAdapter.forward` applies `AvgPool1d(4)` and
then immediately `x.mean(dim=1)` over the whole time axis. The mean of
equal-size means is the global mean, so the downsampling is a no-op; the
adapter is really `mean-pool → Linear → LayerNorm → L2`. That places it at the
weakest point on the axis above, which makes it a clean baseline to beat and a
precise motivation for a token-aligned bridge.

---

## `github.com/193746/CLSR` — upstream, third-party

CC-BY-NC-4.0, 4 commits. The release is thinner than the paper, and several
problems are confirmed rather than suspected.

- **Two variants, neither covering the paper's four datasets.** `CLSR_aqtc`
  (audio question → text context) ships **LibriSQA manifests only**;
  `CLSR_aqac` (audio question → audio context) ships **SLUE-SQA-5 only**. No
  Spoken-SQuAD, no DRCD, no ODSQA, despite the README naming two of them.
- **No trained checkpoints** — only download instructions for
  `BAAI/bge-base-en-v1.5` and
  `iic/speech_paraformer_asr-en-16k-vocab4199-pytorch`. Every reproduced number
  costs a training run.
- **`requirements.txt` cannot resolve as pinned**: `numpy==1.23.5` alongside
  `scipy==1.17.1` (which needs ≥ 1.25), `PyYAML` pinned twice to different
  versions, against an otherwise 2023-era stack (`funasr==0.7.8`,
  `torch==2.0.1+cu118`, `transformers==4.33.3`). The `.pyc` files are
  cpython-38, so the real environment is Python 3.8 + CUDA 11.8.
- **`CLSR_aqtc` is missing `build_model_from_file.py`**, which `CLSR_aqac`
  ships — likely an import failure on the aqtc path.
- **`CLSR_aqtc`'s `validation/` and `test/` manifests are the same git blobs**
  (`con.txt b2614ef2…`, `que.txt 48eb7005…`, `wav.scp 58ac1419…`). There is no
  held-out validation split for LibriSQA.
- **It vendors a modified FunASR**: `MyParaformer.py`, `MySanmDecoder.py`
  (77 KB), `attention.py`, `build_trainer.py` (38 KB). The novel part is small
  and liftable: `src/my_vector_quantizer.py`, 6 KB. CIF comes from Paraformer
  itself.
- **Audited at `1a03529` (2026-11-03).** What the code actually does, which
  settles the questions the paper leaves open:
  - **Direction.** `CLSR_aqtc` is *text question -> spoken context*, despite
    its name: `wav.scp` holds the LibriSpeech reading of the context, `con.txt`
    is both its ASR target and the retrieved side, and `que.txt` is encoded as
    text by frozen BGE (`MyParaformer.py:543-561`). `CLSR_aqac` is genuinely
    audio question -> audio context (`data_names: sque,scon,que,con`).
  - **Pool.** `Trainer.evaluate` (`build_trainer.py`) concatenates every test
    batch and scores the full split, ground truth on the diagonal. So the pool
    is the split, not the batch — but it holds one row per *question*, and a
    context asked about several times appears several times. SLUE-SQA-5 test
    has 2,382 rows over 1,969 distinct contexts; the twins tie with the gold
    row and `argsort` breaks the tie arbitrarily. `lsr.tasks.ManifestTask`
    offers both this `rows` pool and a deduplicated `unique` pool.
  - **Vocabulary.** The bridge does not map Paraformer's 4,199-token
    vocabulary onto BGE's: it replaces it. Contexts are tokenised with BGE's
    tokenizer, the decoder loss runs over `bge_vocab_size = 30522`, and the VQ
    output multiplies `text_encoder.embeddings.word_embeddings.weight`
    (`MyParaformer.py:99,554,575`). So the idea transfers to another text
    encoder by retraining the decoder's output layer over that encoder's
    vocabulary. The exact place the shipped 4,200-entry `token_list` gets
    overridden has not been traced.
  - Two LibriSQA test utterances share a transcript, so the text-side pool has
    2,619 contexts and the audio-side pool 2,620.

Because of the dependency island, CLSR is the motivating case for the
cached-embedding boundary in [DESIGN.md](DESIGN.md): it runs in its own
container and hands over embeddings, never sharing an environment.
