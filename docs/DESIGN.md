# Design

Decisions this repository is built on. Each one exists because of something
that went wrong, or was hard to undo, in an earlier codebase.

## Scope

A research environment for **speech retrieval**, heading toward **retrieval in
long-form audio**. Not a reproduction of any single paper. Papers enter here as
*backends* and *tasks*, never as repository layout.

Three repositories stay frozen and are never edited from here:

| Repo | Role |
|---|---|
| `clasp/CLASP` | read-only source of truth for the original CLASP implementation |
| `ParaSpeechRAG` | the conference artifact; its `docs/GAPS.md` audits a specific draft, so the tree must stop moving |
| `CLSR` (upstream) | third-party, CC-BY-NC-4.0, Python 3.8 dependency island |

Reading from them is fine. Depending on them is not — anything needed here gets
written here.

## The research axis

The systems worth comparing differ in exactly one thing: how frame-level speech
features become something that lives in a text embedding space.

| Bridge | Granularity | Objective |
|---|---|---|
| pooled vector → MLP | one vector per utterance | contrastive (in-batch InfoNCE) |
| pooled vector → linear | one vector per utterance | distillation to a text embedding |
| CIF → VQ → frozen text encoder | one embedding per token | contrastive + ASR losses |

The point of the `retrievers/` protocol is to make that column swappable while
the task, the pool and the metric stay fixed. A comparison that changes two
things at once answers nothing.

## The cached-embedding boundary

The constraint that shapes everything: the interesting backends have **mutually
incompatible dependency stacks**. CLASP wants Python 3.10 + modern transformers;
MSEB wants ≥ 3.12 plus `scann`; CLSR wants 3.8 + `funasr==0.7.8` + cu118 wheels;
NeMo ASR wants its own pins. No single environment resolves all of them, and
trying is how weeks disappear.

So a backend never has to run in the same process as the scorer. It writes

```
runs/<run_id>/
  embeddings.npz      # queries [Nq, D], passages [Np, D], both L2-normalised
  ids.json            # query_ids [Nq], passage_ids [Np] — order matches the arrays
  provenance.json     # see store/
```

and the core — which imports nothing heavier than numpy — scores it. One metric
path over every backend, with no shared environment. In-process backends are a
convenience, not the contract.

## Tasks name their direction explicitly

In the CLASP codebase, `build_spoken_squad_pkl.py` pairs a paragraph's text with
a WAV reading *of that same paragraph*, and the evaluator uses the text as the
query. That is passage → reading-of-itself self-retrieval, not question →
passage retrieval, and nothing in the name said so. It made a published 0.954
look comparable to a published 0.700 measured on a genuinely harder task.

Therefore: a task's identity includes its query side, its document side and its
pool. `spoken_squad_qa` and `spoken_squad_self` are two different tasks that may
never be compared, and the distinction is in the name, not in a docstring.

## Provenance is not optional

Every run records git SHA, config hash, seed, checkpoint sha256, pool size and
query count before it records a metric. A number without those is not a result.
Inherited failure modes being designed out:

- **Seeds.** No global seed anywhere in CLASP's training or candidate sampling,
  so evaluations were not reproducible run to run. Seed everything; record it.
- **Single-run point estimates.** Bootstrap confidence intervals over the query
  pool are cheap and need no retraining.
- **Split integrity.** Assert train/validation disjointness as a test, not as a
  thing someone remembers to check.
- **Positive controls.** A near-zero score is a bug until a control says
  otherwise. CLAP's published R@1 ≈ 0.000 was indistinguishable from a broken
  pipeline because no control existed.

## Execution is remote; the code must not assume otherwise

Experiments run on an SSH machine, not on the laptop. That is a design input,
not an operational detail:

- **No task runner.** Entry points are plain `scripts/*.py` taking `--config`.
  A Make target that wraps a one-line command is indirection that has to be
  read before it can be trusted on an unfamiliar box. Make would earn its place
  for targets with real file dependencies; none exist yet.
- **Configs over flags.** A run has to be reconstructible from what was logged.
  A fifteen-flag command line in shell history is not that.
- **A run is a directory, not a stdout stream.** `runs/<run_id>/` holds config,
  provenance, logs and metrics, so a run can be rsynced back whole, or
  diagnosed after it died, without re-running it.
- **Repo-relative paths only**, resolved from config or environment. Absolute
  paths are how the inherited ASR scripts became unrunnable anywhere but the
  machine they were written on.
- **The split that follows from this**: heavy backends run on the box, the core
  scores locally. Which is the cached-embedding boundary again, arrived at from
  a second direction.

## Layout

```
src/lsr/
  tasks/       a Task is (queries, documents, qrels, modalities, pool)
  retrievers/  one protocol, N backends
  perturb/     acoustic transforms, applicable to any task x retriever
  longaudio/   segmentation -> segment selection -> downstream LALM QA
  eval/        the metric core and bootstrap CIs
  store/       run manifests and the cached-embedding format
```
