# longform-speech-retrieval (`lsr`)

A research environment for speech retrieval, heading toward retrieval in
long-form audio.

Spoken queries, spoken documents, or both. Retrievers are pluggable, acoustic
perturbations apply to any task, and every number carries its provenance. This
is not a reproduction of a single paper — papers enter as backends and tasks.

## Documentation

| | |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | the decisions behind the layout, and what each one defends against |
| [docs/LANDSCAPE.md](docs/LANDSCAPE.md) | the four codebases this grew out of: what to reuse, what is broken |
| [docs/PAPERS.md](docs/PAPERS.md) | CLSR and ParaSpeechRAG, with the exact numbers to reproduce |
| [docs/PLAN_CLSR.md](docs/PLAN_CLSR.md) | the plan to reproduce CLSR and test its claim against CLASP |

Read LANDSCAPE.md before touching anything in a neighbouring repository: three
of them are frozen, and it records why.

## Status

Scaffolding. The interfaces below are stubs with their contracts written down
and their bodies raising `NotImplementedError`. Nothing is implemented yet.

| Module | What it owns | State |
|---|---|---|
| `lsr.tasks` | a Task = queries, documents, qrels, modalities, pool | stub |
| `lsr.retrievers` | one protocol, N backends | stub |
| `lsr.eval` | Hits@k, MRR, MAP, bootstrap CIs | stub |
| `lsr.store` | run manifests, cached-embedding format | stub |
| `lsr.perturb` | noise, reverb, codec, rate, speaker, emotion | stub |
| `lsr.longaudio` | segmentation, segment selection, downstream QA | stub |

**Where to start:** `lsr.eval.metrics` and `lsr.tasks.check_task`. Everything
reports through them, and both are testable against a hand-built five-query
task with no model, no audio and no GPU — so the metric is trustworthy before
there is a backend to blame.

## Install

```sh
uv venv
uv pip install -e ".[dev]"
```

Backends are separate extras and none requires another:

```sh
uv pip install -e ".[clasp]"      # fusion MLP over frozen HuBERT + spectrogram
uv pip install -e ".[speechrag]"  # distilled adapter into a text embedding space
uv pip install -e ".[cascade]"    # ASR -> text retrieval
uv pip install -e ".[longaudio]"  # segmentation + a downstream audio LM
```

CLSR is deliberately absent from the extras: `funasr==0.7.8` on Python 3.8 with
cu118 wheels cannot coexist with anything else here. It runs in its own
container and reaches the metric path through the cached-embedding boundary.

## Run things

No task runner, no Makefile. Entry points are plain scripts in `scripts/`,
parameterised by a config file rather than by long flag lists, so that a command
that worked yesterday can be re-run from a log.

```sh
uv run pytest                        # tests
uv run ruff check src tests          # lint
uv run ruff check --fix src tests    # fix what's fixable
uv run python scripts/<name>.py --config configs/<name>.yaml
```

## Running on a remote box

Experiments run over SSH, which constrains a few things:

- **`uv` must exist on the box**, and it must be able to fetch a 3.10–3.12
  interpreter: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then
  `uv venv && uv pip install -e ".[dev]"`. `uv run` will provision the
  interpreter itself if the system Python is too old.
- **Nothing long-running in a bare SSH session.** A dropped connection kills the
  job and takes the only copy of its output with it. Start under `tmux`, or
  `nohup … &` with stdout redirected into the run directory.
- **Every run writes to `runs/<run_id>/`** — config, provenance, logs, metrics
  in one place, so a finished run can be rsynced back whole and a dead one
  diagnosed without re-running it.
- **No absolute paths in committed code or config.** Paths come from a config
  file or an env var and default to repo-relative. The inherited ASR scripts
  hardcoded `/home/dollis/...` and would not run anywhere else.
- **Headless.** No plotting backend that wants a display; write figures to
  files.
- **The cached-embedding boundary fits this well**: run the heavy backend on the
  box, rsync `runs/<run_id>/` back, score and plot locally where iteration is
  fast and the core installs in seconds.
