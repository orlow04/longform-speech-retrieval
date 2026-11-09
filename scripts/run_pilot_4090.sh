#!/usr/bin/env bash
# Duration pilot on the 4090 (lab150), end to end. Run from the repo root on the box:
#
#   bash scripts/run_pilot_4090.sh
#
# Copy the repo over first, from the Mac:
#   rsync -av --exclude .venv --exclude data --exclude runs ~/dev/lsr/ lab150:~/lsr/
#
# Steps: environment check -> JSONs -> 48-article build from the box's own
# Spoken-SQuAD dev audio -> parity with CLASP's code -> pilot with CLASP's
# released checkpoint -> results/pilot_4090_<stamp>.tar.gz to send back. Re-running is safe:
# the build and the per-chunk features are cached.
set -euo pipefail
# Output goes through tee, i.e. a pipe: unbuffered Python, or nothing shows until exit.
export PYTHONUNBUFFERED=1

CLASP_ROOT="${CLASP_ROOT:-/timevoz/CLASP}"
WAV_DIR="$CLASP_ROOT/data/datasets/spoken_squad/dev_wav"
CONFIG="configs/duration_pilot_4090.yaml"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs results
LOG="logs/pilot_4090_$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

echo "== environment"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
if ! command -v uv >/dev/null; then
    echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
uv sync --extra clasp --extra cascade --extra dev
CKPT="$CLASP_ROOT/models/checkpoints/CLASP_Concat_Final_Fusion_Encoder.pt"
[[ -f "$CKPT" ]] || { echo "missing checkpoint: $CKPT"; exit 1; }
sha256sum "$CKPT"   # official release: 9ad6954a935cc2bbe913055f0a0aaa398300f422c081b74a180b0116f4d9bf47
n_wav=$(find "$WAV_DIR" -name '*.wav' | wc -l)
echo "dev WAVs in $WAV_DIR: $n_wav (full release: 10578)"

echo "== JSONs"
mkdir -p data/datasets/spoken_squad data/datasets/squad
[[ -f data/datasets/spoken_squad/spoken_test-v1.1.json ]] || curl -fsSL -o \
    data/datasets/spoken_squad/spoken_test-v1.1.json \
    https://raw.githubusercontent.com/Chia-Hsuan-Lee/Spoken-SQuAD/master/spoken_test-v1.1.json
[[ -f data/datasets/squad/dev-v1.1.json ]] || curl -fsSL -o data/datasets/squad/dev-v1.1.json \
    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

echo "== build (48 dev articles)"
uv run python scripts/build_longaudio.py --config "$CONFIG"
head -n 30 data/builds/longaudio/squad_dev/build.json

echo "== tests"
uv run pytest -q

echo "== parity with CLASP's own code"
# No `ls | head` under pipefail: head closing the pipe kills ls (SIGPIPE) and,
# with set -e, the whole script, silently.
sample=data/builds/longaudio/squad_dev/audio/paragraphs/0_0.wav
uv run python scripts/check_clasp_parity.py "$sample" --device cuda \
    --clasp-src "$CLASP_ROOT/src" --checkpoint "$CKPT"

echo "== pilot"
time uv run python scripts/run_duration_pilot.py --config "$CONFIG"

summary=$(find runs -maxdepth 1 -name '*_duration_pilot_summary.tsv' | sort | tail -n 1)
run_stamp=$(basename "$summary" _duration_pilot_summary.tsv)
tar czf "results/pilot_4090_$STAMP.tar.gz" "$LOG" runs/"$run_stamp"_duration_* \
    data/builds/longaudio/squad_dev/build.json
echo
column -t "$summary" | cut -c1-200
echo
echo "send back: $(pwd)/results/pilot_4090_$STAMP.tar.gz"
echo "  e.g. from the Mac: scp lab150:$(pwd)/results/pilot_4090_$STAMP.tar.gz ~/dev/lsr/runs/"
