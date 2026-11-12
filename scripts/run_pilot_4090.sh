#!/usr/bin/env bash
# Duration pilot on the 4090 (lab150), end to end. Run from the repo root on the box,
# after `git pull`:
#
#   bash scripts/run_pilot_4090.sh              # Spoken-SQuAD, then SPIRAL
#   bash scripts/run_pilot_4090.sh squad        # one of them
#   bash scripts/run_pilot_4090.sh spiral
#
# Steps: environment check -> tests -> parity with CLASP's code -> per dataset:
# data (Spoken-SQuAD from the box's own dev audio; SPIRAL downloaded, 3.5 GB),
# build, pilot with CLASP's released checkpoint -> results/pilot_4090_<stamp>.tar.gz
# to send back. Re-running is safe: downloads, builds and per-chunk features are
# cached, so only new durations are extracted.
set -euo pipefail
# Output goes through tee, i.e. a pipe: unbuffered Python, or nothing shows until exit.
export PYTHONUNBUFFERED=1

CLASP_ROOT="${CLASP_ROOT:-/timevoz/CLASP}"
WAV_DIR="$CLASP_ROOT/data/datasets/spoken_squad/dev_wav"
DATASETS=("$@")
[[ ${#DATASETS[@]} -gt 0 ]] || DATASETS=(squad spiral)
for d in "${DATASETS[@]}"; do
    [[ "$d" == squad || "$d" == spiral ]] || { echo "unknown dataset: $d (squad, spiral)"; exit 1; }
done
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

echo "== tests"
uv run pytest -q

run_pilot() {   # $1 = config; appends its run stamp to STAMPS
    local rc=0
    time uv run python scripts/run_duration_pilot.py --config "$1" || rc=$?
    if [[ $rc -ne 0 ]]; then    # a killed process leaves no traceback (137 = out of memory)
        echo "pilot failed: exit code $rc ($1)"
        exit "$rc"
    fi
    local summary
    summary=$(find runs -maxdepth 1 -name '*_duration_pilot_summary.tsv' | sort | tail -n 1)
    STAMPS+=("$(basename "$summary" _duration_pilot_summary.tsv)")
    echo
    column -t "$summary" | cut -c1-220
    echo
}
STAMPS=()
BUILDS=()
parity() {      # $1 = a WAV of the dataset: 16 kHz Spoken-SQuAD and 24 kHz SPIRAL differ
    echo "== parity with CLASP's own code: $1"
    uv run python scripts/check_clasp_parity.py "$1" --device cuda \
        --clasp-src "$CLASP_ROOT/src" --checkpoint "$CKPT"
}

for d in "${DATASETS[@]}"; do
    if [[ "$d" == squad ]]; then
        CONFIG="configs/duration_pilot_4090.yaml"
        n_wav=$(find "$WAV_DIR" -name '*.wav' | wc -l)
        echo "== Spoken-SQuAD: dev WAVs in $WAV_DIR: $n_wav (full release: 10578)"
        mkdir -p data/datasets/spoken_squad data/datasets/squad
        [[ -f data/datasets/spoken_squad/spoken_test-v1.1.json ]] || curl -fsSL -o \
            data/datasets/spoken_squad/spoken_test-v1.1.json \
            https://raw.githubusercontent.com/Chia-Hsuan-Lee/Spoken-SQuAD/master/spoken_test-v1.1.json
        [[ -f data/datasets/squad/dev-v1.1.json ]] || curl -fsSL -o \
            data/datasets/squad/dev-v1.1.json \
            https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
        uv run python scripts/build_longaudio.py --config "$CONFIG"
        head -n 30 data/builds/longaudio/squad_dev/build.json
        # A fixed sample, not `ls | head`: under pipefail, head closing the pipe kills
        # ls (SIGPIPE) and, with set -e, the whole script, silently.
        parity data/builds/longaudio/squad_dev/audio/paragraphs/0_0.wav
        BUILDS+=(data/builds/longaudio/squad_dev/build.json)
    else
        CONFIG="configs/spiral_pilot_4090.yaml"
        echo "== SPIRAL"
        uv run python scripts/download_spiral.py
        uv run python scripts/build_longaudio.py --config "$CONFIG"
        cat data/builds/longaudio/spiral/build.json
        parity data/datasets/spiral/wavs/lecture_0.wav
        BUILDS+=(data/builds/longaudio/spiral/build.json)
    fi
    echo "== pilot: $d"
    run_pilot "$CONFIG"
done

files=("$LOG" "${BUILDS[@]}")
for s in "${STAMPS[@]}"; do
    files+=(runs/"$s"_duration_*)
done
tar czf "results/pilot_4090_$STAMP.tar.gz" "${files[@]}"
echo
echo "send back: $(pwd)/results/pilot_4090_$STAMP.tar.gz"
echo "  e.g. from the Mac: scp lab150:$(pwd)/results/pilot_4090_$STAMP.tar.gz ~/dev/lsr/results/"
