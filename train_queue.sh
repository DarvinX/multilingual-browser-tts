#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD:$PWD/src:$PWD/pipeline${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 HF_HUB_OFFLINE=1
speech_cuda_libs="$HOME/miniconda3/envs/torch_env/lib/python3.12/site-packages/nvidia"
speech_library_paths=$(find "$speech_cuda_libs" -maxdepth 2 -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$speech_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
trainer=${TTS_PYTHON:-$HOME/tiny-voca/.venv/bin/python}
scorer=${TTS_MOS_PYTHON:-$HOME/tiny-voca/work/utmos-env/bin/python}
onnx_deps=${TTS_ONNX_DEPS:-$HOME/tiny-voca/runs/bengali-browser/build-deps}
reference=artifacts/bn-model/bengali-2.255-fp32.pt
reference_vocab=artifacts/bn-model/frontend/phone-vocab.json
mapfile -t languages < release-languages.txt
mkdir -p artifacts/logs artifacts/published

gpu() {
  flock -x artifacts/gpu.lock "$@"
}

train_language() {
  local language=$1 marker=$2 release="artifacts/releases/$1" work="artifacts/training/$1"
  local frontend="$work/frontend" data="$work/aligned" stages="$work/stages"
  local mode phase candidate src steps
  if [[ ! -s "$frontend/frontend-source.json" ]]; then
    "$trainer" -u pipeline/g2p_training.py "$language" "$release/metadata.jsonl" "$frontend" || return
  fi
  "$trainer" -u pipeline/prepare_language.py "$release" "$frontend" "$reference" "$data" || return
  if [[ ! -s "$stages/base/step-08000-int8.pt.gz" ]]; then
    gpu "$trainer" -u pipeline/train_compact.py "$data" "$frontend" "$reference" "$reference_vocab" "$stages/base" \
        --steps 8000 || return
  fi
  if [[ ! -s "$stages/perceptual/step-08000-int8.pt.gz" ]]; then
    gpu "$trainer" -u pipeline/train_compact.py "$data" "$frontend" "$reference" "$reference_vocab" "$stages/perceptual" \
        --steps 8000 --init "$stages/base/step-08000-fp32.pt" --feature-weight 15 --gan-ratio .85 \
        --lr 7e-5 --seed 71833 || return
  fi
  if [[ ! -s "$stages/refiner/step-03600-int8.pt.gz" ]]; then
    gpu "$trainer" -u pipeline/train_compact.py "$data" "$frontend" "$reference" "$reference_vocab" "$stages/refiner" \
        --steps 3600 --init "$stages/perceptual/step-08000-fp32.pt" --refiner-only || return
  fi
  local -a candidates=()
  for mode in int8 int6 int4; do
    for phase in base perceptual refiner; do
      steps=08000
      if [[ "$phase" == refiner ]]; then steps=03600; fi
      candidate="$work/candidates/$phase/$mode"
      mkdir -p "$candidate"
      src="$stages/$phase/step-$steps-$mode.pt.gz"
      if [[ "$mode" == int6 && ! -s "$src" ]]; then
        "$trainer" pipeline/repack.py "$stages/$phase/step-$steps-fp32.pt" "${src%.gz}" --bits 6 --protect-heads || return
        gzip -n -c "${src%.gz}" > "$src.tmp"
        mv -T "$src.tmp" "$src"
      fi
      if [[ ! -s "$candidate/checkpoint.pt.gz" ]]; then
        ln "$src" "$candidate/checkpoint.pt.gz" || return
      fi
      if [[ ! -s "$candidate/checkpoint-fp32.pt" ]]; then
        ln "$stages/$phase/step-$steps-fp32.pt" "$candidate/checkpoint-fp32.pt" || return
      fi
      if [[ ! -s "$candidate/browser/size.json" ]]; then
        PYTHONPATH="$PYTHONPATH:$onnx_deps" "$trainer" -u pipeline/export_voice.py "$candidate/checkpoint.pt.gz" \
            "$frontend" "$candidate/browser" || return
      fi
      if ! python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["model_bytes"] >= 10**6)' "$candidate/browser/size.json"; then
        echo "${phase}/${mode} exceeds the 1 MB browser limit"
        continue
      fi
      if [[ ! -s "$candidate/validation/mos.json" ]]; then
        gpu "$scorer" -u pipeline/score_voice.py "$candidate/checkpoint.pt.gz" "$frontend" "$data/validation.jsonl" \
            "$candidate/validation" || return
      fi
    done
    candidate=$(python - "$work/candidates" "$mode" <<'PY'
from pathlib import Path
import json, sys
rows = []
for path in Path(sys.argv[1]).glob(f'*/{sys.argv[2]}/validation/mos.json'):
    score = json.loads(path.read_text())['predicted_mos']
    if score >= 2.25:
        rows.append((score, path.parents[1].as_posix()))
print(max(rows)[1] if rows else '')
PY
) || return
    if [[ -z "$candidate" ]]; then continue; fi
    if [[ ! -s "$candidate/test/mos.json" ]]; then
      gpu "$scorer" -u pipeline/score_voice.py "$candidate/checkpoint.pt.gz" "$frontend" "$data/test.jsonl" "$candidate/test" || return
    fi
    if [[ ! -s "$candidate/test/cer.json" ]]; then
      if [[ "$language" == en ]]; then
        gpu python -u pipeline/score_intelligibility.py "$candidate/test" "$language" || return
      else
        gpu "$trainer" -u pipeline/score_intelligibility.py "$candidate/test" "$language" \
            --indic-models artifacts/indic-asr || return
      fi
    fi
    candidates+=("$candidate")
  done
  if (( ${#candidates[@]} == 0 )); then
    echo "No $language validation candidate met 2.25 predicted MOS and the byte cap. Review before shipping."
    return 2
  fi
  if [[ ! -s "$marker" ]]; then
    echo "Model evaluation completed; waiting for the dataset citation to be published."
    return 3
  fi
  HF_HUB_OFFLINE=0 python -u pipeline/publish_voice.py "$release" "$(cat "$marker")" "$frontend" \
      "artifacts/hf/models/$language" "${candidates[@]}" || return
}

while true; do
  pending=0
  for language in "${languages[@]}"; do
    marker="artifacts/published/$language-dataset.txt"
    published="artifacts/published/$language-model.txt"
    if [[ -s "$published" ]]; then continue; fi
    pending=$((pending + 1))
    if [[ ! -s "$marker" && ! -f "artifacts/releases/$language/.ready" ]]; then continue; fi
    if train_language "$language" "$marker" 2>&1 | tee -a "artifacts/logs/train-$language.log"; then
      tail -n 1 "artifacts/logs/train-$language.log" > "$published.tmp"
      mv -T "$published.tmp" "$published"
    fi
  done
  if (( pending == 0 )); then exit 0; fi
  sleep 120
done
