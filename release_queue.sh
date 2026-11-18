#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
speech_cuda_libs="$HOME/miniconda3/envs/torch_env/lib/python3.12/site-packages/nvidia"
speech_library_paths=$(find "$speech_cuda_libs" -maxdepth 2 -type d -name lib | paste -sd:)
export LD_LIBRARY_PATH="$speech_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
indic_python=${TTS_PYTHON:-$HOME/tiny-voca/.venv/bin/python}
languages=(hi en ta ml te gu kn)
mkdir -p artifacts/logs artifacts/audit artifacts/published

info() {
  python -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

while true; do
  outstanding=0
  for language in "${languages[@]}"; do
    marker="artifacts/published/$language-dataset.txt"
    if [[ -s "$marker" ]]; then
      continue
    fi
    outstanding=$((outstanding + 1))
    generated="artifacts/generated/$language"
    if [[ ! -s "$generated/progress.json" ]]; then
      continue
    fi
    release="artifacts/releases/$language"
    audit="artifacts/audit/$language.jsonl"
    state=$(info "$generated/progress.json" status)
    audit_python=python
    audit_options=()
    if [[ "$language" != en ]]; then
      if ! python -u get_indic_asr.py artifacts/indic-asr "$language"; then continue; fi
      audit_python="$indic_python"
      audit="artifacts/audit/$language-ctc.jsonl"
      audit_options=(--indic-models artifacts/indic-asr)
    fi
    if [[ "$state" == complete ]]; then
      if ! flock -s artifacts/gpu.lock "$audit_python" -u audit.py "$generated" "$audit" --language "$language" --device cuda \
          --release "$release" "${audit_options[@]}" 2>&1 | tee -a "artifacts/logs/audit-$language.log"; then
        echo "Audit for $language needs review. See artifacts/logs/audit-$language.log"
        continue
      fi
      name=$(info "$release/source.json" language_name)
      voice=$(info "$release/source.json" speaker)
      repo="Hansda/${name,,}-maya2-${voice,,}-20h"
      if python -u publish_dataset.py "$release" "artifacts/hf/datasets/$language" \
          --language "$language" --repo "$repo" 2>&1 | tee -a "artifacts/logs/upload-$language.log"; then
        tail -n 1 "artifacts/logs/upload-$language.log" > "$marker.tmp"
        mv -T "$marker.tmp" "$marker"
      fi
    else
      flock -s artifacts/gpu.lock "$audit_python" -u audit.py "$generated" "$audit" --language "$language" --device cuda "${audit_options[@]}" \
          2>&1 | tee -a "artifacts/logs/audit-$language.log" || true
    fi
  done
  if (( outstanding == 0 )); then
    exit 0
  fi
  sleep 45
done
