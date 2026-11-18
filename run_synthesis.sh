#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
maya_dir=${MAYA_DIR:-../maya2-bengali-benchmark}
maya_dir=$(realpath "$maya_dir")
maya_python=${MAYA_PYTHON:-$maya_dir/runtime/reused-venv/bin/python}
maya_engine=${MAYA_ENGINE:-$maya_dir/runtime/engine}
maya_models=${MAYA_MODELS:-$maya_dir/models}
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$maya_engine${PYTHONPATH:+:$PYTHONPATH}"
if (( $# == 0 )); then
  set -- hi en ta ml te gu kn
fi
status=0
mkdir -p artifacts
for language in "$@"; do
  mkdir -p artifacts/logs
  echo "Starting Maya synthesis for $language"
  if ! flock -s artifacts/gpu.lock "$maya_python" -u synthesize.py \
      --models "$maya_models" --prompts "artifacts/prompts/$language" \
      --output "artifacts/generated/$language" --hours "${SYNTHESIS_HOURS:-22.5}" \
      2>&1 | tee -a "artifacts/logs/synthesis-$language.log"; then
    echo "Synthesis for $language stopped with an error. Continuing with the next language."
    status=1
  fi
done
exit "$status"
