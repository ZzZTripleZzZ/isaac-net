#!/usr/bin/env bash
# Run the GPU test suite on a CUDA machine (CI runs only the CPU tests).
#
#   scripts/run_gpu_tests.sh            # gpu tests, including slow ones
#   scripts/run_gpu_tests.sh --all      # the whole suite (CPU + GPU)
#   scripts/run_gpu_tests.sh -k triton  # extra args go to pytest
#
# Uses whatever `python` is active (e.g. `conda activate i5g`). Needs pytest; if it is missing,
# `pip install pytest` or point PYTHONPATH at a directory that has it.
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("CUDA is not available: GPU tests would all be skipped")
print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name()}")
PY

if [[ "${1:-}" == "--all" ]]; then
    shift
    exec python -m pytest -v "$@"
fi
exec python -m pytest -v -m gpu "$@"
