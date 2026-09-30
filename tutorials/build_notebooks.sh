#!/usr/bin/env bash
# Convert the tutorial scripts to notebooks and execute them on a CPU; the executed notebooks go to
# docs/tutorials/, where the docs site renders them (mkdocs-jupyter, execute: false).
#
#   pip install -e ".[docs]" jupytext nbconvert ipykernel    # plus a CPU or CUDA build of torch
#   bash tutorials/build_notebooks.sh
#
# 01-03 run on a CPU in a few minutes. 04 runs its Isaac-free cells (the DirectRLEnv listing is not code).
# 05 needs an ns-3 + 5G-LENA build; without NS3BRIDGE_ROOT it is converted but not executed.
set -euo pipefail
cd "$(dirname "$0")/.."
out=docs/tutorials
mkdir -p "$out"
export CUDA_VISIBLE_DEVICES=""          # the tutorials use device "cpu"; keep them off a shared GPU
for py in tutorials/0*.py; do
    name=$(basename "$py" .py)
    jupytext --quiet --to ipynb --output "$out/$name.ipynb" "$py"
    if [[ "$name" == 05_* && -z "${NS3BRIDGE_ROOT:-}" ]]; then
        echo "$name: converted, not executed (no ns-3 build)"
        continue
    fi
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "$out/$name.ipynb"
    echo "$name: executed"
done
