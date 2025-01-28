#!/bin/bash
set -euxo pipefail

uv sync --verbose
brew upgrade jupyterlab

python -m ipykernel install --user --name=.venv
# Installed kernelspec .venv in /Users/hughbrown/Library/Jupyter/kernels/.venv
