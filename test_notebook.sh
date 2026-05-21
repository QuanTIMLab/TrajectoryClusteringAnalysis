#!/bin/bash
# Test script to execute the exemple_pypi.ipynb notebook in a fresh virtual environment

REPO_PATH="/c/Users/nicolas.grevet/dev/TrajectoryClusteringAnalysis.worktrees/copilot-worktree-2026-04-13T12-30-11"
NOTEBOOK_PATH="$REPO_PATH/Notebooks/exemple_pypi.ipynb"
VENV_PATH="/tmp/tca_venv_test"

echo "Repository: $REPO_PATH"
echo "Notebook: $NOTEBOOK_PATH"
echo "Virtual environment: $VENV_PATH"

# Create virtual environment
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_PATH"
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create venv"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

echo "Virtual environment created and activated"
echo "Python: $(which python)"
echo "Version: $(python --version)"

# Install jupyter and nbconvert
echo ""
echo "============================================================"
echo "Installing jupyter and nbconvert..."
echo "============================================================"
pip install -q jupyter nbconvert
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install jupyter/nbconvert"
    exit 1
fi
echo "✓ Jupyter and nbconvert installed"

# Install requirements
echo ""
echo "============================================================"
echo "Installing dependencies from requirements.txt..."
echo "============================================================"
if [ -f "$REPO_PATH/requirements.txt" ]; then
    pip install -q $(grep -v '^#' "$REPO_PATH/requirements.txt" | grep -v '^$')
    echo "✓ Dependencies installed (some may have failed silently)"
fi

# Execute the notebook
echo ""
echo "============================================================"
echo "Executing notebook with nbconvert..."
echo "============================================================"
echo "Notebook: $NOTEBOOK_PATH"
echo ""

jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=600 "$NOTEBOOK_PATH"
RESULT=$?

echo ""
echo "============================================================"
if [ $RESULT -eq 0 ]; then
    echo "✓ NOTEBOOK EXECUTION SUCCEEDED"
else
    echo "✗ NOTEBOOK EXECUTION FAILED (exit code: $RESULT)"
fi
echo "============================================================"

exit $RESULT
