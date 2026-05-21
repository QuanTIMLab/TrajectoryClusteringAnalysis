#!/usr/bin/env python3
"""Direct execution test - simpler version"""
import subprocess
import sys

# Try to execute the test script directly
test_script = r"C:\Users\nicolas.grevet\dev\TrajectoryClusteringAnalysis.worktrees\copilot-worktree-2026-04-13T12-30-11\run_notebook_test.py"

print(f"Attempting to run: {test_script}")
print(f"Python: {sys.executable}")

# Run with subprocess
result = subprocess.run(
    [sys.executable, test_script],
    capture_output=False,  # Let output go to console
    text=True
)

sys.exit(result.returncode)
