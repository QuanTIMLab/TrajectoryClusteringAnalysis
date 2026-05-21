#!/usr/bin/env python
"""
Test script to execute the exemple_pypi.ipynb notebook in a fresh virtual environment
"""
import subprocess
import sys
import os
import tempfile
import json
import shutil
from pathlib import Path

def main():
    repo_path = Path(r"C:\Users\nicolas.grevet\dev\TrajectoryClusteringAnalysis.worktrees\copilot-worktree-2026-04-13T12-30-11")
    notebook_path = repo_path / "Notebooks" / "exemple_pypi.ipynb"
    
    # Create temporary virtual environment
    temp_dir = Path(tempfile.gettempdir())
    venv_path = temp_dir / "tca_venv_test"
    
    # Clean up if exists
    if venv_path.exists():
        print(f"Removing existing venv at {venv_path}")
        shutil.rmtree(venv_path)
    
    print(f"Creating virtual environment at {venv_path}...")
    result = subprocess.run([sys.executable, "-m", "venv", str(venv_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to create venv")
        print(result.stderr)
        return 1
    
    # Get Python executable from venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    print(f"Python executable: {python_exe}")
    print(f"Pip executable: {pip_exe}")
    
    # Install jupyter and notebook
    print("\n" + "="*60)
    print("Installing jupyter and nbconvert...")
    print("="*60)
    result = subprocess.run([str(pip_exe), "install", "jupyter", "nbconvert"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to install jupyter/nbconvert")
        print(result.stdout)
        print(result.stderr)
        return 1
    print("✓ Jupyter and nbconvert installed")
    
    # Install requirements from requirements.txt
    print("\n" + "="*60)
    print("Installing dependencies from requirements.txt...")
    print("="*60)
    requirements_file = repo_path / "requirements.txt"
    if requirements_file.exists():
        # Read and filter requirements
        with open(requirements_file, 'r') as f:
            reqs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Installing {len(reqs)} packages...")
        for req in reqs:
            print(f"  Installing: {req}")
            result = subprocess.run([str(pip_exe), "install", req], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    WARNING: Failed to install {req}")
                print(f"    Error: {result.stderr[:200]}")
            else:
                print(f"    ✓ Installed")
    
    # Try to execute the notebook
    print("\n" + "="*60)
    print("Executing notebook with nbconvert...")
    print("="*60)
    print(f"Notebook: {notebook_path}")
    
    # Use nbconvert to execute
    output_html = repo_path / "exemple_pypi_output.html"
    cmd = [
        str(python_exe), "-m", "nbconvert",
        "--to", "html",
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        "--output", str(output_html),
        str(notebook_path)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\nSTDOUT:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nSTDERR:")
        print(result.stderr)
        print("\n" + "="*60)
        print("NOTEBOOK EXECUTION FAILED")
        print("="*60)
        
        # Try to parse the error from the output
        if "Error" in result.stderr or "Exception" in result.stderr:
            print("\nError details:")
            print(result.stderr)
        
        return 1
    else:
        print("\n" + "="*60)
        print("✓ NOTEBOOK EXECUTION SUCCEEDED")
        print("="*60)
        if output_html.exists():
            print(f"Output saved to: {output_html}")
        return 0

if __name__ == "__main__":
    sys.exit(main())
