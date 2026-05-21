#!/usr/bin/env python3
"""
Comprehensive test script for executing example_pypi.ipynb notebook
This script:
1. Creates a fresh virtual environment
2. Installs all dependencies
3. Executes the notebook non-interactively
4. Reports results and any errors
"""

import json
import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional

def run_command(cmd, description, env=None, timeout=300):
    """Run a command and return result"""
    print(f"\n{'='*70}")
    print(f"Task: {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return result
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: Command took longer than {timeout} seconds"
    except Exception as e:
        return str(e)

def main():
    # Paths
    REPO_PATH = Path(r"C:\Users\nicolas.grevet\dev\TrajectoryClusteringAnalysis.worktrees\copilot-worktree-2026-04-13T12-30-11")
    NOTEBOOK_PATH = REPO_PATH / "Notebooks" / "exemple_pypi.ipynb"
    TEMP_DIR = Path(tempfile.gettempdir())
    VENV_PATH = TEMP_DIR / "tca_notebook_test_venv"
    
    print("\n" + "="*70)
    print("NOTEBOOK EXECUTION TEST")
    print("="*70)
    print(f"Repository: {REPO_PATH}")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Virtual Environment: {VENV_PATH}")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    
    # Step 0: Validate notebook exists
    if not NOTEBOOK_PATH.exists():
        print(f"\nERROR: Notebook not found at {NOTEBOOK_PATH}")
        return 1
    
    print(f"\n✓ Notebook found: {NOTEBOOK_PATH}")
    
    # Step 1: Clean and create virtual environment
    print(f"\nStep 1: Creating virtual environment...")
    if VENV_PATH.exists():
        print(f"  Removing existing venv...")
        shutil.rmtree(VENV_PATH, ignore_errors=True)
    
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_PATH)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR: Failed to create venv")
        print(result.stderr)
        return 1
    
    print(f"✓ Virtual environment created at {VENV_PATH}")
    
    # Get python and pip paths
    if sys.platform == "win32":
        python_exe = VENV_PATH / "Scripts" / "python.exe"
        pip_exe = VENV_PATH / "Scripts" / "pip.exe"
    else:
        python_exe = VENV_PATH / "bin" / "python"
        pip_exe = VENV_PATH / "bin" / "pip"
    
    print(f"  Python: {python_exe}")
    print(f"  Pip: {pip_exe}")
    
    # Verify venv
    result = subprocess.run(
        [str(python_exe), "--version"],
        capture_output=True,
        text=True
    )
    print(f"  {result.stdout.strip()}")
    
    # Step 2: Install jupyter and nbconvert
    print(f"\nStep 2: Installing jupyter and nbconvert...")
    result = subprocess.run(
        [str(pip_exe), "install", "-q", "jupyter", "nbconvert"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode != 0:
        print(f"  ERROR installing jupyter/nbconvert")
        print(f"  STDOUT: {result.stdout[:500]}")
        print(f"  STDERR: {result.stderr[:500]}")
        return 1
    
    print(f"✓ Jupyter and nbconvert installed")
    
    # Step 3: Install package dependencies from requirements.txt
    print(f"\nStep 3: Installing dependencies from requirements.txt...")
    requirements_file = REPO_PATH / "requirements.txt"
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"  Found {len(lines)} packages to install")
        
        failed_packages = []
        for i, req in enumerate(lines, 1):
            print(f"  [{i}/{len(lines)}] Installing {req[:50]}...", end='', flush=True)
            result = subprocess.run(
                [str(pip_exe), "install", "-q", req],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f" FAILED")
                failed_packages.append((req, result.stderr[:200]))
                # Don't exit, continue with other packages
            else:
                print(f" ✓")
        
        if failed_packages:
            print(f"\n  ⚠ {len(failed_packages)} packages failed to install:")
            for pkg, err in failed_packages[:5]:  # Show first 5
                print(f"    - {pkg}: {err[:100]}")
            if len(failed_packages) > 5:
                print(f"    ... and {len(failed_packages)-5} more")
    
    print(f"✓ Dependencies installed (with {len(failed_packages) if 'failed_packages' in locals() else 0} failures)")
    
    # Step 4: Parse notebook to understand structure
    print(f"\nStep 4: Analyzing notebook structure...")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    print(f"  Total cells: {len(nb['cells'])}")
    print(f"  Code cells: {len(code_cells)}")
    
    for i, cell in enumerate(code_cells, 1):
        source = ''.join(cell.get('source', []))
        first_line = source.split('\n')[0] if source else '(empty)'
        print(f"    [{i}] {first_line[:60]}")
    
    # Step 5: Execute notebook with nbconvert
    print(f"\nStep 5: Executing notebook with nbconvert...")
    output_file = REPO_PATH / "exemple_pypi_output.html"
    
    cmd = [
        str(python_exe), "-m", "nbconvert",
        "--to", "html",
        "--execute",
        "--ExecutePreprocessor.timeout=600",
        "--output", str(output_file),
        str(NOTEBOOK_PATH)
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900  # 15 minutes timeout for notebook execution
    )
    
    print(f"\n  STDOUT:")
    print(f"  {result.stdout[:1000]}")
    if result.stderr:
        print(f"\n  STDERR:")
        print(f"  {result.stderr[:1000]}")
    
    # Step 6: Report results
    print(f"\n" + "="*70)
    if result.returncode == 0:
        print(f"✓ SUCCESS: Notebook executed without errors")
        print(f"="*70)
        if output_file.exists():
            print(f"Output saved to: {output_file}")
        return 0
    else:
        print(f"✗ FAILURE: Notebook execution failed")
        print(f"="*70)
        print(f"Exit code: {result.returncode}")
        print(f"\nFull STDERR:")
        print(result.stderr)
        print(f"\nFull STDOUT:")
        print(result.stdout)
        return 1

if __name__ == "__main__":
    sys.exit(main())
