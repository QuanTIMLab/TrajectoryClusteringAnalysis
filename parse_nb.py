#!/usr/bin/env python
"""
Parse and analyze notebook cells
"""
import json
from pathlib import Path

notebook_path = Path(r"C:\Users\nicolas.grevet\dev\TrajectoryClusteringAnalysis.worktrees\copilot-worktree-2026-04-13T12-30-11\Notebooks\exemple_pypi.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Notebook Cells Analysis")
print("=" * 70)
print(f"Total cells: {len(nb['cells'])}")
print()

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"\n[CELL {i}]")
        source = ''.join(cell.get('source', []))
        print(f"Code length: {len(source)} chars")
        print("First 300 chars:")
        print(source[:300])
        print("..." if len(source) > 300 else "")
