from pathlib import Path
import json

p = Path('Notebooks/unidimensional1.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
# Find the markdown cell with the exact content '# 5. Additions'
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']).strip()
        if source == '# 5. Additions':
            insert_index = i
            break

if insert_index is None:
    raise RuntimeError('Could not find the target markdown cell with # 5. Additions')

new_cell = {
    'cell_type': 'code',
    'metadata': {},
    'source': [
        '# Dictionnaire pour renommer les clusters K-medoids OM\n',
        "cluster_names_OM = {1: 'incomplete', 2: 'Out of care', 3: 'Fast', 4: 'Slow'}\n"
    ],
    'outputs': [],
    'execution_count': None
}

nb['cells'].insert(insert_index, new_cell)
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Inserted new code cell before # 5. Additions at index', insert_index)
