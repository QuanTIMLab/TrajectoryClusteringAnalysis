from pathlib import Path
import json
p = Path('Notebooks/unidimensional1.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'kmedoids_labels_OM' in src:
            print('CELL', i)
            print(src)
            print('---')
