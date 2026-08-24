from pathlib import Path
import json
p = Path('Notebooks/unidimensional1.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i in range(44, 55):
    cell = nb['cells'][i]
    print('CELL', i, 'type', cell['cell_type'])
    if cell['cell_type'] == 'code':
        print(''.join(cell['source']))
    else:
        print('--- markdown ---')
    print('==========')
