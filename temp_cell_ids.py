from pathlib import Path
import json
p = Path('Notebooks/unidimensional1.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if i >= 44 and i <= 46:
        print('INDEX', i, 'TYPE', cell['cell_type'], 'ID', cell.get('id'))
        if cell['cell_type'] == 'code':
            print('SRC:', ''.join(cell['source'])[:200].replace('\n','\\n'))
            print('-----')
