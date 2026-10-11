"""Convierte experimento_features.py (celdas # %%) a notebook y lo ejecuta.

Uso: .venv/Scripts/python.exe ejecutar_notebook_features.py
Requiere jupyter_client/ipykernel ya instalados; no necesita nbformat ni nbclient.
"""
import json
import os
from pathlib import Path
from jupyter_client import KernelManager

root = Path(__file__).resolve().parent
runtime = root / 'resultados_feature_engineering' / 'runtime'
runtime.mkdir(parents=True, exist_ok=True)
for key, folder in [('JUPYTER_CONFIG_DIR','jupyter'), ('JUPYTER_DATA_DIR','data'), ('IPYTHONDIR','ipython')]:
    os.environ[key] = str(runtime / folder)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
source = (root / 'experimento_features.py').read_text(encoding='utf-8')
cells = []
for block in source.split('# %%')[1:]:
    header, _, text = block.partition('\n')
    if '[markdown]' in header:
        content = '\n'.join(line[2:] if line.startswith('# ') else '' if line == '#' else line for line in text.splitlines())
        cells.append(dict(cell_type='markdown', metadata={}, source=content.strip()))
    else:
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.strip()))
for i, cell in enumerate(cells):
    cell['id'] = f'features-{i:03d}'
nb = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata={
    'kernelspec': {'name':'python3', 'display_name':'Python 3.13', 'language':'python'},
    'language_info': {'name':'python', 'version':'3.13'}})
path = root / 'Feature_engineering_modelos.ipynb'


def collect(client, request):
    outputs, count, error = [], None, False
    while True:
        msg = client.get_iopub_msg(timeout=1200)
        if msg.get('parent_header', {}).get('msg_id') != request:
            continue
        kind, c = msg['msg_type'], msg['content']
        if kind == 'status' and c['execution_state'] == 'idle':
            break
        if kind == 'execute_input': count = c['execution_count']
        elif kind == 'stream':
            outputs.append(dict(output_type='stream', name=c['name'], text=c['text']))
            print(c['text'], end='', flush=True)
        elif kind in ['display_data', 'execute_result']:
            out = dict(output_type=kind, data=c['data'], metadata=c.get('metadata', {}))
            if kind == 'execute_result': out['execution_count'] = c['execution_count']
            outputs.append(out)
        elif kind == 'error':
            error = True
            outputs.append(dict(output_type='error', ename=c['ename'], evalue=c['evalue'], traceback=c['traceback']))
            print('\n'.join(c['traceback']), flush=True)
    return outputs, count, error


km = KernelManager(kernel_name='python3')
km.start_kernel(cwd=str(root))
kc = km.client()
kc.start_channels()
try:
    kc.wait_for_ready(timeout=90)
    collect(kc, kc.execute('%matplotlib inline'))
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code': continue
        outputs, count, error = collect(kc, kc.execute(cell['source'], stop_on_error=True))
        cell.update(outputs=outputs, execution_count=count)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'Celda {i+1}/{len(cells)} guardada', flush=True)
        if error: raise RuntimeError(f'Falló celda {i+1}')
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
print('Notebook final:', path)
