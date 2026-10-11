"""Construye y ejecuta el notebook de la sesión a partir de sesion.py."""
import json
import os
import sys
from pathlib import Path
from jupyter_client import KernelManager

sys.stdout.reconfigure(encoding='utf-8',errors='replace')
HERE=Path(__file__).resolve().parent
runtime=HERE/'.runtime'
runtime.mkdir(exist_ok=True)
for key,folder in [('JUPYTER_CONFIG_DIR','jupyter'),('JUPYTER_DATA_DIR','data'),('IPYTHONDIR','ipython')]:
    os.environ[key]=str(runtime/folder)
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
cells=[]
for i,block in enumerate((HERE/'sesion.py').read_text(encoding='utf-8').split('# %%')[1:]):
    header,_,source=block.partition('\n')
    if '[markdown]' in header:
        source='\n'.join(line[2:] if line.startswith('# ') else '' if line=='#' else line for line in source.splitlines())
        cell=dict(cell_type='markdown',metadata={},source=source.strip())
    else: cell=dict(cell_type='code',metadata={},source=source.strip(),execution_count=None,outputs=[])
    cell['id']=f'experimento-{i:02d}'
    cells.append(cell)
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata={
    'kernelspec':{'name':'python3','language':'python','display_name':'Python 3.13'},
    'language_info':{'name':'python','version':'3.13'}})
dest=HERE/'Sesion_experimental.ipynb'


def collect(kc,request):
    outputs,count,failed=[],None,False
    while True:
        msg=kc.get_iopub_msg(timeout=1500)
        if msg.get('parent_header',{}).get('msg_id')!=request: continue
        kind,c=msg['msg_type'],msg['content']
        if kind=='status' and c['execution_state']=='idle': break
        if kind=='execute_input': count=c['execution_count']
        elif kind=='stream':
            outputs.append(dict(output_type='stream',name=c['name'],text=c['text']))
            print(c['text'],end='',flush=True)
        elif kind in ['display_data','execute_result']:
            out=dict(output_type=kind,data=c['data'],metadata=c.get('metadata',{}))
            if kind=='execute_result': out['execution_count']=c['execution_count']
            outputs.append(out)
        elif kind=='error':
            failed=True
            outputs.append(dict(output_type='error',ename=c['ename'],evalue=c['evalue'],traceback=c['traceback']))
            print('\n'.join(c['traceback']),flush=True)
    return outputs,count,failed


km=KernelManager(kernel_name='python3')
km.start_kernel(cwd=str(HERE))
kc=km.client(); kc.start_channels()
try:
    kc.wait_for_ready(timeout=90)
    collect(kc,kc.execute('%matplotlib inline'))
    for i,cell in enumerate(cells):
        if cell['cell_type']!='code': continue
        outputs,count,failed=collect(kc,kc.execute(cell['source'],stop_on_error=True))
        cell.update(outputs=outputs,execution_count=count)
        dest.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
        print(f'Celda {i+1}/{len(cells)} guardada',flush=True)
        if failed: raise RuntimeError(f'Error en celda {i+1}')
finally:
    kc.stop_channels(); km.shutdown_kernel(now=True)
print('Notebook completo:',dest)
