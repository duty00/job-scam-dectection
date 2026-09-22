import json
import os
import sys
import tempfile
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

root=Path(__file__).resolve().parents[1]
base=root/'tmp'/'notebook_validation'
base.mkdir(parents=True,exist_ok=True)
# A generated notebook intentionally refuses to overwrite bundled source files
# that differ from the current payload. Use a new workspace on every validation
# run so an older validation directory can never make `Run all` fail.
work=Path(tempfile.mkdtemp(prefix='work_',dir=base)).resolve()
kernel_dir=base/'kernels'/'emscad-local'
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({
    'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'EMSCAD validation','language':'python'}),encoding='utf-8')
notebook=nbformat.read(root/'notebooks/Job_Scam_Detection.ipynb',as_version=4)
manager=AsyncKernelManager(kernel_name='emscad-local',
    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
destination=root/'notebooks/Job_Scam_Detection_executed.ipynb'
start=time.perf_counter()
def progress(cell,cell_index,**kwargs):
    print(f'Cell {cell_index+1}/{len(notebook.cells)} completed at {time.perf_counter()-start:.1f}s',flush=True)
    nbformat.write(notebook,destination)
client=NotebookClient(notebook,km=manager,timeout=1800,startup_timeout=120,
                      allow_errors=False,on_cell_executed=progress,
                      resources={'metadata':{'path':str(base)}})
env=os.environ.copy()
env.update({'EMSCAD_WORKDIR':str(work),'PYTHONUTF8':'1','MPLBACKEND':'Agg'})
try:
    client.execute(cwd=str(base),env=env)
finally:
    nbformat.write(notebook,destination)
errors=[{'cell':i,'error':out.get('evalue')} for i,c in enumerate(notebook.cells)
        for out in c.get('outputs',[]) if out.output_type=='error']
record={'local_notebook_run_all':not errors,'errors':errors,
        'elapsed_seconds':time.perf_counter()-start,'python':sys.version,
        'fresh_workspace':str(work),'actual_google_colab_execution':False,
        'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
        'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells)}
(root/'results/notebook_validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
if errors:
    raise SystemExit(1)
