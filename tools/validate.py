import json
import os
import sys
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
print((root/'results/run_summary.json').read_text(encoding='utf-8'),flush=True)
result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root)
if result.returncode:
    raise SystemExit(result.returncode)
from tools.build_notebook import build_notebook
print('Notebook:',build_notebook(root),flush=True)
print('VERIFIED_SOURCE',root,flush=True)
