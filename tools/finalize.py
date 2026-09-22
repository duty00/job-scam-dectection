from pathlib import Path
import importlib.metadata
import json
import sys
import zipfile
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from modules.data import digest
from tools.package import package

validation=json.loads((root/'results/notebook_validation.json').read_text(encoding='utf-8'))
if not validation['local_notebook_run_all'] or validation['executed_code_cells']!=validation['code_cells']:
    raise RuntimeError('Notebook validation is incomplete.')
second=Path(validation['fresh_workspace'])
for name in ['data/processed/split_manifest.csv','features/glove_mean.npy','features/glove_idf.npy']:
    if digest(root/name)!=digest(second/name):
        raise AssertionError('Fresh notebook differs: '+name)
first=pd.read_csv(root/'results/test_scores.csv')
other=pd.read_csv(second/'results/test_scores.csv')
pd.testing.assert_frame_equal(first,other,check_exact=False,rtol=1e-10,atol=1e-10)
model_checks={}
for name in ['bow_nb','tfidf_lr','tfidf_svm','glove_lr','glove_svm']:
    model_checks[name]={'exists':(root/f'models/{name}.joblib').exists()}
record={'status':'passed','semantic_and_artifact_tests':6,
        'standalone_notebook_code_cells':validation['code_cells'],
        'fresh_notebook_metrics_match_cli':True,
        'fresh_notebook_features_match_cli':True,
        'actual_google_colab_execution':False,
        'report_pdf_created':(root/'reports/job_scam_report.pdf').exists(),'models':model_checks,
        'source_sha256':{p.relative_to(root).as_posix():digest(p) for p in (root/'modules').glob('*.py')}}
(root/'results/verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
lock=root/'requirements-lock.txt'
try:
    widget_version=importlib.metadata.version('ipywidgets')
except importlib.metadata.PackageNotFoundError:
    widget_version=None
if widget_version and 'ipywidgets==' not in lock.read_text():
    with lock.open('a',encoding='utf-8') as stream:
        stream.write(f'ipywidgets=={widget_version}\n')
archive=package(root)
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:
        raise AssertionError('ZIP CRC check failed')
    names=set(z.namelist())
    required={'notebooks/Job_Scam_Detection.ipynb','notebooks/Job_Scam_Detection_executed.ipynb',
              'modules/experiment.py','features/glove_mean.npy','features/glove_idf.npy',
              'results/verification.json','reports/README.md','reports/job_scam_report.pdf','README.md'}
    if not required.issubset(names):
        raise AssertionError('Incomplete package')
print(json.dumps({'archive':str(archive),'size_mb':round(archive.stat().st_size/1024**2,2),
                  'verified_files':len(names),'verification':record},indent=2),flush=True)
