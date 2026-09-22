import os, sys, json, urllib.request, zipfile, hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / 'data' / 'raw'
out.mkdir(parents=True, exist_ok=True)
print('Python', sys.version, 'CPU', os.cpu_count(), flush=True)
url = 'https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction'
dest = out / 'emscad.zip'
if not dest.exists():
    req = urllib.request.Request(url, headers={'User-Agent': 'MLAssignment/1.0'})
    with urllib.request.urlopen(req, timeout=120) as response, dest.open('wb') as stream:
        while chunk := response.read(1024*1024):
            stream.write(chunk)
if zipfile.is_zipfile(dest):
    with zipfile.ZipFile(dest) as archive:
        print('ZIP', archive.namelist(), flush=True)
        csvs = [n for n in archive.namelist() if n.endswith('.csv')]
        content = archive.read(csvs[0])
        (out/'fake_job_postings.csv').write_bytes(content)
    import pandas as pd
    df = pd.read_csv(out/'fake_job_postings.csv')
    print('SHAPE',df.shape, 'LABELS',df.fraudulent.value_counts().to_dict())
    print('MISSING',df.isna().sum().to_dict())
    print('HASH',hashlib.sha256(content).hexdigest())
else:
    print('Not a ZIP', dest.stat().st_size, dest.read_bytes()[:300])
with urllib.request.urlopen('https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json', timeout=60) as response:
    info = json.load(response)['models']['glove-wiki-gigaword-50']
print('GLOVE',json.dumps(info), flush=True)
(root/'tmp').mkdir(exist_ok=True)
(root/'tmp'/'glove_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
