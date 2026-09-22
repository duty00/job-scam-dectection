"""Download, audit and label-independent grouping before splitting."""
import hashlib
import html
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .config import TEXT_FIELDS, DATA_URL, DATA_SHA256

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def download(url, destination, expected_hash=None, algorithm='sha256'):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (expected_hash is None or digest(destination, algorithm) == expected_hash):
        return destination
    partial = destination.with_suffix(destination.suffix + '.part')
    last_error = None
    for attempt in range(3):
        try:
            print(f'Download: {destination.name} (attempt {attempt+1})', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'EMSCAD-Course-Project/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as output:
                for chunk in iter(lambda: response.read(1024*1024), b''):
                    output.write(chunk)
            if expected_hash and digest(partial, algorithm) != expected_hash:
                raise ValueError('Download checksum mismatch: ' + destination.name)
            partial.replace(destination)
            return destination
        except Exception as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f'Cannot download {url}. Check Internet access and rerun. No login required.') from last_error

def load_raw(config):
    folder = config.path('data/raw')
    csv = folder / 'fake_job_postings.csv'
    if not csv.exists():
        archive = download(DATA_URL, folder / 'emscad.zip')
        if not zipfile.is_zipfile(archive):
            raise ValueError('Dataset endpoint returned a non-ZIP file. Remove data/raw/emscad.zip and retry.')
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if Path(n).name == csv.name]
            if len(names) != 1:
                raise ValueError('Dataset archive does not contain the expected CSV.')
            csv.write_bytes(z.read(names[0]))
    if digest(csv) != DATA_SHA256:
        raise ValueError('Dataset SHA-256 differs from verified source. Inspect the dataset version before proceeding.')
    df = pd.read_csv(csv)
    needed = set(TEXT_FIELDS) | {'job_id', 'fraudulent'}
    if not needed.issubset(df.columns) or len(df) != 17880:
        raise ValueError('Unexpected EMSCAD schema or row count. Inspect source before running.')
    if df.job_id.duplicated().any() or not df.fraudulent.isin([0, 1]).all():
        raise ValueError('Invalid IDs or labels.')
    save_json({'url': DATA_URL, 'paper': 'https://doi.org/10.3390/fi9010006',
               'csv_sha256': digest(csv), 'rows': len(df),
               'label_counts': {str(k): int(v) for k,v in df.fraudulent.value_counts().items()},
               'data_scope': 'English job ads, 2012-2014; EMSCAD mirror on Kaggle'},
              config.path('results') / 'data_manifest.json')
    return df

def clean_text(value):
    if value is None or pd.isna(value):
        return ''
    value = html.unescape(str(value))
    value = re.sub(r'<[^>]+>', ' ', value)
    value = re.sub(r'https?://\S+|www\.\S+', ' urltoken ', value, flags=re.I)
    value = re.sub(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b', ' emailtoken ', value)
    return re.sub(r'\s+', ' ', value).strip().lower()

def combine_fields(frame, fields=TEXT_FIELDS):
    parts = [frame[field].map(clean_text) for field in fields]
    return pd.concat(parts, axis=1).agg(' '.join, axis=1).map(lambda s: re.sub(r'\s+', ' ', s).strip())

def group_rows(frame, min_chars=80):
    # Union-find catches transitive matches. Blank/short generic profiles never group rows.
    parent = list(range(len(frame)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    for field in ('company_profile', 'description'):
        seen = {}
        for i, text in enumerate(frame[field].map(clean_text)):
            if len(text) < min_chars:
                continue
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in seen:
                union(i,seen[key])
            else:
                seen[key] = i
    return np.array([find(i) for i in range(len(frame))])

def prepare_data(raw, config):
    df = raw.copy()
    df['text'] = combine_fields(df)
    df['text_hash'] = df.text.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    empty = df.text.str.len() == 0
    df = df.loc[~empty].copy()
    conflicts = df.groupby('text_hash').fraudulent.nunique()
    conflict_hashes = set(conflicts[conflicts > 1].index)
    conflict_count = int(df.text_hash.isin(conflict_hashes).sum())
    df = df.loc[~df.text_hash.isin(conflict_hashes)].copy()
    duplicates = int(df.text_hash.duplicated().sum())
    df = df.drop_duplicates('text_hash').reset_index(drop=True)
    df['group'] = group_rows(df, config.group_min_chars)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.seed)
    development, test = next(outer.split(df, df.fraudulent, df.group))
    dev = df.iloc[development]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=config.seed + 1)
    train_pos, val_pos = next(inner.split(dev,dev.fraudulent,dev.group))
    split = np.full(len(df), 'test', dtype=object)
    split[development[train_pos]] = 'train'
    split[development[val_pos]] = 'validation'
    df['split'] = split
    for label in ('train','validation','test'):
        if df.loc[df.split == label, 'fraudulent'].nunique() != 2:
            raise ValueError(f'{label} has only one class; review group partition.')
    if df.groupby('group').split.nunique().max() != 1:
        raise AssertionError('Group leakage.')
    manifest = df[['job_id','fraudulent','text_hash','group','split']]
    manifest.to_csv(config.path('data/processed')/'split_manifest.csv',index=False)
    audit = {'raw_rows':len(raw), 'empty_text_rows_removed':int(empty.sum()),
             'conflicting_label_rows_removed':conflict_count,'duplicate_text_rows_removed':duplicates,
             'clean_rows':len(df), 'groups':int(df.group.nunique()),
             'largest_group':int(df.groupby('group').size().max()),
             'group_rule':'Exact normalized description OR company_profile with >=80 chars; transitive union. Not complete near-duplicate or verified-company isolation.',
             'split_counts':manifest.groupby('split').size().astype(int).to_dict(),
             'fraud_counts':manifest.groupby('split').fraudulent.sum().astype(int).to_dict()}
    save_json(audit,config.path('results')/'data_audit.json')
    return df, audit
