"""Frozen pretrained GloVe embeddings; pooling weights learned on train only."""
from collections import Counter
import gzip
import json
import re
import urllib.request
import numpy as np
from .config import GLOVE_URL
from .data import download, digest, save_json

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")

def tokenize(text, max_tokens=1500):
    return TOKEN_RE.findall(text.lower())[:max_tokens]

def load_glove(config):
    folder=config.path('data/pretrained')
    meta=folder/'glove_source.json'
    if meta.exists():
        source=json.loads(meta.read_text(encoding='utf-8'))
    else:
        with urllib.request.urlopen('https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json',timeout=60) as r:
            source=json.load(r)['models']['glove-wiki-gigaword-50']
        save_json(source,meta)
    path=download(GLOVE_URL,folder/'glove-wiki-gigaword-50.gz',source['checksum'],'md5')
    words_path=folder/'glove_words.json'
    vectors_path=folder/'glove_vectors.npy'
    cache_meta=folder/'glove_cache.json'
    valid=False
    if all(p.exists() for p in [words_path,vectors_path,cache_meta]):
        cached=json.loads(cache_meta.read_text())
        valid=(cached.get('source_md5')==source['checksum'] and
               cached.get('vectors_sha256')==digest(vectors_path) and
               cached.get('words_sha256')==digest(words_path))
    if valid:
        words=json.loads(words_path.read_text(encoding='utf-8'))
        vectors=np.load(vectors_path,allow_pickle=False)
    else:
        print('Reading pretrained GloVe 50-dimensional vectors...',flush=True)
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            count,dim=map(int,next(stream).split())
            vectors=np.empty((count,dim),dtype=np.float32)
            words=[]
            for i,line in enumerate(stream):
                word,rest=line.rstrip().split(' ',1)
                values=np.fromstring(rest,sep=' ',dtype=np.float32)
                if len(values)!=dim or i>=count:
                    raise ValueError('Malformed GloVe file')
                words.append(word); vectors[i]=values
        if len(words)!=count or dim!=50 or not np.isfinite(vectors).all():
            raise ValueError('Incomplete GloVe vectors')
        np.save(vectors_path,vectors,allow_pickle=False)
        save_json(words,words_path)
        save_json({'source_md5':source['checksum'],'vectors_sha256':digest(vectors_path),
                   'words_sha256':digest(words_path)},cache_meta)
    save_json({'name':'glove-wiki-gigaword-50','url':GLOVE_URL,'md5':source['checksum'],
               'dimensions':50,'words':len(words),'frozen':True,
               'reference':'https://nlp.stanford.edu/projects/glove/'},
              config.path('results')/'embedding_source.json')
    return {word:i for i,word in enumerate(words)},vectors

def build_embeddings(texts, train_indices, config):
    vocab,vectors=load_glove(config)
    tokens=[tokenize(t,config.max_tokens) for t in texts]
    frequencies=Counter(w for i in train_indices for w in set(tokens[i]))
    n=len(train_indices)
    idf={w:float(np.log((1+n)/(1+count))+1) for w,count in frequencies.items()}
    default_idf=float(np.log(1+n)+1)
    save_json({'idf':idf,'default_idf':default_idf,'fit_partition':'train','max_tokens':config.max_tokens},
              config.path('features')/'glove_pooling.json')
    mean=np.zeros((len(texts),50),dtype=np.float32)
    weighted=np.zeros_like(mean)
    coverage=np.zeros(len(texts),dtype=np.float32)
    for i,terms in enumerate(tokens):
        known=[w for w in terms if w in vocab]
        coverage[i]=len(known)/max(1,len(terms))
        if known:
            rows=vectors[[vocab[w] for w in known]]
            weights=np.array([idf.get(w,default_idf) for w in known],dtype=np.float32)
            mean[i]=rows.mean(axis=0)
            weighted[i]=np.average(rows,weights=weights,axis=0)
    for name,matrix in [('mean',mean),('idf',weighted)]:
        norms=np.linalg.norm(matrix,axis=1,keepdims=True)
        matrix/=np.maximum(norms,1e-12)
        if not np.isfinite(matrix).all():
            raise ValueError('Non-finite embedding values')
        np.save(config.path('features')/f'glove_{name}.npy',matrix,allow_pickle=False)
    np.save(config.path('features')/'glove_coverage.npy',coverage,allow_pickle=False)
    save_json({'max_tokens':config.max_tokens,
               'truncated_documents':sum(len(TOKEN_RE.findall(t))>config.max_tokens for t in texts),
               'mean_known_token_fraction':float(coverage.mean()),
               'zero_vector_documents':int((coverage==0).sum()),
               'pooling':'L2-normalized arithmetic mean and train-IDF-weighted mean'},
              config.path('results')/'embedding_audit.json')
    return {'glove_mean':mean,'glove_idf':weighted}
