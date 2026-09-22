"""Shared experiment workflow for CLI and the self-contained Colab notebook."""
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
from .config import Config
from .data import load_raw,prepare_data,save_json,digest
from .eda import create_eda
from .embeddings import build_embeddings
from .evaluation import scores,choose_threshold,metrics

def prepare(config):
    start=time.perf_counter()
    raw=load_raw(config)
    df,audit=prepare_data(raw,config)
    eda=create_eda(raw,df,config)
    ix={name:np.flatnonzero(df.split.to_numpy()==name) for name in ['train','validation','test']}
    print(json.dumps(audit,indent=2),flush=True)
    packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib']}
    save_json({'python':platform.python_version(),'platform':platform.platform(),'packages':packages,
               'config':config.to_dict(),'protocol':'Train fit; validation model/threshold selection; locked test.'},
              config.path('results')/'environment.json')
    return {'config':config,'df':df,'ix':ix,'y':df.fraudulent.to_numpy(),
            'start':start,'representations':{},'vectorizers':{},'candidates':[],'audit':audit,'eda':eda}

def extract_features(ctx):
    config,df,ix=ctx['config'],ctx['df'],ctx['ix']
    texts=df.text.to_numpy()
    specs=[('bow',CountVectorizer(ngram_range=(1,1),min_df=config.min_df,max_features=config.max_features)),
           ('tfidf_uni',TfidfVectorizer(ngram_range=(1,1),min_df=config.min_df,max_features=config.max_features,sublinear_tf=True,dtype=np.float32)),
           ('tfidf_bi',TfidfVectorizer(ngram_range=(1,2),min_df=config.min_df,max_features=config.max_features,sublinear_tf=True,dtype=np.float32)),
           ('tfidf_bi_stop',TfidfVectorizer(ngram_range=(1,2),min_df=config.min_df,max_features=config.max_features,sublinear_tf=True,stop_words='english',dtype=np.float32))]
    timing={}
    for name,vectorizer in specs:
        t=time.perf_counter()
        vectorizer.fit(texts[ix['train']])
        matrix=vectorizer.transform(texts)
        ctx['representations'][name]=matrix;ctx['vectorizers'][name]=vectorizer
        sparse.save_npz(config.path('features')/f'{name}.npz',matrix)
        timing[name]={'seconds':time.perf_counter()-t,'shape':list(matrix.shape),'fit':'train'}
    t=time.perf_counter()
    ctx['representations'].update(build_embeddings(texts,ix['train'],config))
    timing['glove']={'seconds':time.perf_counter()-t,'shape':[len(df),50],
                     'fit':'Frozen pretrained vectors; IDF pooling weights train only'}
    np.save(config.path('features')/'labels.npy',ctx['y'],allow_pickle=False)
    np.save(config.path('features')/'job_ids.npy',df.job_id.to_numpy(),allow_pickle=False)
    np.save(config.path('features')/'splits.npy',df.split.to_numpy(dtype='U10'),allow_pickle=False)
    manifest={'row_alignment':'All feature rows follow data/processed/split_manifest.csv and job_ids.npy',
              'split_manifest_sha256':digest(config.path('data/processed')/'split_manifest.csv'),
              'timing':timing,'files':{p.name:digest(p) for p in config.path('features').iterdir() if p.is_file() and p.name != 'manifest.json'}}
    save_json(manifest,config.path('features')/'manifest.json')
    ctx['feature_timing']=timing
    return timing

def train_candidates(ctx):
    config,ix,y=ctx['config'],ctx['ix'],ctx['y']
    planned=[]
    for alpha in [.1,1.0]:
        planned.append(('bow_nb','bow',{'alpha':alpha},MultinomialNB(alpha=alpha)))
    for rep in ['tfidf_uni','tfidf_bi','tfidf_bi_stop']:
        for c in [.5,2.0]:
            planned.append(('tfidf_lr',rep,{'C':c,'class_weight':'balanced'},
                            LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=1500,random_state=config.seed)))
            planned.append(('tfidf_svm',rep,{'C':c,'class_weight':'balanced'},
                            LinearSVC(C=c,class_weight='balanced',max_iter=8000,random_state=config.seed)))
    for rep in ['glove_mean','glove_idf']:
        for c in [.1,1.0,10.0]:
            planned.append(('glove_lr',rep,{'C':c,'class_weight':'balanced'},
                            make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=1500,random_state=config.seed))))
            planned.append(('glove_svm',rep,{'C':c,'class_weight':'balanced'},
                            make_pipeline(StandardScaler(),LinearSVC(C=c,class_weight='balanced',max_iter=12000,random_state=config.seed))))
    records=[]
    for n,(family,rep,params,model) in enumerate(planned):
        print(f'Train {n+1}/{len(planned)}: {family} / {rep} / {params}',flush=True)
        matrix=ctx['representations'][rep]
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(matrix[ix['train']],y[ix['train']])
        fit_seconds=time.perf_counter()-t
        val_score=scores(model,matrix[ix['validation']])
        threshold=choose_threshold(y[ix['validation']],val_score)
        warning_messages=[str(w.message) for w in caught]
        record={'candidate':n,'family':family,'representation':rep,'params':json.dumps(params,sort_keys=True),
                'threshold':threshold,'fit_seconds':fit_seconds,
                'warnings':' | '.join(warning_messages),**metrics(y[ix['validation']],val_score,threshold)}
        records.append(record)
        ctx['candidates'].append({'model':model,'record':record})
        # Checkpoint actual completed trials; no placeholder metrics.
        pd.DataFrame(records).to_csv(config.path('results')/'validation_scores.csv',index=False)
    table=pd.DataFrame(records)
    # Freeze family winners and the overall winner BEFORE touching test scores.
    eligible=table[table.warnings=='']
    if eligible.empty:
        raise RuntimeError('All estimators emitted warnings; inspect validation_scores.csv.')
    ordered=eligible.sort_values(['f1_fraud','average_precision','candidate'],ascending=[False,False,True])
    champions=ordered.drop_duplicates('family').candidate.astype(int).tolist()
    if set(ordered.family)!=set(table.family):
        raise RuntimeError('At least one family has no converged candidate.')
    selected=int(ordered.iloc[0].candidate)
    demo=int(ordered[ordered.family=='tfidf_lr'].iloc[0].candidate)
    ctx.update({'champions':champions,'selected':selected,'demo':demo})
    selection={'selected_candidate':selected,'demo_candidate':demo,'family_champions':champions,
               'selection_metric':'validation fraud F1 at validation-selected threshold; ties: average precision, candidate order',
               'test_used_for_selection':False,
               'selected_config':ctx['candidates'][selected]['record']}
    save_json(selection,config.path('results')/'selection.json')
    chosen=ctx['candidates'][demo]
    demo_bundle={'model':chosen['model'],'vectorizer':ctx['vectorizers'][chosen['record']['representation']],
                 'threshold':chosen['record']['threshold'],'candidate':demo,
                 'scope':'English EMSCAD 2012-2014; screening signal, not proof of fraud'}
    joblib.dump(demo_bundle,config.path('models')/'demo_tfidf.joblib',compress=3)
    for candidate in champions:
        item=ctx['candidates'][candidate]
        bundle={'model':item['model'],'record':item['record']}
        rep=item['record']['representation']
        if rep in ctx['vectorizers']:
            bundle['vectorizer']=ctx['vectorizers'][rep]
        joblib.dump(bundle,config.path('models')/f'{item["record"]["family"]}.joblib',compress=3)
    return table.sort_values('f1_fraud',ascending=False)

def evaluate_test(ctx):
    config,ix,y,df=ctx['config'],ctx['ix'],ctx['y'],ctx['df']
    test=ix['test']; y_test=y[test]
    records=[]; predictions=[]
    base_score=np.zeros(len(test))
    baseline={'candidate':-1,'family':'majority_baseline','representation':'none','threshold':1.0,
              'selected_on_validation':False,**metrics(y_test,base_score,1.0)}
    records.append(baseline)
    fig,ax=plt.subplots(figsize=(7,5))
    for candidate in ctx['champions']:
        item=ctx['candidates'][candidate]; r=item['record']
        score=scores(item['model'],ctx['representations'][r['representation']][test])
        result={'candidate':candidate,'family':r['family'],'representation':r['representation'],
                'threshold':r['threshold'],'selected_on_validation':candidate==ctx['selected'],
                **metrics(y_test,score,r['threshold'])}
        records.append(result)
        pred=(score>=r['threshold']).astype(int)
        block=pd.DataFrame({'job_id':df.iloc[test].job_id.to_numpy(),'group':df.iloc[test].group.to_numpy(),
                            'y_true':y_test,'score':score,'prediction':pred,'family':r['family']})
        predictions.append(block)
        errors=block.loc[block.y_true!=block.prediction].copy()
        errors['error_type']=np.where(errors.prediction==1,'false_positive','false_negative')
        errors=errors.merge(df[['job_id','title','text']],on='job_id')
        errors['text_excerpt']=errors.text.str.slice(0,1200)
        errors.drop(columns='text').to_csv(config.path('results/errors')/f'{r["family"]}.csv',index=False)
        PrecisionRecallDisplay.from_predictions(y_test,score,name=r['family'],ax=ax)
        cm_fig,cm_ax=plt.subplots(figsize=(4.8,4.2))
        ConfusionMatrixDisplay.from_predictions(y_test,pred,display_labels=['Legitimate','Fraudulent'],
                                               colorbar=False,ax=cm_ax,cmap='Blues')
        cm_ax.set_title(r['family']+' (locked test)')
        cm_fig.tight_layout();cm_fig.savefig(config.path('results/figures')/f'cm_{r["family"]}.png');plt.close(cm_fig)
    ax.axhline(y_test.mean(),color='gray',linestyle='--',label='Fraud prevalence')
    ax.set_title('Locked test: precision-recall curves');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(config.path('results/figures')/'03_precision_recall.png');plt.close(fig)
    table=pd.DataFrame(records)
    table.to_csv(config.path('results')/'test_scores.csv',index=False)
    pd.concat(predictions).to_csv(config.path('results')/'test_predictions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    table.set_index('family')[['precision_fraud','recall_fraud','f1_fraud']].plot.bar(ax=axes[0],rot=25)
    axes[0].set_ylim(0,1.05);axes[0].set_title('Test metrics at validation-selected thresholds')
    table.set_index('family').average_precision.plot.bar(ax=axes[1],color='#0891b2',rot=25)
    axes[1].set_ylim(0,1.05);axes[1].set_title('Test average precision')
    fig.tight_layout();fig.savefig(config.path('results/figures')/'04_model_comparison.png');plt.close(fig)
    elapsed=time.perf_counter()-ctx['start']
    save_json({'elapsed_seconds':elapsed,'selected_candidate':ctx['selected'],'demo_candidate':ctx['demo'],
               'test_rows':len(test),'test_fraud':int(y_test.sum()),
               'selected_test_result':next(r for r in records if r['selected_on_validation']),
               'limitations':['Historical English dataset; no Vietnamese evaluation.',
                              'Exact profile/description grouping is not verified employer separation and does not catch all paraphrases.',
                              'GloVe average pooling loses word order and negation context.',
                              'Validation is used for both hyperparameters and thresholds; no nested cross-validation.',
                              'No claim of real-world fraud verification or calibrated probabilities.']},
              config.path('results')/'run_summary.json')
    print(f'Completed in {elapsed:.1f} seconds.',flush=True)
    return table

def run_all(config=None):
    ctx=prepare(config or Config())
    extract_features(ctx)
    train_candidates(ctx)
    evaluate_test(ctx)
    return ctx
