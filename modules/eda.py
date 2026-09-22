from collections import Counter
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import save_json

def create_eda(raw, prepared, config):
    folder = config.path('results/figures')
    plt.rcParams.update({'figure.dpi':130,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    counts = raw.fraudulent.value_counts().sort_index()
    axes[0].bar(['Legitimate','Fraudulent'],counts.values,color=['#2563eb','#f97316'])
    axes[0].set_title('Raw dataset: class distribution')
    for i,n in enumerate(counts.values):
        axes[0].text(i,n+100,str(n),ha='center')
    missing = raw.isna().mean().sort_values().tail(10)*100
    axes[1].barh(missing.index,missing.values,color='#0891b2')
    axes[1].set_xlabel('Missing (%)')
    axes[1].set_title('Raw fields with most missing values')
    fig.tight_layout(); fig.savefig(folder/'01_data_overview.png'); plt.close(fig)
    train = prepared.loc[prepared.split == 'train']
    lengths = train.text.str.split().map(len)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label,color in [(0,'#2563eb'),(1,'#f97316')]:
        axes[0].hist(lengths[train.fraudulent == label],bins=40,range=(0,2000),
                     density=True,alpha=.55,color=color,label=['Legitimate','Fraudulent'][label])
    axes[0].legend(); axes[0].set_title('Train only: words per posting'); axes[0].set_xlabel('Words (display capped at 2,000)')
    table=prepared.groupby(['split','fraudulent']).size().unstack(fill_value=0).reindex(['train','validation','test'])
    table.plot.bar(stacked=True,ax=axes[1],color=['#2563eb','#f97316'],rot=0)
    axes[1].set_title('Group-disjoint partitions'); axes[1].set_ylabel('Postings')
    fig.tight_layout(); fig.savefig(folder/'02_lengths_splits.png'); plt.close(fig)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words=[]
    for label in [0,1]:
        counter=Counter(w for text in train.loc[train.fraudulent==label,'text']
                        for w in re.findall(r'\b[a-z]{3,}\b',text) if w not in ENGLISH_STOP_WORDS)
        words.extend({'label':label,'word':word,'count':count} for word,count in counter.most_common(30))
    pd.DataFrame(words).to_csv(config.path('results')/'top_words_train.csv',index=False)
    raw.isna().sum().rename('missing_count').to_csv(config.path('results')/'missing_values.csv')
    summary={'raw_majority_accuracy':float((raw.fraudulent==0).mean()),
             'train_word_length_quantiles':{str(k):float(v) for k,v in lengths.quantile([0,.25,.5,.75,.95,1]).items()},
             'note':'Raw EDA is structural only; text frequencies and length analysis use train only.'}
    save_json(summary,config.path('results')/'eda_summary.json')
    return summary
