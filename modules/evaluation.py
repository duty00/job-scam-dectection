"""Validation-only threshold selection and deterministic held-out evaluation."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix)

def scores(model, X):
    if hasattr(model,'predict_proba'):
        return np.asarray(model.predict_proba(X)[:,1])
    return np.asarray(model.decision_function(X))

def choose_threshold(y, score):
    precision,recall,thresholds=precision_recall_curve(y,score)
    if not len(thresholds):
        raise ValueError('No thresholds available')
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    # Fixed tie-break: largest threshold among identical optimal F1 values.
    best=np.flatnonzero(np.isclose(f1,f1.max(),rtol=0,atol=1e-12))[-1]
    return float(thresholds[best])

def metrics(y, score, threshold):
    pred=(score>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'accuracy':float(accuracy_score(y,pred)),
            'precision_fraud':float(precision_score(y,pred,zero_division=0)),
            'recall_fraud':float(recall_score(y,pred,zero_division=0)),
            'f1_fraud':float(f1_score(y,pred,zero_division=0)),
            'average_precision':float(average_precision_score(y,score)),
            'roc_auc':float(roc_auc_score(y,score)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
