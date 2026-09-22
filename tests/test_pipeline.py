import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from modules.data import clean_text,group_rows,digest
from modules.evaluation import choose_threshold,metrics
from modules.deep_learning import build_vocabulary,encode_texts

class CoreTests(unittest.TestCase):
    def test_html_and_missing_text(self):
        self.assertEqual(clean_text(None),'')
        self.assertEqual(clean_text(float('nan')),'')
        self.assertEqual(clean_text('<p>We do NOT charge &amp; never ask fees.</p>'),
                         'we do not charge & never ask fees.')

    def test_transitive_grouping_and_blank_profiles(self):
        a='Long employer profile '*8
        b='Detailed shared job description '*8
        frame=pd.DataFrame({'company_profile':[a,a,'other company '*10,'',''],
                            'description':['different description '*10,b,b,'short one','short two']})
        groups=group_rows(frame)
        self.assertEqual(groups[0],groups[2])
        self.assertNotEqual(groups[3],groups[4])
        self.assertNotEqual(groups[0],groups[3])

    def test_threshold_matches_brute_force(self):
        y=np.array([0,0,1,0,1,1,0])
        score=np.array([.01,.9,.5,.2,.5,.8,.1])
        threshold=choose_threshold(y,score)
        optimal=max(f1_score(y,score>=t) for t in np.unique(score))
        self.assertAlmostEqual(f1_score(y,score>=threshold),optimal)
        result=metrics(y,score,threshold)
        self.assertEqual(result['tp']+result['fn'],int(y.sum()))

    def test_deep_vocabulary_is_train_only_and_deterministic(self):
        vocab=build_vocabulary(['alpha beta beta','gamma alpha'],max_size=10,min_frequency=1)
        self.assertEqual(vocab,build_vocabulary(['alpha beta beta','gamma alpha'],max_size=10,min_frequency=1))
        encoded=encode_texts(['beta unseen'],vocab,max_length=4)
        self.assertEqual(encoded.shape,(1,4))
        self.assertEqual(encoded[0,1],vocab['<UNK>'])
        self.assertTrue((encoded[0,2:]==0).all())

class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        if not (cls.root/'results'/'run_summary.json').exists():
            raise unittest.SkipTest('Run experiments before artifact validation.')

    def test_partition_and_feature_alignment(self):
        root=self.root
        df=pd.read_csv(root/'data/processed/split_manifest.csv')
        self.assertEqual(df.groupby('group').split.nunique().max(),1)
        self.assertFalse(df.text_hash.duplicated().any())
        np.testing.assert_array_equal(np.load(root/'features/job_ids.npy'),df.job_id.to_numpy())
        np.testing.assert_array_equal(np.load(root/'features/labels.npy'),df.fraudulent.to_numpy())
        np.testing.assert_array_equal(np.load(root/'features/splits.npy'),df.split.to_numpy())
        for name in ['mean','idf']:
            x=np.load(root/f'features/glove_{name}.npy',allow_pickle=False)
            self.assertEqual(x.shape,(len(df),50))
            self.assertTrue(np.isfinite(x).all())
        manifest=json.loads((root/'features/manifest.json').read_text(encoding='utf-8'))
        for name,checksum in manifest['files'].items():
            self.assertEqual(digest(root/'features'/name),checksum)

    def test_selection_and_reported_metrics(self):
        root=self.root
        choice=json.loads((root/'results/selection.json').read_text(encoding='utf-8'))
        validation=pd.read_csv(root/'results/validation_scores.csv').fillna({'warnings':''})
        ordered=validation[validation.warnings==''].sort_values(['f1_fraud','average_precision','candidate'],ascending=[False,False,True])
        self.assertEqual(choice['selected_candidate'],int(ordered.iloc[0].candidate))
        self.assertFalse(choice['test_used_for_selection'])
        predictions=pd.read_csv(root/'results/test_predictions.csv')
        results=pd.read_csv(root/'results/test_scores.csv')
        for _,row in results[results.candidate>=0].iterrows():
            block=predictions[predictions.family==row.family]
            recomputed=metrics(block.y_true.to_numpy(),block.score.to_numpy(),row.threshold)
            for key in ['accuracy','f1_fraud','average_precision','tp','fp','fn','tn']:
                self.assertAlmostEqual(recomputed[key],row[key],places=7)

    def test_demo_rejects_empty_input_and_explanation_is_exact(self):
        import joblib
        from modules.inference import predict_text
        with self.assertRaises(ValueError):
            predict_text(' ',self.root)
        text='We are looking for a software engineer to develop Python applications and work with our team.'
        result,top,bottom=predict_text(text,self.root)
        self.assertTrue(0<=result['model_score']<=1)
        bundle=joblib.load(self.root/'models/demo_tfidf.joblib')
        x=bundle['vectorizer'].transform([clean_text(text)])
        model=bundle['model']
        log_odds=model.intercept_[0]+np.sum(x.data*model.coef_[0,x.indices])
        self.assertAlmostEqual(1/(1+np.exp(-log_odds)),result['model_score'])

if __name__=='__main__':
    unittest.main(verbosity=2)
