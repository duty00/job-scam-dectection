from pathlib import Path
import json,base64
root=Path(__file__).resolve().parents[1]
files=['01_data_overview.png','02_lengths_splits.png','04_model_comparison.png']
result={'test_scores':(root/'results/test_scores.csv').read_text(),
        'embedding_audit':json.loads((root/'results/embedding_audit.json').read_text()),
        'features':{p.name:p.stat().st_size for p in (root/'features').iterdir() if p.is_file()},
        'images':[{'name':name,'data':base64.b64encode((root/'results/figures'/name).read_bytes()).decode()} for name in files]}
print(json.dumps(result))
