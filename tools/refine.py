from pathlib import Path
import importlib.metadata

root=Path(__file__).resolve().parents[1]
p=root/'modules/config.py'
s=p.read_text(encoding='utf-8')
s += "\nDATA_SHA256 = '59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60'\n"
p.write_text(s,encoding='utf-8')
p=root/'modules/data.py'
s=p.read_text(encoding='utf-8').replace('from .config import TEXT_FIELDS, DATA_URL','from .config import TEXT_FIELDS, DATA_URL, DATA_SHA256')
s=s.replace('    df = pd.read_csv(csv)',"    if digest(csv) != DATA_SHA256:\n        raise ValueError('Dataset SHA-256 differs from verified source. Inspect the dataset version before proceeding.')\n    df = pd.read_csv(csv)")
p.write_text(s,encoding='utf-8')
p=root/'modules/experiment.py'
s=p.read_text(encoding='utf-8').replace("if p.is_file()}}","if p.is_file() and p.name != 'manifest.json'}}")
p.write_text(s,encoding='utf-8')
deps=['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','threadpoolctl','nbformat','nbclient','ipykernel']
(root/'requirements-lock.txt').write_text('# Exact local validation environment: Python 3.14\n'+'\n'.join(f'{x}=={importlib.metadata.version(x)}' for x in deps)+'\n',encoding='utf-8')
(root/'requirements.txt').write_text('''numpy>=1.26,<3
pandas>=2.2,<4
scipy>=1.11,<2
scikit-learn>=1.4,<2
matplotlib>=3.8,<4
joblib>=1.3,<2
threadpoolctl>=3.2,<4
nbformat>=5.9,<6
nbclient>=0.9,<1
ipykernel>=6.29,<8
ipywidgets>=8.1,<9
''',encoding='utf-8')
print('Refined checksums, rerun manifest and dependency declarations.')
