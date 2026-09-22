from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass(frozen=True)
class Config:
    seed: int = 42
    max_features: int = 40000
    min_df: int = 2
    group_min_chars: int = 80
    max_tokens: int = 1500
    root: str = '.'

    def path(self, name):
        p = Path(self.root).resolve() / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self):
        return asdict(self)

TEXT_FIELDS = ('title', 'company_profile', 'description', 'requirements', 'benefits')
DATA_URL = 'https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction'
GLOVE_URL = 'https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz'

DATA_SHA256 = '59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60'
