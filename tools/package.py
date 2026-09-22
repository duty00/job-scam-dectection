"""Package code and actual experiment outputs; omit raw/pretrained downloads."""
from pathlib import Path
import zipfile

def package(root, destination=None):
    root=Path(root).resolve()
    destination=Path(destination or root/'output'/'job_scam_project.zip')
    destination.parent.mkdir(parents=True,exist_ok=True)
    include=['modules','notebooks','features','models','results','reports','tests']
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for directory in include:
            for path in (root/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and '.ipynb_checkpoints' not in path.parts:
                    archive.write(path,path.relative_to(root).as_posix())
        for name in ['README.md','requirements.txt','requirements-deep-learning.txt','requirements-lock.txt',
                     'run_experiments.py','run_deep_learning.py',
                     'data/processed/split_manifest.csv','tools/build_notebook.py','tools/package.py']:
            path=root/name
            if path.exists():
                archive.write(path,name)
    return destination

if __name__=='__main__':
    print(package(Path(__file__).resolve().parents[1]))
