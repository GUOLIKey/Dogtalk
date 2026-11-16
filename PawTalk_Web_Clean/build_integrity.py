"""Release maintainer: regenerate hashes after reviewed source changes."""
import hashlib
import json
from pathlib import Path
from app_version import APP_VERSION
ROOT=Path(__file__).resolve().parent
EXCLUDED={'state','models','data','results','__pycache__','.venv','.venv_emotion','.venv_emotion_e4','program_backups'}

def main():
    files={}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if not p.is_file() or any(part in EXCLUDED for part in rel.parts) or p.name=='build_manifest.json':continue
        if p.suffix == '.joblib' and rel.parts[0] == 'emotion_e4_assets':
            files[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();continue
        if p.suffix not in {'.py','.json','.js','.cjs','.txt','.html','.css','.bat','.svg','.webmanifest'}:continue
        if p.name.endswith('_ERROR.txt') or p.name=='requirements_scene_test.txt':continue
        files[rel.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    (ROOT/'build_manifest.json').write_text(json.dumps({'version':APP_VERSION,'sha256':files},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Manifest updated: {len(files)} code/config/static files')

if __name__=='__main__':main()
