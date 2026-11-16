import csv,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
AXES={'arousal':['Low','Medium','High'],'valence':['Negative','Neutral','Positive']}
VERSION='emotion-lab-1.0'
FEATURE_RECIPE='full_mfcc20_delta_spectral_rhythm_v1'

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def app_candidates(config):
    out=[Path(config['app_dir']).expanduser()] if config.get('app_dir') else []
    for p in (ROOT.parent,ROOT.parent.parent):
        out += [p/'PawTalk_Web_Clean',p/'Dogtalk'/'PawTalk_Web_Clean']
    return out

def resolve_app(config):
    return next((p.resolve() for p in app_candidates(config) if (p/'emotion_assets.json').is_file()),None)
