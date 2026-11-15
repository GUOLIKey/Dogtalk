"""One-time DEV-only classifier preparation + cached CLAP feature parity checks."""
import csv,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'2')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def main():
    missing=[package for module,package in [('numpy','numpy'),('sklearn','scikit-learn'),('librosa','librosa'),('soundfile','soundfile'),('torch','torch'),('transformers','transformers'),('joblib','joblib'),('imageio_ffmpeg','imageio-ffmpeg')] if importlib.util.find_spec(module) is None]
    if missing:
        print('[DEPENDENCIES] Installing missing packages only: '+', '.join(missing),flush=True)
        subprocess.check_call([sys.executable,'-m','pip','install',*missing])
    import numpy as np,joblib,sklearn,librosa,transformers,torch
    from imageio_ffmpeg import get_ffmpeg_exe
    from live_audio import Encoder,feature_waves,MODEL_ID,FEATURE_VERSION,MODELS,MODEL,META
    from engine import LABELS
    get_ffmpeg_exe()
    sys.path.insert(0,str(ROOT/'scripts'))
    from step4b1_denoise_abc import prepare
    from step3_svm_tuning import build_model
    print('[1/4] Read original locked DEV feature cache; no holdout inference.',flush=True)
    rows,y,clap,mfcc,durations,keep,old=prepare()
    y=np.asarray(y);durations=np.asarray(durations)
    with (HERE/'dev_manifest.csv').open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
    if len(rows)!=9984 or clap.shape!=(9984,512) or mfcc.shape!=(9984,30):raise RuntimeError('Original DEV feature shapes changed')
    if [r['audio_path'].replace('\\','/') for r in rows]!=[r['audio_path'].replace('\\','/') for r in manifest] or not np.array_equal(y,[r['actual'] for r in manifest]):raise RuntimeError('Original DEV ordering/labels changed')
    print('[2/4] Load original CLAP (reuse local cache; download only if missing).',flush=True)
    encoder=Encoder(allow_download=True)
    print('[DEVICE] '+encoder.device,flush=True)
    pool=np.flatnonzero((durations>=1)&(durations<=60));pool=pool[np.argsort(durations[pool])]
    if len(pool)<3:raise RuntimeError('Not enough DEV clips for feature parity checks')
    selected=[int(pool[0]),int(pool[len(pool)//2]),int(pool[-1])];audit=[]
    for i in selected:
        path=(ROOT/rows[i]['audio_path'].replace('\\','/')).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Invalid DEV audio path')
        m,w=feature_waves(path);c=encoder.vector(w)
        cosine=float(np.dot(c,clap[i])/(np.linalg.norm(c)*np.linalg.norm(clap[i])+1e-12))
        relative=float(np.linalg.norm(c-clap[i])/(np.linalg.norm(clap[i])+1e-12))
        delta=float(np.max(np.abs(m-mfcc[i])))
        record={'audio_path':rows[i]['audio_path'],'mfcc_max_abs_diff':delta,'clap_cosine':cosine,'clap_relative_l2':relative};audit.append(record)
        print('[PARITY] '+json.dumps(record,ensure_ascii=False),flush=True)
        if not np.allclose(m,mfcc[i],rtol=1e-3,atol=0.02) or cosine<0.999 or relative>0.01:
            raise RuntimeError('Fresh features do not match the original training cache. Stop; do not silently normalize/change features. Send LIVE_SETUP_ERROR.txt without API Key.')
    print('[3/4] Fit one final SVM on 9984 DEV clips: C=1, gamma=2, balanced, original 50/50 dimension normalization. CLAP stays frozen.',flush=True)
    model=build_model(y,clap_dim=512,mfcc_dim=30,C=1.,gamma=2.)
    model.named_steps['svc'].set_params(decision_function_shape='ovr')
    x=np.concatenate([clap,mfcc],axis=1);model.fit(x,y)
    if list(model.classes_)!=LABELS:raise RuntimeError('Class order mismatch')
    filenames={Path(r['audio_path'].replace('\\','/')).name.casefold() for r in rows}
    holdout=ROOT/'results'/'optimization_step1'/'splits'/'LOCKED_holdout_DO_NOT_TUNE.csv'
    if holdout.exists():
        with holdout.open(encoding='utf-8-sig',newline='') as f:filenames.update(Path(r['audio_path'].replace('\\','/')).name.casefold() for r in csv.DictReader(f))
    MODELS.mkdir(exist_ok=True)
    tmp=MODEL.with_suffix('.tmp');joblib.dump(model,tmp)
    check=joblib.load(tmp)
    if not np.array_equal(model.predict(x[selected]),check.predict(x[selected])):raise RuntimeError('Saved classifier parity failed')
    metadata={'feature_version':FEATURE_VERSION,'clap_model':MODEL_ID,'train_count':9984,'labels':LABELS,'C':1,'gamma':2,'feature_order':'raw CLAP 512 then MFCC 30','sklearn_version':sklearn.__version__,'librosa_version':librosa.__version__,'transformers_version':transformers.__version__,'torch_version':torch.__version__,'device_checked':encoder.device,'feature_audit':audit,'model_sha256':hashlib.sha256(tmp.read_bytes()).hexdigest(),'dataset_filenames':sorted(filenames),'created':time.time(),'holdout_audio_decoded_or_scored':False}
    meta_tmp=META.with_suffix('.tmp');meta_tmp.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,MODEL);os.replace(meta_tmp,META)
    print('[4/4] READY: new audio inference is enabled. Start 00_START_APP.bat. No DeepSeek calls were made.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        message=traceback.format_exc();(HERE/'LIVE_SETUP_ERROR.txt').write_text(message,encoding='utf-8');print(message,flush=True);sys.exit(1)
