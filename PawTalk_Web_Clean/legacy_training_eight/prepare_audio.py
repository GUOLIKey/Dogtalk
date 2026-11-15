"""Export exact audio-matched, out-of-fold baseline scores for local dataset replay."""
import csv,hashlib,json,os,sys,time,traceback
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'data';OUT.mkdir(exist_ok=True)

def status(message,done=False):
    p=OUT/'prepare_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(message=message,done=done,updated=time.time()),ensure_ascii=False),encoding='utf-8');os.replace(q,p);print(message,flush=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    sys.path.insert(0,str(ROOT/'scripts'))
    from step4b1_denoise_abc import prepare
    from step3_svm_tuning import build_model
    from step1_split_and_weights import CV_FOLDS,SEED
    from engine import LABELS
    status('检查原项目特征、数据划分与基线…')
    rows,y,clap,mfcc,duration,keep,old=prepare();y=np.asarray(y);old=np.asarray(old);X=np.concatenate([clap,mfcc],axis=1)
    with (HERE/'dev_manifest.csv').open(encoding='utf-8-sig',newline='') as f:manifest=list(csv.DictReader(f))
    if len(rows)!=9984 or CV_FOLDS!=3 or int(SEED)!=20261112:raise RuntimeError('Unexpected DEV protocol')
    if [r['audio_path'].replace('\\','/') for r in rows]!=[r['audio_path'].replace('\\','/') for r in manifest]:raise RuntimeError('DEV rows changed')
    if not np.array_equal(y,[r['actual'] for r in manifest]) or not np.array_equal(old,[r['A_pred'] for r in manifest]):raise RuntimeError('Labels/baseline changed')
    folds=list(StratifiedKFold(3,shuffle=True,random_state=SEED).split(X,y));scores=np.empty((len(y),8));pred=np.empty(len(y),object);fid=np.zeros(len(y),int)
    identity=hashlib.sha256(X.tobytes()+json.dumps(y.tolist()).encode()+json.dumps([r['audio_path'] for r in rows]).encode()).hexdigest()
    for f,(train,valid) in enumerate(folds,1):
        target=OUT/(f'baseline_fold{f}_{identity[:16]}.npz')
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                if not np.array_equal(z['valid'],valid) or str(z['identity'])!=identity:raise RuntimeError('Cache mismatch')
                p=z['pred'];s=z['scores']
        else:
            status(f'准备候选排序：训练原基线，第 {f}/3 折（只训练 SVM，不重新提取 CLAP）…')
            model=build_model(y[train],clap_dim=512,mfcc_dim=30,C=1.,gamma=2.)
            model.named_steps['svc'].set_params(decision_function_shape='ovr')
            model.fit(X[train],y[train]);p=model.predict(X[valid]);s=model.decision_function(X[valid])
            if list(model.classes_)!=LABELS:raise RuntimeError('Class ordering mismatch')
            with target.with_suffix('.tmp').open('wb') as handle:np.savez_compressed(handle,pred=np.asarray(p,str),scores=s,valid=valid,identity=identity)
            os.replace(target.with_suffix('.tmp'),target)
        if s.shape!=(len(valid),8) or not np.isfinite(s).all():raise RuntimeError('Invalid decision values')
        if not np.array_equal(p,old[valid]):raise RuntimeError('Baseline OOF no longer reproduces; stop without exporting misleading scores')
        scores[valid]=s;pred[valid]=p;fid[valid]=f
    if not np.array_equal(fid,[int(r['fold']) for r in manifest]):raise RuntimeError('Fold mismatch')
    items={};duplicate=set()
    for i,r in enumerate(rows):
        rel=r['audio_path'].replace('\\','/');path=(ROOT/rel).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('Invalid audio path')
        digest=sha(path)
        if digest in items:duplicate.add(digest)
        items[digest]=dict(audio_path=rel,actual=str(y[i]),predicted=str(pred[i]),scores=dict(zip(LABELS,map(float,scores[i]))),duration=float(duration[i]),fold=int(fid[i]))
        if (i+1)%1000==0:status(f'校验录音指纹：{i+1}/9984…')
    for digest in duplicate:items.pop(digest,None)
    result=dict(version='baseline-oof-v1',identity=identity,items=items,duplicate_hashes_excluded=len(duplicate),
                notes=['All baseline predictions verified against prior OOF.','Duplicate audio hashes excluded from replay, not from model training.','Dataset replay only; no arbitrary new-audio inference.','Holdout not decoded or scored.','OVR scores are not calibrated probabilities.'])
    tmp=OUT/'audio_index.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');os.replace(tmp,OUT/'audio_index.json')
    status(f'准备完成：{len(items)} 条唯一录音可用。排除 {len(duplicate)} 组重复音频指纹。现在运行 00_START_APP.bat。',True)

if __name__=='__main__':
    try:main()
    except Exception:
        e=traceback.format_exc();status('准备失败，请查看 PREPARE_ERROR.txt');(HERE/'PREPARE_ERROR.txt').write_text(e,encoding='utf-8');print(e);sys.exit(1)
