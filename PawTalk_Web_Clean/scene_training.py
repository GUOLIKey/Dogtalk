"""Reproducible fixed-target experiments; no hidden holdout inference or fake metrics."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'2')
HERE=Path(__file__).resolve().parent
from scene_schema import LABELS, ORIGINAL_LABELS, LABEL_VERSION, CONFIG_SHA256, MODEL_VERSION, CACHE_VERSION, SCORE_KIND, target, score_dict
FEATURE_VERSION='mfcc30_full_clap512_first10s_raw_v1'
SEED=20261112

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)

def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def path_in(root,relative):
    p=(Path(root)/relative.replace('\\','/')).resolve()
    if not p.is_relative_to(Path(root).resolve()):raise ValueError('音频路径越界。')
    return p

def build_model():
    import math
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    # Reuse the existing recipe: fit scaling only on each training fold;
    # raw CLAP512 followed by MFCC30, equal block contribution by dimension.
    features=ColumnTransformer([('clap',StandardScaler(),slice(0,512)),('mfcc',StandardScaler(),slice(512,542))],
        transformer_weights={'clap':math.sqrt(.5/512),'mfcc':math.sqrt(.5/30)},sparse_threshold=0)
    return Pipeline([('features',features),('svc',SVC(C=1.,gamma=2.,kernel='rbf',class_weight='balanced',decision_function_shape='ovr',cache_size=512))])

def metrics(y,p,labels):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    if not len(y):return {'n':0,'accuracy':None,'macro_f1':None,'per_class':{},'confusion_matrix':None,'labels':labels}
    return {'n':len(y),'accuracy':float(accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),
            'per_class':classification_report(y,p,labels=labels,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,p,labels=labels).tolist(),'labels':labels}

def report(rows,pred8,pred6,folds,out,provenance):
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    y8=np.array([r['actual'] for r in rows]);y6=np.array([target(y) for y in y8])
    pred8=np.array(pred8);mapped=np.array([target(p) for p in pred8]);folds=np.asarray(folds)
    schemes={'original_eight':(y8,pred8,ORIGINAL_LABELS),'mapped_eight_to_six':(y6,mapped,LABELS)}
    if pred6 is not None:schemes['retrained_six']=(y6,np.array(pred6),LABELS)
    result={'label_version':LABEL_VERSION,'provenance':provenance,'subsets':{}}
    comparison=[]
    for subset,mask in [('all_dev',np.ones(len(rows),dtype=bool)),('primary_ge_1s',np.array([float(r['duration_sec'])>=1 for r in rows]))]:
        result['subsets'][subset]={}
        for name,(y,p,labels) in schemes.items():
            m=metrics(y[mask],p[mask],labels)
            fm=[dict(fold=int(f),**metrics(y[mask&(folds==f)],p[mask&(folds==f)],labels)) for f in sorted(set(folds))]
            m['folds']=fm;m['fold_summary']={}
            for measure in ['accuracy','macro_f1']:
                vals=[r[measure] for r in fm if r['n']]
                m['fold_summary'][measure]={'mean':float(np.mean(vals)) if vals else None,'std_sample':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'fold_count':len(vals)}
            result['subsets'][subset][name]=m
            comparison.append({'subset':subset,'scheme':name,'target_classes':len(labels),'n':m['n'],'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'fold_accuracy_mean':m['fold_summary']['accuracy']['mean'],'fold_accuracy_std':m['fold_summary']['accuracy']['std_sample'],'fold_f1_mean':m['fold_summary']['macro_f1']['mean'],'fold_f1_std':m['fold_summary']['macro_f1']['std_sample']})
            if m['confusion_matrix'] is not None:
                write_csv(out/(subset+'_'+name+'_confusion.csv'),[dict(true_label=l,**dict(zip(labels,counts))) for l,counts in zip(labels,m['confusion_matrix'])])
                write_csv(out/(subset+'_'+name+'_per_class.csv'),[dict(label=l,**m['per_class'][l]) for l in labels])
    write_json(out/'metrics.json',result);write_csv(out/'comparison.csv',comparison)
    return result

def audit(args):
    rows=read_csv(args.manifest)
    if not rows:raise ValueError('DEV 清单为空。')
    for r in rows:
        target(r['actual']);target(r['A_pred'])
        if int(r['fold']) not in (1,2,3):raise ValueError('原始折号无效。')
    result=report(rows,[r['A_pred'] for r in rows],None,[int(r['fold']) for r in rows],args.output,
        {'kind':'real_archived_oof_predictions','manifest_sha256':sha(args.manifest),'model_rerun':False,'retrained_six_available':False,
         'leakage_status':'未提供音频指纹/犬只/源录音分组，此历史统计可能含泄漏；不能当成修正划分后的公平比较。',
         'interpretation':'原八类成绩与映射六类是不同任务；不得相减作为原任务性能提升。'})
    write_csv(Path(args.output)/'mapped_predictions.csv',[dict(r,actual_original=r['actual'],target_six=target(r['actual']),mapped_prediction_six=target(r['A_pred']),label_version=LABEL_VERSION) for r in rows])
    print(json.dumps(result['subsets']['primary_ge_1s']['mapped_eight_to_six']['fold_summary'],ensure_ascii=False))

def audit_groups(rows,root,holdout):
    """Union exact/decoded fingerprints, provided dog IDs and parent recording IDs."""
    import numpy as np
    import soundfile as sf
    n=len(rows);parent=list(range(n));seen={};hashes=[];pcmhashes=[];identities=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    hold_paths={r['audio_path'].replace('\\','/') for r in holdout}
    hold_hashes={r.get('audio_sha256') for r in holdout if r.get('audio_sha256')}
    hold_pcm={r.get('pcm_sha256') for r in holdout if r.get('pcm_sha256')}
    hold_ids={(k,r[k]) for r in holdout for k in ('dog_id','source_recording_id') if r.get(k)}
    for i,r in enumerate(rows):
        rel=r['audio_path'].replace('\\','/')
        if rel in hold_paths:raise ValueError('DEV 与锁定留出集路径重叠。')
        path=path_in(root,rel)
        if not path.is_file():raise FileNotFoundError('缺少 DEV 音频：'+str(path))
        rawsha=sha(path);wave,sr=sf.read(path,dtype='float32',always_2d=True)
        if not np.isfinite(wave).all():raise ValueError('无效音频采样：'+rel)
        pcm=hashlib.sha256(str(sr).encode()+str(wave.shape).encode()+wave.astype('<f4').tobytes()).hexdigest()
        if rawsha in hold_hashes or pcm in hold_pcm:raise ValueError('DEV 与锁定留出集指纹重叠。')
        keys=[('sha',rawsha),('pcm',pcm)]+[(k,r[k]) for k in ('dog_id','source_recording_id') if r.get(k)]
        if set(keys)&hold_ids:raise ValueError('DEV 与留出集犬只/源录音分组重叠，请先修正数据协议。')
        for key in keys:
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
        hashes.append(rawsha);pcmhashes.append(pcm);identities.append(rel)
    groups=[find(i) for i in range(n)];original=[int(r['fold']) for r in rows]
    if set(original)!={1,2,3}:raise ValueError('DEV 清单必须明确包含原始三折划分。')
    group_folds={}
    for g,f in zip(groups,original):group_folds.setdefault(g,set()).add(f)
    crossing={g for g,fs in group_folds.items() if len(fs)>1}
    details={'exact_duplicate_groups':sum(c>1 for c in Counter(hashes).values()),'decoded_duplicate_groups':sum(c>1 for c in Counter(pcmhashes).values()),'original_cross_fold_groups':len(crossing),
             'dog_id_coverage':sum(bool(r.get('dog_id')) for r in rows),'source_recording_id_coverage':sum(bool(r.get('source_recording_id')) for r in rows),
             'limitations':['未提供的犬只/来源身份无法核查；精确指纹不能发现所有重叠切片或重采样副本。','不读取留出集音频；跨集合指纹检查只使用提供的清单字段。'],
             'holdout_hash_coverage':sum(bool(r.get('audio_sha256')) for r in holdout),'holdout_audio_decoded_or_scored':False}
    if crossing:
        from sklearn.model_selection import StratifiedGroupKFold
        folds=np.zeros(n,dtype=int);y=np.array([r['actual'] for r in rows])
        for fold,(_,valid) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=SEED).split(np.zeros(n),y,groups),1):folds[valid]=fold
        details['split_policy']='corrected_stratified_group_kfold_3_seed_20261112'
    else:folds=np.array(original);details['split_policy']='original_manifest_folds_no_detected_group_crossing'
    checked={}
    for g,f in zip(groups,folds):checked.setdefault(g,set()).add(int(f))
    if any(len(fs)!=1 for fs in checked.values()):raise ValueError('分组仍跨折，停止训练。')
    split_id=hashlib.sha256(json.dumps(list(zip(identities,hashes,pcmhashes,groups,folds.tolist())),ensure_ascii=False).encode()).hexdigest()
    return hashes,groups,folds,dict(details,split_id=split_id)

def extract(args):
    import numpy as np
    rows=read_csv(args.manifest);root=Path(args.project_root)
    # Resolve all required files before downloading/loading the encoder.
    paths=[path_in(root,r['audio_path']) for r in rows]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'缺少 {len(missing)} 条 DEV 音频，例如 {missing[0]}')
    import sklearn,librosa,transformers,torch
    from live_audio import Encoder,feature_waves
    encoder=Encoder(allow_download=True);clap=[];mfcc=[]
    for i,path in enumerate(paths):
        m,w=feature_waves(path);mfcc.append(m);clap.append(encoder.vector(w))
        if i%100==0:print(f'[FEATURES] {i+1}/{len(paths)} device={encoder.device}',flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:np.savez_compressed(f,clap=np.array(clap),mfcc=np.array(mfcc),audio_paths=np.array([r['audio_path'] for r in rows]),feature_version=FEATURE_VERSION)
    write_json(str(out)+'.meta.json',{'feature_version':FEATURE_VERSION,'sha256':sha(out),'manifest_sha256':sha(args.manifest),
        'sklearn_version':sklearn.__version__,'librosa_version':librosa.__version__,'transformers_version':transformers.__version__,'torch_version':torch.__version__,
        'data_kind':'real_dev','source':'fresh_existing_live_audio_recipe','device':encoder.device,'original_training_cache_parity':'not_verified'})

def export_existing(args):
    """Use the original project's verified DEV cache, without new CLAP extraction."""
    import numpy as np
    import sklearn
    root=Path(args.project_root).resolve();scripts=root/'scripts'
    for name in ('step4b1_denoise_abc.py','step3_svm_tuning.py'):
        if not (scripts/name).is_file():raise FileNotFoundError('缺少原项目脚本：'+str(scripts/name)+'。可改用 extract 从 DEV 音频提取。')
    meta_path=HERE/'models'/'live_meta.json'
    if not meta_path.is_file():raise FileNotFoundError('导出旧缓存需要原八类 models/live_meta.json 记录特征库版本；不要伪造版本。也可选择 fresh extract。')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('feature_version')!=FEATURE_VERSION:raise ValueError('原特征版本不匹配。')
    sys.path.insert(0,str(scripts))
    from step4b1_denoise_abc import prepare
    from step3_svm_tuning import build_model as original_factory
    rows,y,clap,mfcc,durations,keep,old=prepare();manifest=read_csv(args.manifest)
    if ([r['audio_path'].replace('\\','/') for r in rows]!=[r['audio_path'].replace('\\','/') for r in manifest]
        or not np.array_equal(y,[r['actual'] for r in manifest]) or not np.array_equal(old,[r['A_pred'] for r in manifest])):
        raise ValueError('旧缓存路径、标签或原始预测与固定 DEV 清单不匹配。')
    if clap.shape!=(len(rows),512) or mfcc.shape!=(len(rows),30):raise ValueError('旧缓存维度无效。')
    # Compare the new self-contained factory against the original factory on the same
    # small DEV-only fit. This is a recipe check, never an evaluation metric.
    y=np.array(y);indices=np.concatenate([np.flatnonzero(y==label)[:4] for label in ORIGINAL_LABELS]);x=np.concatenate([clap,mfcc],axis=1)[indices]
    for labels in (y[indices],np.array([target(v) for v in y[indices]])):
        original=original_factory(labels,clap_dim=512,mfcc_dim=30,C=1.,gamma=2.)
        original.named_steps['svc'].set_params(decision_function_shape='ovr')
        original.fit(x,labels);new=build_model().fit(x,labels)
        if (list(original.classes_)!=list(new.classes_) or not np.array_equal(original.predict(x),new.predict(x))
            or not np.allclose(original.decision_function(x),new.decision_function(x),rtol=1e-6,atol=1e-8)):
            raise ValueError('原训练工厂与本次固定超参数实现不一致，停止；请核查真实工厂，不能默默替换流程。')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('wb') as f:np.savez_compressed(f,clap=clap,mfcc=mfcc,audio_paths=np.array([r['audio_path'] for r in manifest]),feature_version=FEATURE_VERSION)
    write_json(str(out)+'.meta.json',{'sha256':sha(out),'feature_version':FEATURE_VERSION,'manifest_sha256':sha(args.manifest),'data_kind':'real_dev',
        'source':'original_verified_DEV_cache','original_factory_parity':'passed_on_small_DEV_fit_both_label_tasks','original_training_cache_parity':'original_cache_reused',
        'sklearn_version':sklearn.__version__,**{k:meta[k] for k in ('librosa_version','transformers_version','torch_version') if k in meta}})
    print('[READY] Exported original DEV features; no new CLAP extraction, no holdout scoring.')

def train(args):
    import numpy as np
    import joblib,sklearn
    if Path(args.output).exists() and any(Path(args.output).iterdir()):raise ValueError('实验输出目录非空；请使用新的运行目录，以保留已有模型和评估。')
    rows=read_csv(args.manifest);holdout=read_csv(args.holdout_manifest)
    if not holdout:raise ValueError('锁定留出清单为空，停止。')
    fpath=Path(args.features)
    if not fpath.is_file():raise FileNotFoundError('缺少原始 CLAP512/MFCC30 特征包：'+str(fpath))
    fm=json.loads(Path(str(fpath)+'.meta.json').read_text(encoding='utf-8'))
    if fm.get('sha256')!=sha(fpath) or fm.get('feature_version')!=FEATURE_VERSION:raise ValueError('特征包校验或版本不匹配。')
    if fm.get('manifest_sha256')!=sha(args.manifest):raise ValueError('特征来源清单与训练清单不一致。')
    for k in ('librosa_version','transformers_version'):
        if not isinstance(fm.get(k),str) or not fm[k]:raise ValueError('特征缺少提取库版本：'+k)
    if fm.get('data_kind') not in ('real_dev','synthetic_test_fixture'):raise ValueError('必须明确特征数据来源 data_kind。')
    if args.install and fm['data_kind']!='real_dev':raise ValueError('禁止安装模拟数据训练的模型。')
    with np.load(fpath,allow_pickle=False) as z:
        clap=z['clap'];mfcc=z['mfcc'];paths=z['audio_paths'].tolist()
        if str(z['feature_version'])!=FEATURE_VERSION or paths!=[r['audio_path'] for r in rows]:raise ValueError('特征顺序或版本与清单不同。')
    if clap.shape!=(len(rows),512) or mfcc.shape!=(len(rows),30) or not np.isfinite(clap).all() or not np.isfinite(mfcc).all():raise ValueError('特征维度或数值非法。')
    if set(r['actual'] for r in rows)!=set(ORIGINAL_LABELS):raise ValueError('训练集必须覆盖原八类。')
    hashes,groups,folds,audit=audit_groups(rows,args.project_root,holdout)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'split_audit.json',audit)
    write_csv(out/'fold_assignments.csv',[dict(audio_path=r['audio_path'],audio_sha256=hashes[i],actual_original=r['actual'],target_six=target(r['actual']),group=groups[i],original_fold=r['fold'],fold=int(folds[i]),label_version=LABEL_VERSION) for i,r in enumerate(rows)])
    x=np.concatenate([clap,mfcc],axis=1);y8=np.array([r['actual'] for r in rows]);y6=np.array([target(y) for y in y8])
    p8=np.empty(len(rows),dtype=object);p6=p8.copy();values=[None]*len(rows);foldmeta={}
    base_meta={'feature_version':FEATURE_VERSION,'clap_model':'laion/clap-htsat-unfused','label_version':LABEL_VERSION,'label_config_sha256':CONFIG_SHA256,
        'model_version':MODEL_VERSION,'score_kind':SCORE_KIND,'split_id':audit['split_id'],'feature_order':'raw CLAP 512 then MFCC 30','C':1,'gamma':2,
        'sklearn_version':sklearn.__version__,'librosa_version':fm['librosa_version'],'transformers_version':fm['transformers_version'],'torch_version':fm.get('torch_version'),
        'created':time.time(),'feature_source':fm,'holdout_audio_decoded_or_scored':False}
    for fold in (1,2,3):
        tr=np.flatnonzero(folds!=fold);va=np.flatnonzero(folds==fold)
        if not len(va) or set(y8[tr])!=set(ORIGINAL_LABELS) or set(y6[tr])!=set(LABELS):raise ValueError('分组后某训练折缺类，需重新设计分组协议；不默默改标签。')
        print(f'[CV] fold={fold} original8 then retrained6; n_train={len(tr)} n_valid={len(va)}',flush=True)
        old=build_model().fit(x[tr],y8[tr]);p8[va]=old.predict(x[va])
        # Save both classifiers in the experiment folder, never overwrite old installed weights.
        joblib.dump(old,out/f'eight_fold{fold}.joblib')
        new=build_model().fit(x[tr],y6[tr]);p6[va]=new.predict(x[va]);s=new.decision_function(x[va])
        joblib.dump(new,out/f'six_fold{fold}.joblib')
        foldmeta[fold]=dict(base_meta,classes_=list(map(str,new.classes_)),train_count=len(tr),model_sha256=sha(out/f'six_fold{fold}.joblib'))
        for i,score in zip(va,s):values[int(i)]=score_dict(new.classes_,score)
    provenance={'kind':'paired_model_rerun','data_kind':fm['data_kind'],'feature_sha256':sha(fpath),'same_samples_and_folds':True,'hyperparameters_changed':False,'split_audit':audit,
        'old_prediction_agreement':float(np.mean(p8==np.array([r['A_pred'] for r in rows]))),
        'note':'原八分类和重训六分类均在同一划分重跑。若原始缓存/训练工厂未提供，不声称逐字节复现旧模型。'}
    report(rows,p8,p6,folds,out,provenance)
    write_csv(out/'paired_oof_predictions.csv',[dict(audio_path=r['audio_path'],audio_sha256=hashes[i],actual_original=r['actual'],target_six=y6[i],prediction_eight=p8[i],prediction_eight_mapped=target(p8[i]),prediction_six=p6[i],fold=int(folds[i])) for i,r in enumerate(rows)])
    counts=Counter(hashes);items={}
    for i,r in enumerate(rows):
        if counts[hashes[i]]>1:continue # ambiguous repeated file bytes are never replayed as unique rows
        items[hashes[i]]=dict(audio_path=r['audio_path'],actual_original=r['actual'],target_six=y6[i],predicted=p6[i],scores=values[i],duration=float(r['duration_sec']),fold=int(folds[i]),prediction_mode='oof',model_metadata=foldmeta[int(folds[i])])
    identity=hashlib.sha256((sha(fpath)+audit['split_id']+CONFIG_SHA256+json.dumps(foldmeta,sort_keys=True)).encode()).hexdigest()
    index=dict(version=CACHE_VERSION,label_version=LABEL_VERSION,label_config_sha256=CONFIG_SHA256,model_version=MODEL_VERSION,identity=identity,split_id=audit['split_id'],items=items,duplicate_hash_groups_excluded=sum(n>1 for n in counts.values()))
    write_json(out/'audio_index.json',index)
    model=build_model().fit(x,y6);joblib.dump(model,out/'live_svm.joblib')
    filenames=sorted({Path(r['audio_path'].replace('\\','/')).name.casefold() for r in rows+holdout})
    metadata=dict(base_meta,classes_=list(map(str,model.classes_)),train_count=len(rows),model_sha256=sha(out/'live_svm.joblib'),dataset_filenames=filenames)
    write_json(out/'live_meta.json',metadata)
    if args.install:
        import shutil
        from live_audio import MODEL,META
        from scene_cache import load_index
        load_index(out/'audio_index.json')
        MODEL.parent.mkdir(parents=True,exist_ok=True);idx=HERE/'data'/'scene_six_v1'/'audio_index.json';idx.parent.mkdir(parents=True,exist_ok=True)
        for src,dst in [(out/'live_svm.joblib',MODEL),(out/'live_meta.json',META),(out/'audio_index.json',idx)]:
            tmp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(src,tmp);os.replace(tmp,dst)
    print('[READY] Six-class training completed; data_kind='+fm['data_kind']+'. No intent accuracy inferred.',flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('audit','extract','export-existing','train'):
        q=sub.add_parser(name);q.add_argument('--manifest',default=str(HERE/'dev_manifest.csv'));q.add_argument('--output',required=True)
        if name in ('extract','export-existing','train'):q.add_argument('--project-root',default=str(HERE.parent))
        if name=='train':
            q.add_argument('--features',required=True);q.add_argument('--holdout-manifest',required=True);q.add_argument('--install',action='store_true')
    a=p.parse_args();{'audit':audit,'extract':extract,'export-existing':export_existing,'train':train}[a.command](a)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('[FAILED] '+str(exc),file=sys.stderr);sys.exit(1)
