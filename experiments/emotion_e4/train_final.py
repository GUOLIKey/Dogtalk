"""Refit the selected E4 procedure on official TRAIN, never on official TEST.

Run with the independent Emotion Lab environment. Evaluation numbers remain
out-of-fold development results, not claimed metrics of this full-data refit.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
APP=ROOT.parents[1]/'PawTalk_Web_Clean'
sys.path.insert(0,str(APP))
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-dir',default='');p.add_argument('--output-dir',default=str(APP/'emotion_e4_assets'));p.add_argument('--cache-dir');args=p.parse_args()
    import numpy as np,joblib,sklearn,librosa,soundfile
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import f1_score
    import data_features as data
    from common import AXES,FEATURE_RECIPE,read_csv,read_json,save_json,sha
    from evaluate import split_groups
    if args.cache_dir:data.ROOT=Path(args.cache_dir).resolve() # Must contain the same pinned manifest; only caches/downloads live here.
    directory,spec=data.prepare_dataset({'dataset_dir':args.dataset_dir,'auto_download_training_data':True,'download_workers':3})
    rows=data.load_rows(directory,spec);basic,x=data.features(rows);groups,audit=data.make_groups(rows)
    reference={r['audio_id']:r for r in read_csv(ROOT/'reference'/'training_manifest.csv')}
    if set(reference)!=set(r['audio_id'] for r in rows):raise ValueError('Training manifest differs from the evaluated E4 dataset')
    for r in rows:
        if any(str(r.get(k,''))!=str(reference[r['audio_id']].get(k,'')) for k in ('arousal','valence','audio_sha256','pcm_sha256','dog_id','source_recording_id')):raise ValueError('Training provenance changed: '+r['audio_id'])
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);trained={};seed=20261116
    folds_by_id={r['audio_id']:r for r in read_csv(ROOT/'reference'/'fold_assignments.csv')};folds=np.array([int(folds_by_id[r['audio_id']]['fold']) for r in rows])
    for g in set(groups):
        if len(set(folds[groups==g]))!=1:raise ValueError('Duplicate group crosses folds')
    for axis,labels in AXES.items():
        y=np.array([r[axis] for r in rows]);oof=np.full(len(rows),'',object)
        # Verify the real-data feature/runtime implementation reproduces the reported E4 OOF predictions.
        previous=read_json(ROOT/'reference'/f'{axis}_selected_parameters.json')
        for choice in previous:
            fold=choice['fold'];pars=choice['params'];tr=folds!=fold;va=folds==fold
            model=Pipeline([('scale',StandardScaler()),('clf',SVC(C=pars['clf__C'],gamma=pars['clf__gamma'],class_weight=pars['clf__class_weight']))])
            model.fit(x[tr],y[tr]);oof[va]=model.predict(x[va])
        expected=read_json(ROOT/'reference'/f'E4_RICH_TUNED_{axis}_metrics.json')
        reproduced=float(f1_score(y,oof,labels=labels,average='macro',zero_division=0))
        if abs(reproduced-expected['macro_f1'])>1e-10:raise ValueError('E4 OOF feature reproduction failed: '+axis)
        cv=split_groups(y,groups,3,seed)
        pipe=Pipeline([('scale',StandardScaler()),('clf',SVC())])
        grid=[{'clf':[LogisticRegression(max_iter=2000,random_state=seed)],'clf__C':[.1,1.,10.],'clf__class_weight':[None,'balanced']},
              {'clf':[SVC(kernel='rbf')],'clf__C':[.1,1.,10.],'clf__gamma':['scale',.01,.1],'clf__class_weight':[None,'balanced']}]
        def scorer(m,xx,yy):return f1_score(yy,m.predict(xx),labels=labels,average='macro',zero_division=0)
        search=GridSearchCV(pipe,grid,cv=cv,scoring=scorer,n_jobs=2,error_score='raise',refit=True);search.fit(x,y)
        best=search.best_estimator_
        if not isinstance(best.named_steps['clf'],SVC):raise ValueError('Final family selection did not choose SVC; review before publishing an SVM release')
        path=out/(axis+'.joblib');joblib.dump(best,path,compress=3)
        trained[axis]={'classes_':list(best.classes_),'filename':path.name,'params':{k:str(v) if k=='clf' else v for k,v in search.best_params_.items()},'full_train_cv_selection_macro_f1':float(search.best_score_),
                       'selection_score_is_not_independent_test':True,'development_oof_macro_f1':expected['macro_f1'],'development_oof_accuracy':expected['accuracy'],'oof_reproduced':True}
        print(axis,trained[axis],flush=True)
    meta={'model_version':'emotion-e4-rich153-svm-v1','recipe':FEATURE_RECIPE,'feature_dimensions':153,'score_kind':'svm_decision_function_ovr','train_count':len(rows),'group_count':len(set(groups)),
          'dataset_repository':spec['repository'],'dataset_revision':spec['revision'],'dataset_manifest_sha256':sha(ROOT/'dataset_manifest.json'),'training_manifest_sha256':sha(ROOT/'reference'/'training_manifest.csv'),
          'official_test_used':False,'trained_on':'official TRAIN only','random_seed':seed,'axes':trained,'versions':{'sklearn':sklearn.__version__,'numpy':np.__version__,'librosa':librosa.__version__,'soundfile':soundfile.__version__},
          'feature_source_sha256':sha(APP/'emotion_e4_features.py'),'usability_policy':'e4-quadrant-gate-unvalidated-v1','limitations':['Development OOF results are not independent deployment accuracy.','Dog/source IDs unavailable; exact duplicate grouping does not establish cross-dog generalization.','Scores are uncalibrated decisions; no validated quadrant confidence gate.']}
    save_json(out/'metadata.json',meta)
    files={name:{'bytes':(out/name).stat().st_size,'sha256':sha(out/name)} for name in ('arousal.joblib','valence.joblib','metadata.json')}
    spec_out={'model_version':meta['model_version'],'recipe':FEATURE_RECIPE,'files':files,'usability_policy':meta['usability_policy']}
    save_json(APP/'emotion_e4_spec.json',spec_out)
    print('FINAL TRAINED:',out,'; both axes refitted on 1200 TRAIN clips; app scene assets untouched.',flush=True)
if __name__=='__main__':main()
