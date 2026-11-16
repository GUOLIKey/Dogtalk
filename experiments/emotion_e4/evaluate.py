"""Fixed grouped outer folds; inner-only tuning/calibration; no test-set access."""
import json,time
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold,GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
import joblib
from common import *

def split_groups(y,groups,n_splits,seed):
    if len(set(groups))<n_splits:raise ValueError('独立组少于折数')
    splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    splits=list(splitter.split(np.zeros((len(y),1)),y,groups))
    classes=set(y)
    for tr,va in splits:
        if set(groups[tr])&set(groups[va]):raise ValueError('分组跨训练/验证折')
        if set(y[tr])!=classes:raise ValueError('分组后训练折缺类；需要重新设计数据协议，不能伪造分数。')
    return splits

def align_probs(model,x,labels):
    if not hasattr(model,'predict_proba'):return None
    p=np.asarray(model.predict_proba(x));classes=list(model.classes_)
    if set(classes)!=set(labels):raise ValueError('模型 classes_ 不完整')
    return p[:,[classes.index(k) for k in labels]]

def metrics(y,pred,labels,probs=None):
    if not len(y):return {'n':0,'accuracy':None,'macro_f1':None}
    out={'n':len(y),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
         'labels':labels,'per_class':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}
    if probs is not None:
        # log_loss assumes its supplied labels are sorted; supply aligned sorted columns.
        order=np.argsort(labels);out['log_loss']=float(log_loss(y,probs[:,order],labels=np.array(labels)[order]))
        target=np.array([[float(v==l) for l in labels] for v in y]);out['multiclass_brier']=float(np.mean(np.sum((probs-target)**2,axis=1)))
        accepted=(probs.max(axis=1)>=.6)&((np.sort(probs,axis=1)[:,-1]-np.sort(probs,axis=1)[:,-2])>=.15)
        out['fixed_gate']={'max_score':.6,'margin':.15,'coverage':float(accepted.mean()),'accepted_n':int(accepted.sum()),'accepted_accuracy':float(accuracy_score(y[accepted],pred[accepted])) if accepted.any() else None,'note':'预先固定的诊断门槛，不是已证明的可靠概率阈值；不用于筛选全体F1样本。'}
    return out

def record_result(out,name,axis,y,pred,folds,probs=None,protocol='nested_group_cv',extra=None):
    labels=AXES[axis];m=metrics(y,pred,labels,probs);m.update(experiment=name,axis=axis,protocol=protocol,extra=extra or {})
    m['folds']=[dict(fold=int(f),**metrics(y[folds==f],pred[folds==f],labels,probs[folds==f] if probs is not None else None)) for f in sorted(set(folds))]
    m['fold_summary']={k:{'mean':float(np.mean([v[k] for v in m['folds']])),'std_sample':float(np.std([v[k] for v in m['folds']],ddof=1))} for k in ('accuracy','macro_f1')}
    save_json(out/f'{name}_{axis}_metrics.json',m)
    write_csv(out/f'{name}_{axis}_per_class.csv',[dict(label=l,**m['per_class'][l]) for l in labels])
    write_csv(out/f'{name}_{axis}_confusion.csv',[dict(actual=l,**dict(zip(labels,row))) for l,row in zip(labels,m['confusion_matrix'])])
    return m

def run_classical(rows,groups,basic,rich,out,config):
    results=[];allpred={};n=len(rows);seed=int(config['seed'])
    # Joint labels define the same outer partition for both axes and every experiment.
    joint=np.array([r['arousal']+'|'+r['valence'] for r in rows]);outer=split_groups(joint,groups,int(config['outer_folds']),seed)
    folds=np.zeros(n,dtype=int)
    for fold,(_,va) in enumerate(outer,1):folds[va]=fold
    write_csv(out/'fold_assignments.csv',[dict(audio_id=r['audio_id'],group=int(groups[i]),fold=int(folds[i]),arousal=r['arousal'],valence=r['valence']) for i,r in enumerate(rows)])
    save_json(out/'split_protocol.json',{'seed':seed,'outer_folds':len(outer),'inner_folds':config['inner_folds'],'same_outer_folds_for_all':True,'official_test_used':False})
    model_dir=out/'fold_models';model_dir.mkdir()
    for axis,labels in AXES.items():
        y=np.array([r[axis] for r in rows])
        experiments=[('DUMMY',basic),('E2_MFCC13_LR',basic),('E2_RICH_LR',rich),('E4_RICH_TUNED',rich)]
        if config.get('run_calibration',True):experiments.append(('E5_CALIBRATED',rich))
        predictions={name:np.full(n,'',dtype=object) for name,_ in experiments}
        probabilities={name:np.zeros((n,3)) for name,_ in experiments if name!='E4_RICH_TUNED'}
        selected=[];started=time.time()
        for fold,(tr,va) in enumerate(outer,1):
            if set(y[tr])!=set(labels):raise ValueError(axis+' 外层训练折缺类')
            inner=split_groups(y[tr],groups[tr],int(config['inner_folds']),seed+fold)
            for name,x in experiments[:3]:
                model=DummyClassifier(strategy='most_frequent') if name=='DUMMY' else make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=seed))
                model.fit(x[tr],y[tr]);predictions[name][va]=model.predict(x[va]);probabilities[name][va]=align_probs(model,x[va],labels)
            print(f'[TRAIN] {axis} fold={fold}/{len(outer)}：内层调参（验证折不参与选择）',flush=True)
            from sklearn.pipeline import Pipeline
            pipe=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=2000,random_state=seed))])
            grid=[{'clf':[LogisticRegression(max_iter=2000,random_state=seed)],'clf__C':[.1,1.,10.],'clf__class_weight':[None,'balanced']},
                  {'clf':[SVC(kernel='rbf')],'clf__C':[.1,1.,10.],'clf__gamma':['scale',.01,.1],'clf__class_weight':[None,'balanced']}]
            if config.get('_test_small_grid'):grid=[{'clf':[LogisticRegression(max_iter=500)],'clf__C':[1.]}]
            def macro_scorer(estimator,x_valid,y_valid):
                return f1_score(y_valid,estimator.predict(x_valid),labels=labels,average='macro',zero_division=0)
            search=GridSearchCV(pipe,grid,scoring=macro_scorer,cv=inner,n_jobs=int(config['n_jobs']),error_score='raise',refit=True)
            search.fit(rich[tr],y[tr]);best=search.best_estimator_;predictions['E4_RICH_TUNED'][va]=best.predict(rich[va])
            params={k:str(v) if k=='clf' else v for k,v in search.best_params_.items()};selected.append({'fold':fold,'params':params,'inner_macro_f1':float(search.best_score_),'train_n':len(tr),'valid_n':len(va)})
            joblib.dump(best,model_dir/f'{axis}_fold{fold}_tuned.joblib')
            if config.get('run_calibration',True):
                calibrated=CalibratedClassifierCV(clone(best),method='sigmoid',cv=inner,ensemble=True,n_jobs=int(config['n_jobs']))
                calibrated.fit(rich[tr],y[tr]);predictions['E5_CALIBRATED'][va]=calibrated.predict(rich[va]);probabilities['E5_CALIBRATED'][va]=align_probs(calibrated,rich[va],labels)
                joblib.dump(calibrated,model_dir/f'{axis}_fold{fold}_calibrated.joblib')
            save_json(out/f'{axis}_selected_parameters.json',selected)
        for name,_ in experiments:
            pred=predictions[name]
            if any(v not in labels for v in pred):raise ValueError('OOF 输出不完整')
            p=probabilities.get(name);m=record_result(out,name,axis,y,pred,folds,p,extra={'selection_inside_outer_train':True,'family_selection_included_in_E4':True,'axis_total_seconds':time.time()-started})
            results.append(m);allpred[(name,axis)]=pred
            export=[]
            for i,r in enumerate(rows):
                item=dict(audio_id=r['audio_id'],fold=int(folds[i]),actual=y[i],predicted=pred[i])
                if p is not None:item.update({('score_'+l):float(p[i,j]) for j,l in enumerate(labels)})
                export.append(item)
            write_csv(out/f'{name}_{axis}_predictions.csv',export)
            print(f'[RESULT] {name} {axis}: Macro-F1={m["macro_f1"]:.4f} Accuracy={m["accuracy"]:.4f}',flush=True)
    return results,folds,allpred
