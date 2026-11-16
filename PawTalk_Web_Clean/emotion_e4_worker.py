"""Isolated E4 inference. Full-clip rich153 features, two frozen refit SVMs."""
import argparse,hashlib,json,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
HERE=Path(__file__).resolve().parent
AXES={'arousal':('Low','Medium','High'),'valence':('Negative','Neutral','Positive')}
NAMES={'Low':'低','Medium':'中','High':'高','Negative':'负向','Neutral':'中性','Positive':'正向'}
MODEL_VERSION='emotion-e4-rich153-svm-v1'
SCORE_KIND='svm_decision_function_ovr'
USABILITY_POLICY='e4-quadrant-gate-unvalidated-v1'

def verify_assets(directory):
    spec=json.loads((HERE/'emotion_e4_spec.json').read_text(encoding='utf-8'))
    if spec.get('model_version')!=MODEL_VERSION:raise ValueError('E4模型版本不匹配。')
    for name,expected in spec['files'].items():
        path=Path(directory)/name
        if Path(name).name!=name or not path.is_file() or path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('E4模型文件校验失败：'+name+'，请运行08准备。')
    return spec

def load_models(directory):
    spec=verify_assets(directory)
    import joblib,sklearn,numpy,librosa,soundfile
    from sklearn.svm import SVC
    from emotion_e4_features import RECIPE
    meta=json.loads((Path(directory)/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('recipe')!=RECIPE or meta.get('feature_dimensions')!=153 or meta.get('model_version')!=MODEL_VERSION:raise ValueError('E4特征/模型元信息不匹配。')
    runtime_versions={'sklearn':sklearn.__version__,'numpy':numpy.__version__,'librosa':librosa.__version__,'soundfile':soundfile.__version__}
    if runtime_versions!=meta['versions']:raise ValueError('E4数值/特征库版本不匹配，请运行08。')
    if hashlib.sha256((HERE/'emotion_e4_features.py').read_bytes()).hexdigest()!=meta['feature_source_sha256']:raise ValueError('E4特征代码与训练时不一致。')
    models={}
    for axis,labels in AXES.items():
        m=joblib.load(Path(directory)/(axis+'.joblib'))
        if not isinstance(m.named_steps.get('clf'),SVC) or m.n_features_in_!=153 or set(m.classes_)!=set(labels) or list(m.classes_)!=meta['axes'][axis]['classes_']:raise ValueError('E4分类器维度或classes_错误。')
        models[axis]=m
    return spec,meta,models

def axis_result(model,vector,labels):
    import numpy as np
    raw=np.asarray(model.decision_function(vector),float)
    actual_classes=list(model.classes_)
    if raw.shape!=(1,3) or not np.isfinite(raw).all() or set(actual_classes)!=set(labels):raise ValueError('E4决策分数/类别无效。')
    values={l:float(raw[0,actual_classes.index(l)]) for l in labels}
    predicted=str(model.predict(vector)[0])
    if predicted not in labels:raise ValueError('E4预测类别无效。')
    ranking=sorted(labels,key=values.get,reverse=True)
    return {'label':predicted,'name':NAMES[predicted],'display':NAMES[predicted]+'（模型预测，待核查）','scores':values,'classes_order':actual_classes,
            'score_kind':SCORE_KIND,'score_ranking':ranking,'top_score_label':ranking[0],
            'original_preferred':{'label':predicted,'score':values[predicted]},'preferred_matches_score_max':predicted==ranking[0],
            'margin':values[ranking[0]]-values[ranking[1]],'margin_kind':'top_two_decision_score_gap','agreement':None,
            'usable':False,'usability_policy':USABILITY_POLICY,'uncertainty_reasons':['uncalibrated_decision_scores','quadrant_usability_not_validated'],
            'score_explanation':'SVM原始决策分数，不是概率或正确率；predict首选可能与分数最大项不同。',
            'usability_note':'模型预测已完成；尚无经验证的四象限可靠性门槛，因此保留待核查线索，不自动确定情绪象限。'}

def predict(path,directory):
    from emotion_e4_features import read_features
    from live_audio import decoded_audio
    spec,meta,models=load_models(directory)
    with decoded_audio(Path(path).read_bytes()) as (decoded,duration):vector,_=read_features(decoded)
    x=vector.reshape(1,-1)
    return {'available':True,'model':'E4 rich153 + RBF SVM','model_version':MODEL_VERSION,'model_commit':None,'recipe':meta['recipe'],
            'duration':duration,'arousal':axis_result(models['arousal'],x,AXES['arousal']),'valence':axis_result(models['valence'],x,AXES['valence']),
            'chunks':[],'aggregation':'whole_clip_statistics','device':'cpu','artifact_sha256':{a:spec['files'][a+'.joblib']['sha256'] for a in AXES},
            'message':'E4整段153维特征＋RBF SVM。完整三类输出为未校准决策分数，不是概率；四象限可靠性门槛尚未验证，DeepSeek仍可读取两轴全部线索进行观察核查。'}

def self_check(directory):
    import numpy as np
    from emotion_e4_features import extract_vector
    spec,meta,models=load_models(directory)
    wave=(.1*np.sin(np.arange(22050)*2*np.pi*440/22050)).astype('float32');_,vector=extract_vector(wave)
    for axis,m in models.items():axis_result(m,vector.reshape(1,-1),AXES[axis])
    return {'ok':True,'recipe':meta['recipe'],'model_version':MODEL_VERSION,'message':'E4模型加载和合成信号推理通过；不代表新犬只上的准确率。'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--audio');p.add_argument('--output',required=True);p.add_argument('--check',action='store_true');args=p.parse_args()
    try:result=self_check(args.assets) if args.check else predict(args.audio,args.assets)
    except Exception as exc:
        import traceback
        traceback.print_exc();Path(args.output).write_text(json.dumps({'available':False,'error':str(exc)[:400]},ensure_ascii=False),encoding='utf-8');raise SystemExit(1)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
