"""Isolated CPU inference. No API keys, Streamlit, Gemini or random demo outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import argparse,hashlib,json,io
from pathlib import Path

HERE=Path(__file__).resolve().parent
AROUSAL=['Low','Medium','High']
VALENCE=['Negative','Neutral','Positive']
NAMES={'Low':'低','Medium':'中','High':'高','Negative':'负向','Neutral':'中性','Positive':'正向'}

def verify_assets(directory):
    spec=json.loads((HERE/'emotion_assets.json').read_text(encoding='utf-8'))
    for name,expected in spec['files'].items():
        path=Path(directory)/name
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('情绪模型文件校验失败，请重新运行 08。')
    return spec

def load_models(directory):
    spec=verify_assets(directory)  # Verify all bytes before joblib deserialization.
    import joblib,tensorflow as tf
    scaler=joblib.load(Path(directory)/'scaler_sequential.pkl')
    if getattr(scaler,'n_features_in_',None)!=1:raise ValueError('情绪特征归一化器维度不符。')
    models=[tf.keras.models.load_model(Path(directory)/name,compile=False,safe_mode=True) for name in ['final_arousal_model.keras','final_valence_model.keras']]
    if any(tuple(m.input_shape)!=(None,25,1) or tuple(m.output_shape)!=(None,3) for m in models):raise ValueError('情绪模型输入输出维度不符。')
    return spec,scaler,models

def features(wave,scaler):
    """Replicate pinned index.py: two-second zero padding, PCM16 roundtrip, first 25 MFCC frames."""
    import librosa,numpy as np,soundfile as sf
    length=22050*2;arrays=[];spans=[]
    for start in range(0,len(wave),length):
        chunk=wave[start:start+length]
        end=start+len(chunk)
        if len(chunk)<length:chunk=np.concatenate([chunk,np.zeros(length-len(chunk))])
        wav=io.BytesIO();sf.write(wav,chunk,22050,format='WAV',subtype='PCM_16');wav.seek(0)
        quantized,_=librosa.load(wav,sr=22050)
        mfcc=librosa.feature.mfcc(y=quantized,sr=22050,n_mfcc=1,n_fft=2048,hop_length=512).T
        mfcc=np.pad(mfcc,((0,max(0,25-len(mfcc))),(0,0)))[:25]
        arrays.append(scaler.transform(mfcc));spans.append((start/22050,end/22050))
    return np.asarray(arrays,dtype=np.float32),spans

def vector_scores(value):
    import numpy as np
    a=np.asarray(value,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or (a<0).any() or (a>1).any() or not np.allclose(a.sum(axis=1),1,atol=1e-4):
        raise ValueError('情绪模型输出无效。')
    return a

def aggregate(scores,classes,spans):
    import numpy as np
    scores=vector_scores(scores)
    weights=np.array([end-start for start,end in spans])
    mean=np.average(scores,axis=0,weights=weights)
    winner=int(mean.argmax());ordered=np.sort(mean)
    margin=float(ordered[-1]-ordered[-2]);agreement=float(weights[np.argmax(scores,axis=1)==winner].sum()/weights.sum())
    # Conservative display gate, NOT a calibrated statistical confidence interval.
    usable=bool(mean[winner]>=0.60 and margin>=0.15 and agreement>=0.60)
    return {'label':classes[winner],'name':NAMES[classes[winner]],'scores':dict(zip(classes,map(float,mean))),
            'usable':usable,'display':NAMES[classes[winner]] if usable else '暂不确定',
            'margin':margin,'agreement':agreement,'score_kind':'uncalibrated_model_score'}

def predict(path,directory):
    import librosa,numpy as np
    from live_audio import decoded_audio
    # Same multi-format decoder as live audio, inside the separate environment.
    with decoded_audio(Path(path).read_bytes()) as (decoded,duration):
        wave,_=librosa.load(str(decoded),sr=22050,mono=True)
    if not np.isfinite(wave).all() or float(np.sqrt(np.mean(wave.astype(float)**2)))<1e-5:raise ValueError('录音接近静音或采样无效。')
    spec,scaler,models=load_models(directory)
    x,spans=features(wave,scaler)
    predictions=[vector_scores(m(x,training=False).numpy()) for m in models]
    chunks=[{'start':start,'end':end,'arousal':dict(zip(AROUSAL,map(float,predictions[0][i]))),'valence':dict(zip(VALENCE,map(float,predictions[1][i])))} for i,(start,end) in enumerate(spans)]
    return {'available':True,'model':'DogBark LSTM','model_commit':spec['commit'],'recipe':spec['recipe'],
            'duration':duration,'arousal':aggregate(predictions[0],AROUSAL,spans),'valence':aggregate(predictions[1],VALENCE,spans),
            'chunks':chunks,'aggregation':'duration_weighted_mean_v1','device':'cpu',
            'message':'实验性情绪线索；分数未经校准，不代表准确率。每两秒分段仅用开头约 0.6 秒特征，情绪标签未在本项目验证。'}

def self_check(directory):
    import numpy as np
    spec,scaler,models=load_models(directory)
    wave=(0.1*np.sin(np.arange(22050)*2*np.pi*440/22050)).astype(np.float32)
    x,_=features(wave,scaler)
    for m in models:vector_scores(m(x,training=False).numpy())
    return {'ok':True,'recipe':spec['recipe'],'message':'模型加载和合成信号推理通过；不代表狗叫情绪准确率。'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--assets',required=True);parser.add_argument('--audio');parser.add_argument('--output',required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    try:result=self_check(args.assets) if args.check else predict(args.audio,args.assets)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        Path(args.output).write_text(json.dumps({'available':False,'error':str(exc)[:300]},ensure_ascii=False),encoding='utf-8')
        raise SystemExit(1)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
