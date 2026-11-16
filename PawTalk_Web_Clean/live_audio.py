"""Local new-audio inference using the original frozen CLAP/MFCC feature recipe."""
import contextlib,csv,hashlib,io,json,os,subprocess,sys,tempfile,threading
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
MODELS=HERE/'models'
from scene_schema import LABEL_VERSION, MODEL_VERSION, validate_metadata, validate_classes, score_dict
META=MODELS/'scene_six_v1'/'live_meta.json'
MODEL=MODELS/'scene_six_v1'/'live_svm.joblib'
MODEL_ID='laion/clap-htsat-unfused'
FEATURE_VERSION='mfcc30_full_clap512_first10s_raw_v1'
MAX_BYTES=64*1024*1024
MAX_SECONDS=60
MIN_SECONDS=0.4
_lock=threading.RLock()
_runtime=None

def status():
    ready=MODEL.is_file() and META.is_file()
    error=''
    if ready:
        try:validate_metadata(json.loads(META.read_text(encoding='utf-8')))
        except (ValueError,KeyError,TypeError) as exc:ready=False;error=str(exc)
    return {'ready':ready,'loaded':_runtime is not None,'device':getattr(_runtime,'device',None),'max_seconds':MAX_SECONDS,'max_mb':64,'label_version':LABEL_VERSION,'model_version':MODEL_VERSION,'message':'六分类新录音识别已准备' if ready else error or '六分类模型缺失：请先运行 scene_training.py train；旧八类模型不会替代' }

def media_format(raw):
    if raw.startswith(b'RIFF') and raw[8:12]==b'WAVE':return 'wav'
    if raw.startswith(b'fLaC'):return 'flac'
    if raw.startswith(b'OggS'):return 'ogg'
    if raw.startswith(b'\x1aE\xdf\xa3'):return 'matroska'
    if len(raw)>=12 and raw[4:8]==b'ftyp':return 'mov'
    if raw.startswith(b'FORM') and raw[8:12] in (b'AIFF',b'AIFC'):return 'aiff'
    if raw.startswith(b'#!AMR'):return 'amr'
    if raw.startswith(b'.snd'):return 'au'
    if raw.startswith(b'ID3'):return 'mp3'
    if len(raw)>2 and raw[0]==255 and raw[1]&0xF6==0xF0:return 'aac'
    if len(raw)>2 and raw[0]==255 and raw[1]&0xE0==0xE0:return 'mp3'
    raise ValueError('无法识别这种录音格式。请上传 WAV、MP3、M4A/AAC、FLAC、OGG、WebM、AIFF 或 AMR。')

def inspect_audio(path):
    import soundfile as sf
    info=sf.info(str(path))
    duration=info.frames/info.samplerate if info.samplerate else 0
    if not MIN_SECONDS<=duration<=MAX_SECONDS:raise ValueError(f'新录音请控制在 {MIN_SECONDS} 到 {MAX_SECONDS} 秒之间，建议 3 到 10 秒。')
    if not 8000<=info.samplerate<=192000 or not 1<=info.channels<=8:raise ValueError('录音采样率或声道数不支持，请转成普通单声道或立体声 WAV。')
    return duration

@contextlib.contextmanager
def decoded_audio(raw):
    if not raw or len(raw)>MAX_BYTES:raise ValueError('录音为空或超过 64 MB。')
    fmt=media_format(raw[:64])
    with tempfile.TemporaryDirectory(prefix='pawtalk_audio_') as directory:
        source=Path(directory)/('input.'+fmt);source.write_bytes(raw)
        import soundfile as sf
        try:sf.info(str(source));native=True
        except RuntimeError:native=False
        path=source
        if not native:
            try:
                from imageio_ffmpeg import get_ffmpeg_exe
                executable=get_ffmpeg_exe()
            except Exception:raise ValueError('当前环境缺少录音解码器，请先运行 07_ENABLE_NEW_AUDIO.bat。') from None
            path=Path(directory)/'decoded.wav'
            # Forced demuxer + protocol whitelist excludes playlists/remote URLs.
            command=[executable,'-nostdin','-hide_banner','-loglevel','error','-protocol_whitelist','file,pipe','-f',fmt,'-i',str(source),'-map','0:a:0','-vn','-t',str(MAX_SECONDS+0.1),'-ac','1','-ar','48000','-c:a','pcm_f32le','-y',str(path)]
            try:result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            except (OSError,subprocess.TimeoutExpired):raise ValueError('录音解码失败或超时，请换成 WAV 再试。') from None
            if result.returncode or not path.exists():raise ValueError('文件无法解码，可能损坏、加密或没有音轨。请换成 WAV 再试。')
        duration=inspect_audio(path)
        yield path,duration

def feature_waves(path):
    import librosa,numpy as np
    # Same loading/resampling defaults as baseline_mfcc.vector and clap_supervised.
    mfcc_wave,sr=librosa.load(str(path),sr=16000,mono=True)
    clap_wave,_=librosa.load(str(path),sr=48000,mono=True,duration=10.0)
    if len(mfcc_wave)<400 or len(clap_wave)<400 or not np.isfinite(mfcc_wave).all() or not np.isfinite(clap_wave).all():raise ValueError('录音过短或包含无效采样。')
    if float(np.sqrt(np.mean(mfcc_wave.astype(np.float64)**2)))<1e-5:raise ValueError('录音接近静音，请靠近小狗重新录制。')
    m=librosa.feature.mfcc(y=mfcc_wave,sr=sr,n_mfcc=13)
    rms=librosa.feature.rms(y=mfcc_wave)[0];zcr=librosa.feature.zero_crossing_rate(y=mfcc_wave)[0]
    vec=np.concatenate([m.mean(axis=1),m.std(axis=1),[rms.mean(),rms.std(),zcr.mean(),zcr.std()]]).astype(np.float32)
    if vec.shape!=(30,) or not np.isfinite(vec).all():raise ValueError('MFCC 特征提取失败。')
    return vec,clap_wave

class Encoder:
    def __init__(self,allow_download=False):
        os.environ.setdefault('HF_HOME',str(PROJECT/'.hf_cache'))
        os.environ.setdefault('HF_HUB_CACHE',str(PROJECT/'.hf_cache'/'hub'))
        import torch
        from transformers import ClapModel,ClapProcessor
        self.torch=torch
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        kwargs={'cache_dir':str(PROJECT/'.hf_cache'/'hub'),'local_files_only':not allow_download}
        self.processor=ClapProcessor.from_pretrained(MODEL_ID,**kwargs)
        self.model=ClapModel.from_pretrained(MODEL_ID,**kwargs).eval().to(self.device)
    def vector(self,wave):
        import numpy as np
        try:inputs=self.processor(audio=[wave],sampling_rate=48000,return_tensors='pt',padding=True)
        except (TypeError,ValueError):inputs=self.processor(audios=[wave],sampling_rate=48000,return_tensors='pt',padding=True)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        with self.torch.inference_mode():
            value=self.model.get_audio_features(**inputs)
            if not isinstance(value,self.torch.Tensor):value=value.pooler_output if hasattr(value,'pooler_output') else value[0]
            vector=value.float().cpu().numpy().reshape(-1)
        if vector.shape!=(512,) or not np.isfinite(vector).all():raise ValueError('CLAP 特征维度异常。请重新运行 07 检查模型版本。')
        return vector.astype(np.float32)

class Runtime:
    def __init__(self):
        import joblib,sklearn,librosa,transformers
        from engine import LABELS
        if not status()['ready']:raise ValueError(status()['message'])
        self.meta=validate_metadata(json.loads(META.read_text(encoding='utf-8')))
        if self.meta.get('feature_version')!=FEATURE_VERSION or self.meta.get('sklearn_version')!=sklearn.__version__:raise ValueError('实时模型或 Python 环境版本已改变，请重新运行 07。')
        if self.meta.get('librosa_version')!=librosa.__version__ or self.meta.get('transformers_version')!=transformers.__version__:raise ValueError('音频特征提取库版本已改变，请重新运行 07 核对特征一致性。')
        if hashlib.sha256(MODEL.read_bytes()).hexdigest()!=self.meta.get('model_sha256'):raise ValueError('实时分类器校验失败，请重新运行 07。')
        self.classifier=joblib.load(MODEL)
        classes=validate_classes(self.classifier.classes_)
        if classes!=self.meta['classes_'] or self.classifier.n_features_in_!=542:raise ValueError('实时分类器类别顺序或维度与元数据不匹配。')
        try:self.encoder=Encoder()
        except Exception as exc:raise ValueError('CLAP 未能加载。请先运行 07；若提示 CUDA 问题，请查看命令窗口。原因：'+str(exc)[:250]) from None
        self.device=self.encoder.device
    def predict_path(self,path,duration):
        import numpy as np
        mfcc,wave=feature_waves(path)
        clap=self.encoder.vector(wave)
        x=np.concatenate([clap,mfcc]).reshape(1,-1)
        pred=str(self.classifier.predict(x)[0]);scores=self.classifier.decision_function(x)[0]
        return {'actual':None,'predicted':pred,'scores':score_dict(self.classifier.classes_,scores),'duration':duration,'prediction_mode':'live','device':self.device,'model_metadata':{'classes_':list(map(str,self.classifier.classes_)),**{k:v for k,v in getattr(self,'meta',{}).items() if k in ('feature_version','clap_model','model_sha256','sklearn_version','librosa_version','transformers_version','torch_version','train_count','C','gamma','created','label_version','label_config_sha256','model_version','score_kind','split_id')}},'note':'新录音实时推理，没有标准标签；CLAP 使用前 10 秒，MFCC 使用整段。该模型没有独立的非狗叫检测能力。'}

class DatasetFilenameError(ValueError):
    pass


def predict_bytes(raw,filename='audio'):
    global _runtime
    if not status()['ready']:raise ValueError(status()['message'])
    with _lock:
        basename=Path(filename.replace('\\','/')).name.casefold()
        # Check file identity BEFORE loading CLAP/torch; this guard does not need model weights in memory.
        metadata = json.loads(META.read_text(encoding='utf-8'))
        if basename in set(metadata.get('dataset_filenames',[])):
            raise DatasetFilenameError('这是数据集中的文件名，但未匹配 DEV 验证索引。请使用原始 DEV 文件；不要用留出集调试网页。若确为家庭新录音，请使用自己的文件名。')
        if _runtime is None:_runtime=Runtime()
        with decoded_audio(raw) as (path,duration):return _runtime.predict_path(path,duration)
