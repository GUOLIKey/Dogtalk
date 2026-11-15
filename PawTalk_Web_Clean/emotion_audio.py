"""Standard-library bridge to the isolated DogBark environment."""
import json,os,subprocess,tempfile,threading
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'models'/'emotion_runtime.json'
ASSETS=HERE/'models'/'emotion'
LOCK=threading.Lock()

def status():
    try:
        config=json.loads(RUNTIME.read_text(encoding='utf-8'))
        expected=json.loads((HERE/'emotion_assets.json').read_text(encoding='utf-8'))
        ready=Path(config['python']).is_file() and config.get('recipe')==expected['recipe'] and all((ASSETS/name).is_file() for name in expected['files'])
    except (OSError,ValueError,KeyError,TypeError):ready=False
    return {'ready':ready,'message':'情绪模型已准备（实验）' if ready else '情绪模块待准备：运行 08_ENABLE_EMOTION.bat；综合意图需要两路均完成。'}

def disabled():return {'enabled':False,'available':False,'message':'本次未使用情绪模型。'}

def predict_bytes(raw):
    if not status()['ready']:return {'enabled':True,'available':False,'message':status()['message']}
    with LOCK,tempfile.TemporaryDirectory(prefix='pawtalk_emotion_') as td:
        directory=Path(td);source=directory/'audio.bin';out=directory/'result.json';source.write_bytes(raw)
        try:
            config=json.loads(RUNTIME.read_text(encoding='utf-8'))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONUTF8='1')
            proc=subprocess.run([config['python'],'-B',str(HERE/'emotion_worker.py'),'--assets',str(ASSETS),'--audio',str(source),'--output',str(out)],
                                cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            result=json.loads(out.read_text(encoding='utf-8')) if out.is_file() else {}
            if proc.returncode or not result.get('available'):
                (HERE/'EMOTION_ERROR.txt').write_text(proc.stderr.decode('utf-8',errors='replace')[-10000:],encoding='utf-8')
                return {'enabled':True,'available':False,'message':'情绪推理失败：'+str(result.get('error','子进程未返回有效结果'))[:300]+'。详情见 EMOTION_ERROR.txt，请运行 08 检查。'}
            if result.get('recipe')!=config['recipe']:raise ValueError('recipe mismatch')
            return dict(result,enabled=True)
        except subprocess.TimeoutExpired:
            return {'enabled':True,'available':False,'message':'情绪推理超时，已停止该进程；本次不能开始综合意图推断。'}
        except (OSError,ValueError,KeyError,TypeError):
            return {'enabled':True,'available':False,'message':'情绪环境配置或输出无效，请运行 08 检查；本次不能开始综合意图推断。'}

def llm_hint(result):
    """Compatibility helper now retains complete validated scores, including unusable axes."""
    from analysis_contract import emotion_result
    if not result or not result.get('available'):return None
    return emotion_result(result)
