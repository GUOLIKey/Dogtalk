"""Enable E4 in its own environment; leave scene and legacy DogBark assets intact."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
from setup_emotion_legacy import python_for_setup,atomic
HERE=Path(__file__).resolve().parent
VENV=HERE/'.venv_emotion_e4'
ASSETS=HERE/'emotion_e4_assets'

def main():
    from emotion_e4_worker import verify_assets
    spec=verify_assets(ASSETS) # Fail clearly before dependency installation if package/model is incomplete.
    base=python_for_setup();python=VENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():subprocess.run([base,'-m','venv',str(VENV)],cwd=HERE,check=True)
    req=HERE/'emotion_e4_requirements.txt';key=hashlib.sha256(req.read_bytes()).hexdigest();marker=VENV/'requirements.sha256'
    if not marker.is_file() or marker.read_text()!=key:
        subprocess.run([str(python),'-m','pip','install','-r',str(req)],cwd=HERE,check=True)
        marker.write_text(key)
    with tempfile.TemporaryDirectory(prefix='pawtalk_e4_setup_') as td:
        out=Path(td)/'check.json'
        subprocess.run([str(python),'-B',str(HERE/'emotion_e4_worker.py'),'--check','--assets',str(ASSETS),'--output',str(out)],cwd=HERE,check=True,env=dict(os.environ,PYTHONUTF8='1'))
        result=json.loads(out.read_text(encoding='utf-8'))
        if result.get('ok') is not True or result.get('model_version')!=spec['model_version']:raise ValueError('E4自检失败。')
    atomic(HERE/'models'/'emotion_e4_runtime.json',json.dumps({'python':str(python.absolute()),'recipe':spec['recipe'],'model_version':spec['model_version']},ensure_ascii=False,indent=2).encode('utf-8'))
    print('READY：E4情绪模型已启用。运行00；场景模型与旧DogBark权重没有修改。',flush=True)
    print('决策分数未经概率校准，未设置可声称可靠的四象限门槛。',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (HERE/'EMOTION_SETUP_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('E4准备失败。请发送EMOTION_SETUP_ERROR.txt，勿发送密钥。',flush=True);raise SystemExit(1)
