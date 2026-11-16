"""Prepare a dedicated CPU TensorFlow environment without touching CLAP/SVM."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
VENV=HERE/'.venv_emotion'
ASSETS=HERE/'models'/'emotion'
SPEC=json.loads((HERE/'emotion_assets.json').read_text(encoding='utf-8'))

def run(command):subprocess.run(command,cwd=HERE,check=True)

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='pawtalk_',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def python_for_setup():
    candidates=[[sys.executable],[str(HERE.parent/'.venv'/'Scripts'/'python.exe')],
                [str(Path(r'D:\PawTalk_DogBark_复现工具包')/'.venv'/'Scripts'/'python.exe')]]
    if os.name=='nt':candidates += [['py','-3.12'],['py','-3.11'],['py','-3.10']]
    else:candidates += [['python3.12'],['python3.11']]
    script='import sys,json; print(json.dumps([list(sys.version_info[:2]),getattr(sys,"_base_executable",sys.executable)]))'
    for candidate in candidates:
        try:
            value=json.loads(subprocess.check_output(candidate+['-c',script],stderr=subprocess.DEVNULL,timeout=15,text=True))
            if value[0][0]==3 and 10<=value[0][1]<=12 and Path(value[1]).is_file():return value[1]
        except (OSError,subprocess.SubprocessError,ValueError,IndexError):continue
    raise ValueError('未找到 Python 3.10–3.12。请安装 Python 3.12（64位）后重跑 08；不要更换原情境模型环境。')

def matches(path,expected):
    return path.is_file() and path.stat().st_size==expected['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']

def prepare_assets(extra=None):
    ASSETS.mkdir(parents=True,exist_ok=True)
    roots=[Path(extra)] if extra else []
    roots += [Path(r'D:\PawTalk_DogBark_复现工具包'),HERE.parent/'PawTalk_DogBark_复现工具包',HERE/'dogbark',HERE.parent/'dogbark']
    for name,expected in SPEC['files'].items():
        target=ASSETS/name
        if matches(target,expected):continue
        found=None
        for root in roots:
            # Only known project folders; do not scan the whole drive.
            for sub in ['', 'dogbark','dogbark-main','models']:
                candidate=root/sub/name
                if matches(candidate,expected):found=candidate;break
            if found:break
        if found:data=found.read_bytes();print('复用已核对的模型：'+name,flush=True)
        else:
            print('下载固定版本模型：'+name,flush=True)
            url='https://raw.githubusercontent.com/itzefyee/dogbark/'+SPEC['commit']+'/'+name
            with urllib.request.urlopen(url,timeout=90) as response:data=response.read(expected['bytes']+1)
        if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('模型文件校验不符：'+name)
        atomic(target,data)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models-dir');args=parser.parse_args()
    print('PawTalk v8 · 情绪模型独立环境准备',flush=True)
    base=python_for_setup();python=VENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():run([base,'-m','venv',str(VENV)])
    run([str(python),'-m','pip','install','-r',str(HERE/'emotion_requirements.txt')])
    prepare_assets(args.models_dir)
    with tempfile.TemporaryDirectory(prefix='pawtalk_emotion_check_') as td:
        out=Path(td)/'check.json'
        run([str(python),'-B',str(HERE/'emotion_worker.py'),'--check','--assets',str(ASSETS),'--output',str(out)])
        result=json.loads(out.read_text(encoding='utf-8'))
        if result.get('ok') is not True:raise ValueError('情绪模型推理自检失败。')
    atomic(HERE/'models'/'emotion_runtime.json',json.dumps({'python':str(python.absolute()),'recipe':SPEC['recipe'],'commit':SPEC['commit']},ensure_ascii=False,indent=2).encode('utf-8'))
    print('READY：情绪模型已启用。运行 00_START_APP.bat；原 CLAP/SVM 环境未修改。',flush=True)
    print('这只验证模型可运行，不代表情绪准确率验证。',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        (HERE/'EMOTION_SETUP_ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print('准备失败。请发送 EMOTION_SETUP_ERROR.txt，勿发送 API Key。原情境识别仍可使用。',flush=True)
        raise SystemExit(1)
