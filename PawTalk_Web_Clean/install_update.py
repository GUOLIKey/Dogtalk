"""Merge verified program files only, backing up replaced code; never touch private runtime assets."""
import argparse,hashlib,json,os,shutil,tempfile,uuid
from datetime import datetime
from pathlib import Path
PROTECTED={'data','models','state','.venv','.venv_emotion','.venv_emotion_e4','.hf_cache','DEEPSEEK_CONFIG.ini'}
def install(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if source==target:raise ValueError('请解压到临时文件夹，再选择原应用文件夹。')
    if not (target/'server.py').is_file() or not (target/'00_START_APP.bat').is_file():raise ValueError('请选择原来的 PawTalk_Web_Clean 程序文件夹，里面应有 server.py 和 00_START_APP.bat。')
    manifest=json.loads((source/'build_manifest.json').read_text(encoding='utf-8'))
    paths=list(manifest['sha256'])+['build_manifest.json']
    for name in paths:
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] in PROTECTED or (target/relative).is_symlink():raise ValueError('更新清单包含受保护或非法路径。')
        if not (source/relative).is_file():raise ValueError('更新程序文件缺失：'+name)
        if name!='build_manifest.json' and hashlib.sha256((source/relative).read_bytes()).hexdigest()!=manifest['sha256'][name]:raise ValueError('更新包内容不完整：'+name)
    backup=target/'program_backups'/('before_emotion_e4_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    backup.mkdir(parents=True);changed=[]
    try:
        for name in paths:
            dst=target/name;old=backup/name;existed=dst.exists()
            if existed:old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,old)
            dst.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dst.parent,delete=False) as tmp:
                tmp.write((source/name).read_bytes());temporary=Path(tmp.name)
            try:os.replace(temporary,dst)
            finally:temporary.unlink(missing_ok=True)
            changed.append((dst,old,existed))
        for name,expected in manifest['sha256'].items():
            if hashlib.sha256((target/name).read_bytes()).hexdigest()!=expected:raise ValueError('安装后校验失败：'+name)
    except Exception:
        for dst,old,existed in reversed(changed):
            if existed:shutil.copy2(old,dst)
            else:dst.unlink(missing_ok=True)
        raise
    return backup

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target');args=parser.parse_args()
    target=args.target
    if not target:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        target=filedialog.askdirectory(title='选择原来的 PawTalk_Web_Clean 文件夹（已关闭旧运行窗口）')
        root.destroy()
    if not target:print('未选择文件夹，没有修改。');return
    backup=install(Path(__file__).resolve().parent,target)
    print('更新完成：PawTalk Web Six Scenes + Emotion E4')
    print('程序备份：',backup)
    print('密钥、模型、数据与历史记录均保留。先在原文件夹运行08启用E4，再运行00。')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('UPDATE FAILED:',exc);raise SystemExit(1)
