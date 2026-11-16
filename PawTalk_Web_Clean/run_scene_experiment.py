"""One-click real DEV six-scene experiment. No emotion or LLM API calls."""
import argparse
import contextlib
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
import webbrowser
from scene_training import read_csv,path_in,write_json,export_existing,extract,train
from scene_results_view import render

HERE=Path(__file__).resolve().parent

@contextlib.contextmanager
def experiment_lock(path):
    """OS lock is released even after interruption; stale lock files are harmless."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if sys.platform=='win32':
            import msvcrt
            try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise RuntimeError('另一个场景训练正在运行，请等待它结束，不要重复双击。') from None
            try:yield
            finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:raise RuntimeError('另一个场景训练正在运行。') from None
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()

def check_resources(root,manifest,holdout):
    rows=read_csv(manifest)
    missing=[str(path_in(root,r['audio_path'])) for r in rows if not path_in(root,r['audio_path']).is_file()]
    problems=[]
    if missing:problems.append(f'缺少 {len(missing)} 条 DEV 音频。例如：{missing[0]}。需要原项目 data/extracted 目录，不能用 Git LFS 指针代替。')
    if not holdout.is_file():problems.append('缺少原锁定留出清单：'+str(holdout)+'。请从原实验项目恢复，不能重新随机划分代替。')
    packages={'numpy':'numpy','sklearn':'scikit-learn','joblib':'joblib','soundfile':'soundfile'}
    missing_dep=[v for k,v in packages.items() if importlib.util.find_spec(k) is None]
    if missing_dep:problems.append('当前 Python 缺少依赖：'+', '.join(missing_dep)+'。请使用之前能运行场景模型的 .venv。')
    if problems:raise RuntimeError('\n'.join(problems))
    return len(rows)

def run(a):
    root=Path(a.project_root).resolve();manifest=HERE/'dev_manifest.csv'
    holdout=Path(a.holdout_manifest).resolve() if a.holdout_manifest else root/'results'/'optimization_step1'/'splits'/'LOCKED_holdout_DO_NOT_TUNE.csv'
    base=HERE/'results'/'scene_six_v1';base.mkdir(parents=True,exist_ok=True)
    with experiment_lock(base/'training.lock'):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');log=base/('train_'+stamp+'.log')
        with log.open('w',encoding='utf-8') as f,contextlib.redirect_stdout(Tee(sys.stdout,f)),contextlib.redirect_stderr(Tee(sys.stderr,f)):
            try:
                print('PawTalk 六分类训练与 F1 评估；本次不调用 DeepSeek、不运行情绪模型。',flush=True)
                print('项目目录：'+str(root));print('日志：'+str(log))
                count=check_resources(root,manifest,holdout);print(f'[1/4] 发现 {count} 条 DEV 音频及原留出清单。',flush=True)
                features=HERE/'data'/'scene_features_v1.npz'
                if a.fresh:features=HERE/'data'/f'scene_features_fresh_{stamp}.npz'
                feature_args=argparse.Namespace(manifest=manifest,project_root=root,output=features)
                if not a.fresh and features.is_file() and Path(str(features)+'.meta.json').is_file():
                    print('[2/4] 复用六类实验特征包；训练入口会检查版本、SHA256 和样本顺序。',flush=True)
                elif not a.fresh and all((root/'scripts'/name).is_file() for name in ('step4b1_denoise_abc.py','step3_svm_tuning.py')) and (HERE/'models'/'live_meta.json').is_file():
                    print('[2/4] 从原项目导出 CLAP/MFCC 特征缓存，核对原训练工厂。',flush=True)
                    try:export_existing(feature_args)
                    except Exception as exc:raise RuntimeError('原缓存导出失败：'+str(exc)+'\n请先把这份日志发给助手核查；不要自动忽略不一致。确认不能复用后可运行 11_TRAIN_FROM_AUDIO.bat 重新提取。') from exc
                else:
                    print('[2/4] 原缓存导出条件不足，将从 DEV 音频提取冻结 CLAP/MFCC 特征。首次下载/提取可能较久；不提供未经计时的时长保证。',flush=True)
                    extract(feature_args)
                output=base/('trained_run_'+stamp)
                print('[3/4] 检查指纹/分组，三折分别训练八类与六类；sklearn SVC 使用 CPU。每折可能几分钟无新输出，请勿重复启动。',flush=True)
                train(argparse.Namespace(manifest=manifest,project_root=root,holdout_manifest=holdout,features=features,output=output,install=not a.no_install))
                measured=json.loads((output/'metrics.json').read_text(encoding='utf-8'))
                if measured['provenance'].get('data_kind')!='real_dev':raise ValueError('一键入口只发布真实 DEV 报告；不能把合成训练标为真实完成。')
                report,summary=render(output)
                write_json(output/'RUN_COMPLETE.json',{'completed_at':time.time(),'data_kind':'real_dev','installed':not a.no_install,'metrics_file':'metrics.json'})
                print('[4/4] 训练和报告已完成。\n'+summary,flush=True)
                print('网页报告：'+str(report));print('需要继续使用 App 时再运行 00_START_APP.bat。')
                if not a.no_browser:webbrowser.open(report.resolve().as_uri())
                return output
            except Exception as exc:
                print('\n[STOPPED] '+str(exc),flush=True)
                print('没有可冒充新六类成绩的结果。请发送本窗口错误或本次日志；不要发送密钥。',flush=True)
                raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',default=str(HERE.parent));p.add_argument('--holdout-manifest');p.add_argument('--fresh',action='store_true');p.add_argument('--no-install',action='store_true');p.add_argument('--no-browser',action='store_true')
    from start_app import verify_install
    verify_install()
    run(p.parse_args())

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print("[FAILED] "+str(exc),flush=True)
        raise SystemExit(1)
