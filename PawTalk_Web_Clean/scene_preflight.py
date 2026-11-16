"""Read-only local resource inventory; does not decode holdout or contact an API."""
import argparse
import importlib.util
import json
from pathlib import Path
from scene_training import read_csv, path_in, write_json
from scene_schema import LABEL_VERSION, MODEL_VERSION

HERE=Path(__file__).resolve().parent

def inspect(root,features=None,holdout=None):
    root=Path(root);rows=read_csv(HERE/'dev_manifest.csv')
    missing=[r['audio_path'] for r in rows if not path_in(root,r['audio_path']).is_file()]
    pointers=[]
    for p in (root/'datasets').rglob('*') if (root/'datasets').is_dir() else []:
        if p.is_file() and p.stat().st_size<1024:
            if p.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):pointers.append(str(p.relative_to(root)))
    return {'label_version':LABEL_VERSION,'model_version':MODEL_VERSION,'dev_count':len(rows),'dev_audio_missing':len(missing),'dev_missing_examples':missing[:3],
        'lfs_pointer_count':len(pointers),'lfs_pointer_examples':pointers[:3],
        'feature_package_available':bool(features and Path(features).is_file()),
        'locked_holdout_manifest_available':bool(holdout and Path(holdout).is_file()),
        'six_model_available':(HERE/'models'/'scene_six_v1'/'live_svm.joblib').is_file(),
        'emotion_runtime_available':(HERE/'models'/'emotion_runtime.json').is_file(),
        'deepseek_config_file_available':(HERE/'DEEPSEEK_CONFIG.ini').is_file(),
        'dependencies':{m:importlib.util.find_spec(m) is not None for m in ['numpy','sklearn','joblib','soundfile','librosa','torch','transformers']},
        'training_executed':False,'api_called':False,'holdout_decoded_or_scored':False,
        'note':'存在 LFS 指针不等于音频已下载；情绪数据不能替代八类活动训练数据。此检查不验证权重/API 的运行有效性。'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project-root',default=str(HERE.parent));p.add_argument('--features');p.add_argument('--holdout-manifest');p.add_argument('--output',required=True)
    a=p.parse_args();r=inspect(a.project_root,a.features,a.holdout_manifest);write_json(a.output,r);print(json.dumps(r,ensure_ascii=False,indent=2))
