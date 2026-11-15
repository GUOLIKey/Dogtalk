"""Validate six-class OOF; never relabel the eight-class cache."""
from pathlib import Path
from scene_cache import load_index

def main():
    path=Path(__file__).parent/'data'/'scene_six_v1'/'audio_index.json'
    if not path.is_file():
        raise FileNotFoundError('六分类 OOF 尚未生成。请按 README_SIX_SCENES.md 运行 scene_training.py train --install。旧八类 audio_index.json 保留但不使用。')
    data=load_index(path)
    print(f"READY: {len(data['items'])} six-class OOF recordings; label_version={data['label_version']}")

if __name__=='__main__':
    try:main()
    except Exception as exc:print('[NOT READY] '+str(exc));raise SystemExit(1)
