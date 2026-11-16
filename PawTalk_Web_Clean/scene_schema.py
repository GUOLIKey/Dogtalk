"""Single versioned scene vocabulary. Intent IDs and historical A IDs are separate."""
import hashlib
import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name('scene_labels.json')
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
LABEL_VERSION = CONFIG['version']
CONFIG_SHA256 = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
SCENES = CONFIG['scenes']
LABELS = [s['label'] for s in SCENES]
NAMES = {s['label']: s['name'] for s in SCENES}
IDS = {s['label']: s['id'] for s in SCENES}
ORIGINAL_TO_SIX = {old:s['label'] for s in SCENES for old in s['original_labels']}
ORIGINAL_LABELS = list(ORIGINAL_TO_SIX)
SCORE_KIND = CONFIG['score_kind']
MODEL_VERSION = 'mfcc-clap-svm-six-v1'
CACHE_VERSION = 'scene-six-oof-v1'

def target(original):
    if original not in ORIGINAL_TO_SIX:
        raise ValueError('未知原始八类标签：' + str(original))
    return ORIGINAL_TO_SIX[original]

def validate_classes(classes):
    classes = list(map(str, classes))
    if len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ValueError('必须是真正六分类模型，classes_ 与 scene-six-v1 不一致。')
    return classes

def score_dict(classes, values):
    classes = validate_classes(classes)
    values = list(values)
    if len(values) != len(classes):
        raise ValueError('决策分数数量与 classes_ 不一致。')
    if any(isinstance(v,bool) or not math.isfinite(float(v)) for v in values):
        raise ValueError('决策分数不是有限数值。')
    mapped = dict(zip(classes, map(float,values)))
    return {label:mapped[label] for label in LABELS}

def validate_metadata(meta):
    if (meta.get('label_version') != LABEL_VERSION or meta.get('label_config_sha256') != CONFIG_SHA256
            or meta.get('model_version') != MODEL_VERSION or meta.get('score_kind') != SCORE_KIND):
        raise ValueError('六分类模型/标签版本不匹配；旧八类模型或缓存不能用于当前推理。')
    validate_classes(meta.get('classes_',[]))
    return meta

def public_config():
    return dict(CONFIG, sha256=CONFIG_SHA256, model_version=MODEL_VERSION)
