"""Six-scene OOF cache validation. Never accepts legacy eight-class caches."""
import json
import re
from pathlib import Path
from scene_schema import CACHE_VERSION, LABEL_VERSION, CONFIG_SHA256, MODEL_VERSION, LABELS, validate_metadata, target, score_dict

def load_index(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if (data.get('version')!=CACHE_VERSION or data.get('label_version')!=LABEL_VERSION
        or data.get('label_config_sha256')!=CONFIG_SHA256 or data.get('model_version')!=MODEL_VERSION):
        raise ValueError('OOF 缓存版本不匹配：请用六分类训练结果生成，不能复用旧八分类缓存。')
    for key in ('identity','split_id'):
        if not isinstance(data.get(key),str) or not re.fullmatch('[a-f0-9]{64}',data[key]):raise ValueError('缓存缺少可追溯标识。')
    if not isinstance(data.get('items'),dict):raise ValueError('缓存条目格式无效。')
    for sha,row in data['items'].items():
        if not re.fullmatch('[a-f0-9]{64}',sha):raise ValueError('缓存音频指纹无效。')
        meta=validate_metadata(row.get('model_metadata',{}))
        if meta.get('split_id')!=data['split_id']:raise ValueError('缓存交叉验证划分不匹配。')
        if row.get('prediction_mode')!='oof' or row.get('fold') not in (1,2,3):raise ValueError('缓存必须为三折 OOF 预测。')
        if row.get('predicted') not in LABELS or set(row.get('scores',{}))!=set(LABELS):raise ValueError('缓存没有完整六类结果。')
        score_dict(meta['classes_'],[row['scores'][k] for k in meta['classes_']])
        if target(row.get('actual_original'))!=row.get('target_six'):raise ValueError('原始标签与六类目标不对应。')
    return data
