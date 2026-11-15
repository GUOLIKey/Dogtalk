"""Whitelist model outputs before they can enter any normal LLM prompt."""
import math
from scene_schema import LABELS, NAMES, LABEL_VERSION, CONFIG_SHA256, MODEL_VERSION, validate_metadata

AROUSAL = ('Low', 'Medium', 'High')
VALENCE = ('Negative', 'Neutral', 'Positive')
SCHEMA_VERSION = 'dual-analysis-six-v1'
MODEL_FIELDS = ('feature_version', 'clap_model', 'model_sha256', 'sklearn_version',
                'librosa_version', 'transformers_version', 'torch_version', 'train_count',
                'C', 'gamma', 'created', 'classes_', 'label_version', 'label_config_sha256', 'model_version', 'score_kind', 'split_id')


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('模型分数必须是有限数值。')
    return float(value)


def scores(value, labels):
    if not isinstance(value, dict) or set(value) != set(labels):
        raise ValueError('模型必须返回全部类别，且不能包含未知类别。')
    return {label: number(value[label]) for label in labels}


def context_result(row, cache_meta=None):
    values = scores(row.get('scores'), LABELS)
    preferred = row.get('predicted')
    if preferred not in LABELS:
        raise ValueError('场景原始首选类别无效。')
    mode = row.get('prediction_mode', 'oof')
    if mode not in ('live', 'oof'):
        raise ValueError('正常流程仅接受实时模型或 OOF 缓存预测。')
    metadata = row.get('model_metadata', {})
    metadata = {key: metadata[key] for key in MODEL_FIELDS if key in metadata}
    validate_metadata(metadata)
    classes = metadata.get('classes_')
    if classes is not None and (not isinstance(classes,list) or len(classes)!=6 or set(classes)!=set(LABELS)):
        raise ValueError('模型实际 classes_ 与业务标签不一致。')
    cache_meta = cache_meta or {}
    return {
        'available': True, 'status': 'ready', 'model': 'MFCC30 + CLAP512 + RBF SVM (six scenes)',
        'label_version': LABEL_VERSION, 'label_config_sha256':CONFIG_SHA256, 'model_version':MODEL_VERSION,
        'scores': values, 'labels': dict(NAMES), 'score_ranking':sorted(values,key=values.get,reverse=True), 'feature_limits':{'clap_seconds':10,'mfcc':'whole_clip'},
        'original_preferred': {'label': preferred, 'name': NAMES[preferred], 'score': values[preferred]},
        'score_kind': 'svm_decision_function_ovr',
        'score_explanation': 'SVC decision_function 原始决策分数，未经概率校准；不是概率或正确率。predict 首选可能与分数最大项不同。',
        'duration': number(row['duration']), 'source': 'realtime' if mode == 'live' else 'cached_oof',
        'device': row.get('device'), 'fold': row.get('fold') if mode == 'oof' else None,
        'model_metadata': metadata, 'classes_order': classes, 'classes_order_status': 'verified' if classes else 'unknown_historical_cache',
        'cache_version': cache_meta.get('version') if mode == 'oof' else None,
        'cache_identity': cache_meta.get('identity') if mode == 'oof' else None,
        'version_note': '元信息缺失的历史缓存保留 unknown，不补造模型版本。' if mode == 'oof' and not cache_meta else None,
    }


def emotion_result(raw):
    if raw.get('available') is not True:
        return {'available': False, 'status': 'unavailable', 'message': raw.get('message', '情绪分析不可用。')}
    result = {'available': True, 'status': 'ready', 'source': 'realtime',
              'model': raw.get('model'), 'model_commit': raw.get('model_commit'),
              'recipe': raw.get('recipe'), 'duration': number(raw['duration']),
              'aggregation': raw.get('aggregation'), 'device': raw.get('device'),
              'message': raw.get('message'), 'chunks': []}
    for axis, labels in [('arousal', AROUSAL), ('valence', VALENCE)]:
        item = raw.get(axis, {})
        values = scores(item.get('scores'), labels)
        if any(v < 0 or v > 1 for v in values.values()) or abs(sum(values.values()) - 1) > 1e-4:
            raise ValueError('情绪三类分数无效。')
        if item.get('label') not in labels or values[item['label']] != max(values.values()):
            raise ValueError('情绪最高类别与分数不一致。')
        if type(item.get('usable')) is not bool:
            raise ValueError('缺少情绪 usable 标记。')
        result[axis] = {k: item[k] for k in ('label', 'name', 'display', 'usable') if k in item}
        result[axis].update(scores=values, margin=number(item['margin']), agreement=number(item['agreement']),
                            score_kind='uncalibrated_model_score')
    for chunk in raw.get('chunks', []):
        result['chunks'].append({'start': number(chunk['start']), 'end': number(chunk['end']),
                                 'arousal': scores(chunk['arousal'], AROUSAL),
                                 'valence': scores(chunk['valence'], VALENCE)})
    return result


def bundle(context, emotion):
    return {'schema_version': SCHEMA_VERSION, 'context': context, 'emotion': emotion,
            'ready': bool(context.get('available') and emotion.get('available')),
            'interpretation': '两路均为同一录音的模型预测，不是已经观察到的事实；分数不跨模型相加或相乘。情绪 usable=false 仍保留全部原始分数，但不能当成可靠情绪结论。'}


def prompt_analysis(analysis):
    """Rebuild at every call, not a recursive blacklist of possibly sensitive fields."""
    if not analysis.get('ready'):
        raise ValueError('两路分析尚未完成，不能生成综合意图。')
    c = analysis['context']
    row = {'scores': c['scores'], 'predicted': c['original_preferred']['label'], 'duration': c['duration'],
           'prediction_mode': 'live' if c['source'] == 'realtime' else 'oof',
           'device': c.get('device'), 'fold': c.get('fold'), 'model_metadata': c.get('model_metadata', {})}
    clean_context = context_result(row, {'version': c.get('cache_version'), 'identity': c.get('cache_identity')})
    return bundle(clean_context, emotion_result(analysis['emotion']))
