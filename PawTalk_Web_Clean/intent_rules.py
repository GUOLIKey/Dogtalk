"""Deterministic, versioned hypothesis routing. Never a learned intent classifier."""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
from scene_schema import SCENES as SCENE_ROWS, LABEL_VERSION
from analysis_contract import validate_emotion_axis
SCENES={r['id']:r for r in SCENE_ROWS};QUADRANTS=load('emotion_quadrants.json')['quadrants']
CONFIG=load('intent_link_rules.json');QUESTIONS=load('question_templates.json')
RULES={r['rule_id']:r for r in CONFIG['rules']}
VERSION=CONFIG['rules_version'];MAPPING_VERSION='quadrant-v0.2';USABLE_VERSION='dogbark-max0.60-margin0.15-agreement0.60-v1'

def reverse_index():
    return {f'I0{i}':{'primary':[r['rule_id'] for r in RULES.values() if r['primary_hint_id']==f'I0{i}'],
                     'pool':[r['rule_id'] for r in RULES.values() if f'I0{i}' in r['candidate_pool']]} for i in range(1,7)}

def validate_config():
    expected={f'S{a:02d}-E{e}' for a in range(1,7) for e in range(1,5)}
    if CONFIG.get('label_version')!=LABEL_VERSION or len(CONFIG['rules'])!=24 or set(RULES)!=expected:raise ValueError('24 条六场景规则缺失、重复或版本非法。')
    ids={i['intent_id'] for i in load('intent_catalog.json')['intents']}
    for key,r in RULES.items():
        if key!=r['scene_id']+'-'+r['emotion_id'] or r['observation_required'] is not True:raise ValueError('规则入口配置不一致。')
        pool=r['candidate_pool']
        if len(pool)!=len(set(pool)) or not set(pool)<=ids or r['question_id'] not in QUESTIONS['templates']:raise ValueError('规则引用非法。')
        if r['primary_hint_id'] is not None and r['primary_hint_id'] not in pool:raise ValueError('优先项不在池中。')
    if load('intent_rule_reverse_index.json')['intents']!=reverse_index():raise ValueError('反向索引与规则不一致。')
validate_config()

def emotion_summary(emotion):
    out={'status':'unavailable','quadrant_id':None,'name':'情绪不可用','axis_labels':{},'central_axes':[],
         'reason_codes':[],'mapping_version':MAPPING_VERSION,'usable_version':USABLE_VERSION}
    if emotion.get('available') is not True:out['reason_codes']=['model_unavailable'];return out
    tied=[];unusable=[]
    for axis,labels in (('arousal',{'Low','Medium','High'}),('valence',{'Negative','Neutral','Positive'})):
        try:item=validate_emotion_axis(emotion.get(axis,{}),tuple(sorted(labels)),emotion.get('model_version'))
        except (ValueError,KeyError,TypeError):out['reason_codes']=['invalid_'+axis];return out
        scores=item['scores']
        if item.get('usability_policy'):out['usable_version']=item['usability_policy']
        out['axis_labels'][axis]=item['label']
        if sum(abs(v-max(scores.values()))<1e-12 for v in scores.values())>1:tied.append(axis+'_tied_max')
        if not item['usable']:unusable.append(axis+'_not_usable')
    if tied or unusable:out.update(status='uncertain',name='情绪不确定',reason_codes=tied+unusable);return out
    out['central_axes']=[a for a,l in out['axis_labels'].items() if l in ('Medium','Neutral')]
    if out['central_axes']:out.update(status='central',name='中间状态（并非不确定）',reason_codes=['central_axis']);return out
    for ident,q in QUADRANTS.items():
        if all(out['axis_labels'][a]==q[a] for a in ('arousal','valence')):
            out.update(status='quadrant',quadrant_id=ident,name=q['name'],reason_codes=['both_axes_usable']);return out
    raise ValueError('象限映射配置错误。')

def rule_context(context,summary,mode='C'):
    out={'rules_version':VERSION,'anchor_scene':None,'primary_rule_id':None,'matched_rule_ids':[], 'matched_rules':[], 'candidate_pool':[], 'fallback_id':None}
    if mode=='A':out['fallback_id']='F-OBSERVATION-ONLY';return out
    if not context.get('available'):out['fallback_id']='F-UNAVAILABLE';return out
    by_label={v['label']:k for k,v in SCENES.items()};anchor=context.get('original_preferred',{}).get('label')
    if anchor not in by_label:raise ValueError('原始 classifier.predict 类别缺失或非法。')
    scores=context['scores']
    if set(scores)!=set(by_label):raise ValueError('场景分数不完整。')
    order=[anchor]+[k for k in sorted(scores,key=scores.get,reverse=True) if k!=anchor]
    out.update(anchor_scene=by_label[anchor],preferred_first_ranking=[by_label[k] for k in order],anchor_source='classifier.predict',prediction_source=context.get('source'))
    if mode=='B':out['fallback_id']='F-SCENE-ONLY';return out
    if summary['status']!='quadrant':out['fallback_id']='F-'+summary['status'].upper();return out
    for i,label in enumerate(order[:3]):
        r=RULES[by_label[label]+'-'+summary['quadrant_id']]
        out['matched_rules'].append(dict(r,role='primary' if i==0 else 'auxiliary',rank=i+1))
        out['matched_rule_ids'].append(r['rule_id'])
        for ident in r['candidate_pool']:
            if ident not in out['candidate_pool']:out['candidate_pool'].append(ident)
    out['primary_rule_id']=out['matched_rule_ids'][0]
    return out

def derive(analysis,mode='C'):
    summary=emotion_summary(analysis.get('emotion',{}))
    return {'emotion_summary':summary,'rule_context':rule_context(analysis.get('context',{}),summary,mode)}
