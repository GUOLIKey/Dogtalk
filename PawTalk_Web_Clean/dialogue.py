"""Versioned catalog interviews, explicit corrections and candidate rounds."""
import copy,json,re
import intent_rules as rules
import context_policy
from pathlib import Path
from analysis_contract import prompt_analysis,AROUSAL,VALENCE
from engine import LABELS,NAMES
from llm_client import chat_json
from legacy_intent_v1 import text,strings,unknown_answer as legacy_unknown_answer
def unknown_answer(answer):
    return legacy_unknown_answer(answer) or bool(re.search(r"未观察到|未看到|没有看到|没有提到|没提到",answer))
ROOT=Path(__file__).resolve().parent
CATALOG=json.loads((ROOT/'intent_catalog.json').read_text(encoding='utf-8'))
CATALOG_BY_ID={i['intent_id']:i for i in CATALOG['intents']}
BASE_PROMPT=(ROOT/'intent_system_v4.txt').read_text(encoding='utf-8')
PROMPT_VERSION='intent-interview-4.0'
SCHEMA_VERSION='intent-session-v3'
METHOD='catalog-intent-v3'
MAX_QUESTIONS=3
STATUSES={'possible','supported','excluded','subsumed'}

def input_analysis(analysis,mode='C'):
    clean=prompt_analysis(analysis)
    if mode=='A':return {'mode':'A','note':'仅使用独立观察和问答，不提供声音模型输出。'}
    if mode=='B':return {'mode':'B','context':clean['context'],'interpretation':clean['interpretation']}
    if mode!='C':raise ValueError('实验方案必须为 A/B/C。')
    return clean

def allowed_signals(analysis,mode):
    clean=input_analysis(analysis,mode)
    if mode=='A':return []
    items=[{'source':'context','label':label,'display_name':NAMES[label],'score':score} for label,score in clean['context']['scores'].items()]
    if mode=='C':
        for axis in ('arousal','valence'):
            items.extend({'source':axis,'label':label,'score':score,'usable':clean['emotion'][axis]['usable']} for label,score in clean['emotion'][axis]['scores'].items())
    return items


def canonical_signal(sig,mode):
    if not isinstance(sig,dict):raise ValueError('模型线索应为包含 source、label、explanation 的对象。')
    source=sig.get('source');label=sig.get('label')
    if not isinstance(source,str) or not isinstance(label,str):raise ValueError('模型线索 source/label 必须是文字。')
    source=source.strip().casefold();label=label.strip()
    aliases={'scene':'context','scene_result':'context','context_result':'context','scene_model':'context','场景':'context','情境':'context','场景模型':'context',
             'emotion.arousal':'arousal','emotion_result.arousal':'arousal','唤醒程度':'arousal','唤醒':'arousal',
             'emotion.valence':'valence','emotion_result.valence':'valence','正负倾向':'valence','效价':'valence'}
    source=aliases.get(source,source)
    labels={k.casefold():k for k in LABELS+list(AROUSAL)+list(VALENCE)}
    labels.update({v:k for k,v in NAMES.items()})
    labels.update({'低':'Low','低唤醒':'Low','中':'Medium','中等唤醒':'Medium','高':'High','高唤醒':'High',
                   '负向':'Negative','中性':'Neutral','正向':'Positive'})
    label=labels.get(label.casefold(),label)
    # A generic emotion branch can be resolved only when its label uniquely identifies an axis.
    if source in ('emotion','emotion_result','情绪'):
        if label in AROUSAL:source='arousal'
        elif label in VALENCE:source='valence'
    allowed={'context':LABELS,'arousal':AROUSAL,'valence':VALENCE}
    if source not in allowed or label not in allowed[source] or mode=='A' or (mode=='B' and source!='context'):
        raise ValueError(f'模型线索格式不符：source={source[:60]!r}, label={label[:80]!r}, 方案={mode}。请重试；标签必须取自本次提供的 allowed_model_signals，unknown 不是类别。')
    return {'source':source,'label':label,'explanation':text(sig.get('explanation'),'线索解释',500)}


def call(stage,analysis,model,extra, schema,mode='C'):
    clean=input_analysis(analysis,mode)
    derived=rules.derive(analysis,mode)
    payload={'stage':stage,'phase':'initial' if stage=='generate_intents' and not extra.get('history') else ('regenerate' if stage=='generate_intents' else 'update'),
             'intent_catalog':CATALOG,'intent_catalog_version':CATALOG['version'],'prompt_version':PROMPT_VERSION,
             'rules_version':rules.VERSION,'question_templates':rules.QUESTIONS,'question_templates_version':rules.QUESTIONS['version'],
             'analysis':clean,'scene_result':clean.get('context'),'emotion_result':clean.get('emotion'),
             'emotion_summary':derived['emotion_summary'] if mode=='C' else None,'rule_context':derived['rule_context'],
             'allowed_model_signals':allowed_signals(analysis,mode),**extra,'output_schema':schema,'previous_contract_error':_REPAIR_FEEDBACK.get()}
    result=chat_json(model,[{'role':'system','content':BASE_PROMPT+'\n字段适配：intents 为初始 candidates；supporting_signals 为 model_signals，必须逐项复制 allowed_model_signals 的 source、label、score 并写 explanation。逐轮 assessments 中 evidence 使用 answer_id、quote、role、brief_reason。主要意图字段 selected_intent_id。初始状态统一 possible；subtype 未确认字段 null。严格复制本次 primary_rule_id，emotion_summary_ref=server_emotion_summary。\n输出契约：'+json.dumps(schema,ensure_ascii=False)},
                           {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],max_output_tokens=6000)
    _MODEL_RETURNED.set(True)
    return result

def check_entry(raw,analysis,mode):
    rc=rules.derive(analysis,mode)['rule_context']
    if raw.get('primary_rule_id','missing')!=rc['primary_rule_id'] or raw.get('emotion_summary_ref')!='server_emotion_summary':
        raise ValueError('规则入口或摘要引用不匹配，必须复制本次后端提供的字段。')
    return rc

def subtype(value,refs):
    empty={'target':None,'interaction_partner':None,'requested_change':None,'supporting_answer_ids':[]}
    if value is None:return empty
    if not isinstance(value,dict):raise ValueError('细化字段结构无效。')
    ids=value.get('supporting_answer_ids',[])
    if not isinstance(ids,list) or any(i not in {r['answer_id'] for r in refs if r['role']=='support'} for i in ids):raise ValueError('细化字段缺少当前有效观察引用。')
    out=dict(empty,supporting_answer_ids=ids)
    for k in ('target','interaction_partner','requested_change'):
        if value.get(k) is not None:
            if not ids:raise ValueError('未确认的细化字段必须为 null。')
            out[k]=text(value[k],'细化字段',200)
    return out


def generate_intents(analysis,model,history=None,mode='C',question_budget=None,phase='initial',context_first=False):
    schema={'intents':[{'intent_id':'I01–I06之一','description':'本次目的假设','supporting_signals':[{'source':'context|arousal|valence','label':'真实模型类别','score':'复制原始数值','explanation':'待验证线索'}],'uncertainties':['缺失信息'],'status':'possible','source_rule_ids':[],'outside_rule_pool_reason':None,'supporting_observations':[],'contradicting_observations':[],'subtype':{'target':None,'interaction_partner':None,'requested_change':None,'supporting_answer_ids':[]}}],'primary_rule_id':rules.derive(analysis,mode)['rule_context']['primary_rule_id'],'emotion_summary_ref':'server_emotion_summary','reason':'少于三个时必须说明原因；零个表示无法提出合理假设'}
    if context_first:
        schema['request_understanding']={'user_question':'用户想知道什么；原文没有问题时为空串','question_quotes':[{'answer_id':'原文ID','quote':'提问原句'}],
                                         'statements':[{'answer_id':'原文ID','quote':'直接引用观察或猜测语句，不包含问题','kind':'observation 或 guess'}]}
    raw=call('generate_intents',analysis,model,{'context_first':context_first,'phase':phase,'history':history or [],'max_questions':MAX_QUESTIONS,'question_budget':question_budget or {'limit':MAX_QUESTIONS,'used':0,'remaining':MAX_QUESTIONS}},schema,mode)
    if not isinstance(raw,dict):raise ValueError('候选返回必须是 JSON 对象。')
    rc=check_entry(raw,analysis,mode)
    xs=raw.get('intents');reason=raw.get('reason','')
    if not isinstance(xs,list) or len(xs)>3:raise ValueError('候选应为零到三个，不能强行凑数。')
    if len(xs)<3:reason=text(reason,'候选不足原因',500)
    clean=[];seen=set()
    for x in xs:
        if not isinstance(x,dict) or x.get('intent_id') not in CATALOG_BY_ID or x['intent_id'] in seen:raise ValueError('意图 ID 非法或重复。')
        ident=x['intent_id'];seen.add(ident)
        if x.get('status')!='possible':raise ValueError('初始状态必须 possible。')
        signals=x.get('supporting_signals',[])
        if not isinstance(signals,list) or (mode!='A' and not signals):raise ValueError('声音辅助方案必须返回模型线索列表。')
        refs=[]
        for sig in signals:
            ref=canonical_signal(sig,mode)
            expected=next(i['score'] for i in allowed_signals(analysis,mode) if i['source']==ref['source'] and i['label']==ref['label'])
            if type(sig.get('score')) not in (int,float) or abs(sig['score']-expected)>1e-8:raise ValueError('模型引用数值与本次原始分数不一致。')
            refs.append(dict(ref,score=expected))
        source_rules=x.get('source_rule_ids',[])
        if not isinstance(source_rules,list) or len(source_rules)!=len(set(source_rules)) or any(r not in rc['matched_rule_ids'] for r in source_rules):raise ValueError('引用了本次未匹配的规则。')
        outside=x.get('outside_rule_pool_reason')
        obs=evidence(typed_observations(x.get('supporting_observations',[]),'support'),history or [])
        if not history and (x.get('supporting_observations') or x.get('contradicting_observations')):raise ValueError('首次无观察不能填入观察证据。')
        if rc['matched_rule_ids'] and ident not in rc['candidate_pool']:
            outside=text(outside,'规则池外依据',500)
            if not refs and not obs:raise ValueError('池外候选缺少模型或用户来源。')
        if outside is not None:outside=text(outside,'规则池外依据',500)
        detail=subtype(x.get('subtype'),obs)
        clean.append({'intent_id':ident,'title':CATALOG_BY_ID[ident]['title'],'description':text(x.get('description'),'描述',500),
                      'supporting_signals':refs,'source_rule_ids':source_rules,'outside_rule_pool_reason':outside,'supporting_observations':obs,'subtype':detail,'predicted_context':None,'uncertainties':strings(x.get('uncertainties'),'缺失信息'),'status':'possible'})
    result={'intents':clean,'reason':reason}
    if context_first:result['request_understanding']=context_policy.understanding(raw.get('request_understanding'),history or [],text)
    return result

def initial_states(intents):
    return [{'intent_id':i['intent_id'],'status':'possible','reason':'等待实际观察。','evidence':[],'subtype':copy.deepcopy(i.get('subtype')),'missing_information':i['uncertainties']} for i in intents]

def evidence_role(value):
    # Only equivalent relation words; never infer polarity from candidate status.
    aliases={'support':'support','supports':'support','supporting':'support','supported':'support',
             'supporting_evidence':'support','支持':'support','支持证据':'support',
             'contradict':'contradict','contradicts':'contradict','contradicting':'contradict',
             'contradiction':'contradict','contradictory':'contradict','contradicting_evidence':'contradict',
             '反对':'contradict','反证':'contradict','矛盾':'contradict','排除证据':'contradict',
             'clarify':'clarify','clarification':'clarify','clarifying':'clarify','澄清':'clarify','补充说明':'clarify'}
    key=value.strip().casefold() if isinstance(value,str) else None
    if key not in aliases:
        raise ValueError('DeepSeek 引用字段 role 不合规：实际值='+repr(value)[:100]+'；必须逐条选择 support（支持）、contradict（反证）或 clarify（澄清），不能填候选状态、整组选项或省略。用户原回答无需修改。')
    return aliases[key]

def typed_observations(items,role):
    if not isinstance(items,list):raise ValueError('观察引用必须为列表。')
    out=[]
    for item in items:
        if not isinstance(item,dict):raise ValueError('观察引用必须为对象。')
        if 'role' in item and evidence_role(item['role'])!=role:raise ValueError('观察所在支持/反证字段与引用作用冲突。')
        out.append(dict(item,role=role))
    return out

def evidence(items,history):
    if not isinstance(items,list):raise ValueError('引用必须是列表。')
    answers={h['answer_id']:h for h in history};clean=[]
    for item in items:
        if not isinstance(item,dict):raise ValueError('引用结构无效。')
        h=answers.get(item.get('answer_id'));q=item.get('quote');role=item.get('role')
        if h is None or not isinstance(q,str) or not q.strip() or q not in h['answer']:raise ValueError('引文或回答 ID 不存在。')
        if item.get('answer_revision',h.get('answer_revision',0))!=h.get('answer_revision',0):raise ValueError('引用的是更正前的旧回答。')
        role=evidence_role(role)
        if 'observation_quotes' in h and not any(q in fact for fact in h['observation_quotes']):continue
        source_text=h['answer']
        if 'observation_quotes' in h:
            # Use the surrounding original clause, not the model's paraphrase.
            source_text='；'.join(part for part in re.split(r'[。！？!?；;\n]',h['answer']) if q in part) or h['answer']
        if h.get('kind') in ('unknown','guess') or unknown_answer(source_text):continue
        # Conservative lexical gate over the ENTIRE answer, not the quoted fragment.
        if re.search(r'假如|如果|可能会|我猜|我觉得|平时|以前|过去|这次没有|现在没有|并没有|不是',source_text):continue
        clean.append({'answer_id':h['answer_id'],'turn':h.get('turn'), 'quote':q,'role':role,'answer_revision':h.get('answer_revision',0),'brief_reason':text(item.get('brief_reason','来源可核对；语义仍需审查。'),'引用依据',500)})
    return clean

def adapt_interview_response(raw):
    """Normalize documented equivalent field names, never fill missing assessments."""
    if not isinstance(raw,dict):raise ValueError('问答返回必须为 JSON 对象。')
    out=copy.deepcopy(raw)
    def alias(obj,target,names):
        present=[k for k in [target,*names] if k in obj]
        if not present:return
        first=obj[present[0]]
        if any(obj[k]!=first for k in present[1:]):raise ValueError('DeepSeek 同时返回互相冲突的字段：'+','.join(present))
        obj[target]=first
    alias(out,'assessments',['candidates','intents','active_candidates'])
    alias(out,'selected_intent_id',['primary_intent_id'])
    for item in out.get('assessments',[]) if isinstance(out.get('assessments'),list) else []:
        if not isinstance(item,dict):continue
        alias(item,'reason',['brief_reason'])
        if 'evidence' not in item and ('supporting_observations' in item or 'contradicting_observations' in item):
            refs=[]
            for key,role in [('supporting_observations','support'),('contradicting_observations','contradict')]:
                refs.extend(typed_observations(item.get(key,[]),role))
            item['evidence']=refs
    return out

def normalize(raw,intents,previous,history,observations=None,question_budget=None,context_first=False):
    if not isinstance(raw,dict):raise ValueError('返回结构不是对象。')
    if raw.get('action')!='ask' and raw.get('question') is not None:raise ValueError('结束时 question 必须为 null。')
    raw=adapt_interview_response(raw)
    ids=[i['intent_id'] for i in intents];items=raw.get('assessments')
    if raw.get('action') not in ('ask','conclude','uncertain','outside_candidates'):raise ValueError('action 无效。')
    if not isinstance(items,list):raise ValueError('DeepSeek 未返回评估列表 assessments。请按 expected_intent_ids 逐项返回，不能只给最可能的一项。')
    received=[x.get('intent_id') if isinstance(x,dict) else '<invalid>' for x in items]
    if len(received)!=len(ids) or set(received)!=set(ids):
        raise ValueError('DeepSeek 候选评估不完整：应返回 '+','.join(ids)+'；实际返回 '+','.join(map(str,received))+'。遗漏项也须评估，证据不足时 possible；不能补造支持或排除证据。')
    rows={}
    for item in items:
        if not isinstance(item,dict):raise ValueError('候选结构无效。')
        ident=item.get('intent_id');state=item.get('status')
        if ident not in ids or ident in rows or state not in STATUSES:raise ValueError('候选 ID 或状态无效。')
        refs=evidence(item.get('evidence',[]),(observations or [])+history);reason=text(item.get('reason'),'依据',800)
        required='contradict' if state=='excluded' else 'support'
        if state!='possible' and not any(r['role']==required for r in refs):state='possible';reason='缺少可用直接观察，需澄清。'
        negatives={'I01':r'(没有|没|无).{0,8}(食物|零食|吃的)', 'I02':r'(没有|没).{0,8}(玩耍|游戏|跟它玩)', 'I04':r'(没有|没).{0,8}(触碰|接触|淋水)', 'I05':r'(没有|没).{0,5}(靠近|接近)', 'I06':r'(没有|没).{0,5}(后退|退开|回避)'}
        by_id={h['answer_id']:h for h in (observations or [])+history}
        if state in ('supported','subsumed') and ident in negatives and any(re.search(negatives[ident],by_id[r['answer_id']]['answer']) for r in refs):
            state='possible';reason='完整回答存在明显否定，需先澄清；不能只引用肯定词。'
        target=item.get('subsumed_by')
        if state=='subsumed' and (ident!='I03' or target not in ('I01','I02') or target not in ids):raise ValueError('涵盖关系无效。')
        rows[ident]={'intent_id':ident,'status':state,'reason':reason,'evidence':refs,'subsumed_by':target if state=='subsumed' else None,
                     'subtype':subtype(item.get('subtype'),refs),'missing_information':strings(item['missing_information'],'缺失信息') if item.get('missing_information') else []}
    for row in rows.values():
        if row['status']=='subsumed' and rows[row['subsumed_by']]['status']!='supported':raise ValueError('具体目标尚未被支持，不能涵盖一般关注。')
    if history and (history[-1].get('kind') in ('unknown','guess') or unknown_answer(history[-1]['answer'])):
        rows={r['intent_id']:copy.deepcopy(r) for r in previous}
    supported=[i for i in ids if rows[i]['status']=='supported'];excluded=[i for i in ids if rows[i]['status']=='excluded']
    unresolved=[i for i in ids if rows[i]['status']=='possible'];subsumed=[i for i in ids if rows[i]['status']=='subsumed']
    selected=raw.get('selected_intent_id');coexist=raw.get('coexisting_intent_ids',[])
    if selected is not None and selected not in ids:raise ValueError('主要意图 ID 无效。')
    if not isinstance(coexist,list) or len(coexist)!=len(set(coexist)) or any(i not in supported or i==selected for i in coexist):raise ValueError('并存意图必须来自受支持的当前候选。')
    if not isinstance(raw.get('remaining_uncertainties',[]),list):raise ValueError('不确定性必须为列表。')
    for field in ('subsumed_intent_ids','unresolved_intent_ids'):
        if field in raw and (not isinstance(raw[field],list) or any(i not in ids for i in raw[field])):raise ValueError('结果引用了当前范围外的候选。')
    if raw['action']=='ask' and (selected is not None or coexist or raw.get('result_status') is not None):raise ValueError('询问时不能同时选择结论。')
    used=(question_budget or {}).get('used',len(history))
    readiness=context_policy.evaluate(raw,rows,selected,text) if context_first else None
    outcome=None;action='ask'
    if (ids and len(excluded)==len(ids)) or raw['action']=='outside_candidates':outcome='outside_candidates'
    elif raw['action']=='conclude' and selected is None and len(supported)>=2 and not unresolved and set(coexist)==set(supported):outcome='coexisting_supported'
    elif context_first and raw['action']=='conclude' and readiness['eligible']:outcome='concluded'
    elif not context_first and raw['action']=='conclude' and selected in supported and not unresolved and set(supported)=={selected,*coexist}:
        # I04/I06 cannot be distinguished by an identical single avoidance quote.
        if 'I04' in supported and 'I06' in supported and rows['I04']['evidence']==rows['I06']['evidence']:outcome='multiple_possible'
        else:outcome='concluded'
    elif used>=MAX_QUESTIONS or raw['action'] in ('uncertain','conclude'):
        outcome='multiple_possible' if len(supported)+len(unresolved)>1 else 'insufficient_evidence'
    result={'action':'uncertain' if outcome else 'ask','result_status':outcome,'selected_intent_id':None,'coexisting_intent_ids':[],
            'subsumed_intent_ids':subsumed,'unresolved_intent_ids':unresolved,'assessments':[rows[i] for i in ids],'question':None,
            'remaining_uncertainties':strings(raw['remaining_uncertainties'],'剩余不确定性') if raw.get('remaining_uncertainties') else [],'validation_notes':[],
            'validation_scope':'校验结构、引用来源及保守的词面风险；不证明语义正确或真实意图。','message':'根据观察保留未决可能。'}
    if context_first:
        result['readiness']=readiness
        result['explanation']=text(raw.get('explanation'),'给用户的解释',1800)
        rank=raw.get('ranked_intent_ids',ids)
        if not isinstance(rank,list) or len(rank)!=len(ids) or set(rank)!=set(ids):raise ValueError('排序必须覆盖当前全部候选 ID。')
        result['ranked_intent_ids']=rank
    if outcome=='concluded':result.update(action='conclude',selected_intent_id=selected,coexisting_intent_ids=coexist,message='根据本次声音线索与观察，更可能存在以下目的；这是暂定解释。')
    elif outcome=='coexisting_supported':result.update(action='conclude',coexisting_intent_ids=coexist,message='多项目的有观察支持，但没有依据确定主要目标。')
    elif outcome=='outside_candidates':result['message']='当前候选范围不足或未覆盖本次目的，可以明确选择重新生成候选。'
    if not outcome:
        q=raw.get('question')
        if not isinstance(q,dict):raise ValueError('缺少下一问。')
        if q.get('template_id') is not None and q['template_id'] not in rules.QUESTIONS['templates']:raise ValueError('问题模板 ID 不存在。')
        sentence=text(q.get('text'),'问题',240);dist=q.get('distinguishes')
        if context_first and (len(re.findall(r'有没有|是否',sentence))>1 or '或者' in sentence or '；' in sentence):raise ValueError('每次只问一个具体未知事实，不要把多个行为用或者/分号合成问题。')
        if not isinstance(dist,list) or (ids and not dist) or any(i not in ids for i in dist):raise ValueError('问题引用了当前范围外的候选。')
        if sentence.count('？')+sentence.count('?')>1 or any(h.get('question')==sentence for h in (observations or [])+history):raise ValueError('问题重复或一轮提出多个问题。')
        result['question']={'text':sentence,'template_id':q.get('template_id'),'focus':q.get('focus','实际观察'),'distinguishes':dist}
    return result

def next_turn(analysis,intents,previous,history,model,mode='C',observations=None,question_budget=None,context_first=False,request_understanding=None):
    schema={'action':'ask|conclude|uncertain|outside_candidates','selected_intent_id':None,'coexisting_intent_ids':[],
            'assessments':[{'intent_id':'当前ID','status':'possible|supported|excluded|subsumed','subsumed_by':None,'reason':'简短依据','missing_information':[],
                            'evidence':[{'answer_id':'回答ID','quote':'连续原文','role':'clarify'}]}],
            'question':{'text':'一个观察问题','focus':'关注点','distinguishes':['当前ID']},'remaining_uncertainties':[],'result_status':'ask 时 null；结束时 concluded/coexisting_supported/multiple_possible/outside_candidates/insufficient_evidence'}
    if context_first:
        schema.update(explanation='直接回答用户的问题；不足时说明已知线索与缺失信息，不报百分比',ranked_intent_ids=[i['intent_id'] for i in intents],
                      readiness={'critical_conflicts':[],'comparisons':[{'intent_id':'除主要选择外的每个ID','reason':'为什么相较于主要解释缺少支持，不等于已经排除'}]})
    budget=question_budget or {'limit':MAX_QUESTIONS,'used':len(history),'remaining':max(0,MAX_QUESTIONS-len(history))}
    schema.update(primary_rule_id=rules.derive(analysis,mode)['rule_context']['primary_rule_id'],emotion_summary_ref='server_emotion_summary')
    schema['assessments'][0]['subtype']={'target':None,'interaction_partner':None,'requested_change':None,'supporting_answer_ids':[]}
    schema['assessments'][0]['evidence'][0].update(brief_reason='与意图的关系',answer_revision=0)
    # Show one actual output slot for EACH active ID, not a one-row placeholder.
    template=copy.deepcopy(schema['assessments'][0])
    schema['assessments']=[dict(copy.deepcopy(template),intent_id=i['intent_id'],evidence=[]) for i in intents]
    schema['question']['distinguishes']=[i['intent_id'] for i in intents]
    raw=call('interview',analysis,model,{'context_first':context_first,'request_understanding':request_understanding,'expected_intent_ids':[i['intent_id'] for i in intents],
             'evidence_role_contract':{'allowed_values':['support','contradict','clarify'],'required':True,'note':'每条引用单选一个值；clarify 不能作为支持/排除依据；候选状态 possible/supported/excluded 不是引用作用枚举。'},
             'response_requirements':{'assessment_count':len(intents),'include_unchanged_and_excluded':True,'free_text_answer':True,
                 'instruction':'问题示例不是选项。结合所问问题理解用户省略句、具体物品及自由描述。每个当前 ID 恰好评估一次；证据不足保留 possible。新目标不在候选中时说明范围不足，不偷偷新增 ID。'},
             'independent_or_previous_round_observations':observations or [],'initial_intents':intents,'current_states':previous,'history':history,'max_questions':MAX_QUESTIONS,'remaining_budget':budget['remaining'],'question_budget':budget},schema,mode)
    check_entry(raw,analysis,mode)
    return normalize(raw,intents,previous,history,observations,budget,context_first)

def conclusion(analysis,intents,decision,history):
    from legacy_intent_v1 import conclusion as base
    out=base(analysis,intents,decision,history)
    out.update(method=METHOD,result_status=decision['result_status'],coexisting_intent_ids=decision.get('coexisting_intent_ids',[]),
               subsumed_intent_ids=decision.get('subsumed_intent_ids',[]),unresolved_intent_ids=decision.get('unresolved_intent_ids',[]))
    out.update(subtypes={i['intent_id']:i.get('subtype') for i in decision['assessments']},emotion_summary=rules.derive(analysis)['emotion_summary'])
    if decision['result_status']=='coexisting_supported':out.update(uncertain=False,title='多个意图获观察支持，主要目标未定',supporting_observations=[r for i in decision['assessments'] if i['status']=='supported' for r in i['evidence']])
    if decision.get('readiness') is not None:
        out['policy']=context_policy.POLICY;out['readiness']=decision['readiness']
        out['explanation']=decision.get('explanation','');out['message']=out['explanation'] or out['message']
        by_id={i['intent_id']:i for i in intents};states={i['intent_id']:i for i in decision['assessments']}
        out['alternatives']=[{'intent_id':ident,'title':by_id[ident]['title'],'status':states[ident]['status'],
                             'reason':states[ident]['reason'],'evidence':states[ident]['evidence'],
                             'missing_information':states[ident].get('missing_information',[])}
                            for ident in decision.get('ranked_intent_ids',list(by_id)) if states[ident]['status'] not in ('excluded','subsumed')][:3]
        out['confidence_note']='没有经过校准的意图概率；按观察支持程度给出暂定解释。'
    return out

def summary(records):
    rows=[s for s in records if (s.get('result') or {}).get('method')==METHOD]
    return {'completed':len(rows),'concluded':sum(s['result'].get('result_status') in ('concluded','coexisting_supported') for s in rows),
            'uncertain':sum(s['result'].get('result_status') not in ('concluded','coexisting_supported') for s in rows),'note':'意图无独立参考标签，不计算准确率；A/B/C 分组仅描述流程。'}

# One bounded schema-repair attempt; transport retries remain bounded in llm_client.
# No fallback candidates, and no API retry after a successful response is committed.
from contextvars import ContextVar
_MODEL_RETURNED=ContextVar('intent_model_returned',default=False)
_REPAIR_FEEDBACK=ContextVar('intent_contract_feedback',default=None)
def _with_contract_repair(fn):
    def wrapped(*args,**kwargs):
        flag=_MODEL_RETURNED.set(False)
        try:
            try:return fn(*args,**kwargs)
            except ValueError as first:
                if not _MODEL_RETURNED.get():raise
                token=_REPAIR_FEEDBACK.set(str(first)[:600])
                _MODEL_RETURNED.set(False)
                try:return fn(*args,**kwargs)
                finally:_REPAIR_FEEDBACK.reset(token)
        finally:_MODEL_RETURNED.reset(flag)
    return wrapped
generate_intents=_with_contract_repair(generate_intents)
next_turn=_with_contract_repair(next_turn)
