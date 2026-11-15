"""Observation-first explanations. Checks provenance, never calibrated confidence."""
POLICY='context-first-v1'

def understanding(raw,observations,text):
    if not isinstance(raw,dict):raise ValueError('缺少用户描述与问题的结构化理解。')
    originals={h['answer_id']:h['answer'] for h in observations}
    facts=raw.get('statements',[])
    if not isinstance(facts,list) or len(facts)>20:raise ValueError('描述拆分字段无效。')
    checked=[]
    for item in facts:
        if not isinstance(item,dict) or item.get('kind') not in ('observation','guess'):raise ValueError('描述必须区分直接观察和猜测。')
        quote=item.get('quote');ident=item.get('answer_id')
        if ident not in originals or not isinstance(quote,str) or not quote.strip() or quote not in originals[ident]:raise ValueError('描述拆分引用不在用户原文中。')
        checked.append({'answer_id':ident,'quote':quote,'kind':item['kind']})
    question=raw.get('user_question','')
    refs=raw.get('question_quotes',[])
    if not isinstance(refs,list):raise ValueError('用户问题来源必须是列表。')
    for ref in refs:
        if not isinstance(ref,dict) or ref.get('answer_id') not in originals or not isinstance(ref.get('quote'),str) or not ref['quote'].strip() or ref['quote'] not in originals[ref['answer_id']]:raise ValueError('用户问题引用无效。')
    if question and not refs:raise ValueError('识别出的用户问题必须有原文来源；未提问时留空。')
    return {'user_question':text(question,'用户问题',500) if question else '这段叫声可能表达什么？',
            'question_quotes':refs,'question_is_default':not bool(question),'statements':checked,
            'note':'语句类型是语言模型的解析，引用核对不证明语义或观察真实性。'}

def evaluate(raw,rows,selected,text):
    """Require supporting observation, comparison with each alternative, no unresolved conflicts."""
    ready=raw.get('readiness',{})
    if not isinstance(ready,dict):raise ValueError('判断依据结构无效。')
    conflicts=ready.get('critical_conflicts',[])
    if not isinstance(conflicts,list):raise ValueError('关键矛盾必须为列表。')
    conflicts=[text(x,'关键矛盾',400) for x in conflicts]
    comparison=ready.get('comparisons',[])
    if not isinstance(comparison,list):raise ValueError('候选比较必须为列表。')
    comp={}
    for item in comparison:
        if not isinstance(item,dict) or item.get('intent_id') not in rows or item['intent_id']==selected or item['intent_id'] in comp:raise ValueError('比较引用无效候选。')
        comp[item['intent_id']]=text(item.get('reason'),'比较理由',500)
    evidence=rows.get(selected,{}).get('evidence',[])
    supported=rows.get(selected,{}).get('status')=='supported' and any(r['role']=='support' for r in evidence)
    # A contradicting quote on the selected intent must be resolved by a later corrected observation.
    contradicted=any(r['role']=='contradict' for r in evidence)
    complete=set(comp)==(set(rows)-{selected}) if selected in rows else False
    return {'policy':POLICY,'eligible':bool(supported and not contradicted and not conflicts and complete),
            'critical_conflicts':conflicts,'comparisons':[{'intent_id':k,'reason':v} for k,v in comp.items()],
            'basis':'直接观察支持＋逐项解释相对适合程度＋没有未解决的关键矛盾；不是概率阈值。'}
