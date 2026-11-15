"""Offline A/B/C paired evaluation with independent, explicitly supplied reference sets."""
import argparse,json,hashlib

def evaluate(records,references):
    refs={r['sha']:r for r in references};eligible={};excluded={}
    for s in sorted(records,key=lambda s:s.get('created',0)):
        ref=refs.get(s.get('sha'));reason=None
        if not ref or not ref.get('reference_clear'):reason='missing_or_unclear_independent_reference'
        elif not ref.get('dog_id') or not ref.get('source_id'):reason='missing_group_identity'
        elif not s.get('result') or s.get('schema_version')!='intent-session-v3':reason='incomplete_or_old_method'
        elif s.get('contaminated') or s.get('source')!='real':reason='not_blind_real_observation'
        if reason:excluded[reason]=excluded.get(reason,0)+1;continue
        eligible.setdefault((s['sha'],s.get('experiment_mode','C')),s)
    paired={sha for sha,_ in eligible if all((sha,m) in eligible for m in 'ABC')}
    out={}
    for mode in 'ABC':
        samples=[eligible[sha,mode] for sha in sorted(paired)];tp=fp=fn=covered=wrong=0;js=[];candidate=[];qs=[];primary=[]
        for s in samples:
            ref=refs[s['sha']];gold=set(ref['intent_ids']);r=s['result'];pred=set(r.get('coexisting_intent_ids',[]))
            if r.get('intent_id'):pred.add(r['intent_id'])
            tp+=len(gold&pred);fp+=len(pred-gold);fn+=len(gold-pred)
            js.append(len(gold&pred)/len(gold|pred) if gold|pred else 1)
            candidate.append(gold.issubset({i['intent_id'] for i in s['initial_intents']}))
            certain=r.get('result_status') in ('concluded','coexisting_supported');covered+=certain;wrong+=certain and pred!=gold
            qs.append(s.get('question_budget',{}).get('used',len((s.get('dialogue') or {}).get('history',[]))))
            if ref.get('primary_id'):primary.append(r.get('intent_id')==ref['primary_id'])
        n=len(samples)
        out[mode]={'n':n,'initial_set_coverage':sum(candidate)/n if n else None,'set_jaccard':sum(js)/n if n else None,
                   'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'conclusion_coverage':covered/n if n else None,
                   'error_given_conclusion':wrong/covered if covered else None,'uncertain_fraction':1-covered/n if n else None,
                   'mean_questions':sum(qs)/n if n else None,'primary_accuracy':sum(primary)/len(primary) if primary else None}
    return {'paired_samples':len(paired),'modes':out,'excluded':excluded,'unpaired_records':len(eligible)-3*len(paired),
            'reference_note':'使用独立行为标注，不是场景标签或内心真值；只比较同一录音完整 A/B/C 三组。'}

def grouped_split(references):
    # Connected components: sharing EITHER dog or source keeps records in the same partition.
    parent=list(range(len(references)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(references):
        for field in ('dog_id','source_id'):
            if not r.get(field):raise ValueError('缺少犬只/来源分组，不能宣称无同源泄漏。')
            key=(field,r[field])
            if key in seen:parent[find(i)]=find(seen[key])
            seen[key]=i
    groups={}
    for i,r in enumerate(references):groups.setdefault(find(i),[]).append(r['sha'])
    if len(groups)<2:raise ValueError('独立分组不足两个，无法划分开发/测试。')
    ordered=sorted(groups.values(),key=lambda xs:hashlib.sha256('|'.join(sorted(xs)).encode()).hexdigest())
    test_count=max(1,len(ordered)//5)
    return {'test':[s for g in ordered[:test_count] for s in g],'dev':[s for g in ordered[test_count:] for s in g]}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('export');p.add_argument('references');p.add_argument('--split',action='store_true');a=p.parse_args()
    records=json.load(open(a.export,encoding='utf-8'))['records'];refs=json.load(open(a.references,encoding='utf-8'))
    print(json.dumps(grouped_split(refs) if a.split else evaluate(records,refs),ensure_ascii=False,indent=2))
