"""Read-only v8 scene metrics; never used for v9 inference."""
from legacy_scene_engine import LABELS
METHOD = "adaptive-dual-v3"

def summary(records):
    seen=set();eligible=[]
    for s in sorted(records,key=lambda s:s['created']):
        if s.get('gold') not in LABELS:continue
        d=s.get('dialogue',{})
        if not s.get('result') or s['result'].get('method')!=METHOD or s.get('demo') or s.get('contaminated') or s.get('source')!='real' or not d.get('history'):continue
        if s['sha'] in seen:continue
        seen.add(s['sha']);eligible.append(s)
    metrics={}
    for key,name in [('audio_pred','audio'),('combined_pred','combined')]:
        f=[];n=len(eligible)
        for label in LABELS:
            tp=sum(s['gold']==label and s['result'][key]==label for s in eligible)
            fp=sum(s['gold']!=label and s['result'][key]==label for s in eligible)
            fn=sum(s['gold']==label and s['result'][key]!=label for s in eligible)
            f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        metrics[name]={'accuracy':sum(s['gold']==s['result'][key] for s in eligible)/n if n else None,'macro_f1':sum(f)/8 if n else None}
    n=len(eligible)
    return {'n':n,'metrics':metrics,'answered':sum(s['result']['combined_pred'] is not None for s in eligible),'top3_recall':sum(s['gold'] in s['dialogue']['candidates'] for s in eligible)/n if n else None,'mean_questions':sum(len(s['dialogue']['history']) for s in eligible)/n if n else None,'expanded_trials':sum(bool(s['dialogue'].get('expansions')) for s in eligible),'recovered_outside_top3':sum(s['gold'] not in s['dialogue']['candidates'] and s['result']['combined_pred']==s['gold'] for s in eligible),'note':'仅统计 v8 自报真实背景探索；不确定按错误计，未完成对话不在统计中。Top-3 召回率描述原声音候选，扩展后可选已有八类中的其他类别；这些观察性记录不能单独证明模型提升。'}


def grouped_summary(records):
    # Different groups are descriptive, not a paired causal effect estimate.
    modes=('context_only','emotion_assisted','emotion_unavailable')
    return {mode:summary([s for s in records if s.get('dialogue',{}).get('mode','context_only')==mode]) for mode in modes}
