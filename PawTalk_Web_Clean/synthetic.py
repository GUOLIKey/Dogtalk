"""Label-conditioned fictional contexts. Never factual annotation or held-out evidence."""
import argparse,csv,json,random
from pathlib import Path
from engine import NAMES,ollama_models
from llm_client import chat_json
import urllib.request
TEMPLATES={
'alerting to sounds':[('屋外传来一阵敲门声，它转向门口，随后发出叫声。','sound'),('家里突然响起提示音，它停下原来的活动，朝声音方向叫了几声。','sound')],
'begging for food':[('主人正在准备食物，它靠近食盆，视线跟着主人手里的食物移动，随后发出叫声。','food'),('主人拿出零食袋，它走近主人并看着袋子，这时叫了几声。','food')],
'playing with human':[('主人正与它进行追逐游戏，两者交替靠近和退开，它在互动过程中发出叫声。','human'),('主人正拉着绳结玩具与它互动，它在来回拉扯时叫了几声。','human')],
'playing with other animals':[('另一只狗正在与它互相追逐，它在靠近同伴时发出叫声。','animal'),('它和另一只动物轮流靠近、退开，在互动过程中出现叫声。','animal')],
'playing with toy':[('一个玩具球滚到地上，它追上去拨弄并叼起玩具，期间发出叫声。','toy'),('它独自在摆弄一个玩具，玩具移动时，它追过去叫了几声。','toy')],
'rest':[('它原本躺在垫子上休息，周围没有明显活动，随后出现一段短叫声。','rest'),('它在安静的角落里卧着，没有人与它游戏，这时发出了叫声。','rest')],
'seeking attention':[('主人停止与它互动，转身做别的事情，它随后开始叫；主人重新回应后，叫声明显减少。','attention'),('主人把注意力转向电脑，它靠近并发出叫声；主人再次与它互动后，它安静了一些。','attention')],
'taking shower':[('主人正在给它淋水、清洗身体，它在清洗过程中发出叫声。','water'),('它站在洗浴区域，主人正用水冲洗它的身体，这时出现了叫声。','water')]}

def generate(label,seed=0,model=None,duration=None):
    if label not in TEMPLATES:raise ValueError('Unknown label')
    text,key=random.Random(seed).choice(TEMPLATES[label])
    generator='template-v1'
    if model:
        if model not in ollama_models():raise ValueError('语言模型尚未配置，请检查 Key 或 Ollama。')
        prompt='你是课堂演示情境生成器。请根据活动标签写一段40至100字的中文假设情境。这不是事实描述。只描述环境和可观察行为，不声称听过录音、不推测品种、体型、内心想法或健康状况。以假设当时开头。不要加入其他活动类别。返回JSON，只有text字段。标签：'+NAMES[label]+'。录音时长：'+str(duration)+'秒。'
        output=chat_json(model,[{'role':'user','content':prompt}],temperature=0.4);text=output.get('text','')
        if not isinstance(text,str) or not 10<=len(text)<=1000:raise ValueError('语言模型未返回有效情境，请重试或使用模板。')
        generator=model if model.startswith('DeepSeek::') else 'ollama:'+model
    return dict(text=text,generator=generator,source='label_conditioned_synthetic',label_used=True,is_observed=False,
                assumed_facts={} if model else {key:'yes'},unavailable=['品种','体型','真实动作','真实情绪'],
                notice='这段情境根据标签编写，动作与环境均是假设，并非从录音中识别。不可用于证明问答提高了真实准确率。')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path(__file__).parent/'synthetic_contexts.jsonl');ap.add_argument('--llm',default=None);ap.add_argument('--limit',type=int,default=None);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Output exists; choose a new --output path')
    with (Path(__file__).parent/'dev_manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if a.limit is not None:
        if a.limit<1:raise ValueError('limit must be positive')
        rows=rows[:a.limit]
    with a.output.open('w',encoding='utf-8') as out:
        for i,r in enumerate(rows):
            item=dict(audio_path=r['audio_path'],label=r['actual'],label_zh=NAMES[r['actual']],duration_sec=float(r['duration_sec']),
                      original_fold=int(r['fold']),audio_model_prediction=r['A_pred'],synthetic=generate(r['actual'],i,a.llm,float(r['duration_sec'])),valid_for_accuracy_evaluation=False)
            out.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(f'Wrote {len(rows)} synthetic scenarios: {a.output}')
    print('Label-conditioned demo data only. No factual captions, no inference of breed or body size.')
if __name__=='__main__':main()
