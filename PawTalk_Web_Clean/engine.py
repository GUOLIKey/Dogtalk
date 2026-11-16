"""Current scene API. Historical eight-class statistics use their own module."""
from scene_schema import LABELS, NAMES, LABEL_VERSION
from llm_client import available_models as ollama_models, chat_json
from legacy_scene_engine import evaluate_records

def order(row):
    if set(row['scores']) != set(LABELS) or row['predicted'] not in LABELS:
        raise ValueError('不是当前六类场景输出。')
    pred=row['predicted']
    return [pred]+[k for k in sorted(LABELS,key=row['scores'].get,reverse=True) if k!=pred]

def cards(labels,values=None):
    return [dict(label=k,name=NAMES[k],rank=i+1,score=values[k] if values else None) for i,k in enumerate(labels)]
