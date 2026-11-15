"""Offline tests: real application handlers/state/transport; mocked ML and DeepSeek responses."""
import copy
import hashlib
import http.client
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import analysis_contract as contract
import dialogue
import server
import start_app
import engine
import llm_client

MODEL = 'DeepSeek::test-model'
ROW = {'actual': 'begging for food', 'predicted': 'seeking attention',
       'scores': dict(zip(engine.LABELS, [6.2, 4.1, 2.2, 1.8, 3.3, .7, 6.1, 1.2])),
       'duration': 3., 'fold': 1}
EMOTION = {'available': True, 'enabled': True, 'model': 'DogBark LSTM', 'model_commit': 'fixture-commit',
           'recipe': 'fixture-recipe', 'duration': 3., 'aggregation': 'duration_weighted_mean_v1', 'device': 'cpu',
           'message': 'fixture only, not real model inference', 'chunks': [],
           'arousal': {'label': 'High', 'scores': {'Low': .2, 'Medium': .3, 'High': .5}, 'usable': False, 'margin': .2, 'agreement': .5},
           'valence': {'label': 'Neutral', 'scores': {'Negative': .1, 'Neutral': .7, 'Positive': .2}, 'usable': True, 'margin': .5, 'agreement': 1.}}

def analysis(): return contract.bundle(contract.context_result(copy.deepcopy(ROW)), contract.emotion_result(copy.deepcopy(EMOTION)))

def intents():
    return [{'intent_id': 'intent_'+str(i+1), 'title': title, 'description': desc,
             'supporting_signals': [{'source': source, 'label': label, 'explanation': '模型仅提示这种目的值得核对。'}],
             'uncertainties': ['需要实际观察，模型线索不能确认。'], 'status': 'possible'}
            for i, (title, desc, source, label) in enumerate([
                ('请主人靠近陪伴', '希望主人恢复接触和陪伴', 'context', 'seeking attention'),
                ('争取正在展示的食物', '希望得到食物而发声', 'arousal', 'High'),
                ('提醒主人留意外界', '希望主人注意环境中的变化', 'valence', 'Neutral')])]


def response(statuses=None, history=None, action='ask', selected=None, question='叫声开始前，狗主要朝向什么对象？'):
    history = history or []; statuses = statuses or ['possible'] * 3
    return {'action': action, 'selected_intent_id': selected,
            'assessments': [{'intent_id': i['intent_id'], 'status': state, 'reason': '根据本次观察作暂定解释。',
                             'evidence': [{'turn': len(history), 'quote': history[-1]['answer']}] if history and state != 'possible' else []}
                            for i, state in zip(intents(), statuses)],
            'question': {'text': question, 'focus': '注意对象'}, 'remaining_uncertainties': ['仍需考虑其他遗漏的目的。']}


def turn(answer='它盯着食物，食物拿走后停止叫，没有门铃或互动。', kind='text', n=1):
    return {'turn': n, 'question': '它看向哪里？', 'focus': '注意对象', 'answer': answer, 'kind': kind}


class Harness(server.Handler):
    def __init__(self, path, data=None, raw=None, headers=None):
        self.path = path; body = raw if raw is not None else json.dumps(data or {}).encode()
        self.rfile = io.BytesIO(body); self.headers = {'Content-Length': str(len(body)), 'Host': 'localhost:8882', **(headers or {})}
    def send(self, status, data, mime='application/json; charset=utf-8'): self.status = status; self.output = data


def request(path, data=None, raw=None, headers=None, method='POST'):
    h = Harness(path, data, raw, headers)
    h.do_GET() if method == 'GET' else h.do_POST()
    return h.status, h.output
