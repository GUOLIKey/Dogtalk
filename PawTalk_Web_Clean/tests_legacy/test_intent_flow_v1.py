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


class DialogueTests(unittest.TestCase):
    def test_first_call_contains_complete_independent_scores_and_no_reference(self):
        a = analysis(); a['gold'] = 'SECRET_GOLD'; a['actual'] = 'SECRET_ACTUAL'
        a['synthetic'] = {'scene': 'SECRET_LABEL_GENERATED_BACKGROUND'}
        a['context']['actual'] = 'SECRET_IN_NESTED_CONTEXT'
        a['emotion']['gold'] = 'SECRET_EMOTION_GOLD'
        with patch('dialogue.chat_json', return_value={'intents': intents()}) as call:
            result = dialogue.generate_intents(a, MODEL)
        payload = json.loads(call.call_args.args[1][1]['content'])
        encoded = json.dumps(payload)
        self.assertNotIn('SECRET_', encoded)
        self.assertEqual(payload['analysis']['context']['scores'], ROW['scores'])
        self.assertEqual(len(payload['analysis']['context']['scores']), 8)
        self.assertEqual(payload['analysis']['context']['original_preferred']['label'], ROW['predicted'])
        self.assertEqual(payload['analysis']['context']['score_kind'], 'svm_decision_function_ovr')
        for axis in ('arousal', 'valence'):
            self.assertEqual(payload['analysis']['emotion'][axis]['scores'], EMOTION[axis]['scores'])
            self.assertEqual(len(payload['analysis']['emotion'][axis]['scores']), 3)
        self.assertFalse(payload['analysis']['emotion']['arousal']['usable'])
        self.assertEqual(payload['analysis']['emotion']['arousal']['label'], 'High')
        self.assertEqual(len({i['intent_id'] for i in result}), 3)
        self.assertNotIn(result[0]['title'], engine.NAMES.values())

    def test_generation_validates_count_ids_status_signals_without_scene_title_enum(self):
        mutations = [lambda xs: xs.pop(), lambda xs: xs[1].update(intent_id=xs[0]['intent_id']),
                     lambda xs: xs[1].update(title=xs[0]['title']), lambda xs: xs[0].update(status='supported'),
                     lambda xs: xs[0]['supporting_signals'][0].update(label='invented')]
        for mutation in mutations:
            xs = intents(); mutation(xs)
            with self.subTest(mutation=mutation), patch('dialogue.chat_json', return_value={'intents': xs}), self.assertRaises(ValueError):
                dialogue.generate_intents(analysis(), MODEL)

    def test_full_history_analysis_and_fixed_initial_ids_sent_every_round(self):
        h = [turn()]
        with patch('dialogue.chat_json', return_value=response(history=h)) as call:
            dialogue.next_turn(analysis(), intents(), dialogue.initial_states(intents()), h, MODEL)
        payload = json.loads(call.call_args.args[1][1]['content'])
        self.assertEqual(payload['history'], h); self.assertEqual(payload['initial_intents'], intents())
        self.assertEqual(payload['analysis']['context']['scores'], ROW['scores'])
        self.assertEqual(payload['analysis']['emotion']['arousal']['scores'], EMOTION['arousal']['scores'])
        self.assertNotIn('taxonomy', payload); self.assertEqual(payload['max_questions'], 8)

    def test_conclusion_requires_supported_one_and_explicit_exclusions(self):
        h = [turn()]
        r = dialogue.normalize(response(['excluded','supported','excluded'], h, 'conclude', 'intent_2'), intents(), dialogue.initial_states(intents()), h)
        self.assertEqual(r['action'], 'conclude'); self.assertEqual(r['selected_intent_id'], 'intent_2')
        final = dialogue.conclusion(analysis(), intents(), r, h)
        self.assertEqual(final['title'], intents()[1]['title'])
        self.assertEqual(len(final['excluded_intents']), 2); self.assertTrue(final['supporting_observations'])
        self.assertNotIn('combined_pred', final)

    def test_multiple_supported_and_all_excluded_never_forced(self):
        for states in [['supported','supported','excluded'], ['excluded']*3, ['possible','excluded','excluded']]:
            h = [turn()]
            r = dialogue.normalize(response(states, h, 'conclude', 'intent_1'), intents(), dialogue.initial_states(intents()), h)
            self.assertEqual(r['action'], 'uncertain'); self.assertIsNone(r['selected_intent_id'])

    def test_missing_evidence_cannot_exclude(self):
        raw = response(['excluded']*3)
        r = dialogue.normalize(raw, intents(), dialogue.initial_states(intents()), [])
        self.assertTrue(all(x['status'] == 'possible' for x in r['assessments']))

    def test_unknown_free_text_and_button_cannot_change_states(self):
        for answer, kind in [('不知道','text'), ('当时我没观察到，所以不确定。','text'), ('没留意呀','text'), ('not sure','text'), ('未知','unknown')]:
            h = [turn(answer, kind)]
            r = dialogue.normalize(response(['excluded','supported','excluded'], h, 'conclude', 'intent_2'), intents(), dialogue.initial_states(intents()), h)
            self.assertTrue(all(x['status'] == 'possible' for x in r['assessments']))
            self.assertEqual(r['action'], 'uncertain')

    def test_unknown_preserves_previous_supported_and_excluded(self):
        h = [turn(), turn('不知道', 'unknown', 2)]
        previous = response(['excluded','possible','possible'], h[:1])['assessments']
        r = dialogue.normalize(response(history=h, question='之后主人位置有没有变化？'), intents(), previous, h)
        self.assertEqual(r['assessments'], previous)

    def test_correction_can_restore_excluded_candidate(self):
        h = [turn('只有一只狗。'), turn('我说错了，另一只狗刚从门口走了进来。', n=2)]
        previous = response(['excluded','possible','possible'], h[:1])['assessments']
        r = dialogue.normalize(response(['supported','possible','possible'], h), intents(), previous, h)
        self.assertEqual(r['assessments'][0]['status'], 'supported')
        self.assertEqual(r['assessments'][0]['evidence'][0]['turn'], 2)

    def test_bad_quote_turn_ids_and_illegal_status_rejected(self):
        h = [turn()]
        for field, value in [('intent_id','activity_food'), ('status','unlikely'), ('evidence',[{'turn':99,'quote':h[0]['answer']}]), ('evidence',[{'turn':1,'quote':'食物'}])]:
            raw = response(history=h);raw['assessments'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                dialogue.normalize(raw, intents(), dialogue.initial_states(intents()), h)

    def test_stale_evidence_cannot_change_state(self):
        h = [turn(), turn('后来狗转向门口。', n=2)]
        raw = response(['excluded','possible','possible'], h)
        raw['assessments'][0]['evidence'] = [{'turn':1,'quote':h[0]['answer']}]
        with self.assertRaises(ValueError):dialogue.normalize(raw, intents(), dialogue.initial_states(intents()), h)

    def test_eight_questions_uncertain_and_no_api_fallback_question(self):
        h = [turn('我只看到它朝向门口。',n=i+1) for i in range(dialogue.MAX_QUESTIONS)]
        r = dialogue.normalize(response(history=h), intents(), dialogue.initial_states(intents()), h)
        self.assertEqual(r['action'], 'uncertain'); self.assertIsNone(r['question'])
        raw = response();raw['question'] = None
        with self.assertRaises(ValueError):dialogue.normalize(raw, intents(), dialogue.initial_states(intents()), [])

    def test_nonfinite_partial_and_invalid_axes_fail_closed(self):
        for score in [float('nan'), float('inf'), True]:
            row = copy.deepcopy(ROW); row['scores'][engine.LABELS[0]] = score
            with self.assertRaises(ValueError):contract.context_result(row)
        raw=copy.deepcopy(EMOTION);del raw['valence']['scores']['Positive']
        with self.assertRaises(ValueError):contract.emotion_result(raw)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        p=patch.object(server,'DB',Path(self.tmp.name)/'sessions.sqlite');p.start();self.addCleanup(p.stop)
        self.audio=b'fixture-recording-not-real-audio'; self.digest=hashlib.sha256(self.audio).hexdigest()
        for p in [patch.object(server,'index',return_value={self.digest:copy.deepcopy(ROW)}),
                  patch.object(server,'_index_meta',{'version':'baseline-oof-v1','identity':'fixture-cache'}),
                  patch.object(server,'deepseek_models',return_value=[MODEL]),
                  patch.object(server.emotion_audio,'predict_bytes',return_value=copy.deepcopy(EMOTION))]:
            p.start();self.addCleanup(p.stop)

    def upload(self):
        code,s=request('/api/upload',raw=self.audio,headers={'X-Filename':'dog.wav'})
        self.assertEqual(code,200);return s

    def generated(self):
        s=self.upload()
        with patch('dialogue.chat_json',return_value={'intents':intents()}):
            code,s=request('/api/intents/generate',{'id':s['id'],'model':MODEL,'source':'real'})
        self.assertEqual(code,200);return s

    def started(self):
        s=self.generated()
        with patch('dialogue.chat_json',return_value=response()):code,s=request('/api/chat/start',{'id':s['id']})
        self.assertEqual(code,200);return s

    def test_upload_both_branches_even_without_toggle_and_cache_skips_live(self):
        with patch.object(server.live_audio,'predict_bytes') as live, patch.object(server.emotion_audio,'predict_bytes',return_value=EMOTION) as emotion:
            s=self.upload()
        live.assert_not_called();emotion.assert_called_once_with(self.audio)
        self.assertTrue(s['analysis']['ready']);self.assertEqual(s['analysis']['context']['cache_version'],'baseline-oof-v1')
        self.assertIsNone(s['scene_reference']);self.assertNotIn('gold',s);self.assertNotIn('actual',json.dumps(s))
        self.assertEqual(s['initial_intents'],[])

    def test_missing_branch_visible_no_synthetic_candidates(self):
        with patch.object(server.emotion_audio,'predict_bytes',return_value={'available':False,'message':'missing weights: run 08'}):s=self.upload()
        self.assertEqual(s['phase'],'blocked');self.assertIn('missing weights',s['analysis']['emotion']['message'])
        with patch('dialogue.chat_json') as call:
            code,out=request('/api/intents/generate',{'id':s['id'],'model':MODEL})
        self.assertEqual(code,400);call.assert_not_called();self.assertEqual(server.get(s['id'])['initial_intents'],[])

    def test_context_failure_still_attempts_emotion(self):
        with patch.object(server,'index',return_value={}),patch.object(server.live_audio,'predict_bytes',side_effect=ValueError('missing CLAP weights')),patch.object(server.emotion_audio,'predict_bytes',return_value=EMOTION) as emotion:
            s=self.upload()
        self.assertEqual(s['phase'],'blocked');self.assertTrue(s['analysis']['emotion']['available']);emotion.assert_called_once()
        self.assertIn('missing CLAP',s['analysis']['context']['message'])

    def test_api_failure_no_candidates_and_retry_preserves_completed_generation(self):
        s=self.upload()
        with patch('dialogue.chat_json',side_effect=ValueError('LLM HTTP 401')):
            self.assertEqual(request('/api/intents/generate',{'id':s['id'],'model':MODEL})[0],400)
        saved=server.get(s['id']);self.assertEqual(saved['initial_intents'],[]);self.assertTrue(saved['errors'])
        s=self.generated()
        with patch('dialogue.chat_json',side_effect=ValueError('LLM timeout')):
            self.assertEqual(request('/api/chat/start',{'id':s['id']})[0],400)
        self.assertEqual(server.get(s['id'])['initial_intents'],intents());self.assertIsNone(server.get(s['id'])['dialogue'])

    def test_full_flow_state_changes_export_and_metric_separation(self):
        s=self.started();h=[turn()]
        with patch('dialogue.chat_json',return_value=response(['excluded','supported','excluded'],h,'conclude','intent_2')):
            code,out=request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':h[0]['answer']})
        self.assertEqual(code,200);self.assertFalse(out['result']['uncertain']);self.assertEqual(out['phase'],'completed')
        saved=server.get(s['id']);self.assertEqual(len(saved['dialogue']['state_changes']),1)
        self.assertEqual(saved['initial_intents'],intents());self.assertEqual(saved['dialogue']['history'][0]['answer'],h[0]['answer'])
        hist=request('/api/history',method='GET')[1]
        self.assertEqual(hist['intent_summary']['concluded'],1)
        self.assertEqual(hist['legacy_scene_summary']['manual']['n'],0)
        self.assertEqual(hist['legacy_scene_summary']['v8_adaptive']['n'],0)
        self.assertEqual(engine.evaluate_records([saved])['n'],0)
        exported=request('/api/export',method='GET')[1]
        self.assertEqual(exported['records'][0]['analysis'],saved['analysis'])
        self.assertNotIn('gold',exported['records'][0]);self.assertNotIn('accuracy',exported['intent_summary'])
        self.assertEqual(request('/api/reveal',{'id':s['id']})[1]['scene_reference'],'讨食')
        self.assertEqual(request('/api/chat/answer',{'id':s['id'],'revision':1,'answer':'change'})[0],400)

    def test_unknown_round_state_and_manual_stop(self):
        s=self.started();h=[turn('不知道 / 没观察到','unknown')]
        with patch('dialogue.chat_json',return_value=response(['excluded']*3,h,question='主人靠近后叫声有没有变化？')):
            code,out=request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':h[0]['answer'],'kind':'unknown'})
        self.assertEqual(code,200);self.assertTrue(all(x['status']=='possible' for x in out['dialogue']['states']))
        code,out=request('/api/chat/stop',{'id':s['id'],'revision':1})
        self.assertEqual(code,200);self.assertTrue(out['result']['uncertain'])

    def test_failed_answer_retry_and_revision_are_atomic(self):
        s=self.started()
        with patch('dialogue.chat_json',side_effect=ValueError('API offline')):
            self.assertEqual(request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'看向门口。'})[0],400)
        saved=server.get(s['id']);self.assertEqual(saved['dialogue']['revision'],0);self.assertEqual(saved['dialogue']['history'],[])
        self.assertEqual(request('/api/chat/answer',{'id':s['id'],'revision':9,'answer':'看向门口。'})[0],400)

    def test_synthetic_early_reveal_and_old_endpoints_blocked(self):
        s=self.upload()
        self.assertEqual(request('/api/reveal',{'id':s['id']})[0],400)
        self.assertEqual(request('/api/intents/generate',{'id':s['id'],'model':MODEL,'source':'simulated'})[0],400)
        for endpoint in ('generate','predict','parse','demo'):
            self.assertEqual(request('/api/'+endpoint,{'id':s['id']})[0],410)
        saved=server.get(s['id']);saved['synthetic']={'text':'label-generated'};server.save(saved)
        with patch('dialogue.chat_json') as call:
            self.assertEqual(request('/api/intents/generate',{'id':s['id'],'model':MODEL,'source':'real'})[0],400)
        call.assert_not_called()

    def test_no_retraining_during_interview(self):
        with patch.object(server.live_audio,'predict_bytes') as scene,patch.object(server.emotion_audio,'predict_bytes',return_value=EMOTION) as emotion:
            s=self.started();emotion.assert_called_once();scene.assert_not_called()
            with patch('dialogue.chat_json',return_value=response(history=[turn()])):
                request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':turn()['answer']})
            emotion.assert_called_once();scene.assert_not_called()

    def test_legacy_history_read_only_and_exports_incomplete_new_records(self):
        old={'id':'old','filename':'old.wav','created':0,'result':None,'gold':'rest','sha':'old'};server.save(old)
        self.assertEqual(request('/api/history',method='GET')[1]['records'][0]['legacy'],True)
        self.assertEqual(request('/api/chat/start',{'id':'old'})[0],400)
        self.upload();out=request('/api/export',method='GET')[1]
        self.assertEqual(len(out['records']),2);self.assertTrue(all('gold' not in x for x in out['records']))

    def test_static_and_origin_guards_and_db_close(self):
        for path in ('/DEEPSEEK_CONFIG.ini','/data/audio_index.json','/models/live_svm.joblib'):
            self.assertEqual(request(path,method='GET')[0],404)
        self.assertEqual(request('/api/upload',raw=b'a',headers={'Origin':'https://other.example'})[0],403)
        self.assertEqual(request('/',method='GET')[0],200)
        with server.connect() as conn:conn.execute('SELECT 1')
        with self.assertRaises(sqlite3.ProgrammingError):conn.execute('SELECT 1')

    def test_real_local_http_with_mocked_deepseek_transport(self):
        """Real TCP/HTTP + llm_client JSON parser. ML and remote responses are fixtures."""
        class Opener:
            def open(self, req, timeout):
                body=json.loads(req.data);payload=json.loads(body['messages'][1]['content'])
                assert req.full_url=='https://api.deepseek.com/chat/completions'
                assert len(payload['analysis']['context']['scores'])==8
                raw={'intents':intents()}
                return io.BytesIO(json.dumps({'choices':[{'message':{'content':json.dumps(raw)},'finish_reason':'stop'}]}).encode())
        httpd=start_app.bind_server(server.Handler,0);thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        def http_request(path,data=None,raw=None):
            conn=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
            try:
                body=raw if raw is not None else json.dumps(data or {})
                conn.request('POST',path,body,{'Content-Type':'application/json'})
                result=conn.getresponse();return result.status,json.loads(result.read())
            finally:conn.close()
        try:
            code,s=http_request('/api/upload',raw=self.audio);self.assertEqual(code,200)
            with patch('llm_client.config',return_value=('test-secret','test-model')),patch('urllib.request.build_opener',return_value=Opener()):
                code,s=http_request('/api/intents/generate',{'id':s['id'],'model':MODEL})
            self.assertEqual(code,200);self.assertEqual(len(s['initial_intents']),3);self.assertNotIn('test-secret',json.dumps(s))
        finally:httpd.shutdown();httpd.server_close();thread.join()


if __name__=='__main__':unittest.main()
