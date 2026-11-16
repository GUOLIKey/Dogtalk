"""Synthetic contract/HTTP fixtures. These tests never measure real prediction quality."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scene_schema as schema
import scene_cache
import analysis_contract as contract
import intent_rules as rules
import dialogue
import server
import live_audio

def metadata():
    return dict(label_version=schema.LABEL_VERSION,label_config_sha256=schema.CONFIG_SHA256,
        model_version=schema.MODEL_VERSION,score_kind=schema.SCORE_KIND,classes_=list(reversed(schema.LABELS)),split_id='b'*64)

def row():
    return dict(actual_original='playing with human',target_six='play related',predicted='play related',
        scores=schema.score_dict(list(reversed(schema.LABELS)),[1,2,3,4,5,6]),duration=3.,prediction_mode='oof',fold=1,model_metadata=metadata())

def emotion(arousal='High',valence='Positive',usable=True):
    e=dict(available=True,model='SYNTHETIC TEST ONLY',duration=3.,model_commit='fixture',recipe='fixture',chunks=[])
    for axis,labels,winner in [('arousal',contract.AROUSAL,arousal),('valence',contract.VALENCE,valence)]:
        e[axis]=dict(label=winner,scores={l:.8 if l==winner else .1 for l in labels},usable=usable,margin=.7,agreement=1.)
    return e

def analysis():return contract.bundle(contract.context_result(row()),contract.emotion_result(emotion()))

def candidates(a=None):
    a=a or analysis();rc=rules.derive(a)['rule_context']
    return [dict(intent_id=i,title=dialogue.CATALOG_BY_ID[i]['title'],description='待核查目的',status='possible',uncertainties=['需要动作观察'],
        supporting_signals=[dict(source='context',label='play related',score=a['context']['scores']['play related'],explanation='这是预测线索')],source_rule_ids=[rc['primary_rule_id']],outside_rule_pool_reason=None,
        supporting_observations=[],contradicting_observations=[],subtype=None) for i in ['I02','I03','I04']]

def envelope(raw,a=None):return dict(raw,primary_rule_id=rules.derive(a or analysis())['rule_context']['primary_rule_id'],emotion_summary_ref='server_emotion_summary')

def answer(text='它把玩具递过来，互动暂停后又主动返回。',kind='text'):
    return dict(answer_id='a1',turn=1,answer=text,question='暂停后它做了什么？',kind=kind,answer_revision=0)

def response(states=('possible','possible','possible'),history=None,action='ask',selected=None):
    h=history or []
    return envelope(dict(action=action,selected_intent_id=selected,coexisting_intent_ids=[],
        assessments=[dict(intent_id=i['intent_id'],status=s,reason='结合直接观察核查',evidence=[dict(answer_id='a1',quote=h[-1]['answer'],role='contradict' if s=='excluded' else 'support')] if h and s!='possible' else []) for i,s in zip(candidates(),states)],
        question=dict(text='它与当前对象的距离后来怎样变化？',focus='距离',distinguishes=['I02','I03','I04']) if action=='ask' else None,remaining_uncertainties=['仍可能遗漏其他目的']))

class Harness(server.Handler):
    def __init__(self,path,data=None,raw=None):
        self.path=path;b=raw if raw is not None else json.dumps(data or {}).encode()
        self.rfile=io.BytesIO(b);self.headers={'Content-Length':str(len(b)),'Host':'localhost:8882','X-Filename':'home.wav'}
    def send(self,status,data,mime=None):self.status=status;self.output=data

def request(path,data=None,raw=None,get=False):
    h=Harness(path,data,raw);h.do_GET() if get else h.do_POST();return h.status,h.output

class SchemaTests(unittest.TestCase):
    def test_exact_mapping_and_legacy_ids(self):
        self.assertEqual(len(schema.ORIGINAL_TO_SIX),8);self.assertEqual(len(set(schema.ORIGINAL_TO_SIX.values())),6)
        for s in ['playing with human','playing with other animals','playing with toy']:self.assertEqual(schema.target(s),'play related')
        self.assertEqual([s['id'] for s in schema.SCENES],[f'S{i:02d}' for i in range(1,7)])
        with self.assertRaises(ValueError):schema.target('A3')
    def test_noncanonical_classes_order(self):
        values=schema.score_dict(list(reversed(schema.LABELS)),range(6))
        self.assertEqual(values['alerting to sounds'],5);self.assertEqual(values['taking shower'],0)
        c=contract.context_result(row());self.assertEqual(c['original_preferred']['label'],'play related')
        self.assertNotEqual(c['score_ranking'][0],c['original_preferred']['label'])
        self.assertEqual(c['classes_order'],list(reversed(schema.LABELS)))
    def test_eight_classes_and_wrong_version_rejected(self):
        for classes in [schema.ORIGINAL_LABELS,schema.LABELS[:-1],schema.LABELS+['rest']]:
            with self.assertRaises(ValueError):schema.score_dict(classes,range(len(classes)))
        for key in ('label_version','model_version','label_config_sha256'):
            r=row();r['model_metadata'][key]='old'
            with self.assertRaises(ValueError):contract.context_result(r)
    def test_nonfinite_rejected(self):
        r=row();r['scores']['rest']=float('nan')
        with self.assertRaises(ValueError):contract.context_result(r)
    def test_24_unique_rules(self):
        rules.validate_config();self.assertEqual(len(rules.RULES),24)
        self.assertEqual(set(rules.RULES),{f'S{s:02d}-E{e}' for s in range(1,7) for e in range(1,5)})
        self.assertEqual(len(dialogue.CATALOG_BY_ID),6)
        self.assertEqual(rules.RULES['S03-E1']['primary_hint_id'],'I02')
        self.assertIsNone(rules.RULES['S03-E2']['primary_hint_id'])
    def test_all_quadrants_and_fallbacks(self):
        for ar,va,q in [('High','Positive','E1'),('High','Negative','E2'),('Low','Positive','E3'),('Low','Negative','E4')]:
            self.assertEqual(rules.emotion_summary(emotion(ar,va))['quadrant_id'],q)
        self.assertEqual(rules.emotion_summary(emotion('Medium'))['status'],'central')
        self.assertEqual(rules.emotion_summary(emotion(valence='Neutral'))['status'],'central')
        self.assertEqual(rules.emotion_summary(emotion(usable=False))['status'],'uncertain')
        self.assertEqual(rules.emotion_summary({'available':False})['status'],'unavailable')
    def test_cache_versions_classes_and_fold(self):
        data=dict(version=schema.CACHE_VERSION,label_version=schema.LABEL_VERSION,label_config_sha256=schema.CONFIG_SHA256,model_version=schema.MODEL_VERSION,identity='a'*64,split_id='b'*64,items={'c'*64:row()})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cache.json';path.write_text(json.dumps(data));self.assertEqual(len(scene_cache.load_index(path)['items']),1)
            for key in ('version','label_version','model_version','label_config_sha256','split_id'):
                bad=copy.deepcopy(data);bad[key]='old';path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):scene_cache.load_index(path)
            bad=copy.deepcopy(data);bad['items']['c'*64]['target_six']='rest';path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):scene_cache.load_index(path)

class DialogueTests(unittest.TestCase):
    def test_first_payload_full_scores_and_no_gold(self):
        a=analysis();a['gold']='SECRET_GOLD';a['synthetic']='SECRET_BACKGROUND';a['context']['actual']='SECRET_GOLD'
        r=envelope({'intents':candidates(a)},a)
        with patch.object(dialogue,'chat_json',return_value=r) as llm:
            generated=dialogue.generate_intents(a,'mock')
        payload=json.loads(llm.call_args.args[1][1]['content'])
        self.assertEqual(set(payload['scene_result']['scores']),set(schema.LABELS))
        for ax in ('arousal','valence'):self.assertEqual(len(payload['emotion_result'][ax]['scores']),3)
        self.assertNotIn('SECRET_',json.dumps(payload));self.assertEqual(len({i['intent_id'] for i in generated['intents']}),3)
        self.assertTrue(all(i['status']=='possible' for i in generated['intents']))
    def test_explicit_observation_updates_and_unknown_retains(self):
        cs=candidates();h=[answer()];previous=dialogue.initial_states(cs)
        d=dialogue.normalize(response(('supported','possible','possible'),h),cs,previous,h)
        self.assertEqual(d['assessments'][0]['status'],'supported')
        h=[answer('不知道 / 没观察到','unknown')]
        d=dialogue.normalize(response(('excluded','excluded','excluded'),h),cs,previous,h)
        self.assertTrue(all(s['status']=='possible' for s in d['assessments']))
    def test_multiple_all_excluded_and_budget_never_force_unique(self):
        cs=candidates();h=[answer()]
        for states in [('supported','supported','possible'),('excluded','excluded','excluded')]:
            d=dialogue.normalize(response(states,h,'conclude','I02'),cs,dialogue.initial_states(cs),h)
            self.assertIsNone(d['selected_intent_id']);self.assertEqual(d['action'],'uncertain')
        d=dialogue.normalize(response(history=h),cs,dialogue.initial_states(cs),h,question_budget={'used':dialogue.MAX_QUESTIONS})
        self.assertEqual(d['action'],'uncertain');self.assertIsNone(d['question'])
    def test_missing_candidate_or_fake_quote_rejected(self):
        cs=candidates();h=[answer()];r=response(history=h);r['assessments'].pop()
        with self.assertRaises(ValueError):dialogue.normalize(r,cs,dialogue.initial_states(cs),h)
        r=response(('supported','possible','possible'),h);r['assessments'][0]['evidence'][0]['quote']='NOT IN ANSWER'
        with self.assertRaises(ValueError):dialogue.normalize(r,cs,dialogue.initial_states(cs),h)
    def test_correction_can_restore_excluded(self):
        cs=candidates();old=dialogue.initial_states(cs);old[0]['status']='excluded'
        h=[answer()];h[0]['answer_revision']=1;r=response(('supported','possible','possible'),h)
        r['assessments'][0]['evidence'][0]['answer_revision']=1
        d=dialogue.normalize(r,cs,old,h);self.assertEqual(d['assessments'][0]['status'],'supported')
    def test_api_failure_has_no_fallback(self):
        with patch.object(dialogue,'chat_json',side_effect=RuntimeError('API unavailable')):
            with self.assertRaisesRegex(RuntimeError,'API unavailable'):dialogue.generate_intents(analysis(),'mock')

class HandlerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.db=patch.object(server,'DB',self.root/'sessions.sqlite');self.db.start()
        self.idx=patch.object(server,'INDEX',self.root/'data'/'scene_six_v1'/'audio_index.json');self.idx.start()
        server._cache={};server._mtime=None;server._index_meta={}
    def tearDown(self):self.idx.stop();self.db.stop();self.temp.cleanup()
    def test_upload_chat_history_export_same_schema(self):
        with patch.object(live_audio,'predict_bytes',return_value=dict(row(),prediction_mode='live')),patch.object(server.emotion_audio,'predict_bytes',return_value=emotion()):
            code,s=request('/api/upload',raw=b'SYNTHETIC AUDIO TEST')
        self.assertEqual(code,200);self.assertEqual(len(s['analysis']['context']['scores']),6);before=copy.deepcopy(s['analysis'])
        with patch.object(server,'deepseek_models',return_value=['DeepSeek::mock']),patch.object(dialogue,'chat_json',return_value=envelope({'intents':candidates()})):
            code,s=request('/api/intents/generate',dict(id=s['id'],model='DeepSeek::mock',source='real',pre_observation='它正盯着主人手里的鸭腿。'))
        self.assertEqual(code,200,s)
        with patch.object(dialogue,'chat_json',return_value=response()):code,s=request('/api/chat/start',dict(id=s['id']))
        self.assertEqual(code,200,s)
        with patch.object(dialogue,'chat_json',side_effect=RuntimeError('API offline')):
            code,err=request('/api/chat/answer',dict(id=s['id'],revision=0,answer='它靠近了食物。'))
        self.assertEqual(code,500);self.assertEqual(server.get(s['id'])['dialogue']['history'],[])
        code,s=request('/api/chat/stop',dict(id=s['id'],revision=0))
        self.assertEqual(code,200);self.assertEqual(s['analysis'],before)
        code,export=request('/api/export',get=True);self.assertEqual(export['scene_schema']['version'],schema.LABEL_VERSION)
        self.assertNotIn('gold',export['records'][0]);self.assertNotIn('original_scene_label',export['records'][0])
        self.assertEqual(export['intent_summary']['completed'],1)
        self.assertEqual(export['legacy_scene_summary']['manual']['n'],0)
    def test_missing_models_both_attempted(self):
        with patch.object(live_audio,'predict_bytes',side_effect=ValueError('six model missing')),patch.object(server.emotion_audio,'predict_bytes',return_value={'available':False,'message':'emotion missing'}) as emotion_call:
            code,s=request('/api/upload',raw=b'x')
        emotion_call.assert_called_once();self.assertEqual(s['phase'],'blocked');self.assertEqual(s['initial_intents'],[])
        code,err=request('/api/intents/generate',dict(id=s['id'],model='mock'));self.assertEqual(code,400)
    def test_old_cache_not_loaded_and_old_history_not_mutated(self):
        old_index=self.root/'data'/'audio_index.json';old_index.parent.mkdir();old_index.write_text('{"items":{"old":{}}}')
        self.assertEqual(server.index(),{})
        old=dict(id='old',created=1,filename='old.wav',schema_version='intent-session-v3',analysis={'context':{'scores':dict.fromkeys(schema.ORIGINAL_LABELS,1)}},result=None)
        server.save(old);public=server.public(old);self.assertTrue(public['legacy']);self.assertEqual(len(public['analysis']['context']['scores']),8)
        self.assertEqual(server.get('old'),old)
    def test_real_missing_six_model_status(self):
        with patch.object(live_audio,'MODEL',self.root/'models'/'scene_six_v1'/'live_svm.joblib'),patch.object(live_audio,'META',self.root/'models'/'scene_six_v1'/'live_meta.json'):
            self.assertFalse(live_audio.status()['ready']);self.assertIn('六分类',live_audio.status()['message'])

if __name__=='__main__':unittest.main()
