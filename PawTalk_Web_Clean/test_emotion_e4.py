"""E4 contracts and real packaged model smoke tests. No live DeepSeek calls."""
import copy,hashlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import analysis_contract as contract
import emotion_e4_worker as worker
import intent_rules,dialogue,server
from test_intent_flow import ROW,EMOTION

class FakeSVM:
    classes_=np.array(['High','Low','Medium'])
    def decision_function(self,x):return np.tile([2.2,-.2,1.1],(len(x),1))
    def predict(self,x):return np.array(['Low']*len(x))

def fixture():
    return dict(available=True,model='E4 fixture',model_version=worker.MODEL_VERSION,recipe='fixture',duration=1.,aggregation='whole_clip_statistics',chunks=[],
                artifact_sha256={'arousal':'a'*64,'valence':'b'*64},
                arousal=worker.axis_result(FakeSVM(),np.zeros((1,153)),worker.AXES['arousal']),
                valence=dict(label='Positive',name='正向',display='正向（模型预测）',scores={'Negative':-.1,'Neutral':.8,'Positive':2.1},classes_order=['Negative','Neutral','Positive'],score_kind=worker.SCORE_KIND,original_preferred={'label':'Positive','score':2.1},margin=1.3,agreement=None,usable=False,usability_policy=worker.USABILITY_POLICY))

class E4ContractTests(unittest.TestCase):
    def test_real_classes_and_predict_not_argmax_are_preserved(self):
        raw=fixture();value=contract.emotion_result(raw)
        self.assertEqual(value['arousal']['label'],'Low');self.assertEqual(value['arousal']['top_score_label'],'High')
        self.assertEqual(value['arousal']['scores'],{'Low':-.2,'Medium':1.1,'High':2.2})
        self.assertIsNone(value['arousal']['agreement']);self.assertFalse(value['arousal']['usable'])
        self.assertEqual(contract.emotion_result(value),value)
    def test_unvalidated_e4_gate_cannot_be_forged(self):
        for change in ('usable','classes','version','scores'):
            raw=fixture()
            if change=='usable':raw['arousal']['usable']=True
            elif change=='classes':raw['arousal']['classes_order']=['Low','Low','High']
            elif change=='version':raw['model_version']='old-or-unknown'
            else:raw['arousal']['scores']['Low']=float('nan')
            with self.assertRaises(ValueError):contract.emotion_result(raw)
    def test_legacy_emotion_history_keeps_original_score_semantics(self):
        raw=copy.deepcopy(EMOTION);result=contract.emotion_result(raw)
        self.assertEqual(result['arousal']['scores'],raw['arousal']['scores']);self.assertEqual(result['arousal']['score_kind'],'uncalibrated_model_score')
        self.assertIsNone(result['model_version'])
    def test_ready_analysis_keeps_all_scores_without_forcing_quadrant(self):
        e=contract.emotion_result(fixture());a=contract.bundle(contract.context_result(copy.deepcopy(ROW)),e)
        self.assertTrue(a['ready']);self.assertEqual(intent_rules.derive(a)['emotion_summary']['status'],'uncertain')
        self.assertEqual(intent_rules.derive(a)['emotion_summary']['usable_version'],worker.USABILITY_POLICY)
        self.assertIsNone(intent_rules.derive(a)['rule_context']['primary_rule_id'])
        raw=copy.deepcopy(a);raw['emotion']['gold']='SECRET';raw['emotion']['arousal']['actual']='SECRET'
        payload=dialogue.input_analysis(raw,'C')
        self.assertNotIn('SECRET',json.dumps(payload));self.assertEqual(len(payload['emotion']['arousal']['scores']),3)
        self.assertEqual(payload['emotion']['arousal']['score_kind'],worker.SCORE_KIND)
        self.assertNotIn('emotion',dialogue.input_analysis(a,'B'))
    def test_missing_assets_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises((ValueError,FileNotFoundError)):worker.load_models(td)
    def test_upload_keeps_e4_provenance_and_scene_scores(self):
        with tempfile.TemporaryDirectory() as td,patch.object(server,'DB',Path(td)/'test.sqlite'),patch.object(server,'index',return_value={}),patch.object(server.live_audio,'predict_bytes',return_value=copy.deepcopy(ROW)),patch.object(server.emotion_audio,'predict_bytes',return_value=fixture()):
            s=server.analyze_upload(b'synthetic upload fixture','home.wav')
            self.assertEqual(s['phase'],'analyzed');self.assertEqual(s['analysis']['emotion']['model_version'],worker.MODEL_VERSION)
            self.assertEqual(s['analysis']['context']['scores'],ROW['scores'])
            self.assertEqual(s['emotion_summary']['status'],'uncertain')
            self.assertEqual(s['analysis']['emotion']['arousal']['label'],'Low')

@unittest.skipUnless((worker.HERE/'emotion_e4_assets'/'arousal.joblib').is_file(),'packaged real models not yet built')
class E4PackagedModelTests(unittest.TestCase):
    def test_real_weights_self_check(self):self.assertTrue(worker.self_check(worker.HERE/'emotion_e4_assets')['ok'])
    def test_real_wav_inference_and_contract(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'signal.wav';sf.write(p,.1*np.sin(np.arange(22050)*.07),22050)
            raw=worker.predict(p,worker.HERE/'emotion_e4_assets');result=contract.emotion_result(raw)
            self.assertTrue(result['available']);self.assertEqual(result['aggregation'],'whole_clip_statistics')
            for axis in worker.AXES:self.assertEqual(set(result[axis]['scores']),set(worker.AXES[axis]));self.assertFalse(result[axis]['usable'])
    def test_asset_tampering_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as td:
            for p in (worker.HERE/'emotion_e4_assets').iterdir():shutil.copy2(p,Path(td)/p.name)
            p=Path(td)/'arousal.joblib';p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):worker.load_models(td)
    def test_http_upload_real_e4_then_mock_llm_interview_and_export(self):
        import sys,threading,urllib.request,soundfile as sf
        import start_app
        from test_context_first_v12 import gen,update
        audio=io.BytesIO();sf.write(audio,.1*np.sin(np.arange(22050)*.07),22050,format='WAV')
        calls=[]
        def llm(model,messages,**kwargs):
            payload=json.loads(messages[1]['content']);calls.append(payload)
            self.assertEqual(len(payload['scene_result']['scores']),6)
            em=payload['emotion_result'];self.assertEqual(em['model_version'],worker.MODEL_VERSION)
            for axis in worker.AXES:
                self.assertEqual(len(em[axis]['scores']),3)
                self.assertEqual(em[axis]['score_kind'],worker.SCORE_KIND)
            return gen('') if payload['stage']=='generate_intents' else update(question=f'第{len(calls)}次观察，它接下来做了什么？')
        with tempfile.TemporaryDirectory() as td:
            runtime=Path(td)/'runtime.json'
            runtime.write_text(json.dumps({'python':sys.executable,'recipe':'full_mfcc20_delta_spectral_rhythm_v1','model_version':worker.MODEL_VERSION}),encoding='utf-8')
            with patch.object(server,'DB',Path(td)/'sessions.sqlite'),patch.object(server,'index',return_value={}),patch.object(server.live_audio,'predict_bytes',return_value=copy.deepcopy(ROW)),patch.object(server.emotion_audio,'RUNTIME',runtime),patch.object(server,'deepseek_models',return_value=['DeepSeek::mock']),patch.object(dialogue,'chat_json',side_effect=llm):
                httpd=start_app.bind_server(server.Handler,0);thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
                base=f'http://127.0.0.1:{httpd.server_port}'
                def http(path,data=None,raw=None):
                    body=raw if raw is not None else json.dumps(data).encode() if data is not None else None
                    req=urllib.request.Request(base+path,data=body,headers={'Content-Type':'audio/wav' if raw else 'application/json','X-Filename':'e4_synthetic_test.wav'})
                    with urllib.request.urlopen(req,timeout=180) as response:return json.loads(response.read())
                try:
                    s=http('/api/upload',raw=audio.getvalue());self.assertTrue(s['analysis']['ready'])
                    original=copy.deepcopy(s['analysis'])
                    s=http('/api/intents/generate',{'id':s['id'],'model':'DeepSeek::mock','source':'real','flow':'context_first','pre_observation':''})
                    self.assertEqual(len(s['initial_intents']),3)
                    for _ in range(3):
                        s=http('/api/chat/answer',{'id':s['id'],'revision':s['dialogue']['revision'],'answer':'不知道','kind':'unknown'})
                    self.assertTrue(s['result']['uncertain']);self.assertEqual(s['analysis'],original)
                    record=http('/api/export')['records'][0]
                    self.assertEqual(record['analysis'],original);self.assertEqual(len(record['dialogue']['history']),3)
                    self.assertTrue(calls)
                finally:httpd.shutdown();httpd.server_close();thread.join()
if __name__=='__main__':unittest.main()
