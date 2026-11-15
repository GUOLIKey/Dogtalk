"""Fast contracts; external LLM responses mocked, real-model parity tested separately."""
import hashlib,json,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import dialogue as d
import emotion_audio as ea
import emotion_worker as ew
import server
import test_app as t
from test_expansion import INITIAL,response

def prediction(usable=True):
    def axis(classes,label):return {'label':label,'name':ew.NAMES[label],'usable':usable,'display':ew.NAMES[label] if usable else '暂不确定','scores':dict(zip(classes,[.1,.1,.8]))}
    return {'enabled':True,'available':True,'recipe':'test','arousal':axis(ew.AROUSAL,'High'),'valence':axis(ew.VALENCE,'Positive'),'message':'test'}

class ModelContractTests(unittest.TestCase):
    def test_low_separation_not_sent_to_llm(self):
        hint=ea.llm_hint(prediction(False))
        self.assertEqual(hint['axes']['arousal']['label'],'unknown')
        self.assertNotIn('scores',json.dumps(hint));self.assertNotIn('Positive',json.dumps(hint))
        self.assertIsNone(ea.llm_hint(ea.disabled()))
    def test_weighted_scores_and_gate(self):
        out=ew.aggregate([[.8,.1,.1],[.1,.1,.8]],ew.AROUSAL,[(0,2),(2,2.4)])
        self.assertAlmostEqual(out['scores']['Low'],1.64/2.4);self.assertTrue(out['usable'])
        for scores in [[[.34,.33,.33]],[[.49,.02,.49]],[[.5,.25,.25]]]:self.assertFalse(ew.aggregate(scores,ew.AROUSAL,[(0,2)])['usable'])
    def test_bad_scores_and_assets_rejected(self):
        for scores in [[[float('nan'),.1,.9]],[[1,1,1]],[[.1,.9]],[[1.1,0,-.1]]]:
            with self.assertRaises(ValueError):ew.vector_scores(scores)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):ew.verify_assets(td)
    def test_features_original_padding_pcm16_25_frames(self):
        import librosa,soundfile as sf
        from sklearn.preprocessing import StandardScaler
        wave=np.random.default_rng(10).normal(0,.08,60000).astype(np.float32);scaler=StandardScaler().fit(np.arange(50).reshape(-1,1))
        actual,spans=ew.features(wave,scaler);expected=[]
        for start in range(0,len(wave),44100):
            chunk=wave[start:start+44100]
            if len(chunk)<44100:chunk=np.concatenate([chunk,np.zeros(44100-len(chunk))])
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'chunk.wav';sf.write(p,chunk,22050);y,sr=librosa.load(p,sr=22050)
                m=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=1,n_fft=2048,hop_length=512).T
                expected.append(scaler.transform(m[:25]))
        np.testing.assert_allclose(actual,np.asarray(expected,dtype=np.float32),rtol=0,atol=0)
        self.assertEqual(actual.shape,(2,25,1));self.assertAlmostEqual(spans[-1][1],len(wave)/22050)
    def test_missing_setup_preserves_context(self):
        with patch.object(ea,'status',return_value={'ready':False,'message':'run 08'}):out=ea.predict_bytes(b'audio')
        self.assertFalse(out['available']);self.assertNotIn('arousal',out)
    def test_timeout_cleans_temp_files(self):
        with tempfile.TemporaryDirectory() as td:
            config=Path(td)/'runtime.json';config.write_text(json.dumps({'python':'test-python','recipe':'test'}));paths=[]
            def timeout(args,**kwargs):
                paths.append(Path(args[args.index('--audio')+1]));self.assertTrue(paths[0].exists());raise subprocess.TimeoutExpired(args,1)
            with patch.object(ea,'RUNTIME',config),patch.object(ea,'status',return_value={'ready':True}),patch.object(ea.subprocess,'run',side_effect=timeout):out=ea.predict_bytes(b'audio')
            self.assertFalse(out['available']);self.assertFalse(paths[0].exists())

class DialogueEmotionTests(unittest.TestCase):
    def test_audio_hint_cannot_eliminate_activity_without_answers(self):
        r=response(states=['unlikely','supported','unlikely'],action='conclude',selected=INITIAL[1])
        r['emotion_review']={'arousal':{'label':'High','basis':'audio'},'valence':{'label':'Positive','basis':'audio'}}
        out=d.normalize(r,INITIAL,[],emotion=ea.llm_hint(prediction()))
        self.assertEqual(out['action'],'ask');self.assertIsNone(out['selected'])
        self.assertEqual(out['emotion_review']['arousal']['basis'],'audio');self.assertTrue(all(x['status']=='possible' for x in out['assessments']))
    def test_model_only_emotion_must_match_available_axis(self):
        r={'arousal':{'label':'Low','basis':'audio'},'valence':{'label':'Positive','basis':'audio'}}
        out=d.emotion_review(r,[],ea.llm_hint(prediction()))
        self.assertEqual(out['arousal']['label'],'unknown');self.assertEqual(out['valence']['label'],'Positive')
        self.assertTrue(all(v['label']=='unknown' for v in d.emotion_review(r,[],ea.llm_hint(prediction(False))).values()))
    def test_observation_correction_requires_exact_quote(self):
        h=[{'answer':'它一直安静趴着，动作很少。','kind':'text'}]
        r={'arousal':{'label':'Low','basis':'observation','reason':'观察与低活动水平相符。','evidence':[{'turn':1,'quote':'动作很少'}]}}
        out=d.emotion_review(r,h,ea.llm_hint(prediction()));self.assertEqual(out['arousal']['label'],'Low')
        r['arousal']['evidence'][0]['quote']='它很高兴';self.assertEqual(d.emotion_review(r,h,ea.llm_hint(prediction()))['arousal']['label'],'unknown')
        h[0]['kind']='unknown';r['arousal']['evidence'][0]['quote']='动作很少';self.assertEqual(d.emotion_review(r,h,ea.llm_hint(prediction()))['arousal']['label'],'unknown')
    def test_off_sends_no_emotion_and_on_no_gold(self):
        captured=[]
        def fake(model,messages,**kwargs):captured.append(messages);return response()
        with patch.object(d,'chat_json',side_effect=fake):
            out=d.next_turn(INITIAL,[],'test');d.next_turn(INITIAL,[],'test',emotion=ea.llm_hint(prediction()))
        self.assertNotIn('emotion_hint',captured[0][1]['content']);self.assertIsNone(out['emotion_review'])
        payload=json.loads(captured[1][1]['content']);self.assertIn('emotion_hint',payload);self.assertNotIn('gold',payload);self.assertNotIn('actual',payload)

class RouteEmotionTests(unittest.TestCase):
    def setUp(self):
        with server.connect() as c:c.execute('DELETE FROM sessions');c.execute('DELETE FROM revealed')
        self.raw=b'unit-test-audio';self.sha=hashlib.sha256(self.raw).hexdigest()
        self.row={'actual':INITIAL[0],'predicted':INITIAL[0],'scores':{x:(10-INITIAL.index(x) if x in INITIAL else 0) for x in server.LABELS},'duration':3}
        self.index=patch.object(server,'index',return_value={self.sha:self.row});self.index.start()
        self.models=patch.object(server,'ollama_models',return_value=['DeepSeek::test']);self.models.start()
    def tearDown(self):self.index.stop();self.models.stop()
    def upload(self,on):
        code,s=t.request('/api/upload',raw=self.raw,headers={'X-Filename':'test.wav','X-Emotion':'1' if on else '0'});self.assertEqual(code,200);return s
    def start(self,s):
        with patch.object(d,'chat_json',return_value=response()):code,out=t.request('/api/chat/start',{'id':s['id'],'source':'real','model':'DeepSeek::test'})
        self.assertEqual(code,200);return out
    def finish(self,s):
        with patch.object(d,'chat_json',return_value=response(action='uncertain')):code,out=t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'不确定','kind':'unknown'})
        self.assertEqual(code,200);return out
    def test_switch_off_skips_emotion_process(self):
        with patch.object(ea,'predict_bytes',side_effect=AssertionError('must not call')):s=self.upload(False)
        s=self.start(s);self.assertEqual(s['dialogue']['mode'],'context_only');self.assertIsNone(s['dialogue']['emotion_hint'])
    def test_on_stores_separate_prediction_and_mode(self):
        with patch.object(ea,'predict_bytes',return_value=prediction()) as model:s=self.upload(True)
        model.assert_called_once_with(self.raw);s=self.finish(self.start(s))
        self.assertEqual(s['result']['audio_pred'],INITIAL[0]);self.assertEqual(s['result']['mode'],'emotion_assisted')
        self.assertEqual(t.request('/api/history',method='GET')[1]['mode_summaries']['emotion_assisted']['n'],1)
        self.assertEqual(t.request('/api/export',method='GET')[1]['records'][0]['emotion']['arousal']['label'],'High')
    def test_failed_emotion_falls_back(self):
        with patch.object(ea,'predict_bytes',return_value={'enabled':True,'available':False,'message':'failed'}):s=self.upload(True)
        s=self.finish(self.start(s));self.assertEqual(s['result']['mode'],'emotion_unavailable')
        groups=t.request('/api/history',method='GET')[1]['mode_summaries'];self.assertEqual(groups['emotion_unavailable']['n'],1);self.assertEqual(groups['emotion_assisted']['n'],0)
    def test_duplicate_record_dedup_per_mode(self):
        with patch.object(ea,'predict_bytes',return_value=prediction()):
            for on in [False,True,True]:self.finish(self.start(self.upload(on)))
        groups=t.request('/api/history',method='GET')[1]['mode_summaries'];self.assertEqual(groups['context_only']['n'],1);self.assertEqual(groups['emotion_assisted']['n'],1)
    def test_private_files_not_public(self):
        for path in ['/models/emotion_runtime.json','/.venv_emotion/pyvenv.cfg','/emotion_assets.json','/EMOTION_ERROR.txt']:self.assertEqual(t.request(path,method='GET')[0],404)

if __name__=='__main__':unittest.main()
