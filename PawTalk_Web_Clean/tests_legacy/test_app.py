"""No network needed. Tests real handlers through an in-memory request harness."""
import unittest,tempfile,pathlib,os,sys,json,io,hashlib
from unittest.mock import patch
TEST_DIR=tempfile.TemporaryDirectory();os.environ['PAWTALK_STATE']=TEST_DIR.name
import server
server.STATE=pathlib.Path(TEST_DIR.name)
server.DB=server.STATE/'sessions.sqlite'
from engine import infer,evaluate_records,LABELS,KEYS,parse_context
from synthetic import generate

class Harness(server.Handler):
    def __init__(self,path,data=None,raw=None,headers=None):
        self.path=path;body=raw if raw is not None else json.dumps(data or {}).encode()
        self.rfile=io.BytesIO(body);self.headers={'Content-Length':str(len(body)),'Host':'localhost:8765',**(headers or {})}
    def send(self,status,data,mime='application/json; charset=utf-8'):
        self.status=status;self.output=data

def request(path,data=None,raw=None,headers=None,method='POST'):
    h=Harness(path,data,raw,headers)
    if method=='GET':h.do_GET()
    else:h.do_POST()
    return h.status,h.output

class Tests(unittest.TestCase):
    def setUp(self):
        with server.connect() as c:c.execute('DELETE FROM sessions');c.execute('DELETE FROM revealed')
        self.audio=b'unit-test-only-not-a-recording'
        self.sha=hashlib.sha256(self.audio).hexdigest()
        self.row=dict(actual='begging for food',predicted='seeking attention',scores=dict(zip(LABELS,[6,5,1,2,3,0,7,4])),duration=3.)
        self.patch=patch('server.index',return_value={self.sha:self.row});self.patch.start()
    def tearDown(self):self.patch.stop()
    def upload(self):
        code,s=request('/api/upload',raw=self.audio,headers={'X-Filename':'test.wav'})
        self.assertEqual(code,200);self.assertIsNone(s['gold']);self.assertNotIn('row',s);return s
    def predict(self,s):return request('/api/predict',dict(id=s['id'],source='real',answers={'food':'yes'},text='主人正在拿食物'))
    def test_audio_context_fusion_independent_of_label(self):
        a=infer(self.row,{'food':'yes'});other=dict(self.row,actual='rest')
        self.assertEqual(a,infer(other,{'food':'yes'}));self.assertEqual(a['combined_pred'],'begging for food')
        self.assertEqual(a['audio_pred'],'seeking attention');self.assertEqual(a['background_pred'],'begging for food')
    def test_real_trial_lock_reveal_and_duplicate(self):
        s=self.upload();code,out=self.predict(s);self.assertEqual(code,200)
        self.assertEqual(request('/api/history',method='GET')[1]['summary']['n'],1)
        self.assertEqual(self.predict(s)[0],400)
        code,label=request('/api/reveal',dict(id=s['id']));self.assertEqual(label['gold'],'讨食')
        new=self.upload();self.assertTrue(new['contaminated']);self.predict(new)
        self.assertEqual(request('/api/history',method='GET')[1]['summary']['n'],1)
    def test_early_reveal_excluded(self):
        s=self.upload();request('/api/reveal',dict(id=s['id']));self.predict(s)
        self.assertEqual(request('/api/history',method='GET')[1]['summary']['n'],0)
    def test_synthetic_stays_excluded_even_if_user_selects_real(self):
        s=self.upload();code,out=request('/api/generate',dict(id=s['id']))
        self.assertEqual(code,200);self.assertFalse(out['scene']['is_observed']);self.assertTrue(out['scene']['label_used'])
        self.predict(s);self.assertEqual(request('/api/history',method='GET')[1]['summary']['n'],0)
    def test_export_does_not_leak_unsubmitted_labels(self):
        s=self.upload();code,out=request('/api/export',method='GET');self.assertEqual(out['records'],[])
        self.predict(s);code,out=request('/api/export',method='GET');self.assertEqual(len(out['records']),1)
        self.assertTrue(self.upload()['contaminated'])
    def test_unknown_audio_and_bad_inputs(self):
        self.assertEqual(request('/api/upload',raw=b'unknown')[0],422)
        self.assertEqual(request('/api/demo',headers={'Origin':'https://untrusted.example'})[0],403)
        s=self.upload();self.assertEqual(request('/api/predict',dict(id=s['id'],source='real',answers={}))[0],400)
        self.assertEqual(request('/api/predict',dict(id=s['id'],source='real',answers={'food':'yes','water':'invalid'}))[0],400)
    def test_demo_is_labeled_and_not_evaluated(self):
        code,s=request('/api/demo');self.assertTrue(s['demo']);self.predict(s)
        self.assertEqual(request('/api/history',method='GET')[1]['summary']['n'],0)
    def test_static_whitelist(self):
        self.assertEqual(request('/',method='GET')[0],200)
        self.assertEqual(request('/data/audio_index.json',method='GET')[0],404)
        self.assertEqual(request('/dev_manifest.csv',method='GET')[0],404)
    def test_llm_parser_contract_no_gold_and_verbatim_evidence(self):
        content=json.dumps({'food':{'value':'yes','evidence':'拿出食物'},'water':{'value':'yes','evidence':'不存在于原文'}})
        captured=[]
        def call(model,messages,temperature=0):captured.append(messages);return json.loads(content)
        with patch('engine.ollama_models',return_value=['test-model']),patch('engine.chat_json',side_effect=call):
            out=parse_context('主人拿出食物','test-model')
        self.assertEqual(out['food']['value'],'yes');self.assertEqual(out['water']['value'],'unknown')
        self.assertNotIn('gold',json.dumps(captured));self.assertEqual(captured[0][1]['content'],'主人拿出食物')
    def test_batch_all_eight_labels_synthetic(self):
        for label in LABELS:
            r=generate(label);self.assertTrue(r['label_used']);self.assertFalse(r['is_observed']);self.assertEqual(r['generator'],'template-v1')

if __name__=='__main__':unittest.main()
