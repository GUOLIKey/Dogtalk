import copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import dialogue as d,server
from test_intent_flow import ROW,EMOTION,analysis,request

def candidates(ids=('I01','I02','I03')):
 return [{'intent_id':i,'title':d.CATALOG_BY_ID[i]['title'],'description':'待验证的目的','supporting_signals':[{'source':'context','label':'seeking attention','score':6.1,'explanation':'仅作为待核查线索。'}], 'uncertainties':['需要观察'],'status':'possible'} for i in ids]
def history(answer='它盯着食物，在人和食物之间来回看。'):
 return [{'answer_id':'a1','turn':1,'question':'它看着什么？','answer':answer,'kind':'text'}]
def response(cs,h=(),states=None,action='ask',selected=None,coexist=None):
 return {'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','action':action,'selected_intent_id':selected,'coexisting_intent_ids':coexist or [],'assessments':[{'intent_id':i['intent_id'],'status':state,'reason':'观察与该目的相关。','missing_information':[], 'subsumed_by':'I01' if state=='subsumed' else None,'evidence':[{'answer_id':h[-1]['answer_id'],'quote':h[-1]['answer'],'role':'contradict' if state=='excluded' else 'support'}] if h and state!='possible' else []} for i,state in zip(cs,states or ['possible']*len(cs))], 'question':{'text':'叫声前发生了什么变化？','distinguishes':[i['intent_id'] for i in cs]} if action=='ask' else None,'remaining_uncertainties':[]}
class Contracts(unittest.TestCase):
 def test_catalog_complete(self):self.assertEqual(set(d.CATALOG_BY_ID),{f'I0{i}' for i in range(1,7)});self.assertEqual(len(d.rules.SCENES),8)
 def test_full_input_and_modes(self):
  for mode in ('A','B','C'):
   xs=candidates()
   if mode=='A':
    for x in xs:x['supporting_signals']=[]
   with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':xs}) as call:d.generate_intents(analysis(),'test',mode=mode)
   payload=json.loads(call.call_args.args[1][1]['content']);a=payload['analysis']
   if mode=='C':self.assertEqual(len(a['context']['scores']),8);self.assertEqual(len(a['emotion']['arousal']['scores']),3);self.assertFalse(a['emotion']['arousal']['usable'])
   if mode=='A':self.assertNotIn('context',a);self.assertNotIn('emotion',a)
   if mode=='B':self.assertNotIn('emotion',a)
   self.assertNotIn('actual',json.dumps(payload));self.assertNotIn('gold',json.dumps(payload))
 def test_fewer_or_zero_need_reason(self):
  for ids in ([],['I01'],['I01','I02']):
   with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':candidates(ids),'reason':'目前缺少足够线索。'}):self.assertEqual(len(d.generate_intents(analysis(),'test')['intents']),len(ids))
  with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':[]}),self.assertRaises(ValueError):d.generate_intents(analysis(),'test')
 def test_bad_id_rejected(self):
  xs=candidates();xs[0]['intent_id']='begging for food'
  with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':xs}),self.assertRaises(ValueError):d.generate_intents(analysis(),'test')
 def test_subsumed_is_not_excluded(self):
  cs=candidates();h=history();r=d.normalize(response(cs,h,['supported','excluded','subsumed'],'conclude','I01'),cs,d.initial_states(cs),h)
  self.assertEqual(r['result_status'],'concluded');self.assertEqual(r['subsumed_intent_ids'],['I03'])
 def test_ambiguous_avoidance_keeps_multiple(self):
  cs=candidates(['I04','I06']);h=history('它持续后退回避当前对象。');r=d.normalize(response(cs,h,['supported','supported'],'conclude','I04',['I06']),cs,d.initial_states(cs),h)
  self.assertEqual(r['result_status'],'multiple_possible')
 def test_negation_condition_guess_do_not_support(self):
  for answer in ('平时爱吃零食，但这次没有食物，它一直看门口','假如给零食可能会安静','我猜它饿了','不知道，没看到'):
   cs=candidates(['I01']);h=history(answer);r=d.normalize(response(cs,h,['supported'],'conclude','I01'),cs,d.initial_states(cs),h)
   self.assertIsNone(r['selected_intent_id'])
 def test_invalid_reference_and_unknown(self):
  cs=candidates();h=history();raw=response(cs,h,['supported','possible','possible']);raw['assessments'][0]['evidence'][0]['answer_id']='missing'
  with self.assertRaises(ValueError):d.normalize(raw,cs,d.initial_states(cs),h)
 def test_outside_and_budget(self):
  cs=candidates();h=history();r=d.normalize(response(cs,h,['excluded']*3),cs,d.initial_states(cs),h);self.assertEqual(r['result_status'],'outside_candidates')
  h=[dict(history()[0],answer_id=str(i),turn=i+1) for i in range(8)];r=d.normalize(response(cs,h),cs,d.initial_states(cs),h);self.assertEqual(r['result_status'],'multiple_possible')
class Routes(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  p=patch.object(server,'DB',Path(self.tmp.name)/'s.sqlite');p.start();self.addCleanup(p.stop)
  for p in [patch.object(server,'index',return_value={}),patch.object(server.live_audio,'predict_bytes',return_value=ROW),patch.object(server.emotion_audio,'predict_bytes',return_value=EMOTION),patch.object(server,'deepseek_models',return_value=['DeepSeek::test'])]:p.start();self.addCleanup(p.stop)
 def start(self):
  _,s=request('/api/upload',raw=b'fixture')
  with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':candidates()}):code,s=request('/api/intents/generate',{'id':s['id'],'model':'DeepSeek::test','source':'real'})
  self.assertEqual(code,200,s)
  with patch.object(d,'chat_json',return_value=response(candidates())):code,s=request('/api/chat/start',{'id':s['id']})
  self.assertEqual(code,200,s);return s
 def test_correction_and_regeneration_archive(self):
  s=self.start()
  def assess(model,msg,**kw):
   p=json.loads(msg[1]['content']);r=response(candidates(),p['history'],['possible']*3);r['question']['text']='随后它关注的对象如何变化？';return r
  with patch.object(d,'chat_json',side_effect=assess):code,s=request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'它看着食物。'})
  self.assertEqual(code,200,s);ident=s['dialogue']['history'][0]['answer_id']
  with patch.object(d,'chat_json',side_effect=assess):code,s=request('/api/chat/correct',{'id':s['id'],'revision':1,'answer_id':ident,'answer':'我说错了，它看着门外。'})
  self.assertEqual(code,200,s);self.assertEqual(len(s['dialogue']['answer_revisions']),1)
  code,s=request('/api/chat/stop',{'id':s['id'],'revision':2});self.assertEqual(s['result']['result_status'],'user_stopped')
  with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':candidates(['I05','I06']),'reason':'新的外界目标观察。'}):code,s=request('/api/intents/regenerate',{'id':s['id'],'revision':3})
  self.assertEqual(code,200,s);self.assertEqual(len(s['candidate_rounds']),1);self.assertEqual(len(s['initial_intents']),2)
  with patch.object(d,'chat_json',return_value=dict(response(candidates(['I05','I06'])),question={'text':'现在能确认哪个外界对象？','distinguishes':['I05','I06']})):code,s=request('/api/chat/start',{'id':s['id']})
  self.assertEqual(code,200,s);self.assertEqual(s['dialogue']['revision'],4)
  self.assertEqual(request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'旧问题的回答'})[0],400)
 def test_old_record_read_only(self):
  server.save({'id':'old','schema_version':'intent-session-v1','created':0,'filename':'old','result':None})
  self.assertTrue(server.public(server.get('old'))['legacy']);self.assertEqual(request('/api/chat/start',{'id':'old'})[0],400)
if __name__=='__main__':unittest.main()
