"""Synthetic model/LLM fixtures: validates routing and provenance, not canine intent accuracy."""
import copy,json,unittest
from unittest.mock import patch
import intent_rules as rules,dialogue as d,server
from test_intent_flow import analysis,ROW,EMOTION,request
from test_catalog_v2 import candidates,history,response,Routes

def quadrant(arousal='High',valence='Positive'):
 a=analysis()
 for axis,label in [('arousal',arousal),('valence',valence)]:
  x=a['emotion'][axis];x.update(label=label,usable=True,scores={k:.8 if k==label else .1 for k in x['scores']})
 return a

def envelope(a,raw,mode='C'):
 return dict(raw,primary_rule_id=rules.derive(a,mode)['rule_context']['primary_rule_id'],emotion_summary_ref='server_emotion_summary')

class Rules(unittest.TestCase):
 def test_all_unique_and_inverse(self):
  rules.validate_config();self.assertEqual(len(rules.RULES),32)
  for ident,idx in rules.reverse_index().items():
   self.assertEqual(set(idx['pool']),{k for k,v in rules.RULES.items() if ident in v['candidate_pool']})
 def test_quadrants(self):
  for ar,va,e in [('High','Positive','E1'),('High','Negative','E2'),('Low','Positive','E3'),('Low','Negative','E4')]:
   q=rules.derive(quadrant(ar,va))['emotion_summary'];self.assertEqual(q['quadrant_id'],e);self.assertNotIn('probability',q)
 def test_central_uncertain_unavailable_priority(self):
  for ar,va in [('Medium','Positive'),('High','Neutral'),('Medium','Neutral')]:
   a=quadrant(ar,va);self.assertEqual(rules.derive(a)['emotion_summary']['status'],'central')
   a['emotion']['arousal']['usable']=False;self.assertEqual(rules.derive(a)['rule_context']['fallback_id'],'F-UNCERTAIN')
  a=analysis();a['emotion']={'available':False};self.assertEqual(rules.derive(a)['emotion_summary']['status'],'unavailable')
  with self.assertRaises(ValueError):d.generate_intents(dict(a,ready=False),'mock')
 def test_ties_invalid_and_no_forced_quadrant(self):
  a=quadrant();a['emotion']['arousal']['scores']={'Low':.4,'Medium':.2,'High':.4}
  self.assertEqual(rules.derive(a)['emotion_summary']['status'],'uncertain')
  a['emotion']['arousal']['label']='Medium';self.assertEqual(rules.derive(a)['emotion_summary']['status'],'unavailable')
 def test_anchor_predict_not_argmax_and_aux_order(self):
  a=quadrant();r=rules.derive(a)['rule_context']
  self.assertEqual(r['primary_rule_id'],'A7-E1');self.assertEqual(r['matched_rule_ids'],['A7-E1','A1-E1','A2-E1'])
  self.assertEqual([x['role'] for x in r['matched_rules']],['primary','auxiliary','auxiliary'])
 def test_empty_rest_and_shared_food_hints(self):
  for e in range(1,5):
   r=rules.RULES[f'A6-E{e}'];self.assertEqual(r['candidate_pool'],[]);self.assertIsNone(r['primary_hint_id'])
  self.assertEqual(rules.RULES['A1-E2']['primary_hint_id'],'I06')
  for e in (1,2):self.assertEqual(rules.RULES[f'A2-E{e}']['primary_hint_id'],'I01')
 def test_full_payload_different_distributions_and_no_answers(self):
  seen=[]
  for value in (.3,2.):
   a=quadrant();a['context']['scores']['rest']=value;a['gold']='secret';a['actual']='secret';a['synthetic']='secret'
   raw=envelope(a,{'intents':candidates()})
   with patch.object(d,'chat_json',return_value=raw) as call:d.generate_intents(a,'mock')
   p=json.loads(call.call_args.args[1][1]['content']);seen.append(p)
   self.assertEqual(len(p['scene_result']['scores']),8);self.assertEqual(len(p['emotion_result']['valence']['scores']),3);self.assertEqual(len(p['emotion_result']['arousal']['scores']),3)
   self.assertNotIn('secret',json.dumps(p));self.assertEqual(p['rule_context']['primary_rule_id'],'A7-E1')
  self.assertNotEqual(seen[0]['scene_result']['scores'],seen[1]['scene_result']['scores'])
 def test_ablations_do_not_leak_emotion_via_rules(self):
  a=quadrant()
  for mode in ('A','B'):
   xs=candidates()
   if mode=='A':
    for x in xs:x['supporting_signals']=[]
   with patch.object(d,'chat_json',return_value=envelope(a,{'intents':xs},mode)) as call:d.generate_intents(a,'mock',mode=mode)
   p=json.loads(call.call_args.args[1][1]['content']);self.assertIsNone(p['emotion_result']);self.assertIsNone(p['emotion_summary']);self.assertEqual(p['rule_context']['matched_rules'],[])
 def test_wrong_rule_and_score_rejected(self):
  a=quadrant()
  for mutation in ('entry','source','score','outside'):
   xs=candidates();raw=envelope(a,{'intents':xs})
   if mutation=='entry':raw['primary_rule_id']='A1-E4'
   if mutation=='source':xs[0]['source_rule_ids']=['A6-E4']
   if mutation=='score':xs[0]['supporting_signals'][0]['score']=.99
   if mutation=='outside':xs[0]['intent_id']='I04'
   with patch.object(d,'chat_json',return_value=raw),self.assertRaises(ValueError):d.generate_intents(a,'mock')
 def test_initial_subtype_and_fake_observation_rejected(self):
  for field in ('subtype','supporting_observations'):
   xs=candidates()
   xs[0][field]={'target':'玩具','supporting_answer_ids':[]} if field=='subtype' else [{'answer_id':'fake','quote':'玩具','role':'support'}]
   with patch.object(d,'chat_json',return_value=envelope(analysis(),{'intents':xs})),self.assertRaises(ValueError):d.generate_intents(analysis(),'mock')
 def test_empty_candidates_neutral_question(self):
  raw=response([]);raw['question']['distinguishes']=[]
  with patch.object(d,'chat_json',return_value=raw):out=d.next_turn(analysis(),[],[],[],'mock')
  self.assertEqual(out['action'],'ask');self.assertEqual(out['question']['distinguishes'],[])
 def test_coexisting_supported_vs_unresolved(self):
  cs=candidates(['I04','I06']);h=history('淋水开始它就叫，暂停后不叫。它持续远离拿水管的人。')
  raw=response(cs,h,['supported','supported'],'conclude',coexist=['I04','I06'])
  out=d.normalize(raw,cs,d.initial_states(cs),h);self.assertEqual(out['result_status'],'coexisting_supported');self.assertIsNone(out['selected_intent_id'])
  raw=response(cs,h,['possible','possible'],'uncertain');self.assertEqual(d.normalize(raw,cs,d.initial_states(cs),h)['result_status'],'multiple_possible')
 def test_subtype_current_revision(self):
  cs=candidates(['I01']);h=history();h[0]['answer_revision']=1;raw=response(cs,h,['supported'],'conclude','I01')
  raw['assessments'][0]['subtype']={'target':'食物','supporting_answer_ids':['a1']}
  out=d.normalize(raw,cs,d.initial_states(cs),h);self.assertEqual(out['assessments'][0]['subtype']['target'],'食物')
  raw['assessments'][0]['evidence'][0]['answer_revision']=0
  with self.assertRaises(ValueError):d.normalize(raw,cs,d.initial_states(cs),h)
 def test_network_error_not_retried_by_semantic_repair(self):
  with patch.object(d,'chat_json',side_effect=ValueError('LLM HTTP 401')) as call:
   with self.assertRaises(ValueError):d.generate_intents(analysis(),'mock')
  self.assertEqual(call.call_count,1)
 def test_repair_once_no_fabrication(self):
  good=envelope(analysis(),{'intents':candidates()});bad=dict(good,primary_rule_id='invented')
  with patch.object(d,'chat_json',side_effect=[bad,good]) as call:out=d.generate_intents(analysis(),'mock')
  self.assertEqual(call.call_count,2);self.assertEqual(len(out['intents']),3)
  self.assertIn('规则入口',json.loads(call.call_args.args[1][1]['content'])['previous_contract_error'])
 def test_unobserved_text_not_negative_evidence(self):
  cs=candidates(['I01']);h=history('未观察到')
  raw=response(cs,h,['excluded'],'uncertain')
  out=d.normalize(raw,cs,d.initial_states(cs),h)
  self.assertEqual(out['assessments'][0]['status'],'possible')
 def test_actual_classes_order_is_saved_and_checked(self):
  from analysis_contract import context_result
  row=copy.deepcopy(ROW);order=list(reversed(list(row['scores'])))
  row['model_metadata']={'classes_':order}
  out=context_result(row);self.assertEqual(out['classes_order'],order)
  row['model_metadata']['classes_']=['rest']*8
  with self.assertRaises(ValueError):context_result(row)
 def test_exhausted_never_asks_again(self):
  raw=response(candidates())
  out=d.normalize(raw,candidates(),d.initial_states(candidates()),[],question_budget={'used':8})
  self.assertIsNone(out['question']);self.assertEqual(out['result_status'],'multiple_possible')

class BudgetRoutes(Routes):
 def test_shared_budget_idempotence_and_empty_regeneration(self):
  s=self.start();self.assertEqual(s['question_budget']['used'],1)
  code,s=request('/api/chat/start',{'id':s['id']});self.assertEqual(s['question_budget']['used'],1)
  def answer(model,msg,**kw):
   p=json.loads(msg[1]['content']);r=response(candidates(),p['history']);r['question']['text']=f"第{len(p['history'])+1}问：这次注意对象有什么变化？";return r
  for n in range(d.MAX_QUESTIONS):
   with patch.object(d,'chat_json',side_effect=answer):code,s=request('/api/chat/answer',{'id':s['id'],'revision':n,'answer':'未观察到','kind':'unknown'})
   self.assertEqual(code,200,s)
  self.assertEqual(s['question_budget']['used'],d.MAX_QUESTIONS);self.assertIsNotNone(s['result'])
  with patch.object(d,'chat_json',return_value=envelope(analysis(),{'intents':[],'reason':'还没有明确目标线索'})):
   code,s=request('/api/intents/regenerate',{'id':s['id'],'revision':d.MAX_QUESTIONS})
  self.assertEqual(code,200,s);self.assertEqual(len(s['candidate_rounds']),1);self.assertEqual(s['question_budget']['remaining'],0)
  with patch.object(d,'chat_json',return_value=response([])):code,s=request('/api/chat/start',{'id':s['id']})
  self.assertEqual(code,200,s);self.assertIsNone(s['dialogue']['decision']['question']);self.assertEqual(s['result']['result_status'],'insufficient_evidence')
 def test_real_http_full_mock_flow(self):
  import http.client,threading
  from http.server import ThreadingHTTPServer
  httpd=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
  thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
  def post(path,data=None,raw=None):
   conn=http.client.HTTPConnection(*httpd.server_address,timeout=5)
   try:
    conn.request('POST',path,body=raw if raw is not None else json.dumps(data).encode())
    r=conn.getresponse();return r.status,json.loads(r.read())
   finally:conn.close()
  try:
   code,s=post('/api/upload',raw=b'mocked audio bytes');self.assertEqual(code,200)
   self.assertEqual(s['emotion_summary']['status'],'uncertain')
   with patch.object(d,'chat_json',return_value=envelope(analysis(),{'intents':candidates()})):
    code,s=post('/api/intents/generate',{'id':s['id'],'model':'DeepSeek::test'})
   self.assertEqual(code,200,s)
   with patch.object(d,'chat_json',return_value=response(candidates())):code,s=post('/api/chat/start',{'id':s['id']})
   self.assertEqual(code,200,s);self.assertEqual(s['question_budget']['used'],1)
   code,s=post('/api/chat/stop',{'id':s['id'],'revision':s['dialogue']['revision']})
   self.assertEqual(code,200,s);self.assertEqual(s['result']['result_status'],'user_stopped')
  finally:httpd.shutdown();httpd.server_close();thread.join()
 def test_zero_can_start_and_generate_retry_dedup(self):
  _,s=request('/api/upload',raw=b'fixture')
  with patch.object(d,'chat_json',return_value=envelope(analysis(),{'intents':[],'reason':'先补充背景'})) as mock:
   data={'id':s['id'],'model':'DeepSeek::test'}
   code,s=request('/api/intents/generate',data);code,s=request('/api/intents/generate',data)
  self.assertEqual(mock.call_count,1);self.assertIsNone(s['result'])
  with patch.object(d,'chat_json',return_value=response([])):code,s=request('/api/chat/start',{'id':s['id']})
  self.assertEqual(code,200,s);self.assertEqual(s['question_budget']['used'],1)

if __name__=='__main__':unittest.main()
