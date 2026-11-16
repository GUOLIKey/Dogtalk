"""Mock LLM outputs; validate application behavior, not language model quality."""
import json,copy,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import dialogue as d,server
from test_intent_flow import analysis,ROW,EMOTION,request
from test_catalog_v2 import candidates,response,history
TEXT='主人手里拿着鸭腿，它一直盯着鸭腿并向它靠近，还发出哼唧声。这是怎么回事？'
FACT=TEXT.split('。')[0]
def gen(text=TEXT,guess=False):
 return {'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':candidates(),
         'request_understanding':{'user_question':'为什么看着食物哼唧？' if text else '',
          'question_quotes':[{'answer_id':'pre_1','quote':'这是怎么回事？'}] if text else [],
          'statements':[{'answer_id':'pre_1','quote':text.split('。')[0],'kind':'guess' if guess else 'observation'}] if text else []}}
def update(h=(),selected=None,states=None,action='ask',question='它当时主要关注什么对象？'):
 r=response(candidates(),h,states,action,selected)
 r.update(explanation='根据你描述的注意对象与靠近行为，目前更值得考虑食物相关诉求；仍可能有其他原因。' if selected else '现有信息还不足以确定目的，需要补充实际观察。',
          ranked_intent_ids=['I01','I03','I02'],readiness={'critical_conflicts':[], 'comparisons':[{'intent_id':i,'reason':'相比具体食物目标，目前没有同样直接的行为支持。'} for i in ['I01','I02','I03'] if selected and i!=selected]})
 if r['question']:r['question']['text']=question
 return r
class Policy(unittest.TestCase):
 def test_three_question_cap_configured(self):self.assertEqual(d.MAX_QUESTIONS,3)
 def test_early_conclusion_without_excluding_other_two(self):
  h=history(FACT);r=update(h,'I01',['supported','possible','possible'],'conclude')
  out=d.normalize(r,candidates(),d.initial_states(candidates()),h,context_first=True)
  self.assertEqual(out['result_status'],'concluded');self.assertEqual(out['unresolved_intent_ids'],['I02','I03'])
 def test_no_support_or_missing_comparison_or_conflict_cannot_conclude(self):
  for kind in ('unsupported','comparison','conflict','contradiction'):
   h=history(FACT);r=update(h,'I01',['supported','possible','possible'],'conclude')
   if kind=='unsupported':r['assessments'][0]['evidence']=[]
   if kind=='comparison':r['readiness']['comparisons']=[]
   if kind=='conflict':r['readiness']['critical_conflicts']=['用户描述前后矛盾']
   if kind=='contradiction':r['assessments'][0]['evidence'].append(dict(r['assessments'][0]['evidence'][0],role='contradict'))
   self.assertNotEqual(d.normalize(r,candidates(),d.initial_states(candidates()),h,context_first=True)['result_status'],'concluded')
 def test_question_not_counted_as_observation_and_guess_not_support(self):
  for quote,allowed in [('这是怎么回事？',[]),('我猜它饿了',[])]:
   h=[{'answer_id':'pre_1','answer':quote,'kind':'text','observation_quotes':allowed}]
   self.assertEqual(d.evidence([{'answer_id':'pre_1','quote':quote,'role':'support'}],h),[])
 def test_third_answer_returns_ranked_possibilities_not_fourth_question(self):
  r=update();out=d.normalize(r,candidates(),d.initial_states(candidates()),[],question_budget={'used':3},context_first=True)
  self.assertIsNone(out['question']);self.assertEqual(out['result_status'],'multiple_possible')
  final=d.conclusion(analysis(),candidates(),out,[])
  self.assertEqual([i['intent_id'] for i in final['alternatives']],['I01','I03','I02'])
  self.assertNotIn('probability',final);self.assertIn('confidence_note',final)
 def test_compound_question_rejected(self):
  r=update(question='它有没有靠近？是否盯着看？')
  with self.assertRaises(ValueError):d.normalize(r,candidates(),d.initial_states(candidates()),[],context_first=True)
 def test_question_extraction_grounded_in_original(self):
  observations=[{'answer_id':'pre_1','answer':TEXT}]
  raw=gen()['request_understanding'];raw['question_quotes'][0]['quote']='它是不是生病了'
  with self.assertRaises(ValueError):d.context_policy.understanding(raw,observations,d.text)
class ContextRoutes(unittest.TestCase):
 def setUp(self):
  td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
  patches=[patch.object(server,'DB',Path(td.name)/'s.sqlite'),patch.object(server,'index',return_value={}),
   patch.object(server.live_audio,'predict_bytes',return_value=ROW),patch.object(server.emotion_audio,'predict_bytes',return_value=EMOTION),
   patch.object(server,'deepseek_models',return_value=['DeepSeek::test'])]
  for p in patches:p.start();self.addCleanup(p.stop)
 def upload(self):return request('/api/upload',raw=b'contextfixture')[1]
 def start(self,s,text,side):
  with patch.object(d,'chat_json',side_effect=side) as calls:
   code,s=request('/api/intents/generate',{'id':s['id'],'model':'DeepSeek::test','source':'real','flow':'context_first','pre_observation':text})
  self.assertEqual(code,200,s);return s,calls
 def test_description_and_question_reach_first_call_and_direct_explanation(self):
  def model(_model,messages,**kwargs):
   p=json.loads(messages[1]['content'])
   if p['stage']=='generate_intents':
    self.assertEqual(p['history'][0]['answer'],TEXT);self.assertEqual(len(p['scene_result']['scores']),6);self.assertEqual(len(p['emotion_result']['arousal']['scores']),3)
    return gen()
   self.assertEqual(p['request_understanding']['user_question'],'为什么看着食物哼唧？')
   prior=p['independent_or_previous_round_observations'];h=[dict(prior[0],answer=FACT)]
   return update(h,'I01',['supported','possible','possible'],'conclude')
  s,calls=self.start(self.upload(),TEXT,model)
  self.assertEqual(calls.call_count,2);self.assertEqual(s['question_budget']['used'],0)
  self.assertEqual(s['result']['intent_id'],'I01');self.assertIn('食物',s['result']['explanation'])
  self.assertEqual(s['analysis']['context']['original_preferred']['label'],'seeking attention')
 def test_three_followups_hard_stop_and_duplicate_no_charge(self):
  s,_=self.start(self.upload(),'', [gen(''),update()]);self.assertEqual(s['question_budget']['used'],1)
  before=copy.deepcopy(s)
  code,again=request('/api/chat/start',{'id':s['id']});self.assertEqual(again['question_budget'],before['question_budget'])
  for n in range(3):
   with patch.object(d,'chat_json',return_value=update(question=f'补充观察{n+1}：它接下来做了什么？')):
    code,s=request('/api/chat/answer',{'id':s['id'],'revision':n,'answer':'不知道','kind':'unknown'})
   self.assertEqual(code,200,s)
  self.assertEqual(s['question_budget']['used'],3);self.assertIsNotNone(s['result']);self.assertEqual(len(s['result']['alternatives']),3)
  self.assertEqual(request('/api/chat/answer',{'id':s['id'],'revision':3,'answer':'第四次'})[0],400)
 def test_one_answer_can_finish_with_alternatives_unexcluded(self):
  s,_=self.start(self.upload(),'', [gen(''),update()])
  def model(_m,messages,**kw):
   p=json.loads(messages[1]['content'])
   return update(p['history'],'I01',['supported','possible','possible'],'conclude')
  with patch.object(d,'chat_json',side_effect=model):code,s=request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':FACT})
  self.assertEqual(code,200,s);self.assertEqual(s['question_budget']['used'],1);self.assertEqual(s['result']['result_status'],'concluded')
 def test_corrected_answer_reopens_supported_conclusion(self):
  s,_=self.start(self.upload(),'', [gen(''),update()])
  def model(_m,messages,**kw):
   p=json.loads(messages[1]['content']);return update(p['history'],'I01',['supported','possible','possible'],'conclude')
  with patch.object(d,'chat_json',side_effect=model):_,s=request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':FACT})
  ident=s['dialogue']['history'][0]['answer_id']
  with patch.object(d,'chat_json',return_value=update(question='修订后它主要关注哪里？')):
   code,s=request('/api/chat/correct',{'id':s['id'],'revision':1,'answer_id':ident,'answer':'我看错了，手里没有食物，它看着门外。'})
  self.assertEqual(code,200,s);self.assertIsNone(s['result']);self.assertEqual(len(s['dialogue']['answer_revisions']),1)
 def test_initial_second_call_failure_atomic(self):
  s=self.upload()
  with patch.object(d,'chat_json',side_effect=[gen(),ValueError('LLM HTTP 401')]):
   code,out=request('/api/intents/generate',{'id':s['id'],'model':'DeepSeek::test','flow':'context_first','pre_observation':TEXT})
  self.assertEqual(code,400);saved=server.get(s['id']);self.assertEqual(saved['initial_intents'],[]);self.assertEqual(saved['question_budget']['used'],0)
 def test_description_guess_cannot_produce_supported_final(self):
  text='我猜它饿了。这是怎么回事？'
  def model(_m,messages,**kw):
   p=json.loads(messages[1]['content'])
   if p['stage']=='generate_intents':return gen(text,True)
   h=[dict(p['independent_or_previous_round_observations'][0],answer='我猜它饿了')]
   return update(h,'I01',['supported','possible','possible'],'conclude')
  s,_=self.start(self.upload(),text,model)
  self.assertNotEqual(s['result']['result_status'],'concluded');self.assertEqual(s['result']['intent_id'],None)
if __name__=='__main__':unittest.main()
