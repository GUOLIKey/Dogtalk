import copy,json,unittest
from unittest.mock import patch
import dialogue as d
from test_catalog_v2 import candidates,history,response
from test_intent_flow import analysis
class Freeform(unittest.TestCase):
 def test_duck_leg_free_text_and_all_candidate_slots(self):
  cs=candidates();h=history('主人手里的鸭腿');h[0]['question']='叫声开始时，它的注意集中在什么对象上？'
  raw=response(cs);raw['question']['text']='当时主人拿着它关注的东西做了什么？'
  with patch.object(d,'chat_json',return_value=raw) as call:out=d.next_turn(analysis(),cs,d.initial_states(cs),h,'mock')
  p=json.loads(call.call_args.args[1][1]['content'])
  self.assertEqual(p['history'][0]['answer'],'主人手里的鸭腿')
  self.assertEqual(p['expected_intent_ids'],['I01','I02','I03'])
  self.assertEqual([i['intent_id'] for i in p['output_schema']['assessments']],p['expected_intent_ids'])
  self.assertEqual(out['action'],'ask')
 def test_documented_aliases_keep_full_evidence(self):
  cs=candidates();h=history('它盯着主人手里的鸭腿。');raw=response(cs,h,['supported','possible','possible'])
  raw['candidates']=raw.pop('assessments');raw['primary_intent_id']=raw.pop('selected_intent_id')
  for i in raw['candidates']:
   i['brief_reason']=i.pop('reason');i['supporting_observations']=i.pop('evidence')
  out=d.normalize(raw,cs,d.initial_states(cs),h)
  self.assertEqual(len(out['assessments']),3);self.assertEqual(out['assessments'][0]['evidence'][0]['quote'],h[0]['answer'])
 def test_partial_reply_repaired_not_filled_locally(self):
  cs=candidates();h=history('主人手里的鸭腿');bad=response(cs);bad['assessments']=bad['assessments'][:1];good=response(cs)
  with patch.object(d,'chat_json',side_effect=[bad,good]) as call:out=d.next_turn(analysis(),cs,d.initial_states(cs),h,'mock')
  self.assertEqual(call.call_count,2);self.assertEqual(len(out['assessments']),3)
  feedback=json.loads(call.call_args.args[1][1]['content'])['previous_contract_error'];self.assertIn('I01,I02,I03',feedback)
 def test_persistent_partial_rejected_and_no_invented_candidate(self):
  cs=candidates();raw=response(cs);raw['assessments']=raw['assessments'][:1]
  with patch.object(d,'chat_json',return_value=raw) as call,self.assertRaisesRegex(ValueError,'评估不完整'):d.next_turn(analysis(),cs,d.initial_states(cs),history(),'mock')
  self.assertEqual(call.call_count,2)
 def test_conflicting_aliases_and_new_id_rejected(self):
  cs=candidates();raw=response(cs);raw['candidates']=[]
  with self.assertRaises(ValueError):d.normalize(raw,cs,d.initial_states(cs),[])
  raw=response(cs);raw['assessments'][0]['intent_id']='I06'
  with self.assertRaises(ValueError):d.normalize(raw,cs,d.initial_states(cs),[])
 def test_no_lexical_option_whitelist(self):
  cs=candidates()
  for answer in ('它看着扫地机器人','它盯着主人手里的鸭腿','它看着阳台飞过去的东西'):
   h=history(answer);raw=response(cs)
   with patch.object(d,'chat_json',return_value=raw) as call:d.next_turn(analysis(),cs,d.initial_states(cs),h,'mock')
   self.assertEqual(json.loads(call.call_args.args[1][1]['content'])['history'][0]['answer'],answer)
if __name__=='__main__':unittest.main()
