import unittest,json
from unittest.mock import patch
import dialogue as d
from test_catalog_v2 import candidates,history,response
from test_intent_flow import analysis
class Roles(unittest.TestCase):
 def test_equivalent_roles_preserve_relation(self):
  for value,expected in [('支持','support'),('supporting','support'),(' Support ','support'),('反证','contradict'),('contradiction','contradict'),('clarification','clarify')]:
   h=history('主人手里的鸭腿');refs=d.evidence([{'answer_id':'a1','quote':h[0]['answer'],'role':value}],h)
   self.assertEqual(refs[0]['role'],expected)
 def test_ambiguous_missing_and_candidate_status_not_inferred(self):
  for value in (None,'possible','excluded','support|contradict|clarify','neutral',[]):
   with self.assertRaisesRegex(ValueError,'实际值='):d.evidence_role(value)
 def test_typed_fields_supply_role_but_reject_conflict(self):
  item={'answer_id':'a1','quote':'主人手里的鸭腿'}
  self.assertEqual(d.typed_observations([item],'support')[0]['role'],'support')
  with self.assertRaises(ValueError):d.typed_observations([dict(item,role='反证')],'support')
 def test_missing_role_is_repaired_once_with_actionable_feedback(self):
  cs=candidates();h=history('主人手里的鸭腿');good=response(cs,h,['supported','possible','possible']);bad=response(cs,h,['supported','possible','possible']);bad['assessments'][0]['evidence'][0].pop('role')
  with patch.object(d,'chat_json',side_effect=[bad,good]) as call:out=d.next_turn(analysis(),cs,d.initial_states(cs),h,'mock')
  self.assertEqual(call.call_count,2);self.assertIn('实际值=None',json.loads(call.call_args.args[1][1]['content'])['previous_contract_error'])
  self.assertEqual(len(out['assessments']),3)
 def test_unknown_answer_still_not_support(self):
  h=history('不知道');self.assertEqual(d.evidence([{'answer_id':'a1','quote':'不知道','role':'支持'}],h),[])
