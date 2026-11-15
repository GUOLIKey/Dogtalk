import unittest
from evaluate_intents import evaluate,grouped_split
class Tests(unittest.TestCase):
 def test_no_references_no_accuracy(self):self.assertEqual(evaluate([],[])['modes']['C']['primary_accuracy'],None)
 def test_paired_sets(self):
  refs=[{'sha':'s','reference_clear':True,'dog_id':'d','source_id':'source','intent_ids':['I01'],'primary_id':'I01'}]
  rows=[{'sha':'s','schema_version':'intent-session-v3','source':'real','experiment_mode':m,'result':{'intent_id':'I01','result_status':'concluded','coexisting_intent_ids':[]},'initial_intents':[{'intent_id':'I01'}],'dialogue':{'history':[{}]}} for m in 'ABC']
  r=evaluate(rows,refs);self.assertEqual(r['paired_samples'],1);self.assertEqual(r['modes']['C']['micro_f1'],1)
 def test_shared_source_cannot_cross_split(self):
  refs=[{'sha':'a','dog_id':'d1','source_id':'s1'},{'sha':'b','dog_id':'d2','source_id':'s1'},{'sha':'c','dog_id':'d3','source_id':'s2'}]
  r=grouped_split(refs);self.assertEqual('a' in r['test'],'b' in r['test'])
