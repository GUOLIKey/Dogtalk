import json,unittest
from unittest.mock import patch
import dialogue as d
from test_catalog_v2 import candidates
from test_intent_flow import analysis
class Tests(unittest.TestCase):
 def test_visible_enumeration(self):
  self.assertEqual(len(d.allowed_signals(analysis(),'C')),14)
  self.assertEqual(len(d.allowed_signals(analysis(),'B')),8)
  self.assertEqual(d.allowed_signals(analysis(),'A'),[])
 def test_equivalent_labels(self):
  for source,label,expected_source,expected_label in [('scene','寻求关注','context','seeking attention'),('emotion_result.arousal','low','arousal','Low'),('emotion','中性','valence','Neutral')]:
   out=d.canonical_signal({'source':source,'label':label,'explanation':'待核查'},'C')
   self.assertEqual((out['source'],out['label']),(expected_source,expected_label))
 def test_unknown_and_invisible_not_invented(self):
  for mode,source,label in [('C','context','hungry'),('C','arousal','unknown'),('B','valence','Neutral'),('A','context','rest'),('C','valence','Low')]:
   with self.assertRaises(ValueError):d.canonical_signal({'source':source,'label':label,'explanation':'test'},mode)
 def test_real_generation_input_contains_exact_references(self):
  xs=candidates();xs[0]['supporting_signals']=[{'source':'scene_result','label':'讨食','score':4.1,'explanation':'值得核查'}]
  with patch.object(d,'chat_json',return_value={'primary_rule_id':None,'emotion_summary_ref':'server_emotion_summary','intents':xs}) as call:out=d.generate_intents(analysis(),'test')
  self.assertEqual(out['intents'][0]['supporting_signals'][0]['label'],'begging for food')
  payload=json.loads(call.call_args.args[1][1]['content']);self.assertEqual(len(payload['allowed_model_signals']),14)
  self.assertFalse(next(x for x in payload['allowed_model_signals'] if x['source']=='arousal')['usable'])
 def test_error_contains_offending_pair(self):
  with self.assertRaisesRegex(ValueError,"source='context'.*label='I01'"):
   d.canonical_signal({'source':'context','label':'I01','explanation':'test'},'C')
