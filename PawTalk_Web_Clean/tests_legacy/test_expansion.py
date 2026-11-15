"""Mocked LLM contract + real application handlers; not an LLM accuracy test."""
import copy
import json
import unittest
from unittest.mock import patch
import dialogue as d
import server
import test_app as t
from engine import LABELS, evaluate_records

INITIAL=['playing with human','alerting to sounds','rest']
FOOD='begging for food'
ANSWER='有人在拿食物逗它。'

def history(answer=ANSWER):
    return [{'turn':1,'question':'它在和人做什么？','topic':'human','answer':answer,'kind':'text'}]

def response(labels=INITIAL,states=None,quote=None,turn=1,action='ask',selected=None):
    return {'action':action,'question':{'topic':'food','text':'它主要盯着食物索要，还是跟着人追逐互动？'},
            'selected':selected,'additions':[],
            'assessments':[{'label':x,'status':state,'reason':'根据实际观察，仍需要确认与叫声的关系。',
                            'evidence':[{'turn':turn,'quote':quote}] if quote else []}
                           for x,state in zip(labels,states or ['possible']*len(labels))]}

def addition(label=FOOD,answer=ANSWER,keyword='食物'):
    return {'label':label,'reason':'有人拿食物，但还需确认狗是否在索要食物。',
            'cues':[{'turn':1,'quote':answer,'keyword':keyword,'polarity':'affirmed','time':'current'}]}

def expand_response():
    r=response(INITIAL+[FOOD],['possible','possible','possible','supported'],ANSWER,
               action='conclude',selected=FOOD)
    r['additions']=[addition()]
    return r

class ExpansionLogicTests(unittest.TestCase):
    def test_add_food_preserves_play_and_requires_confirmation(self):
        out=d.normalize(expand_response(),INITIAL,history())
        self.assertEqual(out['pool'],INITIAL+[FOOD]);self.assertEqual(out['action'],'ask')
        self.assertIsNone(out['selected']);self.assertEqual(out['assessments'][-1]['status'],'possible')
        self.assertEqual(out['assessments'][0]['status'],'possible')
        self.assertEqual(out['assessments'][-1]['origin'],'context')
        self.assertEqual(out['additions'][0]['cues'][0]['keyword'],'食物')
        self.assertEqual(INITIAL,['playing with human','alerting to sounds','rest'])

    def test_no_addition_on_first_question_or_unknown_answer(self):
        with self.assertRaises(ValueError):d.normalize(expand_response(),INITIAL,[])
        h=history();h[0]['kind']='unknown'
        with self.assertRaises(ValueError):d.normalize(expand_response(),INITIAL,h)

    def test_reject_negative_past_hypothetical_or_uncertain_cue_tags(self):
        for field,value,answer in [('polarity','negated','没有食物'),('time','past','昨天拿食物逗过它'),
                                    ('time','hypothetical','假如拿食物会怎样'),('polarity','uncertain','不确定有没有食物')]:
            with self.subTest(value=value):
                r=response();a=addition(answer=answer);a['cues'][0][field]=value;r['additions']=[a]
                with self.assertRaises(ValueError):d.normalize(r,INITIAL,history(answer))
                # A keyword in an answer does not itself trigger a new class.
                r['additions']=[];self.assertEqual(d.normalize(r,INITIAL,history(answer))['pool'],INITIAL)

    def test_reject_shortened_fabricated_or_missing_keyword_quote(self):
        for cue in [{'quote':'食物'},{'quote':'有人给它食物'},{'keyword':'狗盯着食物'},{'turn':2}]:
            r=expand_response();r['additions'][0]['cues'][0].update(cue)
            with self.assertRaises(ValueError):d.normalize(r,INITIAL,history())

    def test_mixed_negation_does_not_block_actual_food_observation(self):
        answer='没有玩具，但有人拿食物逗它。';r=response();r['additions']=[addition(answer=answer)]
        out=d.normalize(r,INITIAL,history(answer));self.assertIn(FOOD,out['pool'])

    def test_only_eight_known_unique_labels(self):
        for label in ['illness',INITIAL[0]]:
            r=response();r['additions']=[addition(label)]
            with self.assertRaises(ValueError):d.normalize(r,INITIAL,history())
        r=response();r['additions']=[addition(),addition()]
        with self.assertRaises(ValueError):d.normalize(r,INITIAL,history())
        r=response(LABELS);r['additions']=[addition(x) for x in LABELS if x not in INITIAL]
        self.assertEqual(set(d.normalize(r,INITIAL,history())['pool']),set(LABELS))

    def test_fourth_candidate_requires_all_three_others_excluded(self):
        h=history();h.append({'turn':2,'topic':'food','question':'在盯什么？','answer':'盯着食物，没和人玩，没有异响，也没休息。','kind':'text'})
        labels=INITIAL+[FOOD]
        r=response(labels,['unlikely','unlikely','possible','supported'],h[-1]['answer'],2,'conclude',FOOD)
        self.assertNotEqual(d.normalize(r,labels,h,INITIAL)['action'],'conclude')
        r['assessments'][2]['status']='unlikely'
        self.assertEqual(d.normalize(r,labels,h,INITIAL)['selected'],FOOD)

    def test_conflicting_correction_can_restore_initial_candidate(self):
        h=history('我刚才说错了，没有食物，是和人玩追逐。')
        r=response(INITIAL+[FOOD],['supported','possible','possible','unlikely'],h[0]['answer'])
        out=d.normalize(r,INITIAL+[FOOD],h,INITIAL)
        self.assertEqual(out['assessments'][0]['status'],'supported')
        self.assertEqual(out['assessments'][-1]['status'],'unlikely');self.assertIsNone(out['selected'])

    def test_prompt_contains_taxonomy_and_preserves_initial_ranks(self):
        captured=[]
        def fake(model,messages):captured.append(messages);return response(INITIAL+[FOOD])
        with patch.object(d,'chat_json',side_effect=fake):
            d.next_turn(INITIAL+[FOOD],history(),'DeepSeek::test',INITIAL,[dict(addition(),turn=1)])
        payload=json.loads(captured[0][1]['content'])
        self.assertEqual([x['label'] for x in payload['candidates']],INITIAL)
        self.assertEqual([x['label'] for x in payload['active_candidates']],INITIAL+[FOOD])
        self.assertEqual([x['label'] for x in payload['taxonomy']],LABELS)
        self.assertNotIn('gold',payload);self.assertNotIn('actual',payload)
        self.assertIn('没有食物',captured[0][0]['content'])

class ExpansionHandlerTests(unittest.TestCase):
    def setUp(self):
        with server.connect() as c:c.execute('DELETE FROM sessions');c.execute('DELETE FROM revealed')
        self.row={'actual':FOOD,'predicted':INITIAL[0],'scores':{x:(10-INITIAL.index(x) if x in INITIAL else 0) for x in LABELS},'duration':3}
        self.s=server.new_session(self.row,'food-outside-test','food.wav')
        self.models=patch.object(server,'ollama_models',return_value=['DeepSeek::test']);self.models.start()
    def tearDown(self):self.models.stop()
    def start(self,sid=None):
        with patch.object(d,'chat_json',return_value=response()):
            code,s=t.request('/api/chat/start',{'id':sid or self.s['id'],'model':'DeepSeek::test','source':'real'})
        self.assertEqual(code,200);return s
    def expand(self,s):
        with patch.object(d,'chat_json',return_value=expand_response()):
            code,s=t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':ANSWER})
        self.assertEqual(code,200);return s
    def finish(self,s):
        answer='它一直盯食物索要，没有和人追逐玩耍，没有异响，也没有休息。'
        r=response(INITIAL+[FOOD],['unlikely']*3+['supported'],answer,2,'conclude',FOOD)
        with patch.object(d,'chat_json',return_value=r):
            code,s=t.request('/api/chat/answer',{'id':s['id'],'revision':1,'answer':answer})
        self.assertEqual(code,200);return s
    def test_end_to_end_expansion_persistence_metrics_and_export(self):
        s=self.expand(self.start());self.assertEqual(s['dialogue']['candidates'],INITIAL)
        self.assertIsNone(s['result']);self.assertIsNone(s['gold'])
        stored=server.get(s['id']);self.assertEqual(stored['dialogue']['pool'],INITIAL+[FOOD])
        self.assertEqual(stored['dialogue']['expansions'][0]['turn'],1)
        s=self.finish(s);self.assertEqual(s['result']['combined_pred'],FOOD)
        self.assertEqual(s['result']['audio_pred'],INITIAL[0]);self.assertEqual(s['result']['method'],d.METHOD)
        summary=t.request('/api/history',method='GET')[1]['adaptive_summary']
        self.assertEqual(summary['top3_recall'],0);self.assertEqual(summary['metrics']['combined']['accuracy'],1)
        self.assertEqual(summary['expanded_trials'],1);self.assertEqual(summary['recovered_outside_top3'],1)
        self.assertEqual(evaluate_records(server.records())['n'],0)
        exported=t.request('/api/export',method='GET')[1]['records'][0]
        self.assertEqual(exported['dialogue']['expansions'][0]['cues'][0]['quote'],ANSWER)
    def test_invalid_expansion_leaves_answer_revision_and_pool_unchanged(self):
        s=self.start();before=server.get(s['id']);r=expand_response();r['additions'][0]['cues'][0]['quote']='食物'
        with patch.object(d,'chat_json',return_value=r):
            code,_=t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':ANSWER})
        self.assertEqual(code,400);self.assertEqual(server.get(s['id']),before)
    def test_live_audio_expansion_has_no_reference_and_no_accuracy_claim(self):
        row=dict(self.row,actual=None,prediction_mode='live')
        s=server.new_session(row,'home-dog','my-dog.wav')
        s=self.finish(self.expand(self.start(s['id'])))
        self.assertFalse(s['has_reference']);self.assertEqual(s['result']['combined_pred'],FOOD)
        self.assertEqual(t.request('/api/history',method='GET')[1]['adaptive_summary']['n'],0)
    def test_previous_method_kept_out_of_new_metrics(self):
        s=self.finish(self.expand(self.start()));saved=server.get(s['id'])
        saved['result']['method']='adaptive-top3-v1';server.save(saved)
        out=t.request('/api/history',method='GET')[1]
        self.assertEqual(out['adaptive_summary']['n'],0);self.assertEqual(out['summary']['n'],0)
        self.assertEqual(len(out['records']),1)

if __name__=='__main__':unittest.main()
