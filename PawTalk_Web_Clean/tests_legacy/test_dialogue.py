import copy
import json
import unittest
from unittest.mock import patch
import dialogue as d
import test_app as t
import server
from engine import LABELS,evaluate_records

LABELS3=['rest','alerting to sounds','playing with other animals']
def raw(action='ask',question='当时有没有其他动物与它互动？',topic='animal',selected=None,statuses=None,quotes=None):
    statuses=statuses or ['possible']*3;quotes=quotes or [None]*3
    return {'action':action,'question':{'topic':topic,'text':question},'selected':selected,'assessments':[{'label':x,'status':state,'reason':'根据回答中的事件与互动情况。','evidence':[] if not ref else [{'turn':ref[0],'quote':ref[1]}]} for x,state,ref in zip(LABELS3,statuses,quotes)]}

class LogicTests(unittest.TestCase):
    def test_first_question_has_candidates_but_no_answers_or_gold(self):
        captured=[]
        def fake(model,messages,temperature=0):captured.append(messages);return raw()
        with patch.object(d,'chat_json',side_effect=fake):out=d.next_turn(LABELS3,[],'DeepSeek::test')
        self.assertEqual(out['action'],'ask')
        payload=json.loads(captured[0][1]['content'])
        self.assertEqual(payload['answers'],[])
        self.assertEqual([x['label'] for x in payload['candidates']],LABELS3)
        self.assertNotIn('gold',payload);self.assertNotIn('actual',payload)

    def test_two_answer_adaptive_finish(self):
        h=[{'turn':1,'topic':'animal','question':'当时有没有其他动物与它互动？','answer':'没有其他动物，只有它在房间里。','kind':'text'}]
        r=raw(question='它是不是听到门外声音后马上转头并开始叫？',topic='sound',statuses=['possible','possible','unlikely'],quotes=[None,None,(1,'没有其他动物')])
        out=d.normalize(r,LABELS3,h);self.assertEqual(out['action'],'ask');self.assertEqual(out['question']['topic'],'sound')
        h.append({'turn':2,'topic':'sound','question':out['question']['text'],'answer':'它原本在走动，不是在休息；门外一响，它马上转头并开始叫。','kind':'text'})
        r=raw(action='conclude',selected=LABELS3[1],statuses=['unlikely','supported','unlikely'],quotes=[(2,'不是在休息'),(2,'门外一响，它马上转头并开始叫'),(1,'没有其他动物')])
        out=d.normalize(r,LABELS3,h);self.assertEqual(out['action'],'conclude');self.assertEqual(out['selected'],LABELS3[1])

    def test_one_answer_can_finish_when_explicitly_sufficient(self):
        h=[{'turn':1,'topic':'sound','question':'当时发生了什么？','answer':'它没有休息，也没有别的动物；敲门后马上朝门口叫。','kind':'text'}]
        r=raw(action='conclude',selected=LABELS3[1],statuses=['unlikely','supported','unlikely'],quotes=[(1,'没有休息'),(1,'敲门后马上朝门口叫'),(1,'没有别的动物')])
        self.assertEqual(d.normalize(r,LABELS3,h)['action'],'conclude')

    def test_unknown_and_fabricated_evidence_cannot_eliminate(self):
        h=[{'turn':1,'topic':'animal','question':'有别的动物吗？','answer':'不知道','kind':'unknown'}]
        r=raw(action='conclude',selected=LABELS3[1],statuses=['unlikely','supported','unlikely'],quotes=[(1,'不知道')]*3)
        out=d.normalize(r,LABELS3,h);self.assertNotEqual(out['action'],'conclude')
        self.assertTrue(all(x['status']=='possible' for x in out['assessments']))
        h[0]['kind']='text';r['assessments'][0]['evidence']=[{'turn':1,'quote':'明确没在休息'}]
        self.assertEqual(d.normalize(r,LABELS3,h)['assessments'][0]['status'],'possible')

    def test_no_forced_choice_when_all_candidates_fail(self):
        h=[{'turn':1,'topic':'water','question':'当时在做什么？','answer':'正在洗澡，没有休息，没有异响，也没其他动物。','kind':'text'}]
        r=raw(action='outside',statuses=['unlikely']*3,quotes=[(1,'没有休息'),(1,'没有异响'),(1,'没其他动物')])
        out=d.normalize(r,LABELS3,h);self.assertEqual(out['action'],'outside');self.assertIsNone(out['selected'])

    def test_eight_answers_end_without_forced_choice(self):
        h=[{'turn':i+1,'topic':topic,'question':d.QUESTIONS[topic],'answer':'不确定','kind':'unknown'} for i,topic in enumerate(d.KEYS)]
        out=d.normalize(raw(),LABELS3,h);self.assertEqual(out['action'],'uncertain');self.assertIsNone(out['selected'])

    def test_duplicate_questions_replaced_and_candidates_can_recover(self):
        q=raw()['question']['text'];h=[{'turn':1,'topic':'animal','question':q,'answer':'刚才说错了，其实有只猫在和它互动。','kind':'text'}]
        out=d.normalize(raw(question=q,statuses=['possible','possible','supported'],quotes=[None,None,(1,'有只猫在和它互动')]),LABELS3,h)
        self.assertNotEqual(out['question']['text'],q);self.assertEqual(out['assessments'][2]['status'],'supported')

    def test_invalid_label_rejected(self):
        r=raw();r['assessments'][0]['label']='taking shower'
        with self.assertRaises(ValueError):d.normalize(r,LABELS3,[])

class HandlerTests(unittest.TestCase):
    def setUp(self):
        with server.connect() as c:c.execute('DELETE FROM sessions');c.execute('DELETE FROM revealed')
        row={'actual':'alerting to sounds','predicted':'rest','scores':dict(zip(LABELS,[7,1,0,6,2,8,3,4])),'duration':3}
        self.s=server.new_session(row,'test-sha','00025-test.wav')
        self.models=patch.object(server,'ollama_models',return_value=['DeepSeek::test']);self.models.start()
    def tearDown(self):self.models.stop()
    def start(self):
        with patch.object(d,'chat_json',return_value=raw()):code,s=t.request('/api/chat/start',{'id':self.s['id'],'model':'DeepSeek::test','source':'real'})
        self.assertEqual(code,200);return s
    def test_start_before_any_answer_and_no_gold_exposure(self):
        s=self.start();self.assertEqual(s['dialogue']['history'],[]);self.assertIsNone(s['gold']);self.assertEqual(s['dialogue']['candidates'],LABELS3)
        self.assertIsNone(s['result'])
    def test_full_two_question_selection_then_reveal(self):
        s=self.start()
        r=raw(question='门外有声音后它有没有马上转头并开始叫？',topic='sound',statuses=['possible','possible','unlikely'],quotes=[None,None,(1,'没有其他动物')])
        with patch.object(d,'chat_json',return_value=r):
            code,s=t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'没有其他动物','kind':'text'})
        self.assertEqual(code,200);self.assertIsNone(s['result'])
        r=raw(action='conclude',selected=LABELS3[1],statuses=['unlikely','supported','unlikely'],quotes=[(2,'没在休息'),(2,'门外响了后马上开始叫'),(1,'没有其他动物')])
        with patch.object(d,'chat_json',return_value=r):
            code,s=t.request('/api/chat/answer',{'id':s['id'],'revision':1,'answer':'它没在休息，门外响了后马上开始叫。','kind':'text'})
        self.assertEqual(code,200);self.assertEqual(s['result']['combined_pred'],LABELS3[1]);self.assertIsNone(s['gold'])
        summary=t.request('/api/history',method='GET')[1]['adaptive_summary']
        self.assertEqual(summary['metrics']['combined']['accuracy'],1);self.assertEqual(summary['mean_questions'],2)
        self.assertEqual(t.request('/api/reveal',{'id':s['id']})[1]['gold_label'],LABELS3[1])
    def test_failed_llm_preserves_question_for_retry(self):
        s=self.start()
        with patch.object(d,'chat_json',side_effect=ValueError('Temporary API error')):
            code,_=t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'没有其他动物','kind':'text'})
        self.assertEqual(code,400);stored=server.get(s['id'])
        self.assertEqual(stored['dialogue']['revision'],0);self.assertEqual(stored['dialogue']['history'],[])
    def test_duplicate_answer_rejected_stop_counted_as_abstention(self):
        s=self.start();data={'id':s['id'],'revision':0,'answer':'没观察到','kind':'unknown'}
        with patch.object(d,'chat_json',return_value=raw(question='有刚响起的门铃吗？',topic='sound')):
            code,s=t.request('/api/chat/answer',data);self.assertEqual(code,200)
            self.assertEqual(t.request('/api/chat/answer',data)[0],400)
        code,s=t.request('/api/chat/stop',{'id':s['id'],'revision':1});self.assertEqual(code,200)
        summary=t.request('/api/history',method='GET')[1]['adaptive_summary']
        self.assertEqual(summary['n'],1);self.assertEqual(summary['answered'],0);self.assertEqual(summary['metrics']['combined']['accuracy'],0)
        self.assertEqual(t.request('/api/history',method='GET')[1]['summary']['n'],0)
        self.assertEqual(t.request('/api/chat/answer',dict(data,revision=2))[0],400)
    def test_early_reveal_excludes_dynamic_trial(self):
        t.request('/api/reveal',{'id':self.s['id']});s=self.start()
        with patch.object(d,'chat_json',return_value=raw(action='uncertain')):
            code,s=t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'不清楚','kind':'unknown'})
        self.assertEqual(code,200);self.assertEqual(t.request('/api/history',method='GET')[1]['adaptive_summary']['n'],0)
    def test_export_keeps_full_qa_but_no_unfinished_gold(self):
        s=self.start();self.assertEqual(t.request('/api/export',method='GET')[1]['records'],[])
        with patch.object(d,'chat_json',return_value=raw(action='uncertain')):
            t.request('/api/chat/answer',{'id':s['id'],'revision':0,'answer':'不知道','kind':'unknown'})
        out=t.request('/api/export',method='GET')[1];self.assertEqual(len(out['records'][0]['dialogue']['history']),1)
        self.assertNotIn('api_key',json.dumps(out))

if __name__=='__main__':unittest.main()
