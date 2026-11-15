"""v9.1 disconnect/background upload regression tests. Model outputs are fixtures."""
import copy
import hashlib
import http.client
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import server
import live_audio
import start_app
from upload_jobs import UploadJobs
from test_intent_flow import ROW, EMOTION, request, Harness


def wait_until(fn, timeout=5):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=fn()
        if value:return value
        time.sleep(.01)
    raise AssertionError('Timed out waiting for test worker')


class UploadJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        p=patch.object(server,'DB',Path(self.tmp.name)/'sessions.sqlite');p.start();self.addCleanup(p.stop)
        self.raw=b'async fixture';self.sha=hashlib.sha256(self.raw).hexdigest()
        for p in [patch.object(server,'index',return_value={self.sha:copy.deepcopy(ROW)}),patch.object(server.emotion_audio,'predict_bytes',return_value=copy.deepcopy(EMOTION))]:
            p.start();self.addCleanup(p.stop)
        self.gates=[]
        self.addCleanup(self.cleanup_workers)

    def cleanup_workers(self):
        for gate in self.gates:gate.set()
        if server._job_manager and server._job_manager.db==Path(server.DB):
            for thread in list(server._job_manager.threads.values()):thread.join(5)

    def done(self, ident):
        return wait_until(lambda: (job if (job:=server.jobs().status(ident))['state'] in ('completed','failed') else None))

    def test_start_returns_202_while_model_still_working_and_partial_visible(self):
        release=threading.Event();entered=threading.Event();self.gates.append(release)
        def slow(raw):entered.set();release.wait(5);return copy.deepcopy(EMOTION)
        with patch.object(server.emotion_audio,'predict_bytes',side_effect=slow):
            code,job=request('/api/upload/start',raw=self.raw)
            self.assertEqual(code,202);self.assertTrue(entered.wait(2))
            code,state=request('/api/upload/status?id='+job['job_id'],method='GET')
            self.assertEqual(code,200);self.assertEqual(state['state'],'running')
            self.assertEqual(state['stage'],'emotion');self.assertEqual(len(state['partial']['context']['scores']),8)
            self.assertNotIn('gold',json.dumps(state));self.assertIsNone(state['session_id'])
            release.set();self.done(job['job_id'])
        result=request('/api/upload/status?id='+job['job_id'],method='GET')[1]
        self.assertTrue(result['session']['analysis']['ready'])
        self.assertEqual(server.records()[0]['id'],result['session_id'])

    def test_real_http_acknowledges_before_model_and_recovers_saved_result(self):
        release=threading.Event();entered=threading.Event();self.gates.append(release)
        def slow(raw):entered.set();release.wait(5);return EMOTION
        httpd=start_app.bind_server(server.Handler,0)
        thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        def http_request(path,raw=None):
            c=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=3)
            try:
                c.request('POST' if raw is not None else 'GET',path,raw,{'X-Filename':'fixture.wav'})
                r=c.getresponse();return r.status,json.loads(r.read())
            finally:c.close()
        try:
            with patch.object(server.emotion_audio,'predict_bytes',side_effect=slow):
                code,job=http_request('/api/upload/start',self.raw)
                self.assertEqual(code,202);self.assertTrue(entered.wait(2));self.assertFalse(release.is_set())
                _,out=http_request('/api/upload/status?id='+job['job_id'])
                self.assertEqual(out['state'],'running');release.set();self.done(job['job_id'])
                code,out=http_request('/api/upload/status?id='+job['job_id'])
                self.assertEqual(code,200);self.assertTrue(out['session']['analysis']['ready'])
        finally:release.set();httpd.shutdown();httpd.server_close();thread.join()

    def test_repeat_active_upload_reuses_task(self):
        gate=threading.Event();self.gates.append(gate)
        with patch.object(server.emotion_audio,'predict_bytes',side_effect=lambda raw:(gate.wait(5),copy.deepcopy(EMOTION))[1]) as infer:
            _,a=request('/api/upload/start',raw=self.raw);_,b=request('/api/upload/start',raw=self.raw)
            self.assertEqual(a['job_id'],b['job_id']);gate.set();self.done(a['job_id'])
        infer.assert_called_once()

    def test_error_is_persisted_not_fabricated(self):
        manager=UploadJobs(Path(self.tmp.name)/'jobs.sqlite',lambda *args,**kw:(_ for _ in ()).throw(ValueError('fixture failure')))
        job=manager.submit(b'broken','a.wav')
        result=wait_until(lambda:(r if (r:=manager.status(job['job_id']))['state']=='failed' else None))
        self.assertIn('fixture failure',result['message']);self.assertIsNone(result['session_id'])
        for thread in manager.threads.values():thread.join(2)

    def test_restart_marks_incomplete_but_retains_completed_job(self):
        path=Path(self.tmp.name)/'restore.sqlite'
        a=UploadJobs(path,lambda *a,**k:None)
        base={'created':time.time(),'updated':time.time(),'partial':{},'filename':'dog.wav','sha':'x','stage':'context','message':'working','session_id':None}
        a.write(dict(base,job_id='unfinished',state='running'))
        a.write(dict(base,job_id='finished',state='completed',session_id='saved-session'))
        b=UploadJobs(path,lambda *a,**k:None)
        self.assertEqual(b.status('unfinished')['state'],'interrupted')
        self.assertEqual(b.status('finished')['session_id'],'saved-session')

    def test_bounded_workers(self):
        gate=threading.Event();self.gates.append(gate)
        def analyze(raw,filename,progress):gate.wait(5);return {'id':filename,'analysis':{'ready':True}}
        manager=UploadJobs(Path(self.tmp.name)/'limit.sqlite',analyze)
        try:
            manager.submit(b'1','1');manager.submit(b'2','2')
            with self.assertRaisesRegex(ValueError,'两个'):manager.submit(b'3','3')
        finally:
            gate.set()
            for thread in list(manager.threads.values()):thread.join(3)

    def test_disconnected_browser_does_not_cancel_task(self):
        release=threading.Event();entered=threading.Event();self.gates.append(release)
        def slow(raw):entered.set();release.wait(5);return EMOTION
        h=Harness('/api/upload/start',raw=self.raw)
        writes=[]
        h.send_response=lambda status:writes.append(status);h.send_header=lambda *a:None
        h.end_headers=lambda:None
        class Broken:
            def write(self,body):raise ConnectionAbortedError(10053,'fixture connection closed')
        h.wfile=Broken();h.send=server.Handler.send.__get__(h)
        with patch.object(server.emotion_audio,'predict_bytes',side_effect=slow):
            h.do_POST();self.assertTrue(h.close_connection);self.assertEqual(writes,[202]);self.assertTrue(entered.wait(2))
            ident=next(iter(server.jobs().threads));release.set();self.done(ident)
        self.assertEqual(len(server.records()),1)

    def test_header_and_body_disconnects_do_not_send_second_500(self):
        for where in ('headers','body'):
            h=Harness('/api/upload',raw=self.raw);statuses=[]
            h.send_response=lambda status:statuses.append(status);h.send_header=lambda *a:None
            def disconnect():raise ConnectionResetError(10054,'fixture disconnected')
            h.end_headers=disconnect if where=='headers' else lambda:None
            class Broken:
                def write(self,body):raise BrokenPipeError('fixture broken pipe')
            h.wfile=Broken();h.send=server.Handler.send.__get__(h)
            h.do_POST();self.assertEqual(statuses,[200]);self.assertTrue(h.close_connection)
        self.assertEqual(len(server.records()),2)

    def test_filename_rejected_before_expensive_runtime_load(self):
        meta=Path(self.tmp.name)/'meta.json';meta.write_text(json.dumps({'dataset_filenames':['00025.wav']}))
        with patch.object(live_audio,'META',meta),patch.object(live_audio,'status',return_value={'ready':True}),patch.object(live_audio,'_runtime',None),patch.object(live_audio,'Runtime') as runtime:
            with self.assertRaises(live_audio.DatasetFilenameError):live_audio.predict_bytes(b'fixture','00025.wav')
        runtime.assert_not_called()

    def test_diagnostic_never_returns_gold_or_assumes_holdout(self):
        root=Path(self.tmp.name);(root/'dev_manifest.csv').write_text('audio_path,actual\ndata/00025.wav,SECRET_LABEL\n')
        with patch.object(server,'ROOT',root),patch.object(server,'index',return_value={'somehash':dict(ROW,audio_path='data/00025.wav')}):
            diag=server.recording_diagnostic('00025.wav','changedhash')
        self.assertEqual(diag['same_filename_in_index'],1);self.assertEqual(diag['dev_paths'],['data/00025.wav'])
        self.assertNotIn('SECRET_LABEL',json.dumps(diag));self.assertIn('上传内容不同',diag['note'])

    def test_context_failure_visible_before_emotion_finishes(self):
        gate=threading.Event();entered=threading.Event();self.gates.append(gate)
        def slow(raw):entered.set();gate.wait(5);return EMOTION
        with patch.object(server,'index',return_value={}),patch.object(live_audio,'predict_bytes',side_effect=live_audio.DatasetFilenameError('filename mismatch')),patch.object(server.emotion_audio,'predict_bytes',side_effect=slow):
            _,job=request('/api/upload/start',raw=self.raw,headers={'X-Filename':'00025.wav'})
            self.assertTrue(entered.wait(2));state=server.jobs().status(job['job_id'])
            self.assertFalse(state['partial']['context']['available'])
            self.assertEqual(state['partial']['context']['diagnostic']['filename'],'00025.wav')
            gate.set();self.done(job['job_id'])
        state=request('/api/upload/status?id='+job['job_id'],method='GET')[1]
        self.assertEqual(state['session']['phase'],'blocked');self.assertEqual(state['session']['initial_intents'],[])


if __name__=='__main__':unittest.main()
