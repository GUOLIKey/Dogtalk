import hashlib
import http.client
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import start_app
import test_app
import server
import llm_client
import engine

class FullTests(unittest.TestCase):
    def test_integrity_detects_old_code(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'engine.py').write_bytes(b'new code')
            (root/'build_manifest.json').write_text(json.dumps({'sha256':{'engine.py':hashlib.sha256(b'new code').hexdigest()}}))
            start_app.verify_install(root)
            (root/'engine.py').write_text('old code')
            with self.assertRaises(ValueError):start_app.verify_install(root)

    def test_busy_port_uses_another_server(self):
        first=start_app.bind_server(server.Handler,0)
        try:
            second=start_app.bind_server(server.Handler,first.server_port)
            try:self.assertNotEqual(first.server_port,second.server_port)
            finally:second.server_close()
        finally:first.server_close()

    def test_real_http_status_and_deepseek_parse_flow(self):
        # Real local HTTP; only the external model response is simulated.
        key='fake-test-secret'
        class Opener:
            def open(self,request,timeout):
                body=json.loads(request.data)
                assert request.full_url=='https://api.deepseek.com/chat/completions'
                assert body['model']=='deepseek-flash'
                assert request.get_header('Authorization')=='Bearer '+key
                content=json.dumps({'food':{'value':'yes','evidence':'拿出食物'}})
                return io.BytesIO(json.dumps({'choices':[{'message':{'content':content},'finish_reason':'stop'}]}).encode())
        httpd=start_app.bind_server(server.Handler,0)
        thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
        def request(path,data=None):
            conn=http.client.HTTPConnection('127.0.0.1',httpd.server_port,timeout=5)
            try:
                conn.request('GET' if data is None else 'POST',path,None if data is None else json.dumps(data),{'Content-Type':'application/json'})
                response=conn.getresponse();raw=response.read()
                self.assertNotIn(key.encode(),raw)
                return response.status,json.loads(raw)
            finally:conn.close()
        try:
            with patch.object(llm_client,'config',return_value=(key,'deepseek-flash')),patch.object(server,'llm_config',return_value=(key,'deepseek-flash')),patch.object(llm_client,'ollama_models',return_value=[]),patch('urllib.request.build_opener',return_value=Opener()):
                status,result=request('/api/status')
                self.assertEqual(status,200);self.assertEqual(result['app_version'],'PawTalk Web Dual v8')
                self.assertEqual(result['llm_models'],['DeepSeek::deepseek-flash'])
                self.assertTrue(result['deepseek_configured'])
                _,session=request('/api/demo',{})
                code,result=request('/api/parse',{'id':session['id'],'text':'主人拿出食物','model':'DeepSeek::deepseek-flash'})
                self.assertEqual(code,200);self.assertEqual(result['facts']['food']['value'],'yes')
                self.assertEqual(request('/DEEPSEEK_CONFIG.ini')[0],404)
        finally:httpd.shutdown();httpd.server_close();thread.join()

    def test_bad_config_status_still_keeps_audio_state(self):
        with patch.object(server,'llm_config',side_effect=ValueError('配置格式错误')):
            code,result=test_app.request('/api/status',method='GET')
        self.assertEqual(code,200);self.assertEqual(result['llm_error'],'配置格式错误')
        self.assertEqual(result['llm_models'],[])

    def test_engine_uses_shared_transport(self):
        self.assertIs(engine.ollama_models,llm_client.available_models)
        self.assertIs(engine.chat_json,llm_client.chat_json)

    def test_database_connections_close_on_exit(self):
        with server.connect() as conn:
            conn.execute('SELECT 1')
        with self.assertRaises(sqlite3.ProgrammingError):conn.execute('SELECT 1')
        try:
            with server.connect() as conn:
                raise ValueError('test failure')
        except ValueError:pass
        with self.assertRaises(sqlite3.ProgrammingError):conn.execute('SELECT 1')

    def test_only_current_folder_key_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'DEEPSEEK_CONFIG.ini'
            path.write_text('[deepseek]\napi_key = new-local-key\nmodel = deepseek-flash\n')
            with patch.object(llm_client,'CONFIG',path),patch.dict('os.environ',{'DEEPSEEK_API_KEY':'old-env-key','DEEPSEEK_MODEL':'old-env-model'}):
                self.assertEqual(llm_client.config(),('new-local-key','deepseek-flash'))

if __name__=='__main__':unittest.main()
