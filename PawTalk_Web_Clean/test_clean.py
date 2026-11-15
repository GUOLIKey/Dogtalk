"""Launcher/config regressions retained for Intent v9."""
import hashlib,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import start_app,server,llm_client,engine
from test_intent_flow import request

class StartupTests(unittest.TestCase):
    def test_integrity_detects_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'engine.py').write_bytes(b'new code')
            (root/'build_manifest.json').write_text(json.dumps({'sha256':{'engine.py':hashlib.sha256(b'new code').hexdigest()}}))
            start_app.verify_install(root);(root/'engine.py').write_bytes(b'old code')
            with self.assertRaises(ValueError):start_app.verify_install(root)
    def test_busy_port_uses_another(self):
        first=start_app.bind_server(server.Handler,0)
        try:
            second=start_app.bind_server(server.Handler,first.server_port)
            try:self.assertNotEqual(first.server_port,second.server_port)
            finally:second.server_close()
        finally:first.server_close()
    def test_only_local_config_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'DEEPSEEK_CONFIG.ini';path.write_text('[deepseek]\napi_key = local-test\nmodel = test-model\n')
            with patch.object(llm_client,'CONFIG',path),patch.dict('os.environ',{'DEEPSEEK_API_KEY':'other-key'}):
                self.assertEqual(llm_client.config(),('local-test','test-model'))
    def test_bad_config_keeps_status_available(self):
        with patch.object(server,'llm_config',side_effect=ValueError('配置格式错误')):
            code,result=request('/api/status',method='GET')
        self.assertEqual(code,200);self.assertEqual(result['llm_error'],'配置格式错误');self.assertEqual(result['llm_models'],[])
    def test_shared_transport(self):
        self.assertIs(engine.ollama_models,llm_client.available_models);self.assertIs(engine.chat_json,llm_client.chat_json)

if __name__=='__main__':unittest.main()
