import io,json,tempfile,unittest,urllib.error
from pathlib import Path
from unittest.mock import patch
import llm_client as c
class Opener:
 def __init__(self,callback):self.callback=callback
 def open(self,req,timeout):return self.callback(req)
class Tests(unittest.TestCase):
 def test_placeholder_not_configured(self):
  with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True),patch.object(c,'CONFIG',Path(d)/'config.ini'),patch.object(c,'ollama_models',return_value=[]):
   c.CONFIG.write_text('[deepseek]\napi_key = PASTE_YOUR_API_KEY_HERE\nmodel = deepseek-flash\n')
   self.assertEqual(c.available_models(),[])
 def test_deepseek_payload_and_bearer(self):
  def callback(req):
   self.assertEqual(req.full_url,'https://api.deepseek.com/chat/completions')
   self.assertEqual(req.get_header('Authorization'),'Bearer fake-test-secret')
   payload=json.loads(req.data);self.assertEqual(payload['response_format'],{'type':'json_object'})
   self.assertNotIn('fake-test-secret',json.dumps(payload));self.assertFalse(payload['stream'])
   return io.BytesIO(json.dumps({'choices':[{'message':{'content':'{"ok":true}'}}]}).encode())
  with patch.object(c,'config',return_value=('fake-test-secret','deepseek-flash')),patch('urllib.request.build_opener',return_value=Opener(callback)):
   self.assertEqual(c.chat_json('DeepSeek::deepseek-flash',[{'role':'user','content':'JSON please'}]),{'ok':True})
 def test_errors_do_not_echo_provider_response_or_key(self):
  def callback(req):raise urllib.error.HTTPError(req.full_url,401,'contains fake-test-secret',{},io.BytesIO(b'fake-test-secret'))
  with patch.object(c,'config',return_value=('fake-test-secret','deepseek-flash')),patch('urllib.request.build_opener',return_value=Opener(callback)):
   with self.assertRaises(ValueError) as e:c.chat_json('DeepSeek::deepseek-flash',[])
   self.assertIn('401',str(e.exception));self.assertNotIn('fake-test-secret',str(e.exception))
 def test_ollama_unchanged_and_no_bearer(self):
  def callback(req):
   self.assertEqual(req.full_url,'http://127.0.0.1:11434/api/chat');self.assertIsNone(req.get_header('Authorization'))
   return io.BytesIO(b'{"message":{"content":"{\\"ok\\":true}"}}')
  with patch('urllib.request.build_opener',return_value=Opener(callback)):
   self.assertEqual(c.chat_json('qwen2.5:3b',[]),{'ok':True})
 def test_config_bom_and_equals(self):
  with tempfile.TemporaryDirectory() as d,patch.dict('os.environ',{},clear=True),patch.object(c,'CONFIG',Path(d)/'config.ini'):
   c.CONFIG.write_text('[deepseek]\napi_key = test-key-with=equals\nmodel = deepseek-flash\n',encoding='utf-8-sig')
   self.assertEqual(c.config(),('test-key-with=equals','deepseek-flash'))
class RecoveryTests(unittest.TestCase):
 def response(self,content,reason='stop'):
  return {'choices':[{'finish_reason':reason,'message':{'content':content}}]}
 def test_fenced_json(self):
  self.assertEqual(c.parse_output(self.response('```json\n{"ok":true}\n```'),True),{'ok':True})
 def test_empty_retry_uses_strict_text_fallback(self):
  calls=[]
  def callback(req):
   calls.append(json.loads(req.data))
   content='   ' if len(calls)==1 else '{"ok":true}'
   return io.BytesIO(json.dumps(self.response(content)).encode())
  original=[{'role':'user','content':'JSON please'}]
  with patch.object(c,'config',return_value=('fake-test-secret','deepseek-flash')),patch('urllib.request.build_opener',return_value=Opener(callback)):
   self.assertEqual(c.chat_json('DeepSeek::deepseek-flash',original),{'ok':True})
  self.assertEqual(len(calls),2);self.assertEqual(len(original),1)
  self.assertNotIn('response_format',calls[1]);self.assertEqual(calls[1]['max_tokens'],3200)
 def test_retry_is_bounded_and_errors_are_private(self):
  calls=[]
  def callback(req):
   calls.append(1)
   return io.BytesIO(json.dumps(self.response('fake-test-secret')).encode())
  with patch.object(c,'config',return_value=('fake-test-secret','deepseek-flash')),patch('urllib.request.build_opener',return_value=Opener(callback)):
   with self.assertRaises(ValueError) as ctx:c.chat_json('DeepSeek::deepseek-flash',[])
  self.assertEqual(len(calls),2);self.assertIn('INVALID_JSON',str(ctx.exception));self.assertNotIn('fake-test-secret',str(ctx.exception))
 def test_length_rejected_even_if_json_complete(self):
  with self.assertRaises(c.OutputError) as ctx:c.parse_output(self.response('{"ok":true}','length'),True)
  self.assertEqual(ctx.exception.code,'OUTPUT_LIMIT')
 def test_no_partial_extraction_or_nonstandard_numbers(self):
  for content in ['prefix {"ok":true}','{"x":NaN}','{"x":1','{"ok":true} {"ok":false}','[]']:
   with self.subTest(content=content),self.assertRaises(c.OutputError):c.parse_output(self.response(content),True)
 def test_blank_and_missing_schema_distinguished(self):
  for result,code in [(self.response('  '),'EMPTY_CONTENT'),({},'RESPONSE_SCHEMA'),({'choices':[]},'RESPONSE_SCHEMA')]:
   with self.subTest(code=code),self.assertRaises(c.OutputError) as ctx:c.parse_output(result,True)
   self.assertEqual(ctx.exception.code,code)
 def test_refusal_not_retried(self):
  with patch.object(c,'_chat_once',side_effect=c.OutputError('REFUSED','refused')) as call:
   with self.assertRaises(c.OutputError):c.chat_json('DeepSeek::deepseek-flash',[])
   self.assertEqual(call.call_count,1)
 def test_non_json_http_body_not_exposed(self):
  with patch.object(c,'config',return_value=('fake-test-secret','deepseek-flash')),patch('urllib.request.build_opener',return_value=Opener(lambda req:io.BytesIO(b'fake-test-secret'))):
   with self.assertRaises(c.OutputError) as ctx:c.chat_json('DeepSeek::deepseek-flash',[])
   self.assertEqual(ctx.exception.code,'HTTP_BODY');self.assertNotIn('fake-test-secret',str(ctx.exception))
if __name__=='__main__':unittest.main()
