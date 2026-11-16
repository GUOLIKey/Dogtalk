"""Real HTTP + Chromium; scene/API mocked; optional real E4 subprocess on a sine WAV."""
import copy
import io
import json
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
import server
import dialogue
import start_app
from test_intent_flow import ROW, EMOTION
from test_context_first_v12 import gen, update

def main():
    import argparse
    from playwright.sync_api import sync_playwright
    parser=argparse.ArgumentParser();parser.add_argument('--browser-executable');parser.add_argument('--real-emotion',action='store_true');args=parser.parse_args()
    calls=[]
    def llm(model,messages,**kwargs):
        payload=json.loads(messages[1]['content']);calls.append(payload)
        assert len(payload['scene_result']['scores'])==6
        assert payload['scene_result']['label_version']=='scene-six-v1'
        for axis in ('arousal','valence'):assert len(payload['emotion_result'][axis]['scores'])==3
        if args.real_emotion:
            assert payload['emotion_result']['model_version']=='emotion-e4-rich153-svm-v1'
            for axis in ('arousal','valence'):
                assert payload['emotion_result'][axis]['score_kind']=='svm_decision_function_ovr'
                assert payload['emotion_result'][axis]['usable'] is False
                assert payload['emotion_result'][axis]['agreement'] is None
        assert 'gold' not in payload and 'actual' not in payload
        if payload['stage']=='generate_intents':return gen('')
        return update(question=f'本次第{len(calls)}项观察：它接下来做了什么？')
    with tempfile.TemporaryDirectory() as td:
        if args.real_emotion:
            import numpy as np,soundfile as sf
            import emotion_e4_worker as e4
            audio=io.BytesIO();sf.write(audio,.1*np.sin(np.arange(22050)*.07),22050,format='WAV');upload=audio.getvalue()
            runtime=Path(td)/'e4_runtime.json'
            runtime.write_text(json.dumps({'python':sys.executable,'recipe':'full_mfcc20_delta_spectral_rhythm_v1','model_version':e4.MODEL_VERSION}),encoding='utf-8')
            emotion_patch=patch.object(server.emotion_audio,'RUNTIME',runtime)
        else:
            upload=b'explicit mock model fixture'
            emotion_patch=patch.object(server.emotion_audio,'predict_bytes',return_value=copy.deepcopy(EMOTION))
        with patch.object(server,'DB',Path(td)/'sessions.sqlite'),patch.object(server,'index',return_value={}),\
             patch.object(server.live_audio,'predict_bytes',return_value=dict(copy.deepcopy(ROW),prediction_mode='live')),\
             emotion_patch,\
             patch.object(server,'deepseek_models',return_value=['DeepSeek::mock']),patch.object(dialogue,'chat_json',side_effect=llm):
            httpd=start_app.bind_server(server.Handler,0);t=threading.Thread(target=httpd.serve_forever,daemon=True);t.start()
            try:
                with sync_playwright() as pw:
                    browser=pw.chromium.launch(executable_path=args.browser_executable,headless=True,args=['--no-sandbox'])
                    page=browser.new_page(viewport={'width':1440,'height':1000});page.set_default_timeout(120000);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                    page.goto(f'http://127.0.0.1:{httpd.server_port}/')
                    page.wait_for_function("document.getElementById('llmSelect').value !== ''")
                    assert page.locator('#buildVersion').inner_text()==server.APP_VERSION
                    assert not page.locator('#versionWarning').is_visible()
                    page.locator('#audioFile').set_input_files({'name':'mock_home.wav','mimeType':'audio/wav','buffer':upload})
                    page.locator('#sceneScores tr').nth(6).wait_for()
                    assert page.locator('#sceneScores tr').count()==7
                    assert '玩耍相关情境' in page.locator('#sceneScores').inner_text()
                    assert '与人玩耍' not in page.locator('#sceneScores').inner_text()
                    page.locator('#generateIntents').click()
                    page.locator('#unknownBtn').wait_for()
                    for n in range(3):
                        page.locator('#unknownBtn').click()
                        page.wait_for_function('(n)=>document.querySelectorAll("#chatLog .chatTurn").length===n',arg=n+1)
                    page.locator('#resultCard').wait_for()
                    assert not page.locator('#questionBox').is_visible()
                    assert page.locator('#turnBadge').inner_text().find('3 / 3')>=0
                    assert '暂不确定' in page.locator('#resultTitle').inner_text()
                    export=page.request.get(f'http://127.0.0.1:{httpd.server_port}/api/export').json()
                    assert export['scene_schema']['version']=='scene-six-v1'
                    assert len(export['records'][0]['analysis']['context']['scores'])==6
                    if args.real_emotion:
                        em=export['records'][0]['analysis']['emotion']
                        assert em['model_version']=='emotion-e4-rich153-svm-v1'
                        assert em['available'] and em['aggregation']=='whole_clip_statistics'
                        assert 'E4 rich153 + RBF SVM' in page.locator('#emotionCard').inner_text()
                    assert len(export['records'][0]['dialogue']['history'])==3
                    assert export['records'][0]['result']['uncertain']
                    assert not errors,errors
                    for width in (1440,390):
                        page.set_viewport_size({'width':width,'height':900})
                        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
                    browser.close()
            finally:httpd.shutdown();httpd.server_close();t.join()
    print('PASS: real Chromium upload, six score rows, context-first generation, three unknown answers, uncertain result, export and desktop/mobile widths. Scene/DeepSeek mocked. Emotion: '+('real packaged E4 subprocess on synthetic WAV' if args.real_emotion else 'mocked')+'. No model quality claim.')

if __name__=='__main__':main()
