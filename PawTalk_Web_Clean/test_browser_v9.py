"""Real Chromium UI + real local HTTP, fixture audio models and mocked DeepSeek only."""
import copy
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
import dialogue
import server
import start_app
from test_intent_flow import ROW, EMOTION, MODEL
from test_catalog_v2 import candidates as intents, response


def main():
    import argparse
    from playwright.sync_api import sync_playwright
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser-executable', default=None)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        def llm(model, messages, **kwargs):
            payload=json.loads(messages[1]['content'])
            assert len(payload['analysis']['context']['scores']) == 8
            assert len(payload['analysis']['emotion']['arousal']['scores']) == 3
            assert len(payload['analysis']['emotion']['valence']['scores']) == 3
            if payload['stage']=='generate_intents':return {'intents':intents()}
            history=payload['history']
            if history and history[-1]['kind']=='unknown':
                r=response(intents(),history,['excluded']*3);r['question']['text']='主人靠近之后，叫声有什么变化？';return r
            if history:return response(intents(),history,['excluded','supported','excluded'],'conclude','I02')
            return response(intents())
        with patch.object(server,'DB',Path(td)/'sessions.sqlite'),patch.object(server,'index',return_value={}), \
             patch.object(server.live_audio,'predict_bytes',return_value=dict(copy.deepcopy(ROW),actual=None,prediction_mode='live')), \
             patch.object(server.emotion_audio,'predict_bytes',return_value=copy.deepcopy(EMOTION)), \
             patch.object(server,'deepseek_models',return_value=[MODEL]),patch.object(dialogue,'chat_json',side_effect=llm):
            httpd=start_app.bind_server(server.Handler,0)
            thread=threading.Thread(target=httpd.serve_forever,daemon=True);thread.start()
            try:
                with sync_playwright() as p:
                    browser=p.chromium.launch(executable_path=args.browser_executable,headless=True,args=['--no-sandbox'])
                    page=browser.new_page(viewport={'width':1440,'height':1000})
                    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                    page.goto(f'http://127.0.0.1:{httpd.server_port}/')
                    page.wait_for_function("document.getElementById('llmSelect').value !== ''")
                    page.locator('#audioFile').set_input_files({'name':'fixture.wav','mimeType':'audio/wav','buffer':b'fixture-not-real-audio'})
                    page.locator('#generateIntents').wait_for(state='visible')
                    assert page.locator('#sceneScores tr').count()==9
                    assert page.locator('#emotionAxes tr').count()==8
                    page.locator('#source').select_option('real')
                    page.locator('#generateIntents').click()
                    page.locator('#startChat').wait_for(state='visible')
                    assert page.locator('#intentCandidates article').count()==3
                    assert page.locator('#intentCandidates [data-status="possible"]').count()==3
                    page.screenshot(path=str(Path(td)/'desktop.png'),full_page=True)
                    page.locator('#startChat').click();page.locator('#questionBox').wait_for(state='visible')
                    page.locator('#unknownBtn').click()
                    page.wait_for_function("document.getElementById('turnBadge').textContent.includes('1 /')")
                    assert page.locator('#intentCandidates [data-status="possible"]').count()==3
                    page.locator('#chatAnswer').fill('它盯着食物，食物拿走就停止叫，主人一直在旁边，没有环境声音变化。')
                    page.locator('#answerBtn').click();page.locator('#resultCard').wait_for(state='visible')
                    assert '希望开始或继续游戏' in page.locator('#resultTitle').inner_text()
                    assert page.locator('#resultEvidence li').count()==1
                    assert page.locator('#resultExcluded li').count()==2
                    page.set_viewport_size({'width':390,'height':844})
                    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                    page.locator('#resultCard').scroll_into_view_if_needed()
                    page.screenshot(path=str(Path(td)/'mobile.png'))
                    page.locator('#tabRecords').click();page.locator('#recordsView').wait_for(state='visible')
                    assert '暂定结论 1 次' in page.locator('#stats').inner_text()
                    with page.expect_download() as download:page.locator('#exportBtn').click()
                    exported=json.loads(Path(download.value.path()).read_text())
                    assert len(exported['records'][0]['dialogue']['history'])==2
                    assert exported['records'][0]['initial_intents'][0]['intent_id']=='I01'
                    assert 'gold' not in exported['records'][0]
                    page.locator('#historyList button').click()
                    page.locator('#resultCard').wait_for(state='visible')
                    assert not errors, errors
                    browser.close()
                print('Chromium desktop/mobile flow passed: upload, full scores, generated intents, unknown protection, conclusion, history, export, reopen. No page errors. Model/API outputs mocked.')
            finally:httpd.shutdown();httpd.server_close();thread.join()


if __name__=='__main__':main()
