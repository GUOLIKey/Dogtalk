"""Fresh installation: no access to old Web folders, no history migration."""
import argparse
import errno
import hashlib
import json
from pathlib import Path
import runpy
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
from app_version import APP_VERSION as VERSION

def verify_install(root=ROOT):
    manifest=json.loads((root/'build_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['sha256'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('程序文件缺失或版本混用：'+name+'。请重新完整解压 E4 更新版本，不要复制旧代码。')

def bind_server(handler,preferred=8882):
    from http.server import ThreadingHTTPServer
    try:return ThreadingHTTPServer(('127.0.0.1',preferred),handler)
    except OSError as exc:
        if exc.errno not in (errno.EADDRINUSE,10048) and getattr(exc,'winerror',None)!=10048:raise
        return ThreadingHTTPServer(('127.0.0.1',0),handler)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    print(VERSION,flush=True)
    print('程序目录：',ROOT,flush=True)
    verify_install()
    import intent_rules
    intent_rules.validate_config()
    import engine
    import llm_client
    if engine.ollama_models is not llm_client.available_models or engine.chat_json is not llm_client.chat_json:
        raise ValueError('语言模型程序版本混用，请完整解压 E4 更新版本。')
    try:
        key,model=llm_client.config()
        print('DeepSeek：'+('已读取本文件夹的密钥，实际调用时验证。' if key else '请运行 04_SET_DEEPSEEK.bat 填写密钥。'),flush=True)
    except ValueError as exc:print('DeepSeek 配置提示：'+str(exc),flush=True)
    index_path=ROOT/'data'/'scene_six_v1'/'audio_index.json'
    if index_path.exists():
        from scene_cache import load_index
        count=len(load_index(index_path)['items'])
        print(f'音频已就绪：{count} 条。',flush=True)
    else:print('数据集索引尚未准备；只有需要原 DEV 验证回放时才运行 01。',flush=True)
    import live_audio
    print('新录音：'+live_audio.status()['message'],flush=True)
    import emotion_audio
    print('情绪模块：'+emotion_audio.status()['message'],flush=True)
    if args.check:
        print('CHECK OK：程序完整性检查通过；没有调用 DeepSeek API。')
        return
    if args.test:
        runpy.run_path(str(ROOT/'llm_client.py'),run_name='__main__')
        return
    import server
    if server.APP_VERSION!=VERSION:raise ValueError("启动器与后端版本不一致，请关闭旧窗口并完整替换程序文件。")
    server.connect().close()
    server.jobs()
    httpd=bind_server(server.Handler)
    url=f'http://127.0.0.1:{httpd.server_port}/'
    print('请使用本次打开的地址：'+url,flush=True)
    print('保持本窗口开启；关闭即停止网页。',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:httpd.serve_forever()
    except KeyboardInterrupt:pass
    finally:httpd.server_close()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('START FAILED：'+str(exc),flush=True)
        sys.exit(1)
