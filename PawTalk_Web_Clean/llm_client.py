"""Backend-only DeepSeek / Ollama transport. No credentials in browser or exports."""
import configparser,json,os,re,urllib.request,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'DEEPSEEK_CONFIG.ini'
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def config():
    p=configparser.ConfigParser(interpolation=None)
    if CONFIG.exists():
        try:p.read(CONFIG,encoding='utf-8-sig')
        except configparser.Error:raise ValueError('DEEPSEEK_CONFIG.ini 格式错误，请对照示例文件填写。') from None
    key=p.get('deepseek','api_key',fallback='').strip()
    model=p.get('deepseek','model',fallback='deepseek-flash').strip()
    if not key or key=='PASTE_YOUR_API_KEY_HERE':key=''
    if key and (not key.isascii() or any(c.isspace() for c in key)):
        raise ValueError('API Key 含有空白或中文。只粘贴密钥，不要加引号。')
    if not model or len(model)>100 or any(c.isspace() for c in model):raise ValueError('模型名格式错误。')
    return key,model

def ollama_models():
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=1) as r:data=json.load(r)
        return [x['name'] for x in data.get('models',[])]
    except Exception:return []

def available_models():
    key,model=config()
    return (['DeepSeek::'+model] if key else [])+ollama_models()

class OutputError(ValueError):
    def __init__(self,code,detail,retryable=False):
        self.code=code
        self.retryable=retryable
        super().__init__(f'LLM {code}：{detail}')

def parse_output(result,cloud):
    try:
        if cloud:
            choice=result['choices'][0]
            reason=choice.get('finish_reason')
            message=choice['message']
        else:
            reason=result.get('done_reason')
            message=result['message']
        content=message.get('content')
    except (KeyError,IndexError,TypeError,AttributeError):
        raise OutputError('RESPONSE_SCHEMA','接口返回结构不符合预期。') from None
    if message.get('refusal') or reason=='content_filter':
        raise OutputError('REFUSED','服务未提供本次回答，请检查输入内容。')
    if reason=='length':
        raise OutputError('OUTPUT_LIMIT','输出达到长度上限，不能作为完整结果使用。',True)
    if content is None or (isinstance(content,str) and not content.strip()):
        raise OutputError('EMPTY_CONTENT','接口有响应，但回答为空或只有空格。',True)
    if not isinstance(content,str):
        raise OutputError('RESPONSE_SCHEMA','回答字段类型不符合预期。')
    content=content.strip().lstrip('\ufeff').strip()
    # Accept a single complete fenced JSON object, never extract a fragment.
    fenced=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n\s*```',content,re.S|re.I)
    if fenced:content=fenced.group(1).strip()
    try:
        def invalid_constant(_):raise ValueError()
        parsed=json.loads(content,parse_constant=invalid_constant)
    except (ValueError,TypeError):
        raise OutputError('INVALID_JSON','回答不是完整的 JSON 对象。',True) from None
    if not isinstance(parsed,dict):
        raise OutputError('JSON_NOT_OBJECT','回答格式不是所需的 JSON 对象。',True)
    return parsed

def chat_json(model,messages,temperature=0,max_output_tokens=1600):
    if type(max_output_tokens) is not int or not 256<=max_output_tokens<=6000:raise ValueError("Invalid output budget")
    # At most two billed requests; never retry authentication/network errors.
    for attempt in range(2):
        try:return _chat_once(model,messages,temperature,attempt,max_output_tokens)
        except OutputError as e:
            if not model.startswith('DeepSeek::') or not e.retryable:raise
            if attempt:
                raise ValueError(f'{e} 已自动重试一次仍失败；请将此错误代码发给助手，不要发送密钥。') from None

def _chat_once(model,messages,temperature,attempt,max_output_tokens=1600):
    if model.startswith('DeepSeek::'):
        key,configured=config()
        if not key:raise ValueError('还没有填写 DeepSeek API Key，请运行 04_SET_DEEPSEEK.bat。')
        if model!='DeepSeek::'+configured:raise ValueError('模型配置已改变，请刷新网页。')
        request_messages=[dict(m) for m in messages]
        if attempt:
            request_messages.append({'role':'user','content':'Formatting reminder: follow the original task and schema. Return one complete JSON object only, without Markdown or explanation. Do not invent missing facts.'})
        body=dict(model=configured,messages=request_messages,stream=False,max_tokens=min(6000,max_output_tokens*2) if attempt else max_output_tokens,temperature=temperature,thinking={'type':'disabled'})
        # JSON mode can replace invalid output with spaces. On one retry use
        # ordinary text output, but apply the same strict local JSON parser.
        if not attempt:body['response_format']={'type':'json_object'}
        req=urllib.request.Request('https://api.deepseek.com/chat/completions',data=json.dumps(body,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
    else:
        body=dict(model=model,messages=messages,format='json',stream=False,options={'temperature':temperature})
        req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(body,ensure_ascii=False).encode('utf-8'),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=180) as r:result=json.load(r)
    except urllib.error.HTTPError as e:
        explanation={400:'请求参数或模型名不兼容，请检查模型配置。',401:'API Key 无效，请检查粘贴内容。',402:'DeepSeek API 余额不足，请到开放平台检查余额。',403:'接口访问被拒绝，请检查账户权限。',404:'模型或接口不存在，请检查模型名。',429:'请求过于频繁，请稍后重试。',500:'模型服务暂时异常，请稍后重试。',503:'模型服务繁忙，请稍后重试。'}.get(e.code,'服务请求失败，请稍后重试。')
        raise ValueError(f'LLM HTTP {e.code}：{explanation}') from None
    except (urllib.error.URLError,TimeoutError,OSError):raise ValueError('无法连接语言模型或请求超时。请检查网络；Ollama 用户请确认本地服务已启动。') from None
    except (ValueError,UnicodeError):raise OutputError('HTTP_BODY','接口响应本身不是有效 JSON，请稍后重试。') from None
    return parse_output(result,model.startswith('DeepSeek::'))

if __name__=='__main__':
    try:
        key,model=config()
        if not key:raise ValueError('先运行 04_SET_DEEPSEEK.bat 填写 API Key。')
        print('PawTalk DeepSeek JSON fix v2',flush=True)
        print('Testing DeepSeek (uses API credits; at most one automatic retry)...',flush=True)
        result=chat_json('DeepSeek::'+model,[{'role':'system','content':'You are an API connectivity checker. Output one JSON object only. Required example: {"ok": true}. No explanation or Markdown.'},{'role':'user','content':'Confirm connectivity by returning {"ok": true} in JSON format.'}])
        if result.get('ok') is not True:raise ValueError('模型有响应，但测试格式不符，请重试。')
        print('SUCCESS: DeepSeek API is connected. Restart PawTalk and refresh the page.')
    except Exception as e:
        print(str(e));raise SystemExit(1)
