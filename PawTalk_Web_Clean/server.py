"""Local v9 orchestration. Audio stays local; only whitelisted analysis reaches DeepSeek."""
import argparse
import csv
import errno
import copy
import hashlib
import json
import secrets
import sqlite3
import threading
import time
import urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import analysis_contract as contract
import dialogue
import legacy_activity
import live_audio
import emotion_audio
from upload_jobs import UploadJobs
from engine import LABELS, NAMES, evaluate_records
from llm_client import config as llm_config

from scene_schema import LABEL_VERSION, ORIGINAL_TO_SIX, public_config
from scene_cache import load_index
from app_version import APP_VERSION
ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'state'; STATE.mkdir(exist_ok=True, parents=True)
DB = STATE / 'sessions.sqlite'
INDEX = ROOT / 'data' / 'scene_six_v1' / 'audio_index.json'
CHAT_LOCK = threading.RLock()
_cache = {}; _mtime = None; _index_meta = {}
_job_manager = None
_job_lock = threading.Lock()


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try: return super().__exit__(*args)
        finally: self.close()


def connect():
    c = sqlite3.connect(DB, factory=ClosingConnection)
    c.execute('CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,body TEXT NOT NULL)')
    c.execute('CREATE TABLE IF NOT EXISTS revealed(sha TEXT PRIMARY KEY)')
    return c


def save(s):
    with connect() as c:
        c.execute('INSERT OR REPLACE INTO sessions VALUES (?,?)', (s['id'], json.dumps(s, ensure_ascii=False, allow_nan=False)))


def get(sid):
    with connect() as c: row = c.execute('SELECT body FROM sessions WHERE id=?', (sid,)).fetchone()
    if not row: raise ValueError('记录不存在，请重新上传。')
    return json.loads(row[0])


def records():
    with connect() as c: return [json.loads(r[0]) for r in c.execute('SELECT body FROM sessions')]


def index():
    global _cache, _mtime, _index_meta
    if not INDEX.exists():
        _cache={};_mtime=None;_index_meta={}
        return {}
    stamp = INDEX.stat().st_mtime_ns
    if stamp != _mtime:
        data = load_index(INDEX)
        _cache = data['items']; _mtime = stamp
        _index_meta = {k: data[k] for k in ('version', 'identity', 'label_version', 'model_version', 'split_id') if k in data}
    return _cache


def public(s):
    if s.get('schema_version') != dialogue.SCHEMA_VERSION:
        return {'id': s['id'], 'filename': s.get('filename'), 'created': s['created'], 'legacy': True,
                'result': s.get('result'), 'analysis':s.get('analysis'), 'label_version':s.get('label_version') or (s.get('analysis') or {}).get('context',{}).get('label_version','scene-eight-v1-legacy'), 'note': '旧版会话按原方法归档；不能直接继续新版意图库问答。'}
    fields = ('id', 'label_version', 'schema_version', 'filename', 'created', 'phase', 'analysis', 'initial_intents',
              'dialogue', 'result', 'errors', 'source', 'prediction_mode', 'contaminated', 'candidate_rounds', 'candidate_version', 'pre_observations', 'round_context', 'catalog_version', 'prompt_version', 'experiment_mode', 'generation_reason', 'emotion_summary', 'rule_context', 'rules_version', 'question_budget', 'prompt_versions_used', 'context_first', 'request_understanding')
    out = {key: s.get(key) for key in fields}
    out['has_reference'] = s.get('gold') in LABELS
    out['scene_reference'] = NAMES.get(s.get('gold')) if s.get('revealed') else None
    return out


def failed(branch, exc):
    return {'available': False, 'status': 'error', 'message': f'{branch}失败：{str(exc)[:600]}', 'error_type': type(exc).__name__}


def analyze_upload(raw, filename, progress=None):
    progress = progress or (lambda *args, **kwargs: None)
    digest = hashlib.sha256(raw).hexdigest()
    row = None
    try:
        progress('index', '正在核对音频指纹与 DEV 索引。')
        row = index().get(digest)
        if row is None:
            progress('context', '场景：未命中缓存，正在检查并执行实时模型；首次加载可能较慢。')
            row = live_audio.predict_bytes(raw, filename)
        else:
            progress('context', '场景：已命中 DEV 缓存，正在读取六类决策分数。')
        context = contract.context_result(row, _index_meta)
    except Exception as exc:
        context = failed('场景分析', exc)
        if isinstance(exc, live_audio.DatasetFilenameError):
            context['diagnostic'] = recording_diagnostic(filename, digest)
    progress('emotion', '情绪：正在排队或运行独立模型；子进程超时上限 180 秒（不含排队）。场景结果已返回。', context=context)
    # Always attempt both branches; a failed context branch does not invent an emotion result.
    try: emotion = contract.emotion_result(emotion_audio.predict_bytes(raw))
    except Exception as exc: emotion = failed('情绪分析', exc)
    progress('saving', '两路分析已返回，正在保存记录。', emotion=emotion)
    analysis = contract.bundle(context, emotion)
    with connect() as c: seen = c.execute('SELECT sha FROM revealed WHERE sha=?', (digest,)).fetchone() is not None
    s = {'id': secrets.token_urlsafe(20), 'schema_version': dialogue.SCHEMA_VERSION,
         'sha': digest, 'filename': filename[:160], 'created': time.time(), 'source': 'unknown',
         'gold': row.get('target_six') if row else None, 'original_scene_label':row.get('actual_original') if row else None, 'label_version':LABEL_VERSION, 'contaminated': seen, 'revealed': False,
         'prediction_mode': context.get('source'), 'analysis': analysis,
         'phase': 'analyzed' if analysis['ready'] else 'blocked',
         'initial_intents': [], 'dialogue': None, 'result': None, 'errors': [], 'catalog_version':dialogue.CATALOG['version'], 'prompt_version':dialogue.PROMPT_VERSION, 'candidate_rounds':[], 'candidate_version':1, 'experiment_mode':'C'}
    s.update(dialogue.rules.derive(analysis),rules_version=dialogue.rules.VERSION,question_budget={'limit':dialogue.MAX_QUESTIONS,'used':0,'remaining':dialogue.MAX_QUESTIONS})
    save(s)
    return public(s)


def jobs():
    global _job_manager
    with _job_lock:
        if _job_manager is None or _job_manager.db != Path(DB):
            _job_manager = UploadJobs(DB, analyze_upload)
        return _job_manager


def recording_diagnostic(filename, digest):
    """Filename/index metadata only. Never expose reference labels or assume a file is holdout."""
    basename = Path(filename.replace('\\', '/')).name.casefold()
    items = index()
    matches = [r for r in items.values() if Path(str(r.get('audio_path', '')).replace('\\', '/')).name.casefold() == basename]
    dev_paths = []
    manifest = ROOT / 'dev_manifest.csv'
    if manifest.exists():
        with manifest.open(encoding='utf-8-sig', newline='') as handle:
            dev_paths = [r['audio_path'] for r in csv.DictReader(handle)
                         if Path(r['audio_path'].replace('\\', '/')).name.casefold() == basename]
    if matches:
        note = '索引中有同名录音，但上传内容不同。请上传原始 DEV 文件，不要先裁剪、转码或重新导出。'
    elif dev_paths:
        note = 'DEV 清单有同名文件，但当前索引无同名条目；可能是排除的重复音频或索引版本不匹配。请换用其他原始 DEV 录音核对。'
    else:
        note = '当前 DEV 清单没有这个文件名。可能来自留出集或与数据集文件重名；不能只凭文件名确定来源。数据集录音不要改名绕过检查。'
    return {'filename': filename[:160], 'uploaded_sha256': digest, 'index_count': len(items),
            'same_filename_in_index': len(matches), 'dev_paths': dev_paths[:3], 'note': note}


def deepseek_models():
    key, model = llm_config()
    return ['DeepSeek::' + model] if key else []


def validate_session(s):
    if s.get('schema_version') != dialogue.SCHEMA_VERSION: raise ValueError('旧版记录不能继续新意图流程，请重新上传录音。')
    if s.get('result'): raise ValueError('本次已结束，请上传录音新建探索。')
    if not s['analysis']['ready']: raise ValueError('两路分析未齐全，请按页面错误准备模型后重新上传。')
    with connect() as c: seen = c.execute('SELECT sha FROM revealed WHERE sha=?', (s['sha'],)).fetchone()
    if s.get('contaminated') or s.get('synthetic') or s.get('demo') or seen:
        raise ValueError('本录音曾查看参考标签或用于模拟，不能作为正常盲测意图问答。请换用未看标签的录音。')


def chat_request(path, data):
    with CHAT_LOCK:
        s = get(str(data.get('id', '')))
        if path in ('/api/intents/regenerate','/api/chat/correct') and s.get('schema_version')==dialogue.SCHEMA_VERSION:
            check=copy.deepcopy(s);check['result']=None;validate_session(check)
        else:validate_session(s)
        s['prompt_versions_used']=list(dict.fromkeys(s.get('prompt_versions_used',[])+[s.get('prompt_version',dialogue.PROMPT_VERSION),dialogue.PROMPT_VERSION]))
        s['prompt_version']=dialogue.PROMPT_VERSION
        before_question=(s.get('dialogue') or {}).get('decision',{}).get('question')
        budget=s.setdefault('question_budget',{'limit':dialogue.MAX_QUESTIONS,'used':0,'remaining':dialogue.MAX_QUESTIONS})
        budget['limit']=dialogue.MAX_QUESTIONS;budget['remaining']=max(0,budget['limit']-budget['used'])
        if s.get('dialogue'):s['dialogue']['max_questions']=dialogue.MAX_QUESTIONS
        try:
            if path == '/api/intents/regenerate':
                if not s.get('result') or s['result'].get('result_status') not in ('outside_candidates','multiple_possible','insufficient_evidence','user_stopped'):raise ValueError('请先结束当前探索，再明确生成新一轮。')
                if data.get('revision')!=(s.get('dialogue') or {}).get('revision'):raise ValueError('记录版本已变化。')
                archived={k:copy.deepcopy(s.get(k)) for k in ('initial_intents','dialogue','result','generation_reason')}
                history=s.get('round_context',[])+(s.get('dialogue') or {}).get('history',[])
                generated=dialogue.generate_intents(s['analysis'],s['llm'],s.get('pre_observations',[])+history,s['experiment_mode'],question_budget=budget,phase='regenerate',context_first=s.get('context_first',False))
                if s.get('context_first'):s['request_understanding']=generated['request_understanding']
                s['next_revision']=(s.get('dialogue') or {}).get('revision',s.get('next_revision',0))+1
                s['candidate_rounds'].append(archived);s['candidate_version']+=1;s['initial_intents']=generated['intents'];s['generation_reason']=generated['reason']
                s['round_context']=history;s['dialogue']=None;s['result']=None;s['phase']='intents_ready'
            elif path == '/api/intents/generate':
                if s.get('phase')=='intents_ready' or s['dialogue']: return public(s)
                model = data.get('model')
                if model not in deepseek_models(): raise ValueError('请先运行 04 配置 DeepSeek，并刷新页面选择当前模型。')
                source = data.get('source', 'unknown')
                if source not in ('real', 'unknown'): raise ValueError('正常意图流程不接受标签生成的模拟背景。')
                mode=data.get('experiment_mode','C')
                if mode not in ('A','B','C'):raise ValueError('实验方案无效。')
                s['context_first']=data.get('flow')=='context_first'
                observation=data.get('pre_observation','')
                if not isinstance(observation,str):raise ValueError('请用文字描述当时情况。')
                observation=observation.strip()
                if len(observation)>1500:raise ValueError('独立观察过长。')
                prior=[{'answer_id':'pre_1','turn':0,'question':'查看候选之前的独立观察','answer':observation,'kind':'text'}] if observation else []
                generated = dialogue.generate_intents(s['analysis'], model,prior,mode,question_budget=budget,context_first=s['context_first'])
                if s['context_first']:
                    s['request_understanding']=generated['request_understanding']
                    for item in prior:
                        item['observation_quotes']=[x['quote'] for x in generated['request_understanding']['statements'] if x['answer_id']==item['answer_id'] and x['kind']=='observation']
                s['rule_context']=dialogue.rules.derive(s['analysis'],mode)['rule_context']
                s['experiment_mode']=mode;s['pre_observations']=prior;s['generation_reason']=generated['reason']
                s['initial_intents'] = generated['intents']; s['source'] = source; s['llm'] = model
                s['phase'] = 'intents_ready'
                if s['context_first']:
                    decision=dialogue.next_turn(s['analysis'],s['initial_intents'],dialogue.initial_states(s['initial_intents']),[],model,mode,prior,
                                                question_budget=budget,context_first=True,request_understanding=s['request_understanding'])
                    s['dialogue']={'model':model,'history':[],'revision':s.get('next_revision',0),'states':decision['assessments'],
                                   'decision':decision,'state_changes':[],'answer_revisions':[],'max_questions':dialogue.MAX_QUESTIONS}
                    s['phase']='interview'
            elif path == '/api/chat/start':
                if s['dialogue']: return public(s)
                if s.get('phase')!='intents_ready': raise ValueError('请先生成意图假设或取得候选不足说明。')
                states = dialogue.initial_states(s['initial_intents'])
                decision = dialogue.next_turn(s['analysis'], s['initial_intents'], states, [], s['llm'],s['experiment_mode'],s.get('pre_observations',[])+s.get('round_context',[]),question_budget=budget,context_first=s.get('context_first',False),request_understanding=s.get('request_understanding'))
                s['dialogue'] = {'model': s['llm'], 'history': [], 'revision': s.get('next_revision',0),
                                 'states': decision['assessments'], 'decision': decision,
                                 'state_changes': [], 'answer_revisions':[], 'max_questions': dialogue.MAX_QUESTIONS}
                s['phase'] = 'interview'
            else:
                d = s.get('dialogue')
                if not d: raise ValueError('请先开始问答。')
                if type(data.get('revision')) is not int or data['revision'] != d['revision']:
                    raise ValueError('问题版本已变化，请重新打开这条记录后继续。')
                if path == '/api/chat/stop':
                    d['decision'] = dict(d['decision'], action='uncertain', selected_intent_id=None, question=None,
                                         message='用户主动结束，本次保留不确定。',result_status='user_stopped',coexisting_intent_ids=[],explanation='你已结束本次分析。以下保留当前可能解释与缺失信息，不确认唯一意图。')
                elif path=='/api/chat/correct':
                    ident=data.get('answer_id');answer=data.get('answer')
                    if not isinstance(answer,str) or not 1<=len(answer.strip())<=1500:raise ValueError('请填写更正后的观察。')
                    history=copy.deepcopy(d['history']);prior=copy.deepcopy(s.get('pre_observations',[]));round_context=copy.deepcopy(s.get('round_context',[]));matches=[h for h in prior+round_context+history if h['answer_id']==ident]
                    if len(matches)!=1:raise ValueError('需要指定现有回答 ID。')
                    old=copy.deepcopy(matches[0]);matches[0].pop('observation_quotes',None);matches[0].update(answer_revision=old.get('answer_revision',0)+1,answer=answer.strip(),kind='unknown' if dialogue.unknown_answer(answer) else 'text')
                    if ident=='pre_1':s['request_understanding']={'user_question':s.get('request_understanding',{}).get('user_question','这段叫声可能表达什么？'),'statements':[],'note':'描述已更正，请根据修订后的原文重新理解；旧语句拆分不再有效。'}
                    decision=dialogue.next_turn(s['analysis'],s['initial_intents'],dialogue.initial_states(s['initial_intents']),history,s['llm'],s['experiment_mode'],prior+round_context,question_budget=budget,context_first=s.get('context_first',False),request_understanding=s.get('request_understanding'))
                    s['pre_observations']=prior;s['round_context']=round_context
                    d['answer_revisions'].append({'answer_id':ident,'before':old,'after':copy.deepcopy(matches[0]),'at':time.time()})
                    d['state_changes'].append({'correction':ident,'before':d['states'],'after':decision['assessments']})
                    d.update(history=history,decision=decision,states=decision['assessments'])
                    s['result']=None;s['phase']='interview'
                else:
                    q = d['decision'].get('question')
                    if budget['used']>dialogue.MAX_QUESTIONS:raise ValueError('已达到整段音频的问题上限。')
                    if not q: raise ValueError('没有待回答的问题。')
                    answer = data.get('answer', '')
                    if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 1500:
                        raise ValueError('请填写 1–1500 字观察，或选择不知道。')
                    answer = answer.strip(); kind = data.get('kind', 'text')
                    if kind not in ('text', 'unknown','guess'): raise ValueError('回答类型无效。')
                    if dialogue.unknown_answer(answer): kind = 'unknown'
                    history = d['history'] + [{'answer_id':secrets.token_urlsafe(12),'turn': len(d['history']) + 1, 'question': q['text'],
                                              'answer_revision':0, 'focus': q['focus'], 'answer': answer, 'kind': kind}]
                    decision = dialogue.next_turn(s['analysis'], s['initial_intents'], d['states'], history, s['llm'],s['experiment_mode'],s.get('pre_observations',[])+s.get('round_context',[]),question_budget=budget,context_first=s.get('context_first',False),request_understanding=s.get('request_understanding'))
                    d['state_changes'].append({'turn': len(history), 'before': copy.deepcopy(d['states']),
                                               'after': copy.deepcopy(decision['assessments']),
                                               'validation_notes': decision['validation_notes']})
                    d['history'] = history; d['decision'] = decision; d['states'] = decision['assessments']
                d['revision'] += 1
            d = s.get('dialogue')
            if d and d['decision']['action']=='ask':
                new_question=d['decision']['question']
                # Charge only a newly issued question. Retries and identical correction prompts are free.
                if path in ('/api/chat/start','/api/chat/answer') or new_question!=before_question:
                    if budget['remaining']<=0:raise ValueError('已达到整段音频的三问上限。')
                    budget['used']+=1;budget['remaining']=max(0,budget['limit']-budget['used'])
            if d and d['decision']['action'] != 'ask':
                s['result'] = dialogue.conclusion(s['analysis'], s['initial_intents'], d['decision'], d['history'])
                s['completed'] = time.time(); s['phase'] = 'completed'
            save(s)
            return public(s)
        except Exception as exc:
            # Failed requests never commit fabricated candidates or a partially applied answer.
            original = get(s['id'])
            original['errors'].append({'at': time.time(), 'stage': path, 'message': str(exc)[:600]})
            save(original)
            raise


def export_record(s):
    out={k:v for k,v in s.items() if k not in ('gold','original_scene_label','row','synthetic','last_parse','text')}
    if s.get('schema_version')!=dialogue.SCHEMA_VERSION:
        out['label_version']=s.get('label_version') or (s.get('analysis') or {}).get('context',{}).get('label_version','scene-eight-v1-legacy')
        if s.get('row'):
            out['legacy_model_row']={k:v for k,v in s['row'].items() if k not in ('actual','gold','target_six','actual_original')}
    return out


def summaries(rs):
    legacy = [s for s in rs if not s.get('schema_version','').startswith('intent-session-')]
    return {'intent_summary': dialogue.summary(rs),
            'legacy_scene_summary': {'manual': evaluate_records(legacy), 'v8_adaptive': legacy_activity.summary(legacy)},
            'evaluation_note': '旧版统计只针对八类场景。新版 意图没有独立标准答案，不能与场景参考标签比较。'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def send(self, status, data, mime='application/json; charset=utf-8'):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
        try:
            self.send_response(status); self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store'); self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(body)
        except OSError as exc:
            disconnected = isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)) or exc.errno in (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED) or getattr(exc, 'winerror', None) in (10053, 10054)
            if not disconnected: raise
            self.close_connection = True
            return False
        return True

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path == '/api/upload/status':
                params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                job = jobs().status(params.get('id', [''])[0])
                if job.get('session_id'): job['session'] = public(get(job['session_id']))
                return self.send(200, job)
            if path == '/api/status':
                error = ''; models = []
                try: models = deepseek_models()
                except ValueError as exc: error = str(exc)
                try: cache_count=len(index());cache_status={'ready':bool(cache_count),'message':'六分类 OOF 缓存已验证' if cache_count else '六分类 OOF 缓存尚未准备'}
                except (ValueError,KeyError,TypeError,OSError) as exc: cache_count=0;cache_status={'ready':False,'message':str(exc)}
                return self.send(200, {'app_version': APP_VERSION, 'llm_models': models, 'deepseek_configured': bool(models),
                                      'llm_error': error, 'live_audio': live_audio.status(), 'emotion': emotion_audio.status(),
                                      'audio_count': cache_count, 'cache_status':cache_status, 'scene_schema':public_config(), 'max_questions': dialogue.MAX_QUESTIONS,'prompt_version':dialogue.PROMPT_VERSION,'policy':dialogue.context_policy.POLICY})
            if path == '/api/history':
                rs = records()
                return self.send(200, {'records': [public(s) for s in sorted(rs, key=lambda x: x['created'], reverse=True)], **summaries(rs)})
            if path == '/api/export':
                rs = records()
                # Export all v9 attempts, including blocked/incomplete sessions, but never hidden reference labels.
                exports = [export_record(s) for s in rs]
                return self.send(200, {'version': 'pawtalk-scene-six-v1', 'scene_schema':public_config(), 'records': exports, **summaries(rs)})
            allowed = {'/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js', '/recorder.js': 'recorder.js',
                       '/style.css': 'style.css', '/manifest.webmanifest': 'manifest.webmanifest', '/icon.svg': 'icon.svg',
                       '/presentation.js': 'presentation.js', '/icons.svg': 'icons.svg'}
            if path not in allowed: return self.send(404, {'error': 'Not found'})
            p = ROOT / 'static' / allowed[path]
            mime = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.webmanifest': 'application/manifest+json'}[p.suffix]
            return self.send(200, p.read_bytes(), mime)
        except Exception as exc: self.send(500, {'error': str(exc)})

    def do_POST(self):
        try:
            origin = self.headers.get('Origin')
            if origin and urllib.parse.urlparse(origin).netloc != self.headers.get('Host'):
                return self.send(403, {'error': 'Cross-origin request rejected'})
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 <= length <= live_audio.MAX_BYTES: return self.send(413, {'error': '文件请控制在 64 MB 以内。'})
            raw = self.rfile.read(length); path = urllib.parse.urlparse(self.path).path
            if path == '/api/upload/start':
                if not raw: return self.send(422, {'error': '上传文件为空。'})
                job = jobs().submit(raw, urllib.parse.unquote(self.headers.get('X-Filename', 'audio.wav')))
                return self.send(202, job)
            if path == '/api/upload':
                if not raw: return self.send(422, {'error': '上传文件为空。'})
                session = analyze_upload(raw, urllib.parse.unquote(self.headers.get('X-Filename', 'audio.wav')))
                # A blocked session is a real recorded outcome, visible and retryable after setup.
                return self.send(200, session)
            data = json.loads(raw or b'{}')
            if not isinstance(data, dict): raise ValueError('请求必须是 JSON 对象。')
            if path in ('/api/intents/generate','/api/intents/regenerate', '/api/chat/start', '/api/chat/answer', '/api/chat/correct', '/api/chat/stop'):
                return self.send(200, chat_request(path, data))
            if path == '/api/reveal':
                with CHAT_LOCK:
                    s = get(str(data.get('id', '')))
                    if s.get('schema_version')!=dialogue.SCHEMA_VERSION:raise ValueError('旧记录保留原标签体系，请从历史导出查看。')
                    if not s.get('result'): raise ValueError('仅在探索结束后可查看场景参考标签。')
                    if s.get('gold') not in LABELS: raise ValueError('此录音没有场景参考标签。')
                    s['revealed'] = True
                    with connect() as c: c.execute('INSERT OR IGNORE INTO revealed VALUES (?)', (s['sha'],))
                    save(s)
                    return self.send(200, {'scene_reference': NAMES[s['gold']], 'note': '仅为六类活动参考，不能作为意图标准答案。'})
            if path in ('/api/demo', '/api/generate', '/api/parse', '/api/predict'):
                return self.send(410, {'error': '旧版模拟背景、预设候选和场景融合接口已停用，当前版本使用真实双路分析与独立意图流程。'})
            self.send(404, {'error': 'Not found'})
        except (ValueError, KeyError, TypeError) as exc: self.send(400, {'error': str(exc)})
        except Exception as exc: self.send(500, {'error': '处理失败：' + str(exc)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8882)
    args = parser.parse_args()
    jobs()  # Mark unfinished tasks from an earlier server run as interrupted.
    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'PawTalk: http://127.0.0.1:{httpd.server_port}', flush=True)
    try: httpd.serve_forever()
    except KeyboardInterrupt: pass
    finally: httpd.server_close()
