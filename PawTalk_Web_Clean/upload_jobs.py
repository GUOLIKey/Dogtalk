"""Bounded, persistent background upload tasks; inference never depends on a browser socket."""
import hashlib
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

MAX_ACTIVE_JOBS = 2


class UploadJobs:
    def __init__(self, db, analyze):
        self.db = Path(db); self.analyze = analyze
        self.lock = threading.RLock(); self.threads = {}
        with self.connection() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS upload_jobs(id TEXT PRIMARY KEY,body TEXT NOT NULL)')
            for ident, body in conn.execute('SELECT id,body FROM upload_jobs').fetchall():
                job = json.loads(body)
                if job['state'] in ('queued', 'running'):
                    job.update(state='interrupted', stage='interrupted', message='程序已重启，之前未完成的任务已中断，请重新上传。', updated=time.time())
                    conn.execute('UPDATE upload_jobs SET body=? WHERE id=?', (json.dumps(job, ensure_ascii=False), ident))

    def connection(self):
        # Caller uses contextlib.closing-style subclass, never leaves SQLite open on Windows.
        class Closing(sqlite3.Connection):
            def __exit__(self, *args):
                try: return super().__exit__(*args)
                finally: self.close()
        return sqlite3.connect(self.db, timeout=15, factory=Closing)

    def get(self, ident):
        with self.connection() as conn:
            row = conn.execute('SELECT body FROM upload_jobs WHERE id=?', (ident,)).fetchone()
        if row is None: raise ValueError('没有找到上传任务。请查看探索记录，或重新上传。')
        return json.loads(row[0])

    def write(self, job):
        with self.connection() as conn:
            conn.execute('INSERT OR REPLACE INTO upload_jobs VALUES (?,?)', (job['job_id'], json.dumps(job, ensure_ascii=False, allow_nan=False)))

    def update(self, ident, **fields):
        with self.lock:
            job = self.get(ident); job.update(fields, updated=time.time()); self.write(job)
            return job

    def submit(self, raw, filename):
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            active = [ident for ident, thread in self.threads.items() if thread.is_alive()]
            for ident in active:
                job = self.get(ident)
                if job['sha'] == digest: return self.public(job)
            if len(active) >= MAX_ACTIVE_JOBS:
                raise ValueError('已有两个上传任务正在处理，请先等待已有任务完成，不要重复上传。')
            self.threads = {ident: self.threads[ident] for ident in active}
            ident = secrets.token_urlsafe(20)
            job = {'job_id': ident, 'sha': digest, 'filename': filename[:160], 'created': time.time(),
                   'updated': time.time(), 'state': 'queued', 'stage': 'queued', 'message': '音频已接收，等待后台分析。',
                   'session_id': None, 'partial': {}}
            self.write(job)
            worker = threading.Thread(target=self.run, args=(ident, raw, filename), daemon=True, name='pawtalk-upload-'+ident[:6])
            self.threads[ident] = worker; worker.start()
            return self.public(job)

    def run(self, ident, raw, filename):
        try:
            self.update(ident, state='running', stage='index', message='正在核对音频指纹与 DEV 索引。')
            def progress(stage, message, **partial):
                job = self.get(ident)
                self.update(ident, stage=stage, message=message, partial={**job['partial'], **partial})
                print('[UPLOAD '+ident[:6]+'] '+message, flush=True)
            session = self.analyze(raw, filename, progress=progress)
            self.update(ident, state='completed', stage='completed', session_id=session['id'],
                        message='分析记录已保存。' if session['analysis']['ready'] else '两路分析已返回，部分分支不可用；具体原因已保存。')
        except Exception as exc:
            self.update(ident, state='failed', stage='failed', message='上传任务失败：'+str(exc)[:600])

    @staticmethod
    def public(job):
        out = {k: v for k, v in job.items() if k != 'sha'}
        end = job['updated'] if job['state'] not in ('queued', 'running') else time.time()
        out['elapsed_seconds'] = max(0, int(end - job['created']))
        return out

    def status(self, ident): return self.public(self.get(ident))
