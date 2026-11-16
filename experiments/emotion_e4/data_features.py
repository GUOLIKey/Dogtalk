"""TRAIN only: pinned downloads, feature cache, duplicate/source grouping."""
import concurrent.futures,hashlib,json,os,time,urllib.request,urllib.parse
from pathlib import Path
import numpy as np
from common import *

DOWNLOAD_VERSION='1.1'

def file_verified(path,expected):
    path=Path(path)
    return path.is_file() and path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']

def retry_url(url,attempt):
    if not attempt:return url
    parts=urllib.parse.urlsplit(url)
    query=urllib.parse.parse_qsl(parts.query,keep_blank_values=True)
    query.append(('pawtalk_retry',str(time.time_ns())))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))

def fetch_verified(urls,dest,expected,attempts=4):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if file_verified(dest,expected):return False
    errors=[];pointer_urls=set();attempts=max(1,min(6,int(attempts)))
    for attempt in range(attempts):
        if attempt:time.sleep(min(2**attempt,8))
        for url in urls:
            if url in pointer_urls:continue
            host=urllib.parse.urlsplit(url).netloc
            try:
                req=urllib.request.Request(retry_url(url,attempt),headers={'User-Agent':'PawTalk-Emotion-Lab/'+DOWNLOAD_VERSION,'Cache-Control':'no-cache','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=35) as r:b=r.read(expected['bytes']+1)
                if b.startswith(b'version https://git-lfs.github.com/spec/v1'):
                    pointer_urls.add(url)
                    raise ValueError('返回 Git LFS 文件指针，不是音频；该备用地址不再重复尝试')
                digest=hashlib.sha256(b).hexdigest()
                if len(b)!=expected['bytes'] or digest!=expected['sha256']:
                    raise ValueError(f'校验不符：读取 {len(b)} 字节，期望 {expected["bytes"]}；SHA256={digest}，期望={expected["sha256"]}')
                tmp=dest.with_suffix(dest.suffix+'.part');tmp.write_bytes(b);os.replace(tmp,dest);return True
            except Exception as exc:
                errors.append(f'round={attempt+1} {host}: {type(exc).__name__}: {exc}')
        if len(pointer_urls)==len(urls):break
    raise RuntimeError(f'下载失败 {dest.name}（未通过校验，不用于训练）：\n'+'\n'.join(errors))

def prepare_dataset(config):
    spec=read_json(ROOT/'dataset_manifest.json');files=spec['files']
    if any('/test/' in x['path'] or '_test_labels' in x['path'] for x in files):raise ValueError('训练清单不得包含测试集。')
    candidates=[]
    if config.get('dataset_dir'):candidates.append(Path(config['dataset_dir']).expanduser())
    else:
        for base in (ROOT.parent,ROOT.parent.parent):
            candidates += [base/'datasets'/'EmotionalCanines',base/'EmotionalCanines',base/'data'/'EmotionalCanines']
        candidates.append(ROOT/'data'/'EmotionalCanines')
    for data in candidates:
        if (data/'husky_train_labels.csv').is_file() and (data/'shiba_train_labels.csv').is_file():
            if all(file_verified(data/e['path'],e) for e in files):
                print('[DATA] 全部本地 TRAIN 文件大小/SHA256 校验通过；不写入原数据目录。',flush=True);return data.resolve(),spec
    if not config.get('auto_download_training_data',True):raise FileNotFoundError('情绪训练文件缺失；在 config.json 设置 dataset_dir 或允许自动下载。')
    # All writes stay inside the laboratory, never repair/overwrite the original app/data.
    data=ROOT/'data'/'EmotionalCanines';revision=spec['revision'];repo=spec['repository'].split('github.com/')[-1]
    print(f'[DATA downloader {DOWNLOAD_VERSION}] 检查/下载 {len(files)-2} 条官方 TRAIN 音频（约57.5 MB）；复用已校验文件，只补下载缺失或损坏文件，不下载测试集。',flush=True)
    def one(e,attempts=4):
        path=e['path'];base=f'{repo}/{revision}/{path}'
        urls=[f'https://media.githubusercontent.com/media/{base}',f'https://raw.githubusercontent.com/{base}'] if path.endswith('.wav') else [f'https://raw.githubusercontent.com/{base}']
        return fetch_verified(urls,data/path,e,attempts=attempts)
    failures={};downloaded=0;reused=0
    # Limit concurrent requests even when an older config still specifies six.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(3,int(config.get('download_workers',3))))) as pool:
        futures={pool.submit(one,e):e for e in files}
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            try:
                if f.result():downloaded+=1
                else:reused+=1
            except Exception as exc:failures[futures[f]['path']]={'entry':futures[f],'error':str(exc)}
            if i%50==0 or i==len(files):print(f'[DATA] checked {i}/{len(files)}; failed={len(failures)}',flush=True)
    # A failed final file must not force users to manually restart the entire job.
    for path,item in list(failures.items()):
        print('[DATA RETRY] 串行补下载：'+Path(path).name,flush=True)
        try:
            if one(item['entry'],attempts=2):downloaded+=1
            else:reused+=1
            del failures[path]
        except Exception as exc:failures[path]['error']+='\n串行补下载：'+str(exc)
    save_json(ROOT/'cache'/'download_status.json',{'download_version':DOWNLOAD_VERSION,'total_files':len(files),'downloaded':downloaded,'reused':reused,'failed_files':{k:v['error'] for k,v in failures.items()},'official_test_used':False})
    if failures:raise RuntimeError('\n'.join(v['error'] for v in list(failures.values())[:6])+'\n已保留成功文件。自动重试后仍失败；详见 cache/download_status.json。重新运行只补下载缺失/损坏文件。')
    print(f'[DATA] 校验完成；复用 {reused}，本次下载 {downloaded}，失败 0。',flush=True)
    return data.resolve(),spec

def load_rows(data,spec):
    expected={x['path']:x for x in spec['files']};rows=[];seen=set()
    for breed in ('husky','shiba'):
        label_path=data/f'{breed}_train_labels.csv'
        if sha(label_path)!=expected[label_path.name]['sha256']:raise ValueError('官方训练标签与固定版本不一致：'+str(label_path))
        for r in read_csv(label_path):
            ident=r['audio_id']
            if ident in seen or not ident.startswith(breed+'_train_'):raise ValueError('重复或非 TRAIN 音频 ID：'+ident)
            seen.add(ident)
            for axis,labels in AXES.items():
                if r[axis] not in labels:raise ValueError('情绪标签非法：'+ident)
            rel=f'{breed}/train/{ident}.wav';path=data/rel
            if rel not in expected or not path.is_file():raise FileNotFoundError(rel)
            digest=sha(path)
            if digest!=expected[rel]['sha256']:raise ValueError('音频 SHA256 不匹配：'+rel)
            rows.append(dict(audio_id=ident,path=str(path),breed=breed,arousal=r['arousal'],valence=r['valence'],audio_sha256=digest,dog_id='',source_recording_id=''))
    groups_file=ROOT/'groups.csv'
    if groups_file.is_file():
        extras=read_csv(groups_file);lookup={r['audio_id']:r for r in extras}
        if len(lookup)!=len(extras) or not set(lookup)<=seen:raise ValueError('groups.csv 有重复/未知 ID。')
        for r in rows:
            for k in ('dog_id','source_recording_id'):r[k]=lookup.get(r['audio_id'],{}).get(k,'').strip()
    return sorted(rows,key=lambda r:r['audio_id'])

from emotion_e4_features import extract_vector

def features(rows):
    import librosa,soundfile as sf
    cache=ROOT/'cache'/'features';cache.mkdir(parents=True,exist_ok=True)
    basic=[];rich=[]
    recipe=FEATURE_RECIPE+'|librosa='+librosa.__version__+'|numpy='+np.__version__
    for i,r in enumerate(rows):
        key=hashlib.sha256((r['audio_sha256']+recipe).encode()).hexdigest();p=cache/(key+'.npz')
        cached=False
        if p.is_file():
            try:
                with np.load(p,allow_pickle=False) as z:
                    if str(z['recipe'])!=recipe:raise ValueError('recipe')
                    b=z['basic'];v=z['rich'];pcm=str(z['pcm_sha256']);duration=float(z['duration']);peak=float(z['peak']);rms=float(z['rms'])
                    if b.shape!=(26,) or v.shape!=(153,) or not np.isfinite(b).all() or not np.isfinite(v).all():raise ValueError('cache shape')
                    cached=True
            except Exception:p.unlink()
        if not cached:
            native,sr=sf.read(r['path'],dtype='float32',always_2d=True)
            if native.size==0 or not np.isfinite(native).all():raise ValueError('损坏音频：'+r['audio_id'])
            pcm=hashlib.sha256(str(sr).encode()+str(native.shape).encode()+native.astype('<f4').tobytes()).hexdigest()
            wave=librosa.to_mono(native.T);duration=len(wave)/sr;peak=float(np.max(np.abs(wave)));rms=float(np.sqrt(np.mean(wave.astype(float)**2)))
            if sr!=22050:wave=librosa.resample(wave,orig_sr=sr,target_sr=22050)
            b,v=extract_vector(wave)
            with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,basic=b,rich=v,pcm_sha256=pcm,duration=duration,peak=peak,rms=rms,recipe=recipe)
            os.replace(p.with_suffix('.tmp'),p)
        r.update(pcm_sha256=pcm,duration=duration,peak=peak,rms=rms)
        basic.append(b);rich.append(v)
        if (i+1)%50==0 or i+1==len(rows):print(f'[FEATURES] {i+1}/{len(rows)}',flush=True)
    return np.asarray(basic),np.asarray(rich)

def make_groups(rows):
    parent=list(range(len(rows)));seen={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for k in ('audio_sha256','pcm_sha256','dog_id','source_recording_id'):
            if r.get(k):
                key=(k,r[k])
                if key in seen:parent[find(i)]=find(seen[key])
                else:seen[key]=i
    groups=np.array([find(i) for i in range(len(rows))]);conflicts=[]
    for g in set(groups):
        members=[rows[i] for i in np.flatnonzero(groups==g)]
        for axis in AXES:
            if len({r[axis] for r in members})>1:conflicts.append({'group':int(g),'axis':axis,'n':len(members),'note':'关联组存在不同标签；原样保留，不按预测对错删除'})
    audit={'n':len(rows),'groups':len(set(groups)),'dog_id_coverage':sum(bool(r.get('dog_id')) for r in rows),'source_id_coverage':sum(bool(r.get('source_recording_id')) for r in rows),'conflicting_groups':conflicts,
      'silent_or_near_silent':sum(r['rms']<1e-5 for r in rows),'clipping_peak_ge_0_999':sum(r['peak']>=.999 for r in rows),
      'official_test_decoded_or_used':False,'limitations':['缺失犬只/来源 ID 时，仅精确文件/PCM 去重分组，不能证明独立犬只泛化。','没有把犬种当作情绪类别或输入特征。','不读取官方测试音频及标签；无法核查未知来源的跨集合重叠。']}
    return groups,audit
