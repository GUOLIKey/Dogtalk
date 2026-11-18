"""PawTalk STEP 4A — reproducible audio duration and signal-health diagnosis.

READ-ONLY wrt dataset, STEP1/2/3 artifacts, and locked holdout. Uses only the
fixed 9,984 development recordings. NO model fitting/inference, no CLAP downloads.
Run in the original Windows project after STEP1-3:
    .venv\\Scripts\\python.exe scripts\\step4a_audio_diagnostics.py --check
    .venv\\Scripts\\python.exe scripts\\step4a_audio_diagnostics.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_dataset import CATEGORIES  # noqa: E402
from step1_split_and_weights import SEED, SPLIT_DIR, lock_split, source_manifest  # noqa: E402

OUT = ROOT / 'results' / 'optimization_step4a'
STEP3_PRED = ROOT / 'results' / 'optimization_step3' / 'diagnostics' / 'c1_g2_oof_predictions.csv'
STEP3_SELECTION = ROOT / 'results' / 'optimization_step3' / 'step3_selection_summary.json'
CLAP_SECONDS = 10.0
DURATION_BINS = [
    ('<1s', 0.0, 1.0), ('1-3s', 1.0, 3.0), ('3-5s', 3.0, 5.0),
    ('5-10s', 5.0, 10.0), ('10-15s', 10.0, 15.0),
    ('15-30s', 15.0, 30.0), ('>=30s', 30.0, float('inf')),
]


def normpath(s):
    return str(s or '').replace('\\', '/').strip()


def write_csv(name, items, columns):
    dest = OUT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(items)
    print(f'[SAVED] {dest}', flush=True)


def fmt(x, nd=3):
    return f'{x:.{nd}f}' if x is not None and math.isfinite(float(x)) else 'n/a'


def bin_of(duration):
    for label, low, high in DURATION_BINS:
        if low <= duration < high:
            return label
    return 'unknown'


def number_summary(values):
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return dict(n=0, mean=None, median=None, p10=None, p90=None, minimum=None, maximum=None)
    return dict(n=len(arr), mean=float(np.mean(arr)), median=float(np.median(arr)),
                p10=float(np.percentile(arr, 10)), p90=float(np.percentile(arr, 90)),
                minimum=float(arr.min()), maximum=float(arr.max()))


def audit_locked_split():
    # Guard: do NOT create a fresh split. STEP4A is only valid after STEP1.
    expected = ('dev_train.csv', 'LOCKED_holdout_DO_NOT_TUNE.csv', 'split_protocol.json')
    missing = [x for x in expected if not (SPLIT_DIR / x).is_file()]
    if missing:
        raise RuntimeError('STEP1 locked split missing. Do NOT recreate it in STEP4A: ' + ', '.join(missing))
    original = source_manifest()
    dev, n_holdout = lock_split(original)  # validates split fingerprint and integrity
    if len(dev) != 9984 or n_holdout != 2496:
        raise RuntimeError(f'Unexpected dev/holdout sizes: {len(dev)}/{n_holdout}. Stop and check STEP1.')
    keys = [normpath(r['audio_path']) for r in dev]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Duplicate audio paths in development split')
    print(f'[LOCK] Verified original split: DEV={len(dev)}; holdout={n_holdout} (NOT inspected).', flush=True)
    return dev, n_holdout


def check_dependencies_and_data(dev):
    if not STEP3_SELECTION.is_file():
        print('[WARN] STEP3 selection JSON missing; duration analysis can proceed but STEP3 context unavailable.')
    else:
        try:
            info = json.loads(STEP3_SELECTION.read_text(encoding='utf-8'))
            selected = info.get('selected', {})
            print(f'[BASELINE] STEP3 selected={selected.get("model", "unknown")}; '
                  f'C={selected.get("C")}; gamma={selected.get("gamma")}', flush=True)
        except (OSError, ValueError) as exc:
            print(f'[WARN] Cannot parse STEP3 JSON: {exc}')
    print(f'[PRED] STEP3 development OOF predictions: {"found" if STEP3_PRED.exists() else "not found (optional)"}.', flush=True)
    print(f'[ENV] soundfile={sf.__version__}; numpy={np.__version__}; '
          f'CLAP policy=first {CLAP_SECONDS:g}s at 48,000Hz; MFCC=full recording at 16,000Hz.', flush=True)
    # A damaged file is precisely what the audit should discover, so sample-read
    # failures warn rather than preventing the full metadata scan.
    for r in dev[:3]:
        p = ROOT / normpath(r['audio_path'])
        try:
            m = sf.info(str(p))
            print(f'[SAMPLE] {p.name}: {m.duration:.2f}s / {m.samplerate}Hz / {m.channels}ch')
        except Exception as exc:
            print(f'[WARN] Preliminary sample cannot be decoded: {p.name}: {exc}')
    print('[PASS] STEP4A preflight passed. No audio edited and no model trained.', flush=True)


def read_metadata(dev):
    data = []
    for index, row in enumerate(dev, 1):
        rel = normpath(row['audio_path'])
        path = ROOT / rel
        record = dict(audio_path=rel, activity=row['activity'], audio_id=row.get('audio_id', ''),
                      status='ok', error='', duration_sec='', sample_rate_hz='', channels='',
                      frames='', subtype='', format='', file_bytes='', duration_bucket='',
                      seconds_beyond_clap_10s='', share_after_clap_10s='')
        try:
            meta = sf.info(str(path))
            st = path.stat()
            dur = float(meta.duration)
            if not math.isfinite(dur) or dur <= 0 or meta.samplerate <= 0 or meta.frames < 1:
                raise ValueError('nonpositive/invalid audio duration or metadata')
            tail = max(0.0, dur - CLAP_SECONDS)
            record.update(duration_sec=round(dur, 6), sample_rate_hz=meta.samplerate,
                          channels=meta.channels, frames=meta.frames, subtype=meta.subtype,
                          format=meta.format, file_bytes=st.st_size,
                          duration_bucket=bin_of(dur), seconds_beyond_clap_10s=round(tail, 6),
                          share_after_clap_10s=round(tail / dur, 6))
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {str(exc)[:240]}')
        data.append(record)
        if index % 1000 == 0 or index == len(dev):
            print(f'[META] {index}/{len(dev)}; unreadable={sum(r["status"] != "ok" for r in data)}', flush=True)
    return data


def sample_quality(rows, count):
    # Stratified sample from *DEV* only; fast partial-window waveform decoding.
    good = [r for r in rows if r['status'] == 'ok']
    if count <= 0 or not good:
        return []
    from sklearn.model_selection import train_test_split
    n = min(count, len(good))
    if n == len(good):
        chosen = good
    else:
        indices, _ = train_test_split(np.arange(len(good)), train_size=n,
                                      random_state=SEED, stratify=[r['activity'] for r in good])
        chosen = [good[int(i)] for i in sorted(indices)]
    probes = []
    for index, item in enumerate(chosen, 1):
        rp = ROOT / item['audio_path']
        record = dict(audio_path=item['audio_path'], activity=item['activity'],
                      duration_sec=item['duration_sec'], status='ok', error='',
                      windows_checked=0, window_sample_count=0, rms='', peak_abs='',
                      dc_abs='', near_fullscale_fraction='',
                      low_level_window_count='', notes='Windowed, heuristic only; NOT a noise label')
        try:
            with sf.SoundFile(str(rp)) as f:
                span = max(1, min(int(f.samplerate), f.frames))  # up to 1 second per window
                positions = sorted({0, max(0, (f.frames-span)//2), max(0, f.frames-span)})
                sampled = []
                per_window_rms = []
                for pos in positions:
                    f.seek(pos)
                    block = f.read(span, dtype='float32', always_2d=True)
                    if not block.size:
                        continue
                    if not np.isfinite(block).all():
                        raise ValueError('NaN/Inf in decoded waveform window')
                    per_window_rms.append(float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))))
                    sampled.append(block.reshape(-1))
                if not sampled:
                    raise ValueError('no readable samples')
                s = np.concatenate(sampled)
                record.update(windows_checked=len(sampled), window_sample_count=s.size,
                              rms=round(float(np.sqrt(np.mean(np.square(s, dtype=np.float64)))), 7),
                              peak_abs=round(float(np.max(np.abs(s))), 7),
                              dc_abs=round(float(abs(np.mean(s))), 7),
                              near_fullscale_fraction=round(float(np.mean(np.abs(s) >= 0.999)), 7),
                              low_level_window_count=sum(v < 0.005 for v in per_window_rms))
        except Exception as exc:
            record.update(status='error', error=f'{type(exc).__name__}: {str(exc)[:240]}')
        probes.append(record)
        if index % 100 == 0 or index == len(chosen):
            print(f'[PROBE] {index}/{len(chosen)} waveform samples reviewed', flush=True)
    return probes


def summarize_metadata(meta):
    valid = [r for r in meta if r['status'] == 'ok']
    bad = [r for r in meta if r['status'] != 'ok']
    lengths = [float(r['duration_sec']) for r in valid]
    stats = number_summary(lengths)
    n_over_10 = sum(d > CLAP_SECONDS + 1e-4 for d in lengths)
    seconds_total = sum(lengths)
    tail_secs = sum(max(d - CLAP_SECONDS, 0) for d in lengths)
    summary = dict(n_scanned=len(meta), n_valid=len(valid), n_unreadable=len(bad),
                   n_over_10s=n_over_10, pct_over_10s=100 * n_over_10 / max(len(valid), 1),
                   tail_seconds_total=tail_secs,
                   pct_total_duration_after_10s=100 * tail_secs / max(seconds_total, 1e-9),
                   overall_duration=stats,
                   sample_rates=dict(Counter(str(r['sample_rate_hz']) for r in valid)),
                   channels=dict(Counter(str(r['channels']) for r in valid)),
                   subtypes=dict(Counter(str(r['subtype']) for r in valid)))
    by_class = []
    for c in CATEGORIES:
        subset = [r for r in valid if r['activity'] == c]
        durations = [float(r['duration_sec']) for r in subset]
        s = number_summary(durations)
        by_class.append(dict(activity=c, count=s['n'], median_s=fmt(s['median']),
                             mean_s=fmt(s['mean']), p10_s=fmt(s['p10']), p90_s=fmt(s['p90']),
                             min_s=fmt(s['minimum']), max_s=fmt(s['maximum']),
                             above_10s=sum(d > CLAP_SECONDS + 1e-4 for d in durations),
                             above_10s_pct=round(100 * sum(d > CLAP_SECONDS + 1e-4 for d in durations)
                                                 / max(len(durations), 1), 2)))
    by_bucket = []
    for label, _, _ in DURATION_BINS:
        subset = [r for r in valid if r['duration_bucket'] == label]
        by_bucket.append(dict(bucket=label, count=len(subset), percent=round(100 * len(subset) / max(len(valid), 1), 2),
                              **{c: sum(r['activity'] == c for r in subset) for c in CATEGORIES}))
    return summary, by_class, by_bucket


def join_oof(meta):
    if not STEP3_PRED.is_file():
        print('[WARN] STEP3 OOF predictions missing; skipping error-vs-duration analysis.')
        return [], [], 'missing'
    expected = {r['audio_path']: r for r in meta if r['status'] == 'ok'}
    entries = {}
    with STEP3_PRED.open('r', newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        need = {'audio_path', 'actual', 'predicted'}
        if not need.issubset(rd.fieldnames or []):
            print('[WARN] STEP3 OOF schema unexpected; not comparing predictions with durations.')
            return [], [], 'unexpected-schema'
        for r in rd:
            key = normpath(r['audio_path'])
            # Ignore any entry not on dev; NEVER inspect holdout for this analysis.
            if key in entries:
                print(f'[WARN] Repeated OOF audio_path {key}; skipping diagnostic.')
                return [], [], 'duplicate-oof-key'
            entries[key] = r
    missing = [p for p in expected if p not in entries]
    if missing:
        print(f'[WARN] Missing {len(missing)} valid DEV OOF predictions; reporting matched subset only.')
    joined = []
    for key, item in expected.items():
        pred = entries.get(key)
        if pred is None:
            continue
        if pred.get('actual') != item['activity'] or pred.get('predicted') not in CATEGORIES:
            print(f'[WARN] OOF label mismatch at {key}; skipping OOF analysis entirely.')
            return [], [], 'label-mismatch'
        joined.append(dict(audio_path=key, activity=item['activity'],
                           bucket=item['duration_bucket'], correct=int(pred['predicted'] == item['activity']),
                           predicted=pred['predicted']))
    bucket_performance = []
    class_bucket_performance = []
    for label, _, _ in DURATION_BINS:
        subset = [r for r in joined if r['bucket'] == label]
        labels_present = {r['activity'] for r in subset}
        accuracy = sum(r['correct'] for r in subset) / len(subset) if subset else None
        # Only report an 8-class Macro-F1 if the bin actually contains every class.
        macro = None
        if len(labels_present) == len(CATEGORIES):
            from sklearn.metrics import f1_score
            macro = float(f1_score([r['activity'] for r in subset], [r['predicted'] for r in subset],
                                  labels=CATEGORIES, average='macro', zero_division=0))
        bucket_performance.append(dict(bucket=label, n=len(subset), classes_present=len(labels_present),
                                       accuracy=fmt(accuracy, 4), macro_f1_all8=fmt(macro, 4)))
        for c in CATEGORIES:
            ss = [r for r in subset if r['activity'] == c]
            class_bucket_performance.append(dict(bucket=label, activity=c, n=len(ss),
                                                recall=fmt(sum(r['correct'] for r in ss) / len(ss) if len(ss) >= 10 else None, 4),
                                                note='n<10: suppressed' if len(ss) < 10 else 'per-class recall, NOT F1'))
    print(f'[OOF] STEP3 prediction-vs-duration joined {len(joined)}/{len(expected)} valid DEV clips.', flush=True)
    return bucket_performance, class_bucket_performance, 'ok' if not missing else 'partially-matched'


def plot_charts(meta, class_rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[OPTIONAL] matplotlib not installed; CSV and markdown still generated. No installation needed.')
        return []
    valid = [float(r['duration_sec']) for r in meta if r['status'] == 'ok']
    if not valid:
        return []
    artifacts = []
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(valid, bins=45, color='#5094D5', alpha=0.85)
    ax.axvline(CLAP_SECONDS, color='#CC563B', ls='--', lw=2, label='CLAP first 10 seconds')
    ax.set(xlabel='Audio duration (seconds)', ylabel='Number of DEV clips', title='PawTalk / DEV audio duration distribution')
    ax.legend()
    fig.tight_layout()
    p = OUT / '01_DEV_duration_histogram.png'
    fig.savefig(p, dpi=150);plt.close(fig);artifacts.append(p)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    names = [r['activity'] for r in class_rows]
    medians = [float(r['median_s']) if r['median_s'] != 'n/a' else 0 for r in class_rows]
    ax.barh(names[::-1], medians[::-1], color='#5094D5')
    ax.axvline(CLAP_SECONDS, color='#CC563B', ls='--', lw=2)
    ax.set(xlabel='Median audio duration (seconds)', title='PawTalk / DEV median duration by activity')
    fig.tight_layout()
    p = OUT / '02_DEV_duration_by_activity.png'
    fig.savefig(p, dpi=150);plt.close(fig);artifacts.append(p)
    for p in artifacts:
        print(f'[SAVED] {p}', flush=True)
    return artifacts


def report(meta, summary, class_rows, bucket_rows, probes, perf_rows, perf_status):
    valid_probes = [r for r in probes if r['status'] == 'ok']
    low_windows = sum(int(r['low_level_window_count']) for r in valid_probes)
    sampled_windows = sum(int(r['windows_checked']) for r in valid_probes)
    near_full = sum(float(r['near_fullscale_fraction']) > .001 for r in valid_probes)
    invalid_probes = sum(r['status'] != 'ok' for r in probes)
    med = summary['overall_duration']['median']
    lines = [
        '# PawTalk STEP 4A — 音频时长与质量诊断（仅开发集）', '',
        '## 实验边界', '',
        '- 固定数据：STEP1 原始开发集 9,984 条；锁定测试集 2,496 条**未读取音频，也未做标签分析或模型评估**。',
        '- **没有训练模型、没有重新计算 CLAP、没有改变 STEP1–3 实验结果。**',
        '- 真实特征提取口径：CLAP 对每条音频只加载前 10 秒，重采样至 48kHz；MFCC 对全长音频加载并重采样至 16kHz。',
        '- 音频时长在此仅由文件元数据测量；超过 10 秒表示 CLAP *可能*漏掉后段内容，**不代表后段一定包含有效狗叫**。',
        '- 原始数据缺少犬只/录音来源身份信息；本诊断不能证明不存在来源泄漏。', '',
        '## 1. 时长与格式（全部 DEV 音频）', '',
        f'- 扫描 {summary["n_scanned"]} 条，元数据可读 {summary["n_valid"]} 条，失败 {summary["n_unreadable"]} 条。',
        f'- 音频时长：平均 {fmt(summary["overall_duration"]["mean"])}s；中位数 {fmt(med)}s；'
        f'P10={fmt(summary["overall_duration"]["p10"])}s；P90={fmt(summary["overall_duration"]["p90"])}s。',
        f'- 超过 CLAP 10 秒限制的音频：{summary["n_over_10s"]} 条（{summary["pct_over_10s"]:.2f}%）。',
        f'- 如果按前 10 秒截断，DEV 总录音时长中约 {summary["pct_total_duration_after_10s"]:.2f}% '
        '位于截断点之后（**只是时间比例，并非信息损失率**）。',
        f'- 采样率分布：`{json.dumps(summary["sample_rates"], ensure_ascii=False)}`。',
        f'- 声道数分布：`{json.dumps(summary["channels"], ensure_ascii=False)}`。',
        f'- 编码 subtype 分布：`{json.dumps(summary["subtypes"], ensure_ascii=False)}`。', '',
        '## 2. 各活动类别时长（秒）', '',
        '| 活动 | n | 中位数 | P90 | >10s 条数 | >10s 占比 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in class_rows:
        lines.append(f'| {r["activity"]} | {r["count"]} | {r["median_s"]} | {r["p90_s"]} | {r["above_10s"]} | {r["above_10s_pct"]:.2f}% |')
    lines.extend(['', '## 3. 时长分布', '', '| 时长段 | 音频数 | 占有效 DEV 比例 |', '|---|---:|---:|'])
    for r in bucket_rows:
        lines.append(f'| {r["bucket"]} | {r["count"]} | {r["percent"]:.2f}% |')
    lines += ['', '## 4. 波形抽检（启发式，不是噪声分类）', '',
              f'- 分层随机抽检 {len(probes)} 条（可读 {len(valid_probes)}；解码失败 {invalid_probes}）。',
              f'- 每条只读取开头/中间/结尾最多 3 个约 1 秒的窗口；共读取 {sampled_windows} 个窗口。',
              f'- RMS <0.005 的低电平窗口：{low_windows} 个。低电平不等于坏音频，可能是正常停顿。',
              f'- 在已解码抽检音频中，接近满幅值（|x|≥0.999）样本比例超过 0.1% 的文件：{near_full} 条。'
              '这是疑似高幅值提醒，**不能仅凭这个指标断言 clipping**。',
              '- 未计算客观噪声分数（没有干净参考音频），不能把 RMS 或 ZCR 当成噪声真值。', '',
              '## 5. STEP3 开发集 OOF 错误率与时长', '',
              f'- OOF 状态：`{perf_status}`；数据来源仅为 `optimization_step3/diagnostics/c1_g2_oof_predictions.csv`。',
              '- 按时长分组的 Accuracy 仅作探索，可能受类别比例影响；样本少或类别不全时不报告 8 类 Macro-F1。',
              '', '| 时长段 | n | Accuracy | 8类 Macro-F1（仅8类齐全） |',
              '|---|---:|---:|---:|']
    for r in perf_rows:
        lines.append(f'| {r["bucket"]} | {r["n"]} | {r["accuracy"]} | {r["macro_f1_all8"]} |')
    lines += ['', '## 6. 下一步决策规则（不是已证实的结果）', '',
              '- 若大量音频超过 10 秒，并且开发集 OOF 错误与时长有关：在 STEP4B 可考虑**仅针对长音频**比较前段/中段/尾段或多段 CLAP 表示。',
              '- 若多数音频不超过 10 秒：优先考虑 MFCC 的帧统计增强，而非全面重新提取 CLAP。',
              '- 若短音频错误率明显更高：先区分真实短叫声与录制不完整，再考虑适用的预处理，不要盲目补长。',
              '- 若采样率或声道分布不同：检查模型重采样和 mono 转换是否统一，无需直接改变已经验证的基线。',
              '- STEP4B 中所有新特征和超参数必须只在开发集上比较，**禁止查看锁定测试集的评估结果**。', '',
              '## 输出文件', '',
              '- `DEV_audio_metadata.csv`：每条开发音频的元数据、10 秒截断标志与可能截断时间。',
              '- `DEV_duration_by_activity.csv`、`DEV_duration_bins.csv`：汇总表。',
              '- `DEV_waveform_probe.csv`：小样本波形健康抽检；不是全数据质量评分。',
              '- `DEV_STEP3_OOF_by_duration.csv` 与 `DEV_STEP3_OOF_by_class_duration.csv`：若 STEP3 OOF 可用。',
              '- `01_DEV_duration_histogram.png`、`02_DEV_duration_by_activity.png`：有 matplotlib 时生成。', '']
    dest = OUT / 'STEP4A_诊断报告.md'
    dest.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[SAVED] {dest}', flush=True)


def main():
    ap = argparse.ArgumentParser(description='PawTalk STEP4A DEV-only audio duration and signal audit')
    ap.add_argument('--check', action='store_true', help='Check the locked split and 3 representative audio files; do not scan')
    ap.add_argument('--probe-count', type=int, default=400, help='Stratified DEV audio waveforms to sample (default 400; 0 disables)')
    args = ap.parse_args()
    if args.probe_count < 0:
        ap.error('--probe-count must be >= 0')
    print('[STEP4A] DEV-only audio metadata/quality audit; no model training or CLAP inference.', flush=True)
    dev, n_holdout = audit_locked_split()
    check_dependencies_and_data(dev)
    if args.check:
        return 0
    meta = read_metadata(dev)
    summary, by_class, by_bucket = summarize_metadata(meta)
    print(f'[DURATION] readable={summary["n_valid"]}/{len(dev)}; '
          f'median={fmt(summary["overall_duration"]["median"])}s; '
          f'>10s={summary["n_over_10s"]} ({summary["pct_over_10s"]:.2f}%); '
          f'post-10s time share={summary["pct_total_duration_after_10s"]:.2f}%', flush=True)
    probes = sample_quality(meta, args.probe_count)
    perf_rows, per_class_perf, perf_status = join_oof(meta)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv('DEV_audio_metadata.csv', meta, list(meta[0]))
    write_csv('DEV_duration_by_activity.csv', by_class, list(by_class[0]))
    write_csv('DEV_duration_bins.csv', by_bucket, list(by_bucket[0]))
    if probes:
        write_csv('DEV_waveform_probe.csv', probes, list(probes[0]))
    if perf_rows:
        write_csv('DEV_STEP3_OOF_by_duration.csv', perf_rows, list(perf_rows[0]))
        write_csv('DEV_STEP3_OOF_by_class_duration.csv', per_class_perf, list(per_class_perf[0]))
    plot_charts(meta, by_class)
    report(meta, summary, by_class, by_bucket, probes, perf_rows, perf_status)
    if summary['n_unreadable']:
        print(f'[WARN] {summary["n_unreadable"]} unreadable metadata records; inspect DEV_audio_metadata.csv.', flush=True)
    print('[PASS] STEP4A finished. Locked holdout untouched. STEP1/2/3 files unchanged.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}', flush=True)
        raise
