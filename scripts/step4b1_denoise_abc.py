"""PawTalk STEP4B-1: A/B/C audio preprocessing comparison, original locked DEV only.

A: original MFCC CSV + original CLAP SQLite (reproduce STEP3 C=1/gamma=2).
B: gentle 120Hz second-order high-pass MFCC + ORIGINAL CLAP.
C: conservative spectral-gate MFCC + ORIGINAL CLAP.

All DEV audio durations are used for every fold's training and OOF prediction.
Primary score: fixed >=1.0 second DEV subset, with same evaluation rows for A/B/C.
Also report whole DEV scores and coverage. DO NOT access locked holdout audio or scores.
This is exploratory research, not production inference or dog-independent evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'scripts'
sys.path.insert(0, str(SRC))
OUT = ROOT / 'results' / 'optimization_step4b1_abc_denoise'
CACHE = OUT / 'feature_cache'
METRICS = OUT / 'ABC_cv_metrics.csv'
REPORT = OUT / 'STEP4B1_ABC降噪对照报告.md'
STEP3_OOF = ROOT / 'results' / 'optimization_step3' / 'diagnostics' / 'c1_g2_oof_predictions.csv'
DEV_N = 9984
HOLDOUT_N = 2496
THRESHOLD = 1.0
DIM = 30
METHODS = ('B_highpass_120Hz', 'C_mild_spectral_gate')
FIELDS = ['audio_path', 'activity', 'source_size', 'source_mtime_ns'] + [f'f{i:02d}' for i in range(DIM)]
FEATURE_CONFIG = {
    'B_highpass_120Hz': {'method': 'butterworth_highpass', 'cutoff_hz': 120, 'order': 2, 'zero_phase': True},
    'C_mild_spectral_gate': {'method': 'time_quantile_spectral_floor', 'sr': 16000, 'fft': 512, 'hop': 128,
                             'noise_quantile': 0.20, 'subtraction_strength': 0.65, 'gain_floor': 0.55,
                             'gain_smoothing': 'median_filter_size_3x3'},
}
CACHE_VERSION = 'pawtalk-step4b1-mfcc-v1-librosa-baseline-13mfcc-2moments-rms-zcr'


def normpath(s):
    return str(s or '').strip().replace('\\', '/')


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, names):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite previous experiment: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    print(f'[SAVED] {path}', flush=True)


def highpass(y, sr=16000):
    from scipy.signal import butter, sosfiltfilt
    if y.size < 32:
        raise ValueError('Audio too short for safe zero-phase highpass')
    sos = butter(2, 120.0, btype='highpass', fs=sr, output='sos')
    return sosfiltfilt(sos, y).astype(np.float32)


def spectral_gate(y, sr=16000):
    import librosa
    from scipy.ndimage import median_filter
    # Low-strength attenuation; no artificial "clean" ground truth, no use of labels.
    # Use STFT to estimate per-frequency floor from quieter time frames.
    n_fft, hop = 512, 128
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)
    power = np.abs(spec) ** 2
    floor = np.quantile(power, 0.20, axis=1, keepdims=True)
    gain = 1.0 - 0.65 * floor / (power + 1e-9)
    gain = np.clip(gain, 0.55, 1.0)
    gain = median_filter(gain, size=(3, 3), mode='nearest')
    result = librosa.istft(spec * gain, hop_length=hop, n_fft=n_fft, center=True, length=len(y))
    return result.astype(np.float32)


def mfcc_vector_from_wave(y, sr=16000):
    import librosa
    if y.size < 400:
        raise ValueError('Audio too short for original MFCC baseline (<400 samples)')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    vector = np.concatenate((mfcc.mean(axis=1), mfcc.std(axis=1),
                             [rms.mean(), rms.std(), zcr.mean(), zcr.std()])).astype(np.float32)
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise RuntimeError('Bad 30-dimensional MFCC vector')
    return vector


def compute_two_features(path):
    import librosa
    # Original MFCC baseline uses librosa.load(...sr=16000,mono=True).
    # Never alter original WAV on disk. Read each WAV once for BOTH treatments.
    wave, sr = librosa.load(str(path), sr=16000, mono=True)
    if len(wave) < 400 or not np.isfinite(wave).all():
        raise RuntimeError(f'Bad/short waveform: {path}')
    return {METHODS[0]: mfcc_vector_from_wave(highpass(wave, sr), sr),
            METHODS[1]: mfcc_vector_from_wave(spectral_gate(wave, sr), sr)}


def row_fingerprint(root_path, rel):
    path = root_path / rel
    st = path.stat()
    return str(st.st_size), str(st.st_mtime_ns)


def read_cache(method, dev_rows, *, require_all=False):
    path = CACHE / f'{method}_mfcc30.csv'
    expected = {normpath(r['audio_path']): str(r['activity']) for r in dev_rows}
    if not path.exists():
        if require_all:
            raise FileNotFoundError(f'Missing {path}; run 23_STEP4B1_EXTRACT_FEATURES.bat first')
        return {}
    saved = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if set(FIELDS) - set(reader.fieldnames or []):
            raise RuntimeError(f'Invalid cache header in {path}')
        for record in reader:
            rel = normpath(record.get('audio_path'))
            if rel not in expected:
                raise RuntimeError(f'Cache contains non-DEV or unknown audio path: {rel}')
            if rel in saved:
                raise RuntimeError(f'Duplicate audio cache entry: {rel}')
            if record['activity'] != expected[rel]:
                raise RuntimeError(f'Cache label mismatch: {rel}')
            fsize, fmtime = row_fingerprint(ROOT, rel)
            if record['source_size'] != fsize or record['source_mtime_ns'] != fmtime:
                raise RuntimeError(f'Source WAV changed after extraction; cache invalid: {rel}')
            try:
                vec = np.asarray([float(record[f'f{i:02d}']) for i in range(DIM)], dtype=np.float32)
            except (ValueError, TypeError) as e:
                raise RuntimeError(f'Invalid numeric cache record for {rel}') from e
            if not np.isfinite(vec).all():
                raise RuntimeError(f'Non-finite cache entry: {rel}')
            saved[rel] = vec
    if require_all and len(saved) != len(expected):
        missing = set(expected) - set(saved)
        raise RuntimeError(f'{method}: cache missing {len(missing)} DEV rows; example={list(sorted(missing))[:3]}')
    return saved


def ensure_config():
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = CACHE / 'feature_config.json'
    value = {'cache_version': CACHE_VERSION, 'config': FEATURE_CONFIG,
             'train_split_rule': 'Original locked 9984 DEV only; no holdout audio decoded'}
    if meta.exists():
        saved = json.loads(meta.read_text(encoding='utf-8'))
        if saved != value:
            raise RuntimeError('Cache processing config changed; do not mix different MFCC pipelines')
    else:
        if any(CACHE.glob('*_mfcc30.csv')):
            raise RuntimeError('Existing feature cache lacks feature_config.json; refusing to mix versions')
        meta.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare():
    # Reuse STEP4B-0's independently checked DEV audio, labels, durations and STEP3 OOF.
    # Never read holdout labels as input to fit/evaluate. Original source manifest
    # integrity checks may STAT file existence, but no holdout WAV is decoded.
    from step4b_duration_noise import prepare as upstream_prepare
    from step2_feature_fusion import score
    rows, y, clap, mfcc, durations, keep, old_pred = upstream_prepare()
    if not (len(rows) == DEV_N and clap.shape == (DEV_N, 512) and mfcc.shape == (DEV_N, DIM)):
        raise RuntimeError('DEV original shapes changed; refusing experiment')
    if int(keep.sum()) != 4954 or not np.array_equal(keep, durations >= THRESHOLD):
        raise RuntimeError('1s mask differs from STEP4B-0; refusing experiment')
    baseline = score(y, old_pred)
    eligible = score(y[keep], old_pred[keep])
    if abs(baseline['macro_f1'] - 0.3226) > 0.0002 or abs(eligible['macro_f1'] - 0.3570) > 0.0002:
        raise RuntimeError(f'STEP3 reference differs: {baseline}, >=1s={eligible}')
    print('[BASELINE] STEP3 OOF all {:.4f}; >=1s {:.4f}; coverage={:.2f}%'.format(
        baseline['macro_f1'], eligible['macro_f1'], 100*keep.mean()), flush=True)
    print('[SAFETY] Original DEV 9984 used for training, >=1s DEV 4954 primary evaluation.', flush=True)
    print('[LOCK] Holdout 2496 NOT decoded, fitted, scored or included in CV.', flush=True)
    return rows, y, clap, mfcc, durations, keep, old_pred


def check():
    rows, y, clap, original, dur, keep, old = prepare()
    import librosa
    import scipy
    import soundfile
    print(f'[ENV] numpy={np.__version__}; librosa={librosa.__version__}; '
          f'scipy={scipy.__version__}; soundfile={soundfile.__version__}', flush=True)
    # Smoke-check both approaches without saving files or labels.
    for i in np.flatnonzero(keep)[:2]:
        path = ROOT / normpath(rows[i]['audio_path'])
        got = compute_two_features(path)
        for method, v in got.items():
            if len(v) != DIM:
                raise RuntimeError(f'Bad smoke test vector shape for {method}')
        # Comparing current library raw extractor to cache is diagnostic only;
        # cache may have been created under an earlier librosa version.
        raw, sr = librosa.load(str(path), sr=16000, mono=True)
        discrepancy = float(np.max(np.abs(mfcc_vector_from_wave(raw, sr) - original[i])))
        print(f'[SAMPLE] {path.name}: {dur[i]:.3f}s; B/C MFCC=30D; raw-cache maxabsdiff={discrepancy:.4g}', flush=True)
        if discrepancy > 1.0:
            print('[WARN] Current librosa raw features differ from original baseline cache. '
                  'Do not interpret B/C as denoise-only effects before reviewing this difference.', flush=True)
    print('[PASS] STEP4B-1 preflight complete. No model trained.', flush=True)


def extract():
    rows, y, clap, original, durations, keep, old = prepare()
    import librosa
    ensure_config()
    caches = {name: read_cache(name, rows) for name in METHODS}
    fields = {name: CACHE / f'{name}_mfcc30.csv' for name in METHODS}
    handles = {}
    writers = {}
    try:
        for name in METHODS:
            file = fields[name]
            new = not file.exists()
            handles[name] = file.open('a', newline='', encoding='utf-8-sig' if new else 'utf-8')
            writers[name] = csv.DictWriter(handles[name], fieldnames=FIELDS)
            if new:
                writers[name].writeheader()
        count = 0
        start = time.monotonic()
        for i, row in enumerate(rows, 1):
            rel = normpath(row['audio_path'])
            pending = [m for m in METHODS if rel not in caches[m]]
            if not pending:
                continue
            fpath = ROOT / rel
            try:
                # Compute both when cache is missing for both; if one already cached,
                # this still computes both but writes only missing, preserving consistency.
                feats = compute_two_features(fpath)
                size, mtime = row_fingerprint(ROOT, rel)
                for method in pending:
                    record = {'audio_path': rel, 'activity': str(row['activity']),
                              'source_size': size, 'source_mtime_ns': mtime}
                    record.update({f'f{k:02d}': repr(float(feats[method][k])) for k in range(DIM)})
                    writers[method].writerow(record)
                    handles[method].flush()  # each row survives interruption
                count += 1
            except Exception as e:
                raise RuntimeError(f'Audio extraction failed at {rel}: {e}') from e
            if count % 200 == 0:
                print(f'[EXTRACT] processed-new={count}, scanned={i}/{len(rows)} / {len(rows)}; '
                      f'elapsed={time.monotonic()-start:.1f}s', flush=True)
        print(f'[EXTRACT] new audio processed={count}; cache is resumable.', flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    for method in METHODS:
        done = read_cache(method, rows, require_all=True)
        print(f'[CACHE] {method}: {len(done)}/{len(rows)} aligned 30D vectors', flush=True)
    print('[PASS] STEP4B-1 B/C MFCC features extracted, original WAV files unchanged.', flush=True)


def evaluate_model(method, X, y, keep, expected_old_pred=None):
    from step1_split_and_weights import SEED, CV_FOLDS
    from step3_svm_tuning import build_model
    from step2_feature_fusion import score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    preds = np.empty(len(y), dtype=object)
    t0 = time.monotonic()
    for fold_id, (train, valid) in enumerate(cv.split(X, y), 1):
        model = build_model(y[train], clap_dim=512, mfcc_dim=DIM, C=1.0, gamma=2.0)
        model.fit(X[train], y[train])
        preds[valid] = model.predict(X[valid])
        fold_f1 = float(f1_score(y[valid], preds[valid], labels=None, average='macro', zero_division=0))
        print(f'[CV] {method} fold {fold_id}/{CV_FOLDS}: '
              f'train={len(train)}; val={len(valid)}; Macro-F1={fold_f1:.4f}', flush=True)
    if expected_old_pred is not None:
        n_wrong = int(np.sum(preds != expected_old_pred))
        if n_wrong:
            raise RuntimeError(f'A baseline does not reproduce STEP3 stored OOF exactly: '
                               f'{n_wrong} predictions differ. Stop before B/C training.')
        print(f'[VERIFY] A original OOF matches STEP3 saved predictions exactly ({len(y)}/{len(y)}).', flush=True)
    all_scores = score(y, preds)
    selected_scores = score(y[keep], preds[keep])
    print(f'[RESULT] {method}: >=1s Macro-F1={selected_scores["macro_f1"]:.4f}, '
          f'Acc={selected_scores["accuracy"]:.4f}; full DEV Macro-F1={all_scores["macro_f1"]:.4f}', flush=True)
    return preds, all_scores, selected_scores, round(time.monotonic()-t0, 1)


def run():
    rows, y, clap, mfcc, durations, keep, old_pred = prepare()
    ensure_config()  # confirm cache extraction parameters were not changed
    cache = {method: read_cache(method, rows, require_all=True) for method in METHODS}
    if METRICS.exists() or REPORT.exists():
        raise RuntimeError(f'Final STEP4B-1 results already exist; do not overwrite them: {OUT}')
    features = {'A_original': mfcc}
    for method in METHODS:
        features[method] = np.stack([cache[method][normpath(row['audio_path'])] for row in rows]).astype(np.float32)
    full_oof = {}
    summary = []
    for method, x_mfcc in features.items():
        X = np.concatenate((clap, x_mfcc), axis=1)
        print(f'[CV] {method}: three folds, all DEV training, >=1s primary evaluation', flush=True)
        pred, score_all, score_one, elapsed = evaluate_model(method, X, y, keep,
                                 expected_old_pred=old_pred if method == 'A_original' else None)
        full_oof[method] = pred
        summary.append({'variant': method, 'n_dev_train_pool': len(y), 'n_dev_eval_primary': int(keep.sum()),
                        'coverage': round(float(keep.mean()), 6),
                        'primary_macro_f1': score_one['macro_f1'], 'primary_accuracy': score_one['accuracy'],
                        'primary_attention_f1': score_one['attention_f1'], 'primary_food_f1': score_one['food_f1'],
                        'full_macro_f1': score_all['macro_f1'], 'full_accuracy': score_all['accuracy'],
                        'seconds_cv': elapsed})
    best = max(summary, key=lambda r: (r['primary_macro_f1'], r['full_macro_f1']))
    print(f'[SELECT] DEV exploratory best: {best["variant"]}; primary Macro-F1={best["primary_macro_f1"]:.4f}', flush=True)
    # Save one preprocessor-matched DEV-only model; immutable holdout remains untouched.
    from step3_svm_tuning import build_model
    import joblib
    selected_X = np.concatenate([clap, features[best['variant']]], axis=1)
    selected_model = build_model(y, clap_dim=512, mfcc_dim=DIM, C=1.0, gamma=2.0)
    selected_model.fit(selected_X, y)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, OUT / 'selected_ABC_MODEL_DEV_ONLY.joblib')
    write_csv(METRICS, summary, list(summary[0].keys()))
    oof_rows = []
    for i, row in enumerate(rows):
        item = {'audio_path': normpath(row['audio_path']), 'actual': str(y[i]),
                'duration_sec': f'{durations[i]:.6f}', 'eligible_1s': int(keep[i])}
        for name in features:
            item[f'{name}_pred'] = str(full_oof[name][i])
            item[f'{name}_correct'] = int(full_oof[name][i] == y[i])
        oof_rows.append(item)
    write_csv(OUT / 'ABC_dev_oof_predictions.csv', oof_rows, list(oof_rows[0].keys()))
    details = {'variants': FEATURE_CONFIG, 'selected_variant': best['variant'],
               'primary_evaluation_min_seconds': THRESHOLD,
               'training_includes_all_dev_durations': True,
               'train_cv': {'seed': 20261112, 'folds': 3, 'class_weight': 'balanced fold only',
                            'fusion': '50pct each dimension normalized', 'C': 1.0, 'gamma': 2.0},
               'CLAP_input': 'ORIGINAL audio, previous 512D embeddings from SQLite, first 10 seconds',
               'MFCC_input': 'B/C denoised audio, 30D computed from same original WAV without overwriting',
               'holdout_evaluated': False, 'known_limitations': [
                   'Comparing A/B/C on repeatedly used DEV folds has model-selection optimism',
                   'No dog/video IDs; within-dog or within-session leakage may remain',
                   'The spectral gate may remove meaningful activity context such as water, bells or human voices',
                   'External deployment needs exact same MFCC transform and raw CLAP embedding extraction',
                   'These A/B/C settings were predefined; noise intensity not yet tuned'],
               'results': summary}
    (OUT / 'ABC_experiment_metadata.json').write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding='utf-8')
    report = ['# PawTalk STEP4B-1：A / B / C 温和降噪对照（仅 DEV OOF）', '',
              '- **训练**：所有 9,984 条原 DEV，含 <1 秒音频。',
              f'- **主要评价**：相同的 >=1 秒 DEV {keep.sum()} 条（覆盖率 {100*keep.mean():.2f}%）。',
              '- **辅助评价**：同一模型在全部 DEV 的 OOF 指标。',
              '- **CLAP**：三组均采用相同的原始音频 512 维缓存，**未**对 CLAP 音频降噪。',
              '- **MFCC**：A 原始 CSV；B 120Hz 二阶零相位高通；C 轻度频谱门限。',
              '- **模型固定**：MFCC/CLAP 各 50% 融合 + RBF SVM (C=1,gamma=2), 各折重新估计 balanced 类别权重。',
              '- 三折成员与 STEP3 完全相同，标准化在各折训练集拟合。',
              '- 未触碰锁定的 2,496 条测试集。', '',
              '| Variant | >=1s Macro-F1 | >=1s Acc | attention F1 | food F1 | Full DEV Macro-F1 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        report.append(f'| {r["variant"]} | {r["primary_macro_f1"]:.4f} | {r["primary_accuracy"]:.4f} | '
                      f'{r["primary_attention_f1"]:.4f} | {r["primary_food_f1"]:.4f} | {r["full_macro_f1"]:.4f} |')
    reference = summary[0]
    report.extend(['',f'**开发集主要评价最优：{best["variant"]}，Macro-F1={best["primary_macro_f1"]:.4f}。**',
                   f'相对 A：{best["primary_macro_f1"]-reference["primary_macro_f1"]:+.4f}，仍须独立测试验证。',
                   '', '## 研究边界',
                   '- A/B/C 目前仅改 MFCC 分支，而非端到端 CLAP 降噪。',
                   '- 背景声可能包含行为语境，不可以从声音变干净推断分类会提高。',
                   '- 该实验基于多次使用的开发集，不能作为独立最终成绩。',
                   '- 如果 B/C 最好，最终部署需同时应用同样 MFCC 预处理；现有仅原始音频 CLAP。',
                   '- 不建议为了追求更高开发集得分而重新划分或查看 holdout。'])
    REPORT.write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(f'[SAVED] {REPORT}', flush=True)
    print('[PASS] STEP4B-1 A/B/C completed. Holdout untouched, originals unchanged.', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='validate DEV split and audio preprocessing without training')
    ap.add_argument('--extract', action='store_true', help='compute and cache B/C 30D MFCC features, resumable')
    ap.add_argument('--run', action='store_true', help='run 3-fold ABC evaluation, all DEV training, >=1s primary')
    args = ap.parse_args()
    if sum([args.check, args.extract, args.run]) != 1:
        ap.error('Choose exactly one of --check, --extract, --run')
    print('[STEP4B-1] A original vs B highpass vs C mild spectral gate.', flush=True)
    if args.check:
        check()
    elif args.extract:
        extract()
    elif args.run:
        run()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}', flush=True)
        sys.exit(2)
