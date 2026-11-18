"""PawTalk STEP4B-0: fixed DEV >=1.0s rerun and cautious audio-noise screening.

Never deletes/moves WAVs; never trains or scores locked holdout.
Old DEV OOF and same 3 folds provide a matched-cohort comparison.
No denoising is performed here: environmental cues may be part of the task.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_dataset import CATEGORIES
from step1_split_and_weights import CV_FOLDS, SEED
from step2_feature_fusion import load_dev, score
from step3_svm_tuning import build_model

META = ROOT / 'results' / 'optimization_step4a' / 'DEV_audio_metadata.csv'
OLD_OOF = ROOT / 'results' / 'optimization_step3' / 'diagnostics' / 'c1_g2_oof_predictions.csv'
OUT = ROOT / 'results' / 'optimization_step4b_duration_noise'
THRESHOLD = 1.0
EXPECTED_DEV, EXPECTED_HOLD = 9984, 2496
EXPECTED_OLD_F1, EXPECTED_OLD_ACC = 0.3226, 0.3208


def normpath(x):
    return str(x or '').strip().replace('\\', '/')


def read_indexed_csv(filename, needed):
    if not filename.is_file():
        raise FileNotFoundError(f'Missing CSV: {filename}')
    with filename.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        missing = set(needed) - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f'{filename.name} lacks columns: {sorted(missing)}')
        rows = {}
        for item in reader:
            key = normpath(item.get('audio_path'))
            if not key or key in rows:
                raise RuntimeError(f'Missing or duplicate audio_path in {filename}: {key!r}')
            rows[key] = item
    return rows


def write_csv(path, rows, fields):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'[SAVED] {path}', flush=True)


def prepare():
    rows, y, X_clap, X_mfcc, n_hold = load_dev()  # unchanged STEP1 locked partition
    if len(rows) != EXPECTED_DEV or n_hold != EXPECTED_HOLD or X_clap.shape != (EXPECTED_DEV, 512) or X_mfcc.shape != (EXPECTED_DEV, 30):
        raise RuntimeError('STEP1-3 sizes differ from verified history; stop for audit.')
    paths = [normpath(r['audio_path']) for r in rows]
    if len(set(paths)) != len(paths):
        raise RuntimeError('Duplicate DEV paths')
    metadata = read_indexed_csv(META, {'audio_path', 'activity', 'duration_sec', 'status'})
    old = read_indexed_csv(OLD_OOF, {'audio_path', 'actual', 'predicted'})
    if set(paths) != set(metadata) or set(paths) != set(old):
        raise RuntimeError('DEV, STEP4A metadata, STEP3 OOF are not identical (path mismatch).')
    dur = np.empty(len(rows), dtype=np.float64)
    old_pred = np.empty(len(rows), dtype=object)
    for i, (path, row, label) in enumerate(zip(paths, rows, y)):
        m, p = metadata[path], old[path]
        if m['status'] != 'ok' or m['activity'] != label or p['actual'] != label or p['predicted'] not in CATEGORIES:
            raise RuntimeError(f'Missing audio data or activity mismatch: {path}')
        dur[i] = float(m['duration_sec'])
        old_pred[i] = p['predicted']
    if not np.isfinite(dur).all() or (dur <= 0).any():
        raise RuntimeError('Invalid audio durations.')
    full_scores = score(y, old_pred)
    if abs(full_scores['macro_f1'] - EXPECTED_OLD_F1) > 0.00015 or abs(full_scores['accuracy'] - EXPECTED_OLD_ACC) > 0.00015:
        raise RuntimeError(f'STEP3 old OOF metrics mismatch: {full_scores}')
    keep = dur >= THRESHOLD  # exactly 1.000s is kept
    retained_classes = Counter(y[keep])
    if not set(CATEGORIES).issubset(retained_classes) or min(retained_classes.values()) < CV_FOLDS:
        raise RuntimeError(f'Too few samples for 3-fold CV in one class after threshold: {retained_classes}')
    print(f'[VERIFY] Old STEP3 OOF Macro-F1={full_scores["macro_f1"]:.4f}; Accuracy={full_scores["accuracy"]:.4f}', flush=True)
    print(f'[FILTER] >=1.000s kept={keep.sum()}/{len(rows)} ({keep.mean()*100:.2f}%); frozen <1s={len(rows)-keep.sum()}', flush=True)
    print(f'[CLASS] retained by class={dict(retained_classes)}', flush=True)
    print('[LOCK] Original split unchanged; holdout is never fitted or scored.', flush=True)
    return rows, y, X_clap, X_mfcc, dur, keep, old_pred


def noise_probe(rows, y, keep, durations, old_pred, sample_size=400):
    """Lightweight signal indicators. No SNR estimate or automatic noise labels."""
    import soundfile as sf
    eligible = np.flatnonzero(keep)
    take = min(sample_size, len(eligible))
    if take < 8:
        return []
    if take == len(eligible):
        chosen = eligible
    else:
        subset, _ = train_test_split(eligible, train_size=take, random_state=SEED, stratify=y[eligible])
        chosen = np.asarray(sorted(subset), dtype=int)
    out = []
    for k, i in enumerate(chosen, 1):
        path = normpath(rows[i]['audio_path'])
        fpath = ROOT / path
        entry = {'audio_path': path, 'actual': str(y[i]), 'duration_sec': round(float(durations[i]), 6),
                 'step3_correct': int(old_pred[i] == y[i]), 'sampled_seconds': '', 'status': 'ok',
                 'rms': '', 'clipping_fraction': '', 'quietest_20pct_to_loudest_20pct_rms_ratio': '',
                 'low_freq_below_120hz_energy_ratio': '', 'spectral_flatness': '', 'error': ''}
        try:
            with sf.SoundFile(str(fpath)) as audio:
                length = min(audio.frames, max(1, int(audio.samplerate * 3.0)))
                block = audio.read(length, dtype='float32', always_2d=True)
                x = np.mean(block, axis=1).astype(np.float64)
                sr = int(audio.samplerate)
            if x.size < 64 or not np.isfinite(x).all():
                raise ValueError('Unreadable waveform or too few samples')
            frame = max(64, int(sr * 0.025))
            n_frames = len(x) // frame
            if n_frames < 2:
                raise ValueError('Too few 25ms frames')
            shaped = x[:n_frames*frame].reshape(n_frames, frame)
            frame_rms = np.sqrt(np.mean(shaped**2, axis=1) + 1e-12)
            low = np.mean(np.sort(frame_rms)[:max(1, int(0.2*n_frames))])
            high = np.mean(np.sort(frame_rms)[-max(1, int(0.2*n_frames)):])
            window = np.hanning(min(len(x), sr))
            segment = x[:len(window)] * window
            freqs = np.fft.rfftfreq(len(segment), d=1.0/sr)
            power = np.abs(np.fft.rfft(segment))**2
            power = power + 1e-12
            use = (freqs >= 30) & (freqs <= min(7500, sr/2))
            mid = power[use]
            flatness = float(np.exp(np.mean(np.log(mid))) / np.mean(mid)) if mid.size else float('nan')
            entry.update(sampled_seconds=round(len(x)/sr, 4),
                         rms=round(float(np.sqrt(np.mean(x*x))), 7),
                         clipping_fraction=round(float(np.mean(np.abs(x) >= .98)), 7),
                         quietest_20pct_to_loudest_20pct_rms_ratio=round(float(low/max(high, 1e-12)), 6),
                         low_freq_below_120hz_energy_ratio=round(float(power[(freqs < 120)].sum()/power.sum()), 6),
                         spectral_flatness=round(flatness, 6))
        except Exception as exc:
            entry['status'] = 'error'
            entry['error'] = f'{type(exc).__name__}: {exc}'[:200]
        out.append(entry)
        if k % 100 == 0 or k == len(chosen):
            print(f'[NOISE-PROBE] {k}/{len(chosen)} sampled (no modification)', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Check alignment and exact >=1s count; do not train/write')
    args = parser.parse_args()
    print('[STEP4B-0] >=1s DEV experimental branch; no original audio/data/split is changed.', flush=True)
    rows, y, clap, mfcc, dur, mask, original_pred = prepare()
    if args.check:
        print('[PASS] STEP4B-0 preflight passed. No files written, no model trained.', flush=True)
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f'Non-empty output directory {OUT}; refusing to overwrite previous experiment.')
    OUT.mkdir(parents=True, exist_ok=True)
    retained_idx = np.flatnonzero(mask)
    frozen_idx = np.flatnonzero(~mask)
    write_csv(OUT/'DEV_kept_at_least_1s.csv',
              [{'audio_path':normpath(rows[i]['audio_path']), 'activity':str(y[i]), 'duration_sec':f'{dur[i]:.6f}'} for i in retained_idx],
              ['audio_path','activity','duration_sec'])
    write_csv(OUT/'DEV_frozen_under_1s.csv',
              [{'audio_path':normpath(rows[i]['audio_path']), 'activity':str(y[i]), 'duration_sec':f'{dur[i]:.6f}'} for i in frozen_idx],
              ['audio_path','activity','duration_sec'])
    probe = noise_probe(rows, y, mask, dur, original_pred)
    if probe:
        write_csv(OUT/'DEV_noise_heuristics_sample.csv', probe, list(probe[0]))
    print('[PROBE] Indicators describe signal characteristics; they are NOT ground-truth noise/SNR labels.', flush=True)
    X = np.concatenate([clap, mfcc], axis=1)
    old_scores = score(y[mask], original_pred[mask])
    print(f'[COMPARE] Original STEP3 OOF on SAME >=1s cohort: Macro-F1={old_scores["macro_f1"]:.4f}; Accuracy={old_scores["accuracy"]:.4f}', flush=True)
    # Build folds from FULL original DEV indices to keep val memberships identical to STEP3.
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    preds = np.empty(len(y), dtype=object)
    preds[:] = None
    t0 = time.monotonic()
    for fi, (train_all, val_all) in enumerate(folds.split(X, y), 1):
        train = train_all[mask[train_all]]
        valid = val_all[mask[val_all]]
        if min(Counter(y[train]).values()) < 2:
            raise RuntimeError(f'Fold {fi}: fewer than 2 training samples in a class')
        model = build_model(y[train], clap.shape[1], mfcc.shape[1], C=1.0, gamma=2.0)
        model.fit(X[train], y[train])
        preds[valid] = model.predict(X[valid])
        fold_score = f1_score(y[valid], preds[valid], labels=CATEGORIES, average='macro', zero_division=0)
        print(f'[CV] filtered fold {fi}/{CV_FOLDS}: train={len(train)}, val={len(valid)}, Macro-F1={fold_score:.4f}', flush=True)
    if any(v is None for v in preds[mask]):
        raise RuntimeError('Incomplete >=1s out-of-fold predictions')
    filt_scores = score(y[mask], preds[mask])
    print(f'[RESULT] Filtered-only retrained OOF Macro-F1={filt_scores["macro_f1"]:.4f}; Accuracy={filt_scores["accuracy"]:.4f}', flush=True)
    print(f'[DELTA] On same >=1s cohort: Macro-F1 {filt_scores["macro_f1"]-old_scores["macro_f1"]:+.4f}; Accuracy {filt_scores["accuracy"]-old_scores["accuracy"]:+.4f}', flush=True)
    write_csv(OUT/'DEV_1s_retrained_oof_predictions.csv',
              [{'audio_path':normpath(rows[i]['audio_path']), 'actual':str(y[i]),
                'old_step3_prediction':str(original_pred[i]), 'new_filtered_prediction':str(preds[i]),
                'old_correct':int(original_pred[i]==y[i]), 'new_correct':int(preds[i]==y[i]),
                'duration_sec':f'{dur[i]:.6f}'} for i in retained_idx],
              ['audio_path','actual','old_step3_prediction','new_filtered_prediction','old_correct','new_correct','duration_sec'])
    final_model = build_model(y[mask], clap.shape[1], mfcc.shape[1], C=1.0, gamma=2.0)
    final_model.fit(X[mask], y[mask])
    joblib.dump(final_model, OUT/'MODEL_FILTERED_DEV_ONLY.joblib')
    values = {'threshold_sec':THRESHOLD,'dev_total':len(y),'kept_dev':int(mask.sum()),'frozen_dev':int((~mask).sum()),
              'kept_class_counts':dict(Counter(y[mask])), 'original_step3_all_dev_metrics':score(y, original_pred),
              'old_step3_oof_on_kept_cohort':old_scores,'filtered_retrained_oof_on_same_cohort':filt_scores,
              'delta_f1_on_same_cohort':round(filt_scores['macro_f1']-old_scores['macro_f1'],6),
              'delta_acc_on_same_cohort':round(filt_scores['accuracy']-old_scores['accuracy'],6),
              'fold_rule':'Original full-DEV StratifiedKFold memberships, apply >=1s mask inside each train and valid fold',
              'model':'STEP3 fixed 50% MFCC+CLAP fused RBF SVM C=1 gamma=2 class weights fold-balanced',
              'noise_probe_n':len(probe),'noise_probe_invalid':sum(r['status']!='ok' for r in probe),
              'holdout_scored':False,'elapsed_training_sec':round(time.monotonic()-t0,1)}
    (OUT/'STEP4B_1s_comparison.json').write_text(json.dumps(values,indent=2,ensure_ascii=False),encoding='utf-8')
    report = [
        '# PawTalk STEP4B-0：≥1 秒训练实验 + 噪声初筛','',
        f'- 原始 DEV: {len(y)}；保留 ≥1 秒: {mask.sum()} ({mask.mean()*100:.1f}%)；冻结 <1 秒: {(~mask).sum()}。',
        '- 此处只冻结原始 DEV 的短音频，不删除 WAV、不改写原始分层划分，不读取测试音频进行模型评估。',
        '- 仍然使用 STEP3 模型结构和预处理：CLAP512 + MFCC30，融合 50%，SVM C=1, gamma=2。',
        '- 用 STEP3 原始折次，只在每折 train/valid 内排除 <1 秒；模型预处理仍严格在各折训练集内拟合。',
        '', '| 比较（完全相同的 ≥1 秒验证样本） | Accuracy | Macro-F1 | Seeking Attention F1 | Begging for Food F1 |',
        '|---|---:|---:|---:|---:|',
        f'| 旧 STEP3 OOF（训练含短录音） | {old_scores["accuracy"]:.4f} | {old_scores["macro_f1"]:.4f} | {old_scores["attention_f1"]:.4f} | {old_scores["food_f1"]:.4f} |',
        f'| 新 ≥1秒 OOF（训练排除短录音） | {filt_scores["accuracy"]:.4f} | {filt_scores["macro_f1"]:.4f} | {filt_scores["attention_f1"]:.4f} | {filt_scores["food_f1"]:.4f} |',
        f'| 差值（新-旧） | {filt_scores["accuracy"]-old_scores["accuracy"]:+.4f} | {filt_scores["macro_f1"]-old_scores["macro_f1"]:+.4f} | {filt_scores["attention_f1"]-old_scores["attention_f1"]:+.4f} | {filt_scores["food_f1"]-old_scores["food_f1"]:+.4f} |',
        '', '## 噪声初筛（不能直接当作噪音真值）','',
        f'- 从 ≥1 秒 DEV 中按八类分层抽取 {len(probe)} 条，读取每条前 3 秒左右的波形。',
        '- 输出 `DEV_noise_heuristics_sample.csv`：电平、接近削波比例、低频能量占比、频谱平坦度、安静段相对电平。',
        '- 这些只是可疑声音条件的线索，无法单凭指标证明是噪声；需要人工听辨少量样本。',
        '- **背景声音可能本来就携带活动情境**：去掉背景可能导致八类活动识别变差。',
        '- 真正的下一实验应同时保留原音对照，尝试温和高通/频谱降噪，逐一比较相同折次的 OOF 指标。',
        '- 若对 CLAP 输入进行降噪，就必须重新提取相应 CLAP 特征，不能把旧缓存当作降噪后的表示。',
        '', '## 研究解释边界','',
        '- 删除难样本会机械性改变测评样本和覆盖率，**不能**仅比较筛选后的准确率与全部 9,984 条准确率。',
        '- 上表只比较相同 ≥1 秒 cohort 上的 OOF，才能更接近测量“训练中剔除短样本”的实际效果。',
        '- 仍然是探索性 DEV 结果，数据缺少犬只/视频身份，未消除个体相关数据泄漏。',
        '- 原 2,496 条 holdout 保持锁定，最终模型选择完成后才进行一次独立评估。',
    ]
    (OUT/'STEP4B_1秒筛选与噪声初筛报告.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'[SAVED] {OUT / "STEP4B_1秒筛选与噪声初筛报告.md"}',flush=True)
    print('[PASS] STEP4B-0 finished. Holdout never evaluated; originals unchanged.',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}', flush=True)
        raise
