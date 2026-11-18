"""PawTalk | Barkopedia supervised CLAP embeddings + SVM.

Reuse LAION CLAP weights, DO NOT fine-tune the full neural network. Audio embeddings
are persisted in SQLite for re-use across interrupted runs, and extracted independently
of ground-truth labels. Train and validation manifests from 02_AUDIT_DATA.bat.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Works whether this file is copied into the original project's scripts folder.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_dataset import CATEGORIES

MODEL_ID = 'laion/clap-htsat-unfused'
FEATURE_VERSION = 'clap_htsat_unfused_first10s_sr48000_v1'
CACHE_DB = ROOT / 'results' / 'clap_embeddings_cache.sqlite3'


def read_manifest(split: str):
    path = ROOT / 'results' / f'{split}_manifest.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. Run 02_AUDIT_DATA.bat first.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get('audio_match') == 'MATCH'
            and row.get('activity') in CATEGORIES and (ROOT / row.get('audio_path', '')).is_file()]


def choose_training(rows, per_class):
    if per_class <= 0:
        return rows
    counts = Counter()
    selected = []
    # Manifest preserves deterministic dataset ordering; class limit is fixed and reproducible.
    for row in rows:
        label = row['activity']
        if counts[label] < per_class:
            selected.append(row)
            counts[label] += 1
    return selected


def make_key(rel_path: str):
    path = ROOT / rel_path
    stat = path.stat()
    seed = f'{FEATURE_VERSION}|{rel_path}|{stat.st_size}|{stat.st_mtime_ns}'
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()


def open_cache():
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_DB)
    db.execute('''CREATE TABLE IF NOT EXISTS embeddings (
        cache_key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL
    )''')
    db.commit()
    return db


def get_cached(db, key):
    row = db.execute('SELECT dim,vector FROM embeddings WHERE cache_key=?', (key,)).fetchone()
    if row is None:
        return None
    dim, raw = row
    x = np.frombuffer(raw, dtype=np.float32)
    if len(x) != dim or not np.isfinite(x).all():
        return None
    return x.copy()


def get_embeddings(rows, split, db, batch_size):
    """Extract each clip's first 10 seconds; cache by file modification metadata."""
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    print(f'[CLAP] Loading {MODEL_ID}, using existing local cache if present', flush=True)
    model = ClapModel.from_pretrained(MODEL_ID).eval()
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[DEVICE] {device}; audio truncation: first 10 seconds; sampling rate: 48000 Hz', flush=True)
    model = model.to(device)
    results = [None] * len(rows)
    pending = []
    hit = 0
    for i, r in enumerate(rows):
        key = make_key(r['audio_path'])
        old = get_cached(db, key)
        if old is not None:
            results[i] = old
            hit += 1
        else:
            pending.append((i, r, key))
    print(f'[FEATURE] {split}: {len(rows)} clips, {hit} cached, {len(pending)} remaining', flush=True)

    failures = []
    processed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        waveforms, valid = [], []
        for index, row, key in batch:
            try:
                audio, _ = librosa.load(ROOT / row['audio_path'], sr=48000, mono=True, duration=10.0)
                if len(audio) < 400 or not np.isfinite(audio).all():
                    raise ValueError('empty/too short/non-finite audio')
                waveforms.append(audio)
                valid.append((index, row, key))
            except Exception as exc:
                failures.append({'split': split, 'audio_path': row['audio_path'], 'error': str(exc)})
        if valid:
            # 'audio' is the non-deprecated name; some older Transformers versions only accept 'audios'.
            try:
                inputs = processor(audio=waveforms, sampling_rate=48000,
                                   return_tensors='pt', padding=True)
            except (TypeError, ValueError):
                inputs = processor(audios=waveforms, sampling_rate=48000,
                                   return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                feats = model.get_audio_features(**inputs)
                if not isinstance(feats, torch.Tensor):
                    feats = feats.pooler_output if hasattr(feats, 'pooler_output') else feats[0]
                feats = feats.float().cpu().numpy()
            if feats.ndim != 2 or feats.shape[0] != len(valid):
                raise RuntimeError(f'Unexpected CLAP embedding shape: {feats.shape}')
            with db:
                for (index, row, key), vec in zip(valid, feats):
                    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
                    if not np.isfinite(vec).all():
                        failures.append({'split': split, 'audio_path': row['audio_path'], 'error': 'Nonfinite embedding'})
                        continue
                    db.execute('INSERT OR REPLACE INTO embeddings (cache_key,dim,vector) VALUES (?,?,?)',
                               (key, vec.size, vec.tobytes()))
                    results[index] = vec
        processed += len(batch)
        if processed % 20 < batch_size or processed == len(pending):
            print(f'    {split}: processed {processed}/{len(pending)} new clips', flush=True)

    kept = [(r, v) for r, v in zip(rows, results) if v is not None]
    if not kept:
        raise RuntimeError(f'No embeddings extracted for {split}')
    if failures:
        out = ROOT / 'results' / 'clap_supervised' / f'{split}_audio_failures.csv'
        out.parent.mkdir(exist_ok=True, parents=True)
        with out.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['split', 'audio_path', 'error'])
            w.writeheader()
            w.writerows(failures)
        print(f'[WARNING] {len(failures)} audio failures; report {out}', flush=True)
    return np.vstack([v for _, v in kept]), np.array([r['activity'] for r, _ in kept]), [r for r, _ in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-per-class', type=int, default=20,
                    help='20 for smoke test; 0 for all 12,480 train samples')
    ap.add_argument('--val-limit', type=int, default=80,
                    help='80 for smoke test; 0 for all 500 validation samples')
    ap.add_argument('--batch-size', type=int, default=2, help='Lower if out of memory')
    args = ap.parse_args()
    if args.train_per_class < 0 or args.val_limit < 0 or args.batch_size < 1:
        ap.error('train-per-class and val-limit must be >= 0; batch-size must be >= 1')

    # Preserve model cache on D: even if run via normal shell.
    os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))
    os.environ.setdefault('HF_HUB_CACHE', str(ROOT / '.hf_cache' / 'hub'))

    train_rows = choose_training(read_manifest('train'), args.train_per_class)
    val_rows = read_manifest('validation')
    if args.val_limit:
        val_rows = val_rows[:args.val_limit]
    print(f'[SAMPLES] train={len(train_rows)} / validation={len(val_rows)}', flush=True)
    if args.train_per_class and len(train_rows) < len(CATEGORIES) * args.train_per_class:
        print('[WARNING] Some classes have fewer than requested samples', flush=True)

    db = open_cache()
    try:
        X_train, y_train, used_train = get_embeddings(train_rows, 'train', db, args.batch_size)
        X_val, y_val, used_val = get_embeddings(val_rows, 'validation', db, args.batch_size)
    finally:
        db.close()

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC

    results = ROOT / 'results' / 'clap_supervised'
    results.mkdir(parents=True, exist_ok=True)
    models = {
        'clap_svm_rbf': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, class_weight='balanced')),
        'clap_linear_svm': make_pipeline(StandardScaler(), LinearSVC(C=1.0, class_weight='balanced', max_iter=10000)),
    }
    summary = []
    mode = 'FULL' if args.train_per_class == 0 and args.val_limit == 0 else 'SMOKE_TEST'
    for name, model in models.items():
        print(f'[MODEL] Fitting {name} ...', flush=True)
        started = time.monotonic()
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        row = {
            'mode': mode, 'model': name,
            'train_samples': len(y_train), 'validation_samples': len(y_val),
            'accuracy': round(float(accuracy_score(y_val, pred)), 6),
            'balanced_accuracy': round(float(balanced_accuracy_score(y_val, pred)), 6),
            'macro_f1': round(float(f1_score(y_val, pred, labels=CATEGORIES, average='macro', zero_division=0)), 6),
            'train_seconds': round(time.monotonic() - started, 2),
        }
        summary.append(row)
        print('  ' + str(row), flush=True)
        with (results / f'{mode}_{name}_confusion_matrix.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['actual\\predicted'] + list(CATEGORIES))
            for lab, cm_row in zip(CATEGORIES, confusion_matrix(y_val, pred, labels=CATEGORIES)):
                w.writerow([lab] + list(cm_row))
        (results / f'{mode}_{name}_classification_report.txt').write_text(
            classification_report(y_val, pred, labels=CATEGORIES, zero_division=0), encoding='utf-8')
        with (results / f'{mode}_{name}_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['audio_path', 'true_activity', 'predicted_activity', 'correct'])
            for item, truth, prediction in zip(used_val, y_val, pred):
                w.writerow([item['audio_path'], truth, prediction, int(truth == prediction)])

    filename = f'{mode}_metrics.csv'
    with (results / filename).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f'[SAVED] {results / filename}', flush=True)
    if mode != 'FULL':
        print('[CAUTION] SMOKE TEST = tiny, nonrepresentative subsample. Do not compare to 500-record MFCC baseline.', flush=True)
    else:
        print('[CAUTION] Public validation results only; not independent locked test accuracy.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}', flush=True)
        raise
