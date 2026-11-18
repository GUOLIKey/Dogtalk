"""MFCC + Majority/SVM/RandomForest baselines for Barkopedia activity labels.
No test-set peeking. This evaluates the published validation split only.
"""
import argparse
import csv
from pathlib import Path
import sys
import time

import librosa
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audit_dataset import ROOT, CATEGORIES

FEATURE_SIZE = 26 + 4
FIELDS = ['audio_path', 'activity'] + [f'f{i:02d}' for i in range(FEATURE_SIZE)]


def vector(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size < 400:
        raise ValueError('empty / too short recording')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                           [rms.mean(), rms.std(), zcr.mean(), zcr.std()]]).astype('float32')


def load_manifest(split):
    manifest = ROOT / 'results' / f'{split}_manifest.csv'
    if not manifest.exists():
        raise FileNotFoundError(f'{manifest} not found. Run 02_AUDIT_DATA.bat first.')
    with manifest.open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if r['activity'] in CATEGORIES and r['audio_match'] == 'MATCH']
    if not selected:
        raise RuntimeError(f'No matched {split} audio. Download and audit it first.')
    return selected


def features(split):
    rows = load_manifest(split)
    cache_path = ROOT / 'results' / f'{split}_mfcc_features.csv'
    known = {}
    if cache_path.exists():
        with cache_path.open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if all(f'f{i:02d}' in r for i in range(FEATURE_SIZE)):
                    known[r['audio_path']] = r
    print(f'[FEATURE] {split}: {len(rows)} recordings; {len(known)} cached')
    X, Y = [], []
    cache_path.parent.mkdir(exist_ok=True)
    needs_header = not cache_path.exists() or cache_path.stat().st_size == 0
    with cache_path.open('a', encoding='utf-8-sig' if needs_header else 'utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header: wr.writeheader()
        for ix, row in enumerate(rows, 1):
            rp = row['audio_path']
            if rp in known and known[rp].get('activity') == row['activity']:
                try:
                    x = np.array([float(known[rp][f'f{i:02d}']) for i in range(FEATURE_SIZE)], dtype='float32')
                except (ValueError, KeyError):
                    x = None
            else:
                x = None
            if x is None:
                try:
                    x = vector(ROOT / rp)
                    wr.writerow({'audio_path': rp, 'activity': row['activity'],
                                 **{f'f{i:02d}': float(x[i]) for i in range(FEATURE_SIZE)}})
                    f.flush()  # interrupted runs preserve progress
                except Exception as exc:
                    print(f'[SKIP AUDIO] {rp}: {exc}')
                    continue
            X.append(x)
            Y.append(row['activity'])
            if ix % 200 == 0: print(f'    processed {ix}/{len(rows)}', flush=True)
    if not X: raise RuntimeError(f'No readable {split} audio')
    return np.stack(X), np.asarray(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-limit', type=int, default=0, help='Optional small smoke test')
    args = ap.parse_args()
    X_train, y_train = features('train')
    X_val, y_val = features('validation')
    if args.train_limit: X_train, y_train = X_train[:args.train_limit], y_train[:args.train_limit]
    print(f'[TRAIN] {X_train.shape} / [VALIDATION] {X_val.shape}')
    results = ROOT / 'results' / 'baseline'
    results.mkdir(parents=True, exist_ok=True)
    models = {
        'majority_class': DummyClassifier(strategy='most_frequent'),
        'mfcc_svm_rbf': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', class_weight='balanced')),
        'mfcc_random_forest': RandomForestClassifier(n_estimators=250, random_state=42,
                                                      class_weight='balanced_subsample', n_jobs=-1),
    }
    summary = []
    for name, model in models.items():
        print(f'[MODEL] Training {name}...', flush=True)
        started = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        row = {'model': name,
               'accuracy': round(float(accuracy_score(y_val, pred)), 6),
               'balanced_accuracy': round(float(balanced_accuracy_score(y_val, pred)), 6),
               'macro_f1': round(float(f1_score(y_val, pred, labels=CATEGORIES, average='macro', zero_division=0)), 6),
               'train_seconds': round(time.time() - started, 2),
               'train_samples': len(y_train), 'validation_samples': len(y_val)}
        summary.append(row)
        print(f'  VALIDATION: Accuracy={row["accuracy"]}, BalancedAcc={row["balanced_accuracy"]}, Macro-F1={row["macro_f1"]}')
        cm = confusion_matrix(y_val, pred, labels=CATEGORIES)
        with (results / f'{name}_confusion_matrix.csv').open('w', encoding='utf-8-sig', newline='') as f:
            wr = csv.writer(f); wr.writerow(['actual\\predicted'] + list(CATEGORIES))
            for label, r in zip(CATEGORIES, cm): wr.writerow([label] + list(r))
        (results / f'{name}_classification_report.txt').write_text(
            classification_report(y_val, pred, labels=list(CATEGORIES), zero_division=0), encoding='utf-8')
    with (results/'validation_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader(); wr.writerows(summary)
    (results/'READ_ME_FIRST.txt').write_text(
      'These scores use the public validation set for comparison/model selection. '
      'They are NOT locked, independent final test metrics. '
      'Do not tune on and then call the same validation scores unbiased final accuracy.\n', encoding='utf-8')
    print(f'[SAVED] {results / "validation_metrics.csv"}')
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as e:
        print(f'[FAILED] {type(e).__name__}: {e}')
        sys.exit(2)
