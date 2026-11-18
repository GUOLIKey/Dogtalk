"""PawTalk STEP 2 - development-only MFCC/CLAP feature fusion.

This script:
- reuses the *immutable* STEP1 dev/holdout split (never touches locked holdout rows);
- reuses the original 30-dimensional MFCC CSV and 512-dimensional CLAP SQLite cache;
- aligns the two feature spaces by the normalized relative audio_path;
- fits all scalers within each CV training fold;
- compares CLAP-only, MFCC-only, and three dimension-normalized fusions;
- selects by development 3-fold out-of-fold Macro-F1, saves diagnostic reports
  and a dev-only refitted joblib model. No independent test is evaluated.
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_dataset import CATEGORIES  # noqa: E402
from step1_split_and_weights import (  # noqa: E402
    source_manifest, lock_split, get_weights, SEED, CV_FOLDS,
)
from clap_supervised import CACHE_DB, FEATURE_VERSION, make_key, open_cache, get_cached  # noqa: E402

OUT = ROOT / 'results' / 'optimization_step2'
MFCC_FILE = ROOT / 'results' / 'train_mfcc_features.csv'
MFCC_DIM = 30
CANDIDATES = {
    'clap_only_reference': None,
    'mfcc_only_reference': 1.0,
    'fusion_mfcc_25pct': 0.25,
    'fusion_mfcc_50pct': 0.50,
    'fusion_mfcc_75pct': 0.75,
}


def normpath(p: str) -> str:
    return (p or '').replace('\\', '/').strip()


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def read_mfcc(dev_rows):
    """Load MFCC vectors ONLY for the locked development rows. No holdout labels used."""
    if not MFCC_FILE.is_file():
        raise FileNotFoundError(
            f'No MFCC CSV: {MFCC_FILE}\n'
            'The original baseline caches MFCC features as CSV (not .npy/.npz). '
            'If missing, run 05_RUN_BASELINE.bat to rebuild the original MFCC cache. '
            'Do not delete the existing CLAP cache or locked split.'
        )
    expected = {normpath(r['audio_path']): r['activity'] for r in dev_rows}
    found = {}
    colnames = [f'f{i:02d}' for i in range(MFCC_DIM)]
    total = 0
    with MFCC_FILE.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        needed = set(['audio_path', 'activity'] + colnames)
        if not needed.issubset(fields):
            raise RuntimeError(f'MFCC cache columns missing: {sorted(needed - fields)}')
        for r in reader:
            total += 1
            key = normpath(r['audio_path'])
            if key not in expected:  # ignore locked holdout/cache rows
                continue
            if key in found:
                raise RuntimeError(f'Duplicate MFCC cache record for development audio: {key}')
            if r['activity'] != expected[key]:
                raise RuntimeError(f'MFCC activity label mismatch for {key}: {r["activity"]} != {expected[key]}')
            try:
                vec = np.asarray([float(r[k]) for k in colnames], dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f'Invalid MFCC numeric data for {key}') from exc
            if vec.size != MFCC_DIM or not np.isfinite(vec).all():
                raise RuntimeError(f'Invalid MFCC vector for {key}')
            found[key] = vec
    missing = [r['audio_path'] for r in dev_rows if normpath(r['audio_path']) not in found]
    if missing:
        raise RuntimeError(
            f'MFCC cache is missing {len(missing)} development rows, examples: {missing[:3]}. '
            'Re-run the original 05_RUN_BASELINE.bat to fill the MFCC CSV, '
            'then rerun STEP2. Existing CLAP embeddings remain unchanged.'
        )
    X = np.stack([found[normpath(r['audio_path'])] for r in dev_rows])
    print(f'[MFCC] Reused {len(X)} dev vectors; shape={X.shape}; scanned CSV records={total}', flush=True)
    return X


def read_clap(dev_rows):
    if not CACHE_DB.is_file():
        raise FileNotFoundError(f'CLAP cache not found: {CACHE_DB}')
    db = open_cache()
    X = []
    missing = []
    try:
        for r in dev_rows:
            path = normpath(r['audio_path'])
            vec = get_cached(db, make_key(path))
            if vec is None:
                missing.append(path)
            else:
                X.append(vec)
    finally:
        db.close()
    if missing:
        raise RuntimeError(f'Missing CLAP cache for {len(missing)} dev recordings, examples: {missing[:3]}')
    dims = {len(v) for v in X}
    if len(dims) != 1:
        raise RuntimeError(f'Inconsistent CLAP dimensions: {dims}')
    result = np.stack(X).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError('CLAP vectors contain NaN/Inf')
    print(f'[CLAP] Reused {len(result)} dev vectors; shape={result.shape}; no CLAP inference', flush=True)
    return result


def load_dev():
    # lock_split validates fingerprint, reuses all three immutable files,
    # checks dev/holdout disjointness, and returns DEVELOPMENT rows only.
    rows = source_manifest()
    dev_rows, holdout_n = lock_split(rows)
    if holdout_n <= 0:
        raise RuntimeError('Locked holdout split is missing')
    X_mfcc = read_mfcc(dev_rows)
    X_clap = read_clap(dev_rows)
    y = np.asarray([r['activity'] for r in dev_rows])
    if not (len(y) == len(X_mfcc) == len(X_clap)):
        raise RuntimeError('MFCC/CLAP row counts do not agree')
    return dev_rows, y, X_clap, X_mfcc, holdout_n


def make_model(candidate_name, y_fit, clap_dim, mfcc_dim):
    weights = get_weights(y_fit, 1.0, 1.0)  # STEP1 best policy: balanced
    svm = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight=weights, cache_size=600)
    if candidate_name == 'clap_only_reference':
        return make_pipeline(StandardScaler(), svm)
    if candidate_name == 'mfcc_only_reference':
        return make_pipeline(StandardScaler(), svm)
    frac = CANDIDATES[candidate_name]
    # Divide by sqrt(feature count) so one 512-D group does not automatically
    # dominate the 30-D group merely due to dimensionality.
    # All standardizers are FIT ON CV-TRAIN ONLY, no cross-fold leakage.
    ct = ColumnTransformer(
        transformers=[
            ('clap', StandardScaler(), slice(0, clap_dim)),
            ('mfcc', StandardScaler(), slice(clap_dim, clap_dim + mfcc_dim)),
        ],
        transformer_weights={
            'clap': math.sqrt(1.0 - frac) / math.sqrt(clap_dim),
            'mfcc': math.sqrt(frac) / math.sqrt(mfcc_dim),
        },
        sparse_threshold=0,
    )
    return make_pipeline(ct, svm)


def features_for(name, X_clap, X_mfcc, X_concat):
    if name == 'clap_only_reference':
        return X_clap
    if name == 'mfcc_only_reference':
        return X_mfcc
    return X_concat


def score(y, pred):
    per_f1 = f1_score(y, pred, labels=CATEGORIES, average=None, zero_division=0)
    per_p = precision_score(y, pred, labels=CATEGORIES, average=None, zero_division=0)
    per_r = recall_score(y, pred, labels=CATEGORIES, average=None, zero_division=0)
    by_label = {c: (float(f), float(p), float(r)) for c, f, p, r in zip(CATEGORIES, per_f1, per_p, per_r)}
    return {
        'accuracy': round(float(accuracy_score(y, pred)), 6),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, pred)), 6),
        'macro_f1': round(float(f1_score(y, pred, labels=CATEGORIES, average='macro', zero_division=0)), 6),
        'attention_f1': round(by_label['seeking attention'][0], 6),
        'attention_precision': round(by_label['seeking attention'][1], 6),
        'attention_recall': round(by_label['seeking attention'][2], 6),
        'food_f1': round(by_label['begging for food'][0], 6),
        'food_precision': round(by_label['begging for food'][1], 6),
        'food_recall': round(by_label['begging for food'][2], 6),
    }


def save_diagnostics(name, rows, y, pred):
    diag = OUT / 'diagnostics'
    diag.mkdir(parents=True, exist_ok=True)
    write_csv(diag / f'{name}_oof_predictions.csv', [
        {'audio_path': r['audio_path'], 'actual': act, 'predicted': pre, 'correct': int(act == pre)}
        for r, act, pre in zip(rows, y, pred)
    ], ['audio_path', 'actual', 'predicted', 'correct'])
    cm = confusion_matrix(y, pred, labels=CATEGORIES)
    write_csv(diag / f'{name}_confusion_matrix.csv', [
        {'actual\\predicted': c, **{name_: int(n) for name_, n in zip(CATEGORIES, cm_row)}}
        for c, cm_row in zip(CATEGORIES, cm)
    ], ['actual\\predicted'] + list(CATEGORIES))
    (diag / f'{name}_classification_report.txt').write_text(
        classification_report(y, pred, labels=CATEGORIES, zero_division=0), encoding='utf-8')


def run_cv(dev_rows, y, X_clap, X_mfcc):
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    X_concat = np.concatenate([X_clap, X_mfcc], axis=1)
    summary = []
    for name in CANDIDATES:
        X = features_for(name, X_clap, X_mfcc, X_concat)
        pred = np.empty(len(y), dtype=object)
        start = time.monotonic()
        print(f'[CV] {name}: {CV_FOLDS} folds...', flush=True)
        for fold_i, (idx_train, idx_valid) in enumerate(folds.split(X, y), 1):
            model = make_model(name, y[idx_train], X_clap.shape[1], X_mfcc.shape[1])
            model.fit(X[idx_train], y[idx_train])
            pred[idx_valid] = model.predict(X[idx_valid])
            fold_f1 = f1_score(y[idx_valid], pred[idx_valid], labels=CATEGORIES, average='macro', zero_division=0)
            print(f'    fold {fold_i}/{CV_FOLDS}: Macro-F1={fold_f1:.4f}', flush=True)
        metrics = score(y, pred)
        metrics.update({
            'model': name, 'n_dev': len(y), 'folds': CV_FOLDS,
            'mfcc_weight_fraction': '' if CANDIDATES[name] is None else CANDIDATES[name],
            'duration_sec': round(time.monotonic() - start, 1),
        })
        save_diagnostics(name, dev_rows, y, pred)
        summary.append(metrics)
        print('    OOF Macro-F1={:.4f}; Accuracy={:.4f}; attention F1={:.4f}; food F1={:.4f}'.format(
            metrics['macro_f1'], metrics['accuracy'], metrics['attention_f1'], metrics['food_f1']), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Only verify cache alignment; do not train')
    args = parser.parse_args()
    print('[STEP2] Feature fusion on development data only. Locked holdout NEVER evaluated.', flush=True)
    rows, y, X_clap, X_mfcc, holdout_n = load_dev()
    print(f'[CHECK] Feature alignment passed: dev={len(rows)}, CLAP={X_clap.shape}, MFCC={X_mfcc.shape}; locked holdout={holdout_n}', flush=True)
    if args.check:
        print('[PASS] STEP2 feature cache check finished. No model trained.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    cv_results = run_cv(rows, y, X_clap, X_mfcc)
    write_csv(OUT / 'step2_cv_metrics.csv', cv_results, list(cv_results[0]))
    best = max(cv_results, key=lambda r: (r['macro_f1'], (r['attention_f1'] + r['food_f1']) / 2))
    best_name = best['model']
    x_full = features_for(best_name, X_clap, X_mfcc, np.concatenate([X_clap, X_mfcc], axis=1))
    model = make_model(best_name, y, X_clap.shape[1], X_mfcc.shape[1])
    model.fit(x_full, y)
    joblib.dump(model, OUT / 'selected_STEP2_MODEL_DEV_ONLY.joblib')
    manifest = {
        'selected_by': 'development 3-fold OOF Macro-F1, target-F1 exact tie-break',
        'selected_model': best_name,
        'locked_holdout_evaluated': False,
        'locked_holdout_count': holdout_n,
        'dev_count': len(y),
        'seed': SEED, 'folds': CV_FOLDS,
        'CLAP_feature_version': FEATURE_VERSION,
        'input_feature_order_for_selected': ('clap_512_and_mfcc_30' if 'fusion_' in best_name else
                                             'clap_512_only' if best_name.startswith('clap_') else 'mfcc_30_only'),
        'model_grid': cv_results,
    }
    (OUT / 'step2_selection_summary.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    ref = next(r for r in cv_results if r['model'] == 'clap_only_reference')
    lines = [
        '# PawTalk STEP 2: MFCC + CLAP 特征融合（开发集内部交叉验证）', '',
        f'- Dev rows: {len(y)}. Locked holdout: {holdout_n} rows (not read for training/evaluation).',
        '- Same train/dev/holdout split and three StratifiedKFold folds as STEP1.',
        '- CLAP first 10 seconds at 48 kHz; MFCC original full recording at 16 kHz. Different audio windows remain a limitation.',
        '- MFCC and CLAP matched by relative audio_path and verified against label for dev records.',
        '- Per-fold train-only scaling, SVM C=1, RBF gamma=scale, balanced class weights from fold train.',
        '- Fusion weights refer to relative squared feature-block contribution AFTER per-block standardization and dimensional normalization.',
        '- Existing train MFCC/CLAP caches reused; no full-model fine-tuning.',
        '- No performance claims from locked holdout or original public validation split.', '',
        '| Candidate | Accuracy | Macro-F1 | Seeking Attention F1 | Begging for Food F1 |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in cv_results:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['attention_f1']:.4f} | {r['food_f1']:.4f} |")
    lines += [
        '', f"**Selected: `{best_name}`**, DEV OOF Macro-F1 = {best['macro_f1']:.4f}.",
        f"Difference vs same-dev CLAP reference = {best['macro_f1'] - ref['macro_f1']:+.4f} (NOT locked-test improvement).",
        '', 'If fusion fails to outperform CLAP-only, keep it as a negative result and do not force the method.',
        'Before final testing, resolve the dog/video-group leakage limitation where possible.',
    ]
    (OUT / 'STEP2_结果摘要.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[SELECT] Best on dev CV: {best_name} (Macro-F1={best["macro_f1"]:.4f})', flush=True)
    print(f'[SAVED] {OUT / "step2_cv_metrics.csv"}', flush=True)
    print(f'[SAVED] {OUT / "STEP2_结果摘要.md"}', flush=True)
    print('[CAUTION] Locked holdout untouched. No final independent test score claimed.', flush=True)
    print('[PASS] STEP2 finished.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(f'[FAILED] {type(err).__name__}: {err}', flush=True)
        raise
