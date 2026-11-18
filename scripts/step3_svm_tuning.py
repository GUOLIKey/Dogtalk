"""PawTalk STEP3: development-only SVM C/gamma tuning on STEP2's best fusion.

- Reuses STEP1 locked 9984-row development set and identical StratifiedKFold.
- Reuses STEP2's strictly audio-aligned MFCC (30D) + CLAP (512D) features.
- Uses STEP2's fixed fusion_mfcc_50pct preprocessing inside EACH train fold.
- Reproduces STEP2 reference (C=1, gamma='scale') as first model.
- Searches a modest, predefined SVM C/gamma grid; balanced fold weights only.
- Scores OOF predictions, then refits the selected model on development only.
- NEVER trains or predicts using locked holdout features/labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_dataset import CATEGORIES  # noqa: E402
from step1_split_and_weights import CV_FOLDS, SEED  # noqa: E402
from step2_feature_fusion import (  # noqa: E402
    load_dev, make_model, score, write_csv,
)

OUT = ROOT / 'results' / 'optimization_step3'
STEP2_CV_METRICS = ROOT / 'results' / 'optimization_step2' / 'step2_cv_metrics.csv'
FUSION_NAME = 'fusion_mfcc_50pct'
REFERENCE = 'c1_gscale_reference'
# 12 predefined candidates = 36 sequential SVM fits. Gamma values are absolute
# *after* STEP2's dimension-normalized preprocessing; gamma='scale' reproduces
# the actual STEP2 reference exactly. For the fusion, scale is usually near 1.
GRID = [
    {'model': REFERENCE, 'C': 1.0, 'gamma': 'scale'},
    {'model': 'c0p3_gscale', 'C': 0.3, 'gamma': 'scale'},
    {'model': 'c3_gscale', 'C': 3.0, 'gamma': 'scale'},
    {'model': 'c10_gscale', 'C': 10.0, 'gamma': 'scale'},
    {'model': 'c0p3_g0p5', 'C': 0.3, 'gamma': 0.5},
    {'model': 'c1_g0p5', 'C': 1.0, 'gamma': 0.5},
    {'model': 'c3_g0p5', 'C': 3.0, 'gamma': 0.5},
    {'model': 'c10_g0p5', 'C': 10.0, 'gamma': 0.5},
    {'model': 'c0p3_g2', 'C': 0.3, 'gamma': 2.0},
    {'model': 'c1_g2', 'C': 1.0, 'gamma': 2.0},
    {'model': 'c3_g2', 'C': 3.0, 'gamma': 2.0},
    {'model': 'c10_g2', 'C': 10.0, 'gamma': 2.0},
]


def checked_step2_reference():
    """Read only STEP2's CV summary to verify experiment comparability."""
    if not STEP2_CV_METRICS.exists():
        raise FileNotFoundError(
            f'STEP2 metrics file missing: {STEP2_CV_METRICS}. '
            'Please preserve your completed STEP2 results before STEP3.'
        )
    with STEP2_CV_METRICS.open('r', encoding='utf-8-sig', newline='') as fh:
        entries = list(csv.DictReader(fh))
    ref = [r for r in entries if r.get('model') == FUSION_NAME]
    if len(ref) != 1:
        raise RuntimeError('Could not uniquely find fusion_mfcc_50pct in STEP2 metrics.')
    return float(ref[0]['macro_f1'])


def build_model(y_fit, clap_dim, mfcc_dim, C, gamma):
    """Reuse the exact STEP2 fused pipeline; change ONLY SVM C and gamma."""
    model = make_model(FUSION_NAME, y_fit, clap_dim, mfcc_dim)
    model.named_steps['svc'].set_params(C=C, gamma=gamma)
    return model


def evaluate_one(candidate, X, y, clap_dim, mfcc_dim):
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), dtype=object)
    t0 = time.monotonic()
    print(f'[CV] {candidate["model"]}: C={candidate["C"]}, gamma={candidate["gamma"]}', flush=True)
    for fold_i, (idx_train, idx_valid) in enumerate(folds.split(X, y), 1):
        model = build_model(y[idx_train], clap_dim, mfcc_dim, candidate['C'], candidate['gamma'])
        model.fit(X[idx_train], y[idx_train])
        pred[idx_valid] = model.predict(X[idx_valid])
        f = f1_score(y[idx_valid], pred[idx_valid], labels=CATEGORIES,
                     average='macro', zero_division=0)
        print(f'    fold {fold_i}/{CV_FOLDS}: Macro-F1={f:.4f}', flush=True)
    result = score(y, pred)
    result.update({
        'model': candidate['model'], 'C': candidate['C'],
        'gamma': candidate['gamma'], 'n_dev': len(y),
        'folds': CV_FOLDS, 'mfcc_feature_fraction': 0.5,
        'duration_sec': round(time.monotonic() - t0, 1),
    })
    print('    OOF Macro-F1={:.4f}; Accuracy={:.4f}; attention F1={:.4f}; food F1={:.4f}'.format(
        result['macro_f1'], result['accuracy'], result['attention_f1'], result['food_f1']), flush=True)
    return result, pred


def save_predictions(name, rows, y, pred):
    folder = OUT / 'diagnostics'
    folder.mkdir(parents=True, exist_ok=True)
    write_csv(folder / f'{name}_oof_predictions.csv', [
        {'audio_path': r['audio_path'], 'actual': truth, 'predicted': guess, 'correct': int(truth == guess)}
        for r, truth, guess in zip(rows, y, pred)
    ], ['audio_path', 'actual', 'predicted', 'correct'])
    cm = confusion_matrix(y, pred, labels=CATEGORIES)
    write_csv(folder / f'{name}_confusion_matrix.csv', [
        {'actual\\predicted': category, **{c: int(n) for c, n in zip(CATEGORIES, row)}}
        for category, row in zip(CATEGORIES, cm)
    ], ['actual\\predicted'] + list(CATEGORIES))
    (folder / f'{name}_classification_report.txt').write_text(
        classification_report(y, pred, labels=CATEGORIES, zero_division=0), encoding='utf-8'
    )


def main():
    parser = argparse.ArgumentParser(description='PawTalk STEP3 DEV-only SVM optimization')
    parser.add_argument('--check', action='store_true', help='Check environment/data; do not train')
    args = parser.parse_args()
    print('[STEP3] C/gamma tuning on locked DEV only. Locked holdout NEVER evaluated.', flush=True)
    previous_ref = checked_step2_reference()
    print(f'[BASELINE] STEP2 fused 50pct OOF Macro-F1={previous_ref:.4f}', flush=True)
    rows, y, clap, mfcc, holdout_n = load_dev()
    if clap.shape[1] != 512 or mfcc.shape[1] != 30 or len(rows) != 9984 or holdout_n != 2496:
        raise RuntimeError('Feature size or locked split differs from completed STEP2; stop to audit before tuning.')
    if len(set(row['audio_path'] for row in rows)) != len(rows):
        raise RuntimeError('Duplicate development audio paths: refusing to train.')
    X = np.concatenate([clap, mfcc], axis=1)
    print(f'[CHECK] Aligned dev features={X.shape}; holdout={holdout_n}; candidates={len(GRID)};',
          f'expected SVM fits={len(GRID) * CV_FOLDS}', flush=True)
    if args.check:
        print('[PASS] STEP3 preflight completed. No model trained.', flush=True)
        return
    # Refuse to overwrite prior experiment: preserves baseline for later audit.
    if any((OUT / fn).exists() for fn in ('step3_cv_metrics.csv', 'STEP3_结果摘要.md')):
        raise RuntimeError(f'STEP3 output already exists in {OUT}; back it up first if you intentionally wish to rerun.')
    OUT.mkdir(parents=True, exist_ok=True)

    # First verify the identical reference model, fold seed and audio alignment.
    reference_result, reference_pred = evaluate_one(GRID[0], X, y, clap.shape[1], mfcc.shape[1])
    delta = reference_result['macro_f1'] - previous_ref
    print(f'[VERIFY] STEP3 reference vs STEP2: {delta:+.6f}', flush=True)
    if abs(delta) > 0.0001:
        raise RuntimeError(
            'STEP3 baseline does NOT reproduce STEP2 within 0.0001 Macro-F1. '
            'Check Python environment, cache, split, or code versions; refusing parameter search.'
        )
    all_results = [reference_result]
    predictions = {REFERENCE: reference_pred}
    for candidate in GRID[1:]:
        row, pred = evaluate_one(candidate, X, y, clap.shape[1], mfcc.shape[1])
        all_results.append(row)
        predictions[candidate['model']] = pred
    # Metric selection occurs only after all predetermined candidates are scored.
    best = max(all_results, key=lambda r: (r['macro_f1'], (r['attention_f1'] + r['food_f1']) / 2))
    chosen = next(c for c in GRID if c['model'] == best['model'])
    reference = all_results[0]
    model = build_model(y, clap.shape[1], mfcc.shape[1], chosen['C'], chosen['gamma'])
    model.fit(X, y)  # DEV only; holdout excluded.
    joblib.dump(model, OUT / 'selected_STEP3_MODEL_DEV_ONLY.joblib')
    write_csv(OUT / 'step3_cv_metrics.csv', all_results, list(all_results[0].keys()))
    save_predictions(REFERENCE, rows, y, predictions[REFERENCE])
    if best['model'] != REFERENCE:
        save_predictions(best['model'], rows, y, predictions[best['model']])

    record = {
        'selection_method': 'DEV 3-fold pooled out-of-fold Macro-F1 (target F1 mean as tie-break)',
        'selected': chosen, 'selected_cv_metrics': best,
        'step2_reference_macro_f1': previous_ref,
        'step3_reproduced_reference_macro_f1': reference['macro_f1'],
        'development_count': len(y), 'holdout_count': holdout_n,
        'holdout_predictions_or_scores_computed': False,
        'folds': CV_FOLDS, 'seed': SEED, 'weight_policy': 'fold-balanced from STEP1',
        'feature_fusion': 'STEP2 fusion_mfcc_50pct; standardizers refitted inside every CV fit',
        'fixed_candidates': GRID,
        'known_limitations': [
            'Original audio labels lack reliable dog/video identity for grouped splits',
            'Parameter search over the same DEV folds incurs selection optimism; final holdout should be used once',
            'CLAP uses first 10 sec; baseline MFCC uses full recording',
            'This model needs exactly the same CLAP and MFCC extraction pipelines at inference',
        ],
    }
    (OUT / 'step3_selection_summary.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    lines = [
        '# PawTalk STEP 3: SVM C / gamma 优化（仅开发集）', '',
        f'- Dev: {len(y)} 条；锁定 Holdout: {holdout_n} 条（未进行模型训练/预测/评分）。',
        f'- 固定 STEP2 最佳方案：MFCC 50% + CLAP 50%，总输入维度 {X.shape[1]}。',
        f'- 相同的 StratifiedKFold 三折与种子 {SEED}；所有标准化均在各折训练集拟合。',
        f'- STEP2 参考 OOF Macro-F1: {previous_ref:.4f}；STEP3 复现: {reference["macro_f1"]:.4f}。',
        '- 不再测试 STEP1 中已发现不理想的困难类定向权重提升；固定 fold-balanced 权重。',
        '- gamma=scale 精确复现参考；其余 gamma 为预处理后特征空间中的绝对数值。',
        '- 最优结果来自有限的开发集搜索，不代表独立测试提升。', '',
        '| Candidate | C | gamma | Accuracy | Macro-F1 | Seeking Attention F1 | Begging for Food F1 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in all_results:
        lines.append(f'| {r["model"]} | {r["C"]} | {r["gamma"]} | '
                     f'{r["accuracy"]:.4f} | {r["macro_f1"]:.4f} | '
                     f'{r["attention_f1"]:.4f} | {r["food_f1"]:.4f} |')
    lines.extend([
        '', f'**Best DEV CV: `{best["model"]}`**，Macro-F1={best["macro_f1"]:.4f}。',
        f'相对 STEP2 固定融合参考：Macro-F1 {best["macro_f1"]-previous_ref:+.4f}（非最终测试增益）。',
        '', '**下一步：**仅在开发集上进行 STEP4 预处理与音频片段对照；不要提前动用锁定测试集。',
        '', '特别提示：无可靠狗只/视频分组 ID，当前随机分层切分不能排除个体或视频关联泄漏。',
    ])
    (OUT / 'STEP3_结果摘要.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'[SELECT] Best dev CV: {best["model"]}; Macro-F1={best["macro_f1"]:.4f}', flush=True)
    print(f'[SAVED] {OUT / "step3_cv_metrics.csv"}', flush=True)
    print(f'[SAVED] {OUT / "STEP3_结果摘要.md"}', flush=True)
    print('[CAUTION] Locked holdout untouched. No final independent test score claimed.', flush=True)
    print('[PASS] STEP3 finished.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}', flush=True)
        raise
