"""PawTalk STEP 1: immutable stratified holdout + CLAP SVM class-weight search.

- Uses ONLY existing train_manifest.csv and CLAP SQLite feature cache.
- Does NOT compute, train on, or evaluate locked holdout embeddings.
- Selects weight policy with out-of-fold predictions on development-train data.
- Outputs predictions, confusion matrices, and a dev-only refit model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_dataset import CATEGORIES  # noqa: E402
from clap_supervised import CACHE_DB, FEATURE_VERSION, make_key, open_cache, get_cached  # noqa: E402

OUT = ROOT / "results" / "optimization_step1"
SPLIT_DIR = OUT / "splits"
SEED = 20261112
TEST_FRACTION = 0.20
CV_FOLDS = 3
TARGETS = ("seeking attention", "begging for food")
POLICIES = {
    "clap_balanced_reference": (1.0, 1.0),
    "clap_focus_mild": (1.25, 1.50),
    "clap_focus_strong": (1.50, 2.00),
}
FIELDS = ("audio_path", "audio_id", "activity", "environment", "split")


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_manifest():
    path = ROOT / "results" / "train_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. First run 02_AUDIT_DATA.bat")
    rows = read_csv(path)
    eligible = []
    for row in rows:
        if row.get("audio_match") != "MATCH":
            raise RuntimeError(f"Training manifest contains unmatched audio: {row.get('audio_path')}")
        if row.get("activity") not in CATEGORIES:
            raise RuntimeError(f"Unexpected activity label {row.get('activity')!r}")
        relative_path = row.get("audio_path", "").replace("\\", "/").strip()
        if not relative_path or not (ROOT / relative_path).is_file():
            raise RuntimeError(f"Audio file missing on D: {relative_path}")
        eligible.append({
            "audio_path": relative_path,
            "audio_id": row.get("audio_id", ""),
            "activity": row["activity"],
            "environment": row.get("environment", ""),
            "split": "train",
        })
    paths = [r["audio_path"].casefold() for r in eligible]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate audio_path in train_manifest.csv; do not split until reviewed")
    if len(eligible) < 100:
        raise RuntimeError(f"Expected full dataset; found only {len(eligible)} training rows")
    counts = Counter(r["activity"] for r in eligible)
    if len(counts) != len(CATEGORIES) or min(counts.values()) < 15:
        raise RuntimeError(f"Cannot do stratified 3-fold evaluation with counts {dict(counts)}")
    return eligible


def fingerprint(rows):
    canonical = "\n".join(sorted(f"{r['audio_path']}\t{r['activity']}" for r in rows))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def lock_split(rows):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_file = SPLIT_DIR / "dev_train.csv"
    holdout_file = SPLIT_DIR / "LOCKED_holdout_DO_NOT_TUNE.csv"
    protocol_file = SPLIT_DIR / "split_protocol.json"
    files_exist = [p.exists() for p in (train_file, holdout_file, protocol_file)]
    if any(files_exist) and not all(files_exist):
        raise RuntimeError("Only some split files exist. Refusing to overwrite; restore all three or inspect manually.")
    digest = fingerprint(rows)
    if all(files_exist):
        proto = json.loads(protocol_file.read_text(encoding="utf-8"))
        if proto.get("source_fingerprint_sha256") != digest:
            raise RuntimeError("Source dataset changed after split was locked! Refusing to proceed.")
        dev_rows, hold_rows = read_csv(train_file), read_csv(holdout_file)
        print("[LOCK] Reusing the original split; NEVER reshuffling it.", flush=True)
    else:
        # This is a row-stratified split, NOT dog/video-group split (IDs unavailable).
        indexes = np.arange(len(rows))
        train_idx, hold_idx = train_test_split(
            indexes, test_size=TEST_FRACTION, random_state=SEED,
            stratify=[r["activity"] for r in rows],
        )
        dev_rows = [rows[int(i)] for i in train_idx]
        hold_rows = [rows[int(i)] for i in hold_idx]
        write_csv(train_file, dev_rows, FIELDS)
        write_csv(holdout_file, hold_rows, FIELDS)
        protocol = {
            "seed": SEED,
            "test_fraction": TEST_FRACTION,
            "source_fingerprint_sha256": digest,
            "source_rows": len(rows),
            "development_train_rows": len(dev_rows),
            "locked_holdout_rows": len(hold_rows),
            "purpose": "CV and model selection use development set ONLY; do not evaluate holdout before final model is frozen",
            "known_limitations": [
                "No reliable dog/video identity for strict grouped split",
                "Prior exploratory full-training experiments saw all Barkopedia train samples; fresh models must be retrained using dev only",
                "Official 500-record validation split was previously used during exploration and is not untouched final test",
            ],
            "feature_version": FEATURE_VERSION,
        }
        protocol_file.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[LOCK] Fixed stratified development/holdout split written. Keep it unchanged.", flush=True)
    dev_set = {r["audio_path"] for r in dev_rows}
    hold_set = {r["audio_path"] for r in hold_rows}
    orig_set = {r["audio_path"] for r in rows}
    if dev_set & hold_set or (dev_set | hold_set) != orig_set:
        raise RuntimeError("Locked split consistency failure")
    for group, subset in (("dev", dev_rows), ("LOCKED holdout", hold_rows)):
        print(f"[SPLIT] {group}: {len(subset)}; per-class={dict(Counter(r['activity'] for r in subset))}", flush=True)
    return dev_rows, len(hold_rows)


def load_dev_clap(rows):
    if not CACHE_DB.exists():
        raise FileNotFoundError(f"No CLAP SQLite feature cache: {CACHE_DB}. Run 10_CLAP_SUPERVISED_FULL.bat first.")
    print(f"[CACHE] Reading existing CLAP embeddings from: {CACHE_DB}", flush=True)
    db = open_cache()
    X, missing, dimensions = [], [], set()
    try:
        for r in rows:
            path = r["audio_path"]
            vec = get_cached(db, make_key(path))
            if vec is None:
                missing.append(path)
            else:
                X.append(vec)
                dimensions.add(int(vec.size))
    finally:
        db.close()
    if missing:
        examples = "\n".join(missing[:5])
        raise RuntimeError(f"Missing {len(missing)} cached CLAP embeddings. Examples:\n{examples}\n"
                           "If audio files were moved/modified, re-run 10_CLAP_SUPERVISED_FULL.bat to refill cache.")
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent CLAP embedding dims: {dimensions}")
    if not X:
        raise RuntimeError("No CLAP embeddings in cache")
    arr = np.stack(X).astype(np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError("Nonfinite CLAP embedding values")
    print(f"[CACHE] Reused {arr.shape[0]} vectors; shape={arr.shape}; NO CLAP audio inference needed", flush=True)
    return arr


def get_weights(labels, attention_mult, food_mult):
    # Compute inverse-frequency weights from THIS FIT'S training fold only.
    counts = Counter(labels)
    n, k = len(labels), len(counts)
    weights = {name: n / (k * count) for name, count in counts.items()}
    weights["seeking attention"] *= attention_mult
    weights["begging for food"] *= food_mult
    return weights


def build_model(y_fit, mults):
    weights = get_weights(y_fit, mults[0], mults[1])
    return make_pipeline(
        StandardScaler(),
        SVC(C=1.0, kernel="rbf", gamma="scale", class_weight=weights, cache_size=600),
    )


def evaluate(y, predictions):
    per_f1 = f1_score(y, predictions, average=None, labels=CATEGORIES, zero_division=0)
    per_recall = recall_score(y, predictions, average=None, labels=CATEGORIES, zero_division=0)
    per_precision = precision_score(y, predictions, average=None, labels=CATEGORIES, zero_division=0)
    lookup = {lab: (float(f), float(r), float(p)) for lab, f, r, p in zip(CATEGORIES, per_f1, per_recall, per_precision)}
    result = {
        "accuracy": round(float(accuracy_score(y, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 6),
        "macro_f1": round(float(f1_score(y, predictions, average="macro", labels=CATEGORIES, zero_division=0)), 6),
        "attention_f1": round(lookup["seeking attention"][0], 6),
        "attention_recall": round(lookup["seeking attention"][1], 6),
        "attention_precision": round(lookup["seeking attention"][2], 6),
        "food_f1": round(lookup["begging for food"][0], 6),
        "food_recall": round(lookup["begging for food"][1], 6),
        "food_precision": round(lookup["begging for food"][2], 6),
    }
    return result


def output_diagnostics(name, rows, y, predictions):
    subdir = OUT / "diagnostics"
    subdir.mkdir(parents=True, exist_ok=True)
    write_csv(subdir / f"{name}_oof_predictions.csv", [
        {"audio_path": r["audio_path"], "true_activity": t, "predicted_activity": p, "correct": int(t == p)}
        for r, t, p in zip(rows, y, predictions)
    ], ("audio_path", "true_activity", "predicted_activity", "correct"))
    matrix = confusion_matrix(y, predictions, labels=CATEGORIES)
    write_csv(subdir / f"{name}_confusion_matrix.csv", [
        {"actual\\predicted": label, **{c: int(v) for c, v in zip(CATEGORIES, matrix_row)}}
        for label, matrix_row in zip(CATEGORIES, matrix)
    ], ("actual\\predicted", *CATEGORIES))
    (subdir / f"{name}_classification_report.txt").write_text(
        classification_report(y, predictions, labels=CATEGORIES, zero_division=0), encoding="utf-8"
    )


def run_cv(rows, X, y, folds):
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    summary = []
    for name, mults in POLICIES.items():
        print(f"[CV] {name} attention_x{mults[0]}, food_x{mults[1]}", flush=True)
        predictions = np.empty(len(rows), dtype=object)
        start = time.monotonic()
        for fold, (fit_idx, pred_idx) in enumerate(kf.split(X, y), 1):
            model = build_model(y[fit_idx], mults)
            model.fit(X[fit_idx], y[fit_idx])
            predictions[pred_idx] = model.predict(X[pred_idx])
            fold_macro = f1_score(y[pred_idx], predictions[pred_idx], labels=CATEGORIES, average="macro", zero_division=0)
            print(f"    fold {fold}/{folds}: Macro-F1={fold_macro:.4f}", flush=True)
        scores = evaluate(y, predictions)
        scores.update({"policy": name, "attention_multiplier": mults[0], "food_multiplier": mults[1],
                       "n_dev": len(y), "folds": folds, "elapsed_sec": round(time.monotonic()-start, 2)})
        summary.append(scores)
        output_diagnostics(name, rows, y, predictions)
        print("    OOF Macro-F1={:.4f} | attention F1={:.4f} | food F1={:.4f} | accuracy={:.4f}".format(
            scores["macro_f1"], scores["attention_f1"], scores["food_f1"], scores["accuracy"]), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-only", action="store_true", help="Lock and audit holdout without fitting SVM")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = source_manifest()
    dev_rows, holdout_count = lock_split(rows)
    if args.split_only:
        print("[PASS] Split locked. No model fitted; holdout untouched.")
        return
    X = load_dev_clap(dev_rows)
    y = np.asarray([r["activity"] for r in dev_rows])
    cv_results = run_cv(dev_rows, X, y, CV_FOLDS)
    write_csv(OUT / "cv_weight_results.csv", cv_results, cv_results[0].keys())
    # Main selection criterion is overall macro F1. Attention & food scores break exact ties.
    best = max(cv_results, key=lambda v: (v["macro_f1"], (v["attention_f1"] + v["food_f1"]) / 2))
    best_name = best["policy"]
    print(f"[SELECT] Highest development CV Macro-F1: {best_name}", flush=True)
    final_model = build_model(y, POLICIES[best_name])
    final_model.fit(X, y)
    joblib.dump(final_model, OUT / "selected_CLAP_SVM_DEV_ONLY.joblib")
    manifest = {
        "selected_policy": best_name, "selected_by": "development 3-fold OOF Macro-F1, with target-F1 tie-break",
        "CLAP_feature_version": FEATURE_VERSION, "locked_holdout_evaluated": False,
        "holdout_size": holdout_count, "development_size": len(dev_rows),
        "cv_weight_results": cv_results,
    }
    (OUT / "selection_summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ref = next(v for v in cv_results if v["policy"] == "clap_balanced_reference")
    lines = [
        "# PawTalk STEP 1 — 冻结数据划分 & 类别权重优化（仅开发集）", "",
        f"- 开发训练集：{len(dev_rows)}；锁定保留测试集：{holdout_count}（未参与训练/调参/测试）",
        "- 评价：仅使用开发训练集的 3 折 out-of-fold 预测。",
        "- 未使用原先 500 条官方验证集做本次调参；未查看锁定测试集表现。",
        "- 由于缺少狗/视频身份，划分无法保证跨犬或跨视频独立。", "",
        "| 方案 | Accuracy | Macro-F1 | Seeking Attention F1 | Begging for Food F1 |", "|---|---:|---:|---:|---:|",
    ]
    for v in cv_results:
        lines.append(f"| {v['policy']} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {v['attention_f1']:.4f} | {v['food_f1']:.4f} |")
    lines += ["", f"**本轮按总体 Macro-F1 选出的模型：`{best_name}`**", "",
              f"相对同一划分下的 CLAP balanced 对照，Macro-F1 差值：{best['macro_f1']-ref['macro_f1']:+.4f}。", "",
              "重要：此差值不是最终测试集提升。旧实验使用全部 12,480 条训练音频，不能与新的 OOF 分数直接比较。",
              "下一步才进行 MFCC + CLAP 特征融合，同样只用开发训练集/内部交叉验证选择。", ""]
    (OUT / "STEP1_结果摘要.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {OUT / 'STEP1_结果摘要.md'}", flush=True)
    print(f"[SAVED] {OUT / 'cv_weight_results.csv'}", flush=True)
    print(f"[SAVED] {OUT / 'selected_CLAP_SVM_DEV_ONLY.joblib'}", flush=True)
    print("[CAUTION] DO NOT evaluate locked holdout or claim final test improvement now.", flush=True)
    print("[PASS] STEP1 finished.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)
        raise
