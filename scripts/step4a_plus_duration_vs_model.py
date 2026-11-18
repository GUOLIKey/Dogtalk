"""PawTalk STEP4A+: duration vs STEP3 OOF prediction quality (DEV only).

Uses previous STEP4A metadata and STEP3 best-model OOF predictions.
Does not read WAVs, fit models, extract features, or evaluate locked holdout.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, recall_score

CATEGORIES = [
    "rest", "taking shower", "playing with toy", "playing with other animals",
    "playing with human", "alerting to sounds", "begging for food", "seeking attention",
]
BINS = [
    ("<0.5s", 0.0, 0.5),
    ("0.5-0.96s", 0.5, 0.96),
    ("0.96-2s", 0.96, 2.0),
    ("2-4s", 2.0, 4.0),
    ("4-10s", 4.0, 10.0),
    (">=10s", 10.0, math.inf),
]
THRESHOLDS = (0.0, 0.5, 0.96, 1.0, 2.0, 4.0)
EXPECTED_DEV = 9984
EXPECTED_STEP3_MACRO = 0.3226
EXPECTED_STEP3_ACCURACY = 0.3208


def normpath(s):
    return str(s or "").strip().replace("\\", "/")


def load_csv(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not set(required).issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Unexpected columns in {path.name}: {reader.fieldnames}; required={required}")
        return list(reader)


def unique_by_path(rows, label):
    out = {}
    for row in rows:
        key = normpath(row["audio_path"])
        if not key or key in out:
            raise RuntimeError(f"{label}: empty or duplicate audio_path: {key!r}")
        out[key] = row
    return out


def load_and_verify(root):
    split_dir = root / "results" / "optimization_step1" / "splits"
    dev_file = split_dir / "dev_train.csv"
    hold_file = split_dir / "LOCKED_holdout_DO_NOT_TUNE.csv"
    if not hold_file.exists():
        raise RuntimeError("Locked holdout file missing; refusing to continue. This tool will not read its contents.")
    dev = unique_by_path(load_csv(dev_file, ("audio_path", "activity")), "STEP1 DEV split")
    meta = unique_by_path(load_csv(root / "results" / "optimization_step4a" / "DEV_audio_metadata.csv",
                                   ("audio_path", "activity", "status", "duration_sec")), "STEP4A metadata")
    preds = unique_by_path(load_csv(root / "results" / "optimization_step3" / "diagnostics" /
                                    "c1_g2_oof_predictions.csv",
                                    ("audio_path", "actual", "predicted")), "STEP3 selected OOF predictions")
    if len(dev) != EXPECTED_DEV or len(meta) != EXPECTED_DEV or len(preds) != EXPECTED_DEV:
        raise RuntimeError(f"Expected {EXPECTED_DEV} DEV records per file; found split={len(dev)}, "
                           f"metadata={len(meta)}, OOF={len(preds)}. Do not compute partial metrics.")
    if set(dev) != set(meta) or set(dev) != set(preds):
        raise RuntimeError("DEV split, metadata and OOF prediction paths are NOT identical. "
                           "Do not proceed: this may mix different experiments.")
    records = []
    for key, item in dev.items():
        m, p = meta[key], preds[key]
        actual = item["activity"]
        predicted = p["predicted"]
        if actual not in CATEGORIES or predicted not in CATEGORIES or m["activity"] != actual or p["actual"] != actual:
            raise RuntimeError(f"Label mismatch or invalid label at {key}")
        if m["status"] != "ok":
            raise RuntimeError(f"STEP4A metadata failed for {key}; refusing partial result")
        try:
            duration = float(m["duration_sec"])
        except ValueError as exc:
            raise ValueError(f"Invalid duration for {key}: {m['duration_sec']!r}") from exc
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError(f"Invalid/nonpositive duration for {key}")
        records.append({"audio_path": key, "duration_sec": duration, "actual": actual,
                        "predicted": predicted, "correct": int(actual == predicted)})
    overall = evaluate(records)
    if abs(overall["macro_f1"] - EXPECTED_STEP3_MACRO) > 0.00015 or \
       abs(overall["accuracy"] - EXPECTED_STEP3_ACCURACY) > 0.00015:
        raise RuntimeError(f"STEP3 metric mismatch! Expected approx Macro-F1 0.3226, accuracy 0.3208; "
                           f"computed Macro-F1={overall['macro_f1']:.6f}, accuracy={overall['accuracy']:.6f}. "
                           "Check that c1_g2_oof_predictions.csv is the selected STEP3 model.")
    print(f"[LOCK] Verified original DEV paths only: {len(records)}; holdout file NOT opened.", flush=True)
    print(f"[VERIFY] STEP3 OOF matched: Macro-F1={overall['macro_f1']:.4f}; "
          f"Accuracy={overall['accuracy']:.4f}; correct={overall['correct']}/{len(records)}", flush=True)
    return records, overall


def evaluate(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "correct": 0, "accuracy": None, "macro_f1": None,
                "balanced_accuracy": None, "attention_f1": None, "food_f1": None,
                "classes_present": 0}
    y = [r["actual"] for r in rows]
    pred = [r["predicted"] for r in rows]
    present = len(set(y))
    # Avoid presenting 8-class F1 for bins with absent true classes.
    have_all = present == len(CATEGORIES)
    per_f1 = f1_score(y, pred, labels=CATEGORIES, average=None, zero_division=0)
    result = {
        "n": n, "correct": sum(r["correct"] for r in rows),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CATEGORIES, average="macro", zero_division=0)) if have_all else None,
        "balanced_accuracy": float(recall_score(y, pred, labels=CATEGORIES, average="macro", zero_division=0)) if have_all else None,
        "attention_f1": float(per_f1[CATEGORIES.index("seeking attention")]) if have_all else None,
        "food_f1": float(per_f1[CATEGORIES.index("begging for food")]) if have_all else None,
        "classes_present": present,
    }
    return result


def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def bucket(dur):
    for name, lo, hi in BINS:
        if lo <= dur < hi:
            return name
    raise ValueError(f"Unexpected duration {dur}")


def write_csv(dest, rows, fields):
    if dest.exists():
        raise FileExistsError(f"Output exists: {dest}. Stop: do not overwrite prior results.")
    with dest.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVED] {dest}", flush=True)


def build_tables(rows, overall):
    total = len(rows)
    total_correct = overall["correct"]
    by_bin = []
    by_class = []
    for label, _, _ in BINS:
        ss = [r for r in rows if bucket(r["duration_sec"]) == label]
        scores = evaluate(ss)
        counts = Counter(r["actual"] for r in ss)
        by_bin.append({
            "duration_bin": label, "n": len(ss), "share_of_dev_pct": 100 * len(ss) / total,
            "correct": scores["correct"], "share_of_all_correct_pct": 100 * scores["correct"] / max(1, total_correct),
            "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "attention_f1": scores["attention_f1"], "food_f1": scores["food_f1"],
            "classes_present": scores["classes_present"], "minimum_class_n": min(counts.values(), default=0),
            "caution": "n<30" if len(ss) < 30 else ("n<100" if len(ss) < 100 else
                      "some class n<10" if len(counts) < 8 or min(counts.values()) < 10 else ""),
        })
        for c in CATEGORIES:
            subset = [r for r in ss if r["actual"] == c]
            true_n = len(subset)
            correct = sum(r["correct"] for r in subset)
            pred_n = sum(r["predicted"] == c for r in ss)
            precision = correct / pred_n if pred_n else None
            recall = correct / true_n if true_n else None
            if true_n == 0:
                f1 = None  # No true class members in this duration group.
            elif correct == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
            by_class.append({"duration_bin": label, "activity": c, "true_n": true_n,
                             "predicted_n": pred_n, "correct": correct, "recall": recall,
                             "precision": precision, "f1": f1, "caution": "small support (<10)" if true_n < 10 else ""})
    thresholds = []
    for minimum in THRESHOLDS:
        ss = [r for r in rows if r["duration_sec"] >= minimum]
        stats = evaluate(ss)
        thresholds.append({
            "min_duration_sec": minimum, "kept_n": len(ss), "removed_n": total - len(ss),
            "coverage_pct": 100 * len(ss) / total,
            "kept_correct": stats["correct"],
            "pct_original_correct_kept": 100 * stats["correct"] / max(1, total_correct),
            "accuracy_on_kept": stats["accuracy"], "macro_f1_on_kept": stats["macro_f1"],
            "classes_present": stats["classes_present"],
        })
    return by_bin, by_class, thresholds


def generate_report(dest, by_bin, by_class, thresholds, overall):
    lines = [
        "# PawTalk STEP 4A+ — 录音时长 × STEP3 分类表现（仅开发集）", "",
        "## 研究目的与边界", "",
        "- 使用 STEP4A 已保存的音频时长和 STEP3 最佳模型的 **3-fold OOF 预测**。",
        "- **不训练模型，不读取音频，不重新计算 CLAP，不截断任何录音。**",
        "- 先与 STEP1 固定开发集名单逐条匹配、核对标签，拒绝缺失或重复；完全不打开锁定测试集文件。",
        "- 分段评估统计的是**不同原始录音总时长**对应的表现，并非把同一条长录音截成 0.96 秒后测试。",
        "- 本次为探索性子群评估，类别分布、样本数量、同一录音来源相关性和开发集调参均可能影响解释。", "",
        "## 1. 全体开发集参考", "",
        f"- n={overall['n']}，正确预测 {overall['correct']} 条，Accuracy={fmt(overall['accuracy'])}，Macro-F1={fmt(overall['macro_f1'])}。",
        "- 对照 STEP3 日志 Macro-F1 ≈ 0.3226、Accuracy ≈ 0.3208，已自动核对。", "",
        "## 2. 原始录音时长分组表现", "",
        "| 时长 | n | 占DEV | 正确数 | 贡献全部正确预测 | Accuracy | Macro-F1 | Balanced Accuracy | 关注F1 | 讨食F1 | 提醒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in by_bin:
        lines.append(f"| {r['duration_bin']} | {r['n']} | {r['share_of_dev_pct']:.1f}% | {r['correct']} | "
                     f"{r['share_of_all_correct_pct']:.1f}% | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | "
                     f"{fmt(r['balanced_accuracy'])} | {fmt(r['attention_f1'])} | {fmt(r['food_f1'])} | {r['caution'] or '—'} |")
    lines.extend([
        "", "**解释：**即使短音频 Accuracy 较低，它也可能因为数量多而贡献很大比例的正确预测；"
        "因此不能只按某个区间的准确率决定是否删除它们。",
        "不同区间如果缺少某个类别，就不报告该区间的完整八类 Macro-F1。样本量少的 F1 仅作为线索。", "",
        "## 3. 按录音总时长设置最低门槛：描述性模拟（不是重新训练）", "",
        "| 最低时长 | 保留 n | 删除 n | 覆盖率 | 保留原有正确预测 | 保留样本 Accuracy | 保留样本 Macro-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in thresholds:
        lines.append(f"| {r['min_duration_sec']:g}s | {r['kept_n']} | {r['removed_n']} | "
                     f"{r['coverage_pct']:.1f}% | {r['pct_original_correct_kept']:.1f}% | "
                     f"{fmt(r['accuracy_on_kept'])} | {fmt(r['macro_f1_on_kept'])} |")
    lines.extend([
        "", "**重要：**筛除短录音后仅重新统计原模型在剩余样本上的成绩；"
        "它**不能证明**删掉短录音重新训练模型会提高准确率，也不能证明用户录长音频会获得相同提升。", "",
        "## 4. 各时长段各类别的样本量、F1 和召回率", "",
        "完整表格见 `DEV_duration_by_class_detailed.csv`，尤其检查 `begging for food` 是否有足够样本。", "",
        "## 5. 下一步怎么决策", "",
        "- 短录音差且覆盖率损失可接受：下一阶段可做**产品端最短录音提醒**，但仍需独立验证，不要立即删训练数据。",
        "- 短录音好或贡献大量正确样本：保留短录音；考虑用户上传较长录音后提取多个狗叫事件，属于新方案，需要新实验。",
        "- 所有时长组表现都差：应优先研究标签含义、背景声音与跨狗泛化，而非单纯加长录音。",
        "- 更长的新数据集先保留为候选；标签体系不同或只有 dog-bark 检测标签的不可直接合并八分类。",
        "- **不要**提前评估 STEP1 的 2,496 条锁定测试集。", "",
    ])
    if dest.exists():
        raise FileExistsError(f"Output exists: {dest}. Stop: do not overwrite prior results.")
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVED] {dest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="STEP4A+ duration performance from existing DEV-only OOF records")
    parser.add_argument("--check", action="store_true", help="Validate existing CSVs and reference metric; no outputs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root (optional)")
    args = parser.parse_args()
    root = args.root.resolve()
    print("[STEP4A+] Analyzing recording duration vs selected STEP3 OOF predictions (DEV only).", flush=True)
    print("[SAFETY] No model training, no audio decoding, no locked holdout evaluation.", flush=True)
    rows, overall = load_and_verify(root)
    if args.check:
        print("[PASS] STEP4A+ input validation passed. No results changed.", flush=True)
        return
    out = root / "results" / "optimization_step4a_plus"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"STEP4A+ output directory is not empty: {out}. Will not overwrite files.")
    out.mkdir(parents=True, exist_ok=True)
    grouped, classed, thresholds = build_tables(rows, overall)
    for r in grouped:
        print(f"[BIN] {r['duration_bin']}: n={r['n']}, correct={r['correct']}, "
              f"Accuracy={fmt(r['accuracy'])}, Macro-F1={fmt(r['macro_f1'])}, "
              f"share_correct={r['share_of_all_correct_pct']:.1f}%", flush=True)
    write_csv(out / "DEV_duration_performance_detailed.csv", grouped, list(grouped[0]))
    write_csv(out / "DEV_duration_by_class_detailed.csv", classed, list(classed[0]))
    write_csv(out / "DEV_duration_threshold_simulation.csv", thresholds, list(thresholds[0]))
    generate_report(out / "STEP4A_PLUS_时长与模型表现报告.md", grouped, classed, thresholds, overall)
    print("[PASS] STEP4A+ finished. All STEP1-4A outputs and locked holdout unchanged.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAILED] {type(exc).__name__}: {exc}", flush=True)
        raise
