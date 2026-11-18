"""Barkopedia label and audio audit, using Python standard library only."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    'rest', 'alerting to sounds', 'seeking attention', 'playing with human',
    'playing with other animals', 'playing with toy', 'begging for food', 'taking shower'
)
NORMAL_MAP = {x: x for x in CATEGORIES}
# Only safe, obvious string normalization. Unknown labels remain flagged for review.
NORMAL_MAP.update({
    'alerting to sound': 'alerting to sounds',
    'playing with humans': 'playing with human',
    'playing with toys': 'playing with toy',
    'taking a shower': 'taking shower',
})
AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}


def normalize(raw):
    s = ' '.join(str(raw or '').strip().casefold().split())
    return NORMAL_MAP.get(s, s)


def label_rows(path, split):
    if not path.exists():
        return [], [f'MISSING labels: {path}'], Counter(), Counter()
    problems = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not {'audio_id', 'activity'}.issubset(cols):
            return [], [f'INVALID columns in {path.name}: {cols}'], Counter(), Counter()
        env_col = 'environement' if 'environement' in cols else 'environment' if 'environment' in cols else ''
        if not env_col:
            problems.append(f'No environment column found in {path.name}')
        rows = []
        seen = Counter()
        raw_cases = defaultdict(set)
        env_counts = Counter()
        for idx, row in enumerate(reader, 2):
            audio_id = (row.get('audio_id') or '').strip()
            original = (row.get('activity') or '').strip()
            activity = normalize(original)
            env = (row.get(env_col) or '').strip() if env_col else ''
            raw_cases[activity].add(original)
            env_counts[env] += 1
            seen[audio_id] += 1
            if not audio_id:
                problems.append(f'{split} line {idx}: empty audio_id')
            if activity not in CATEGORIES:
                problems.append(f'{split} line {idx}: UNKNOWN activity {original!r}')
            rows.append({'split': split, 'audio_id': audio_id,
                         'activity': activity, 'activity_original': original,
                         'environment': ' '.join(env.casefold().split()),
                         'environment_original': env})
    for key, times in seen.items():
        if times > 1:
            problems.append(f'{split}: repeated audio_id inside split: {key} x{times}')
    variant_rows = {key: sorted(vals) for key, vals in raw_cases.items() if len(vals) > 1}
    return rows, problems, variant_rows, env_counts


def safe_unzip(archive, destination):
    """Extract only safe regular files; stay within split-specific directories."""
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / '.unpack_complete.json'
    state = {'bytes': archive.stat().st_size, 'mtime_ns': archive.stat().st_mtime_ns}
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding='utf-8')) == state:
                print(f'[SKIP] Extracted earlier: {archive.name}')
                return
        except (ValueError, OSError):
            pass
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f'{archive}: corrupt member {bad}')
        root = destination.resolve()
        num = 0
        for info in z.infolist():
            filename = info.filename.replace('\\', '/')
            p = Path(filename)
            if p.is_absolute() or '..' in p.parts or filename.startswith('/') or ':' in p.parts[0]:
                raise ValueError(f'Unsafe ZIP member: {filename}')
            out = (root / p).resolve()
            if not out.is_relative_to(root):
                raise ValueError(f'ZIP path escapes destination: {filename}')
            if info.is_dir():
                out.mkdir(parents=True, exist_ok=True)
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'ZIP symlink not supported: {filename}')
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, out.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=1024*1024)
            num += 1
    marker.write_text(json.dumps(state), encoding='utf-8')
    print(f'[OK] Extracted {archive.name}: {num} files')


def scan_wavs(split):
    base = ROOT / 'data' / 'extracted' / split
    ids = defaultdict(list)
    wav_count = 0
    for path in base.rglob('*') if base.exists() else []:
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            wav_count += 1
            stem = path.stem
            ids[stem.casefold()].append(path)
            digits = re.fullmatch(r'(?:audio[_-]?)?(\d+)', stem.casefold())
            if digits:
                alias = 'audio_' + digits.group(1).zfill(5)
                if alias != stem.casefold():
                    ids[alias].append(path)
    return ids, wav_count


def get_audio_stats(path):
    if not path or path.suffix.lower() != '.wav':
        return '', '', ''
    try:
        with wave.open(str(path), 'rb') as f:
            frames, sr, channels = f.getnframes(), f.getframerate(), f.getnchannels()
            if not sr or not frames:
                return '', '', 'EMPTY_AUDIO'
            return f'{frames / sr:.4f}', str(sr), str(channels)
    except Exception as exc:
        return '', '', f'WAVE_ERROR:{type(exc).__name__}'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def audit(deep_hash=False, unpack=True):
    raw = ROOT / 'data' / 'raw'
    result = ROOT / 'results'
    result.mkdir(exist_ok=True)
    if unpack:
        for item, out in [
            (raw/'validation'/'valid_audios.zip', ROOT/'data'/'extracted'/'validation'),
            (raw/'train'/'split1.zip', ROOT/'data'/'extracted'/'train'/'split1'),
            (raw/'train'/'split2.zip', ROOT/'data'/'extracted'/'train'/'split2'),
        ]:
            if item.exists():
                safe_unzip(item, out)
            else:
                print(f'[INFO] ZIP not downloaded yet (okay for small mode): {item.name}')
    summary = {'sources': 'ArlingtonCL2/Barkopedia_Dog_Act_Env', 'splits': {}, 'warnings': []}
    all_hashed = {}
    all_audio = {}
    for split, csv_path in [
        ('train', raw/'train'/'train_label.csv'),
        ('validation', raw/'validation'/'valid_labels.csv'),
    ]:
        rows, problems, variants, env_counts = label_rows(csv_path, split)
        lookup, file_count = scan_wavs(split)
        matched = 0
        invalid = 0
        ambiguous = 0
        rates = Counter()
        duration_values = []
        files = []
        manifest = result / f'{split}_manifest.csv'
        with manifest.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['split', 'audio_id', 'activity', 'activity_original',
                          'environment', 'environment_original', 'audio_path', 'audio_match',
                          'duration_sec', 'sample_rate_hz', 'channels', 'audio_issue'])
            writer.writeheader()
            for row in rows:
                options = sorted(set(lookup.get(row['audio_id'].casefold(), [])), key=str)
                path = options[0] if len(options) == 1 else None
                match = 'MATCH' if path else ('AMBIGUOUS' if options else ('MISSING' if file_count else 'NOT_DOWNLOADED'))
                matched += bool(path)
                ambiguous += len(options) > 1
                duration, sr, channels = get_audio_stats(path)
                issue = '' if not path else (channels if channels.startswith('WAVE_ERROR') or channels == 'EMPTY_AUDIO' else '')
                if issue: invalid += 1
                if sr: rates[sr] += 1
                if duration: duration_values.append(float(duration))
                if path: files.append(path)
                writer.writerow({**row, 'audio_path': path.relative_to(ROOT).as_posix() if path else '',
                                 'audio_match': match, 'duration_sec': duration,
                                 'sample_rate_hz': sr, 'channels': '' if issue else channels,
                                 'audio_issue': issue})
        activities = Counter(r['activity'] for r in rows)
        unknown = {k: v for k, v in activities.items() if k not in CATEGORIES}
        extra_audio = file_count - matched if file_count else 0
        stats = {
            'label_rows': len(rows), 'audio_files_extracted': file_count, 'matched_audio': matched,
            'unmatched_labels': len(rows) - matched if file_count else None,
            'ambiguous_audio_ids': ambiguous, 'invalid_wav_headers': invalid,
            'unmatched_audio_files_approx': extra_audio if file_count else None,
            'activity_counts': dict(sorted(activities.items())),
            'environment_counts': dict(sorted(env_counts.items())),
            'unknown_activities': unknown, 'case_and_spacing_variants': variants,
            'sample_rate_counts': dict(sorted(rates.items())),
            'duration_min_sec': round(min(duration_values), 3) if duration_values else None,
            'duration_max_sec': round(max(duration_values), 3) if duration_values else None,
            'warnings': problems[:300], 'warning_count': len(problems),
        }
        if not file_count: summary['warnings'].append(f'{split}: No extracted audio yet; cannot verify audio matching.')
        if unknown: summary['warnings'].append(f'{split}: Unknown activity values: {unknown}. REVIEW before training.')
        if ambiguous: summary['warnings'].append(f'{split}: {ambiguous} ambiguous audio matches.')
        summary['splits'][split] = stats
        all_audio[split] = files
        print(f'[AUDIT] {split}: {len(rows)} labels, {file_count} audio files, {matched} uniquely matched')
        print(f'        activity distribution: {dict(activities)}')
        if problems: print(f'        label warnings: {len(problems)} (first: {problems[0]})')
    if deep_hash and all_audio['train'] and all_audio['validation']:
        print('[DEEP] SHA256 matching across train and validation (may take a while)...')
        train_hashes = defaultdict(list)
        for f in all_audio['train']: train_hashes[digest(f)].append(f)
        collisions = []
        for f in all_audio['validation']:
            hs = digest(f)
            for other in train_hashes.get(hs, []):
                collisions.append([other.relative_to(ROOT).as_posix(), f.relative_to(ROOT).as_posix()])
        summary['cross_split_identical_audio'] = collisions
        if collisions:
            summary['warnings'].append(f'Exact audio duplicates across train/validation: {len(collisions)}')
        print(f'[DEEP] Exact train/validation duplicate pairs: {len(collisions)}')
    else:
        summary['cross_split_identical_audio'] = 'NOT_CHECKED; use --hash after full download'
    # The same audio_id across splits may refer to DIFFERENT clips; do not auto-deduplicate.
    summary['warnings'].append('audio_id may repeat across train/validation. Split-qualified paths are mandatory.')
    (result/'audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# PawTalk · Barkopedia data audit report', '',
             'Data source: https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env', '',
             'IMPORTANT: This report audits existing downloaded data only. It does not measure model accuracy.', '',
             '| Split | Label rows | Audio files | Matched | Unknown activity labels |',
             '|---|---:|---:|---:|---:|']
    for split, s in summary['splits'].items():
        lines.append(f'| {split} | {s["label_rows"]} | {s["audio_files_extracted"]} | {s["matched_audio"]} | {sum(s["unknown_activities"].values())} |')
        lines.extend(['', f'## {split} activity counts', ''])
        for act, n in s['activity_counts'].items(): lines.append(f'- {act}: {n}')
        if s['unknown_activities']: lines.extend(['', f'**REVIEW UNKNOWN LABELS:** {s["unknown_activities"]}'])
        if s['case_and_spacing_variants']: lines.extend(['', f'Case/style variants: `{s["case_and_spacing_variants"]}`'])
        if s['warning_count']: lines.extend(['', f'Label warnings ({s["warning_count"]}):', *[f'- {w}' for w in s['warnings'][:20]]])
    lines.extend(['', '## Warnings and limitations', '', *[f'- {x}' for x in summary['warnings']], '',
      'The public validation split is for model selection; it is NOT an independent final test set.',
      'Checking exact duplicate hashes cannot guarantee no same-dog or same-video leakage.',
      'Audio labels may reflect video-assisted annotation rather than information recoverable from bark alone.',
      '', 'Files: `results/train_manifest.csv`, `results/validation_manifest.csv`, `results/audit_summary.json`.'])
    (result/'audit_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'\n[SAVED] {result / "audit_report.md"}')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hash', action='store_true', help='Expensive cross-split SHA256 duplicate check')
    ap.add_argument('--skip-extract', action='store_true')
    args = ap.parse_args()
    try:
        audit(deep_hash=args.hash, unpack=not args.skip_extract)
        return 0
    except Exception as exc:
        print(f'[FAILED] {type(exc).__name__}: {exc}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
