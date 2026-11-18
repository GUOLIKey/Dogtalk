"""Download the official Barkopedia archives. Standard library only."""
import argparse
import csv
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://huggingface.co/datasets/ArlingtonCL2/Barkopedia_Dog_Act_Env/resolve/main/'
FILES = {
    'small': [
        ('train/train_label.csv', 'train_label.csv'),
        ('validation/valid_labels.csv', 'valid_labels.csv'),
        ('validation/valid_audios.zip', 'valid_audios.zip'),
    ],
    'full_extra': [
        ('train/split1.zip', 'split1.zip'),
        ('train/split2.zip', 'split2.zip'),
    ],
}


def valid_file(path):
    if not path.exists() or path.stat().st_size < 50:
        return False
    effective_name = path.name.lower().removesuffix('.part')
    if effective_name.endswith('.zip'):
        try:
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None and len(z.namelist()) > 0
        except (OSError, zipfile.BadZipFile):
            return False
    if effective_name.endswith('.csv'):
        try:
            with path.open('r', encoding='utf-8-sig', newline='') as f:
                header = next(csv.reader(f))
                return {'audio_id', 'activity'}.issubset(set(header))
        except (OSError, UnicodeError, StopIteration):
            return False
    return True


def download(rel, name):
    out = ROOT / 'data' / 'raw' / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + rel
    if valid_file(out):
        print(f'[SKIP] Already downloaded and verified: {out.name} ({out.stat().st_size/1024/1024:.1f} MiB)', flush=True)
        return True
    part = out.with_suffix(out.suffix + '.part')
    for attempt in range(1, 4):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'PawTalk-Research-Dataset-Downloader/1.0'}
            if start:
                headers['Range'] = f'bytes={start}-'
            print(f'[GET {attempt}/3] {rel}' + (f' (resume at {start} bytes)' if start else ''), flush=True)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as response:
                code = response.status
                append = start > 0 and code == 206
                total_header = response.headers.get('Content-Length')
                total = (int(total_header) if total_header and total_header.isdigit() else None)
                written = start if append else 0
                mode = 'ab' if append else 'wb'
                with part.open(mode) as f:
                    last_notice = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        if written - last_notice >= 25 * 1024 * 1024:
                            print(f'   {written / 1024 / 1024:.1f} MiB downloaded', flush=True)
                            last_notice = written
            if not valid_file(part):
                raise RuntimeError('Downloaded file failed CSV / ZIP integrity validation')
            os.replace(part, out)
            print(f'[OK] {rel} ({out.stat().st_size / 1024 / 1024:.1f} MiB)', flush=True)
            return True
        except Exception as exc:
            print(f'[WARN] Download attempt {attempt} failed: {exc}', flush=True)
            if attempt < 3:
                time.sleep(attempt * 2)
    print(f'[FAILED] Direct download: {url}')
    print(f'         Download from your browser, then place it at:\n         {out}')
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['small', 'full'], required=True)
    args = ap.parse_args()
    print(f'[PawTalk] Data target folder: {ROOT / "data" / "raw"}', flush=True)
    print('Source: Official ArlingtonCL2/Barkopedia_Dog_Act_Env repository', flush=True)
    files = FILES['small'] + (FILES['full_extra'] if args.mode == 'full' else [])
    errors = []
    for rel, name in files:
        if not download(rel, name):
            errors.append(rel)
    if errors:
        print(f'\n[INCOMPLETE] {len(errors)} file(s) could not be downloaded; see links above.')
        return 2
    print('\n[PASS] All requested files downloaded and checked. Run 02_AUDIT_DATA.bat next.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
