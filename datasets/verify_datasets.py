"""Verify both pinned datasets, including LFS payloads, without running models."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
count = 0
total = 0
for name in ('Barkopedia_Dog_Act_Env', 'EmotionalCanines'):
    folder = root / name
    manifest = json.loads((folder / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = folder / item['path']
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Invalid manifest path')
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f'Missing or incorrect size: {path}; run git lfs pull')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            head = handle.read(12)
            if path.suffix == '.wav' and (head[:4] != b'RIFF' or head[8:12] != b'WAVE'):
                raise ValueError(f'Not a WAV payload: {path}')
            if path.suffix == '.zip' and head[:4] != b'PK\x03\x04':
                raise ValueError(f'Not a ZIP payload: {path}')
            digest.update(head)
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != item['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path}')
        count += 1
        total += item['bytes']
    print(f'{name}: verified {len(manifest["files"])} files at {manifest["revision"]}')
print(f'OK: {count} files, {total:,} bytes; all LFS payloads present.')
