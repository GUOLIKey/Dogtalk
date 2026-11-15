"""Restore the reviewed release and import SHA-256-pinned public datasets."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_BRANCH = 'pawtalk-v12-two-datasets-20261115'

def run(*args, cwd=ROOT, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)

def valid(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == item['sha256']

def restore_payload():
    request = json.loads((ROOT / '.github/bootstrap/import_request.json').read_text())
    archive = ROOT / '.github/bootstrap/pawtalk_payload.zip'
    with archive.open('wb') as out:
        for part in sorted(archive.parent.glob('payload.part*')):
            out.write(part.read_bytes())
    if not valid(archive, request['archive']):
        raise ValueError('Release archive checksum mismatch')
    allowed = {'.gitattributes', '.gitignore', 'README_PAWTALK.md', 'PawTalk_Web_Clean', 'datasets', 'docs'}
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = Path(item.filename)
            if path.is_absolute() or '..' in path.parts or path.parts[0] not in allowed:
                raise ValueError('Unexpected archive path')
            destination = ROOT / path
            if destination.exists():
                raise ValueError('Refusing to overwrite an existing file: ' + item.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(z.read(item))
    print('Release archive restored and checksum verified', flush=True)

def download_barkopedia():
    folder = ROOT / 'datasets/Barkopedia_Dog_Act_Env'
    manifest = json.loads((folder / 'SOURCE_MANIFEST.json').read_text())
    def download(item):
        dest = folder / item['path']
        if valid(dest, item):
            return
        expected = 'https://huggingface.co/datasets/' + manifest['repository'] + '/resolve/' + manifest['revision'] + '/'
        if not item['source_url'].startswith(expected):
            raise ValueError('Unexpected source URL')
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + '.part')
        with urllib.request.urlopen(item['source_url'], timeout=180) as response, tmp.open('wb') as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if not valid(tmp, item):
            raise ValueError('Dataset checksum mismatch: ' + item['path'])
        tmp.replace(dest)
        print('Verified Barkopedia ' + item['path'], flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, manifest['files']))

def download_emotionalcanines():
    folder = ROOT / 'datasets/EmotionalCanines'
    manifest = json.loads((folder / 'SOURCE_MANIFEST.json').read_text())
    with tempfile.TemporaryDirectory(prefix='emotionalcanines-') as td:
        checkout = Path(td) / 'source'
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
        run('git', 'clone', '--no-checkout', manifest['repository'], str(checkout), env=env)
        run('git', 'checkout', '--detach', manifest['revision'], cwd=checkout, env=env)
        run('git', 'lfs', 'install', '--local', cwd=checkout)
        run('git', 'lfs', 'pull', cwd=checkout)
        for item in manifest['files']:
            source = checkout / item['path']
            if not valid(source, item):
                raise ValueError('Emotion dataset checksum mismatch: ' + item['path'])
            dest = folder / item['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    print('Verified all EmotionalCanines files', flush=True)

def main():
    if os.environ.get('GITHUB_REPOSITORY') != 'GUOLIKey/Dogtalk' or os.environ.get('GITHUB_REF_NAME') != ALLOWED_BRANCH:
        raise RuntimeError('This importer is restricted to the requested repository and branch')
    restore_payload()
    download_barkopedia()
    download_emotionalcanines()
    run('python3', 'datasets/verify_datasets.py')
    run('git', 'lfs', 'install', '--local')
    run('git', 'add', '.gitattributes', '.gitignore', 'README_PAWTALK.md', 'PawTalk_Web_Clean', 'datasets', 'docs')
    modified = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=MD'], cwd=ROOT, text=True)
    if modified.strip():
        raise RuntimeError('Import must only add files; existing content must remain unchanged')
    run('git', '-c', 'user.name=github-actions[bot]', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com',
        'commit', '-m', 'Import verified PawTalk v12 release and both complete datasets')
    run('git', 'lfs', 'fsck')
    run('git', 'push', 'origin', 'HEAD:refs/heads/' + ALLOWED_BRANCH)

if __name__ == '__main__':
    main()
