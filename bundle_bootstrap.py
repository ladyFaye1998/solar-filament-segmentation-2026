"""Load included weights, or fetch explicitly published weights by SHA-256."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source, cache):
    source, cache = Path(source).resolve(), Path(cache).resolve()
    manifest = json.loads((source / 'models.json').read_text(encoding='utf-8'))
    members = manifest['detectors'] + list(manifest['refiners'].values())
    missing = [m for m in members if not (source / m['path']).is_file()]
    if missing:
        locations = json.loads((source / 'weights-location.json').read_text(encoding='utf-8'))
        for member in missing:
            url = locations.get(member['path'])
            if not isinstance(url, str) or not url.startswith('https://'):
                raise ValueError('No published HTTPS location for ' + member['path'])
        cache.mkdir(parents=True, exist_ok=False)
        for path in source.rglob('*'):
            relative = path.relative_to(source)
            if not path.is_file() or 'weights' in relative.parts or '__pycache__' in relative.parts:continue
            target = cache / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for member in members:
            target = cache / member['path']
            assert target.resolve().is_relative_to(cache) and target.parent.name == 'weights'
            target.parent.mkdir(parents=True, exist_ok=True)
            included = source / member['path']
            if included.is_file():shutil.copy2(included, target)
            else:
                temporary = target.with_suffix('.download')
                size = 0
                with urllib.request.urlopen(locations[member['path']], timeout=90) as response, temporary.open('wb') as stream:
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > member['bytes']:raise ValueError('Oversized checkpoint response')
                        stream.write(block)
                assert size == member['bytes'] and digest(temporary) == member['sha256'], member['path']
                temporary.replace(target)
        source = cache
    for member in members:
        path = source / member['path']
        assert path.stat().st_size == member['bytes'] and digest(path) == member['sha256'], member['path']
    for name, expected in {**manifest['code_sha256'], **manifest.get('assets_sha256', {})}.items():
        assert digest(source / name) == expected, name
    return source
