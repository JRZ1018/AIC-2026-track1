"""Exclusive local assembly from the exact reviewed 554d inventory; no remote operations."""
from pathlib import Path
import hashlib, json, shutil

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT.parent / 'fresh_route_diagnostic_source'
PIN = '554d7c8f839d4ab6fee1ce427c5e2b9b73435d4766ac5347794ceecace53fbc2'

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def main():
    source = UPSTREAM / 'source_manifest.json'
    if sha(source) != PIN:
        raise ValueError('upstream 554d source manifest drift')
    inventory = json.loads(source.read_text(encoding='utf-8'))
    for name, digest in inventory['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(UPSTREAM / rel) != digest:
            raise ValueError('upstream member drift or invalid relative path: ' + name)
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output, (UPSTREAM / rel).open('rb') as handle:
            shutil.copyfileobj(handle, output)
    target = ROOT / 'reference' / 'upstream_554d_source_manifest.json'
    with target.open('xb') as output, source.open('rb') as handle:
        shutil.copyfileobj(handle, output)
    print(json.dumps({'copied_members': len(inventory['files']), 'upstream_source_sha256': PIN,
                      'server_used': False, 'gpu_used': False, 'test_access': False}))

if __name__ == '__main__':
    main()
