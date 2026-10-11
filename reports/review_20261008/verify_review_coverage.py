from pathlib import Path
import hashlib
import json
import re

ROOT = Path(r'D:\aicomp\AICtrack1')
OUT = ROOT / 'reports/review_20261008'
inventory = json.loads((OUT / 'source_inventory.json').read_text(encoding='utf-8-sig'))

def norm(p):
    p = p.replace('\\', '/')
    prefix = ROOT.as_posix() + '/'
    return p[len(prefix):] if p.lower().startswith(prefix.lower()) else p

original = {norm(f['path']): f for f in inventory}
root_paths = ['docs/compliance_questions.md', 'docs/HANDOFF_FOR_NEW_AI_20260929.md',
              'docs/upstream_registry.yaml', 'docs/技术报告.docx', 'docs/群通知1.txt',
              'docs/赛题规则.pdf']
root_files = []
for p in root_paths:
    f = original[p]
    root_files.append(dict(f, reader='/root', read_status='read_full',
                          method='full text' if p.endswith(('.md', '.yaml', '.txt')) else
                          ('all six PDF pages text and rendered pages inspected' if p.endswith('.pdf') else
                           'all OOXML paragraphs/header/footer/notes extracted; sole embedded logo inspected'),
                          note='original read before revision; C27-C33 authored and new full text inspected' if 'compliance' in p else ''))
(OUT / 'root_coverage.json').write_text(json.dumps({'files': root_files}, ensure_ascii=False, indent=2), encoding='utf-8')

records = {}
for name in ['root_coverage.json', 'history_coverage.json', 'recent_coverage.json', 'literature_extra_coverage.json']:
    obj = json.loads((OUT / name).read_text(encoding='utf-8-sig'))
    for f in obj['files']:
        key = norm(f.get('relative_path', f['path']))
        if key in original:
            records.setdefault(key, []).append(dict(f, coverage_source=name))

missing = sorted(set(original) - set(records))
identity_mismatch = []
drift = []
for p, f in original.items():
    if p in records and not any(r.get('sha256') == f['sha256'] for r in records[p]):
        identity_mismatch.append(p)
    current_sha = hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    if current_sha != f['sha256']:
        drift.append({'path': p, 'original_sha256': f['sha256'], 'current_sha256': current_sha})

handoff_records = records.get('docs/HANDOFF.md', [])
handoff_ranges = []
for f in handoff_records:
    for value in f.get('read_ranges', []):
        if isinstance(value, list):
            handoff_ranges.append(value)
        elif isinstance(value, str) and re.fullmatch(r'\d+-\d+', value):
            handoff_ranges.append([int(x) for x in value.split('-')])
    if 'read_start_line' in f:
        handoff_ranges.append([f['read_start_line'], f['read_end_line']])
    if isinstance(f.get('read_range'), list):
        handoff_ranges.append(f['read_range'])
    if isinstance(f.get('line_range'), list):
        handoff_ranges.append(f['line_range'])

handoff_read_lines = {n for a, b in handoff_ranges for n in range(a, b + 1)}
handoff_missing_lines = sorted(set(range(1, original['docs/HANDOFF.md']['lines'] + 1)) - handoff_read_lines)
not_full = []
for p, rows in records.items():
    if p == 'docs/HANDOFF.md':
        continue
    statuses = [r.get('read_status', r.get('status', '')) for r in rows]
    if not any(s in ('read_full', 'full_text_read', 'read_full_sha_reuse', 'read_full_sha_identical_reuse')
               or 'sha' in s and ('reuse' in s or 'full' in s) for s in statuses):
        not_full.append({'path': p, 'statuses': statuses})

new_files = sorted(str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'docs').rglob('*')
                   if p.is_file() and str(p.relative_to(ROOT)).replace('\\', '/') not in original)
result = {
    'snapshot_files': len(inventory), 'snapshot_bytes': sum(f['bytes'] for f in inventory),
    'covered_files': len(records), 'missing': missing, 'identity_mismatch': identity_mismatch,
    'changed_since_snapshot': drift, 'new_docs_since_snapshot': new_files,
    'handoff_ranges_reported': handoff_ranges,
    'handoff_missing_lines': handoff_missing_lines, 'not_full': not_full,
    'method': 'all original document contents reviewed; exact SHA-256 duplicates reused only after original full text reading; HANDOFF split between readers',
    'files': [{'path': p, 'sha256': original[p]['sha256'], 'evidence': records[p]} for p in sorted(records)]
}
(OUT / 'coverage_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))
assert not missing and not identity_mismatch and not handoff_missing_lines and not not_full
