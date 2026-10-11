from pathlib import Path
import hashlib
import json
import re

root = Path(r'D:\aicomp\AICtrack1')
plan = root / 'docs/240_复赛24小时冲71重新审视与执行计划_20261009.md'
text = plan.read_text(encoding='utf-8')
links = re.findall(r'\[[^\]]*\]\(([^)]+)\)', text)
local_links = [u for u in links if re.match(r'^[A-Za-z]:[/\\]', u)]
missing_links = []
bad_lines = []
for u in local_links:
    match = re.match(r'^(.*?):(\d+)$', u)
    p = Path(match.group(1) if match else u)
    if not p.is_file():
        missing_links.append(u)
    elif match and int(match.group(2)) > len(p.read_text(encoding='utf-8-sig').splitlines()):
        bad_lines.append(u)
method_numbers = [int(x) for x in re.findall(r'^\| (\d+) \|', text, re.M)]
directions = re.findall(r'^### (P\d)[：:]', text, re.M)
result = {
    'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
    'plan_lines': len(text.splitlines()), 'local_links': len(local_links),
    'missing_local_links': missing_links, 'out_of_range_line_links': bad_lines,
    'method_numbers': method_numbers, 'directions': directions,
    'replacement_character_count': text.count('\ufffd'),
    'gpu_budget_hours': 16, 'mandatory_final_gpu_reserve_hours': 4,
    'training_started': False,
    'independent_reviews': ['recent_audit resource/protocol/code facts', 'literature_audit all 20 methods and paper mechanisms'],
    'peer_review_fixes': ['fresh S initialization', 'AFA gradient path and LBLS', 'actual GPU-time substitution for Jo-SRC',
                         '108 epochs ceiling and budget shrink', 'new split H1/cache cost and provenance',
                         'FV2 implementation source', 'joint clipping diagnostic', 'shared confirmation control'],
    'rules_sha256': hashlib.sha256((root / 'AGENTS.md').read_bytes()).hexdigest()
}
(root / 'reports/review_20261008/plan_validation.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
assert not missing_links and not bad_lines and method_numbers == list(range(1, 21))
assert directions == ['P0', 'P1', 'P2', 'P3', 'P4'] and not result['replacement_character_count']
