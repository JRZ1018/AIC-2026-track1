"""Repeatable FV2 package verification and bounded CPU model restoration.

All writes stay in acceptance_source. No network, CUDA, training, test image
decoding, or final-inference entry point is used. Historic receipts stay intact.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ASSETS = Path(r'D:\aicomp\stage2-server-transfer\fv2_full_results_20261008')
REPO = Path(r'D:\aicomp\AICtrack1')
FULL = ASSETS / 'full'
DELIVERY = ASSETS / 'delivery'
OFFICIAL = ASSETS / 'official'
EXPECTED = dict(
    clip='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af',
    mapping='811b767e5689fbed5ed9ed7f328e2e1458b02b208647f3ec1b3c3e99c7e4e879',
    manifest='1e912b3a73ffddfa418a885fb1e20c3c15012e6f3fae2203c81dd9dd3c52887d',
    split='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c',
    head='fa4623670a6f48731cf54e62dfb5ce5df44b232f4530f6c13b886efe0579e605',
    final='85f4a7c50da95e78397ed0a1067632fe0fbfc410b46a4cd35bcf61df621cb8b0',
    last='226557f79475b8e2f02c69738972aad2d75c87b82bb14150ce1f7076daaaa450',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def doc(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save(name, value):
    path = (OUT / name).resolve()
    require(path.is_relative_to(OUT), 'output must remain inside acceptance_source')
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    if path.exists():
        require(path.read_bytes() == payload, f'output already exists with different content: {path}')
        return path
    with path.open('xb') as handle:
        handle.write(payload)
    return path


def mapping_identity(path):
    entries = sorted(rows(path), key=lambda r: int(r['internal_class_index']))
    require([int(r['internal_class_index']) for r in entries] == list(range(750)), '750 contiguous class IDs required')
    official = [r['official_class_id'] for r in entries]
    require(len(set(official)) == 750 and all(re.fullmatch(r'\d{4}', value) for value in official), 'invalid official ID mapping')
    digest = hashlib.sha256('\n'.join(f"{r['internal_class_index']}\t{r['official_class_id']}\t{r.get('class_name','')}" for r in entries).encode()).hexdigest()
    return entries, digest


def validate_package(package, test_manifest, mapping_csv, expected_checkpoint=None):
    package = Path(package)
    manifest = doc(package / 'submission_manifest.json')
    spec = manifest['spec']
    require(spec['csv_name'] == 'pred_results.csv' and spec['include_header'] is False and spec['filename_mode'] == 'basename', 'unexpected CSV contract')
    require(spec['official_class_id_pattern'] == r'\d{4}', 'unexpected explicit Stage2 ID pattern')
    test_rows = rows(test_manifest)
    require(len(test_rows) == 37444, 'test manifest must have 37444 rows')
    require(all(r['source_split'] == 'test' and not r['internal_class_index'] and not r['official_class_id'] for r in test_rows), 'test manifest must have no labels')
    expected = [Path(r['relative_path']).name for r in test_rows]
    require(len(set(expected)) == len({n.casefold() for n in expected}) == 37444, 'test filename collision')
    entries, mapping_digest = mapping_identity(mapping_csv)
    raw = (package / 'pred_results.csv').read_bytes()
    require(not raw.startswith(b'\xef\xbb\xbf'), 'CSV BOM forbidden')
    text = raw.decode('utf-8')
    require('\r' not in text and text.endswith('\n'), 'CSV must use LF and final newline')
    parsed = list(csv.reader(io.StringIO(text, newline='')))
    require(len(parsed) == 37444 and all(len(r) == 2 for r in parsed), 'CSV row count or field count invalid')
    require([r[0] for r in parsed] == expected, 'CSV filename order/set differs from manifest')
    allowed = {r['official_class_id'] for r in entries}
    require(all(r[1] in allowed for r in parsed), 'unknown or altered official label ID')
    with zipfile.ZipFile(package / 'pred_results.zip') as archive:
        info = archive.infolist()
        require(len(info) == 1 and info[0].filename == 'pred_results.csv' and not info[0].is_dir(), 'ZIP must contain one root CSV')
        require(not info[0].flag_bits & 1 and archive.read(info[0]) == raw, 'ZIP encrypted or CSV bytes differ')
    csv_sha, zip_sha = sha(package/'pred_results.csv'), sha(package/'pred_results.zip')
    require(manifest['record_count'] == 37444 and manifest['csv_sha256'] == csv_sha and manifest['zip_sha256'] == zip_sha, 'package manifest hashes/count differ')
    require(manifest['class_mapping_sha256'] == manifest['run_metadata']['class_mapping_sha256'] == mapping_digest, 'package class mapping differs')
    metadata = manifest['run_metadata']
    required = {'run_id','checkpoint_sha256','config_hash','data_hash','git_commit','inference_command'}
    require(required.issubset(metadata) and all(str(metadata[key]).strip() for key in required), 'required submission provenance is missing')
    require(all(re.fullmatch(r'[0-9a-f]{64}', str(metadata[key])) for key in ['checkpoint_sha256','config_hash','data_hash']), 'provenance SHA format invalid')
    require(metadata['data_hash'] == metadata['test_manifest_sha256'], 'data hash/test manifest differ')
    require(manifest['expected_filenames_sha256'] == hashlib.sha256('\n'.join(expected).encode()).hexdigest(), 'ordered filename digest differs')
    require(manifest['run_metadata']['test_manifest_sha256'] == sha(test_manifest), 'test manifest SHA differs')
    if expected_checkpoint:
        require(manifest['run_metadata']['checkpoint_sha256'] == expected_checkpoint, 'wrong model in package')
    return dict(complete=True, package_path=str(package), csv_sha256=csv_sha, zip_sha256=zip_sha,
                manifest_sha256=sha(package/'submission_manifest.json'), record_count=37444,
                official_class_count=750, exact_filename_order=True, zip_single_csv_bytes_equal=True,
                checkpoint_sha256=manifest['run_metadata']['checkpoint_sha256'],
                test_image_forward=False, model_reconstructed=False, online_submitted=False)


def verify_sources():
    inventory = {}
    for tag, root, filename in [('full', FULL, 'mainline_sources_sha256.json'), ('delivery', DELIVERY, 'inference_sources_sha256.json')]:
        source_doc = doc(root/filename)
        checked = {}
        for relative, expected in source_doc['files'].items():
            path = (root/relative).resolve()
            require(path.is_relative_to(root) and path.is_file(), f'unsafe or missing source file {relative}')
            actual = sha(path)
            require(actual == expected, f'{tag} source SHA mismatch: {relative}')
            checked[relative] = dict(sha256=actual, bytes=path.stat().st_size)
        inventory[tag] = dict(root=str(root), source_manifest_sha256=sha(root/filename), files=checked, all_raw_source_sha_pass=True)
    selected = ['scripts/infer_and_build_submission.py', 'scripts/train_peft.py', 'scripts/train_n1_peft.py',
                'src/aictrack1/models/clip_vit_b32.py', 'src/aictrack1/models/clip_resolution.py',
                'src/aictrack1/models/heads.py', 'src/aictrack1/models/peft.py',
                'src/aictrack1/engine/peft.py', 'src/aictrack1/engine/training_queue.py',
                'src/aictrack1/evaluation/four_view.py', 'src/aictrack1/submission/package.py']
    inventory['current_repository'] = {relative: dict(sha256=sha(REPO/relative), bytes=(REPO/relative).stat().st_size) for relative in selected if (REPO/relative).is_file()}
    inventory['comparison'] = {}
    for relative in selected:
        variants = {}
        for tag, root in [('full', FULL), ('delivery', DELIVERY), ('current', REPO)]:
            path = root/relative
            if path.is_file():
                raw = path.read_bytes()
                variants[tag] = dict(sha256=hashlib.sha256(raw).hexdigest(), lf_text_sha256=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
        inventory['comparison'][relative] = variants
    metrics = FULL/'artifacts/experiments/fullvisual20/metrics.json'
    value = doc(metrics)
    inventory['full_metrics'] = dict(path=str(metrics), sha256=sha(metrics), bytes=metrics.stat().st_size,
        duration_seconds=value['duration_seconds'], peak_vram_mib=value['peak_vram_mib'],
        history_count=len(value['history']), final_checkpoint_sha256=value['final_checkpoint_sha256'],
        source_git_commit=value['git_commit'], source_git_note=value['git_commit_source'], checkpoint_provenance=value['checkpoint_provenance'])
    return inventory


def restore_cpu():
    import torch
    import numpy as np
    require(resource_info()['available_bytes'] >= 6 * 1024**3, 'need at least 6 GiB available RAM')
    require(not (REPO/'artifacts/queues/training.lock.json').exists(), 'local repository training lock present; do not alter it')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    paths = dict(last=FULL/'artifacts/experiments/fullvisual20/last.pt', final=FULL/'artifacts/experiments/fullvisual20/final.pt',
                 head=FULL/'artifacts/experiments/mainline-h1/last.pt', clip=OFFICIAL/'ViT-B-32.pt')
    actual_hashes = {key: sha(path) for key, path in paths.items()}
    require(actual_hashes == {key: EXPECTED[key] for key in paths}, 'frozen model SHA mismatch')
    script_sha = sha(__file__)
    fingerprint = hashlib.sha256(json.dumps(dict(inputs=actual_hashes, script_sha256=script_sha), sort_keys=True).encode()).hexdigest()
    result_name = f'cpu_restore_{fingerprint[:16]}.json'
    result_path = OUT/result_name
    if result_path.exists():
        old = doc(result_path)
        require(old['complete'] and old['input_sha256'] == actual_hashes and old['script_sha256'] == script_sha, 'invalid existing CPU receipt')
        print(json.dumps(dict(reused_complete=True, result=str(result_path)), ensure_ascii=False), flush=True)
        return old
    stage_path = OUT/f'cpu_stage_{fingerprint[:16]}.json'
    require(not stage_path.exists(), 'previous CPU stage exists without completion; inspect its failure/owner, do not blindly restart')
    started = time.time()
    existing_deadlines = [doc(path)['deadline'] for path in OUT.glob('cpu_stage_*.json')]
    deadline = min(existing_deadlines + [started+600])
    require(started < deadline, 'original CPU acceptance budget is exhausted; do not reset it during repairs')
    save(stage_path.name, dict(actual=started, deadline=deadline, owner_pid=os.getpid(), owner_create_time=resource_info()['process_create_time'],
        input_sha256=actual_hashes, script_sha256=script_sha, scope='CPU restore, two official train images; no CUDA, training or test inference'))
    def budget():
        require(time.time() <= deadline, 'CPU restoration deadline exceeded')
        require(resource_info()['rss_bytes'] <= 5*1024**3, 'CPU restoration RSS exceeded 5 GiB')
    def cuda_forbidden(*args, **kwargs):
        raise RuntimeError('CUDA operation forbidden during CPU restore')
    try:
        inventory = verify_sources()
        # All inputs are immutable, already returned files. Revalidate the transport receipt.
        fetched = doc(ASSETS/'LOCAL_FETCH_RECEIPT.json')
        require(fetched['complete'] and fetched['sha256_all_pass'], 'incomplete transport receipt')
        for relative, item in fetched['files'].items():
            path = (ASSETS/relative).resolve()
            require(path.is_relative_to(ASSETS) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], f'fetched input mismatch: {relative}')
        budget()
        sys.path[:0] = [str(FULL/'scripts'), str(FULL/'src')]
        from mainline_h1_contract import load_new_fold_head, validate_final, digest_rows
        from aictrack1.models.clip_vit_b32 import load_clip_vit_b32
        from aictrack1.models.clip_resolution import enable_interpolated_visual_resolution
        from aictrack1.models.peft import configure_visual_peft, PEFTImageClassifier
        from aictrack1.engine.peft import load_trainable_state_dict
        from aictrack1.engine.provenance import CheckpointProvenance
        from aictrack1.methods.confidence import ClasswiseHistoryWeighting
        from aictrack1.data.contracts import TransformConfig
        from aictrack1.data.transforms import build_split_transforms
        from PIL import Image
        official_rows = rows(OFFICIAL/'train_samples.csv')
        assignments = {r['sample_id']:r['split'] for r in rows(FULL/'inputs/split.csv')}
        entries, mapping_digest = mapping_identity(OFFICIAL/'classes.csv')
        require(mapping_digest == EXPECTED['mapping'] and sha(OFFICIAL/'train_samples.csv') == EXPECTED['manifest'] and sha(FULL/'inputs/split.csv') == EXPECTED['split'], 'full model provenance mismatch')
        require(len(official_rows) == len(assignments) == 148695 and set(assignments) == {r['sample_id'] for r in official_rows}, 'full train IDs mismatch')
        bound = [[r['sample_id'], int(r['internal_class_index']), assignments[r['sample_id']]] for r in official_rows]
        provenance = CheckpointProvenance('stage2', EXPECTED['mapping'], EXPECTED['manifest'], EXPECTED['split'], EXPECTED['clip'], 750)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
            for name in ['init', 'get_rng_state_all', 'set_rng_state_all']:
                stack.enter_context(patch.object(torch.cuda, name, side_effect=cuda_forbidden))
            head, h1 = load_new_fold_head(paths['head'], feature_dim=512, num_classes=750,
                expected_clip_weight_sha256=EXPECTED['clip'], expected_class_mapping_sha256=EXPECTED['mapping'],
                expected_split_sha256=EXPECTED['split'], expected_provenance=provenance, forbid_full_train_initializer=False)
            validate_final(h1, h1['row_binding'])
            require(h1['row_binding']['all_rows_sha256'] == h1['row_binding']['train_rows_sha256'] == digest_rows(bound), 'full H1 binding differs')
            head_opt = torch.optim.Adam(head.parameters(), lr=.01)
            head_opt.load_state_dict(h1['optimizer'])
            compare_optimizer(torch, head_opt, list(head.parameters()), h1['optimizer'], 2)
            head_sched = torch.optim.lr_scheduler.LambdaLR(head_opt, lambda _:1)
            head_sched.load_state_dict(h1['scheduler'])
            require(head_sched.last_epoch == 100, 'H1 scheduler did not restore')
            head_scaler = torch.amp.GradScaler('cpu', enabled=bool(h1['scaler']))
            head_scaler.load_state_dict(h1['scaler'])
            require(head_scaler.state_dict() == h1['scaler'], 'H1 scaler did not restore')
            for parameter in head.parameters(): parameter.requires_grad_(False)
            clip, _, metadata = load_clip_vit_b32(paths['clip'], EXPECTED['clip'], torch.device('cpu'))
            visual = clip.visual.float()
            del clip
            gc.collect()
            enable_interpolated_visual_resolution(visual, target_resolution=336)
            configure_visual_peft(visual, 'full_visual')
            model = PEFTImageClassifier(visual, head, normalize_features=True)
            last = torch.load(paths['last'], map_location='cpu', weights_only=True, mmap=True)
            final = torch.load(paths['final'], map_location='cpu', weights_only=True, mmap=True)
            require(last['epoch'] == 20 and last['training_scope'] == 'full_train_fixed_epochs' and last['best_state'] is None and last['initial_validation'] is None, 'incorrect full checkpoint scope')
            require([r['epoch'] for r in last['history']] == list(range(1,21)) and last['history'] == doc(FULL/'artifacts/experiments/fullvisual20/metrics.json')['history'], 'full epoch history differs')
            require(last['config_sha256'] == final['config_sha256'] == doc(FULL/'full_model_frozen.json')['config_sha256'], 'last/final/config differ')
            require(final['head_initializer_sha256'] == EXPECTED['head'], 'final head initializer differs')
            load_trainable_state_dict(model, last['trainable_state'])
            trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
            require(len(trainable) == 152 and sum(p.numel() for _,p in trainable) == 87849216, 'incorrect full visual trainability')
            require({n for n,_ in trainable} == set(last['trainable_state']) == set(final['trainable_state']), 'last/final state names differ')
            for name, parameter in trainable:
                require(torch.equal(parameter, last['trainable_state'][name]) and torch.equal(parameter, final['trainable_state'][name]) and bool(torch.isfinite(parameter).all()), f'restored parameter differs: {name}')
            params = [p for _,p in trainable]
            optimizer = torch.optim.AdamW(params, lr=1.75e-6, weight_decay=.01)
            optimizer.load_state_dict(last['optimizer'])
            compare_optimizer(torch, optimizer, params, last['optimizer'], 152)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
            scheduler.load_state_dict(last['scheduler'])
            require(scheduler.last_epoch == scheduler.T_max == 20, 'visual scheduler did not restore')
            scaler = torch.amp.GradScaler('cpu', enabled=bool(last['scaler']))
            scaler.load_state_dict(last['scaler'])
            require(scaler.state_dict() == last['scaler'], 'visual scaler did not restore')
            for state in [h1,last]:
                saved = random.getstate(); random.setstate(state['python_rng_state']); random.setstate(saved)
                saved = np.random.get_state(); value = numpy_rng_tuple(state['numpy_rng_state'], np); np.random.set_state(value)
                restored = np.random.get_state()
                require(restored[0] == value[0] and np.array_equal(restored[1],value[1]) and restored[2:] == value[2:], 'NumPy RNG did not restore exactly')
                np.random.set_state(saved)
                saved = torch.random.get_rng_state(); torch.random.set_rng_state(state['torch_rng_state']); torch.random.set_rng_state(saved)
                require(all(value.dtype == torch.uint8 for value in state['cuda_rng_state_all']), 'CUDA RNG serialization invalid')
            confidence = last['confidence_weighting']
            labels = [int(r['internal_class_index']) for r in official_rows]
            require(confidence['sample_indices'].tolist() == list(range(148695)) and confidence['class_ids'].tolist() == labels, 'AC sample/label binding differs')
            ac = ClasswiseHistoryWeighting(confidence['sample_indices'],confidence['class_ids'],momentum=.8,history_mix=.5,min_weight=.02,magnitude_mix=.75)
            ac.load_state_dict(confidence)
            require(ac.epochs_updated == 20 and torch.equal(ac.weights, confidence['weights']) and bool(torch.isfinite(ac.weights).all()), 'AC state did not restore')
            transform = build_split_transforms(TransformConfig(image_size=336,resize_size=336,train_crop_scale_min=.8,train_horizontal_flip_probability=.5))['validation']
            model.eval()
            forward = []
            for row in [official_rows[0], next(r for r in official_rows if r['internal_class_index'] != official_rows[0]['internal_class_index'])]:
                budget()
                relative = Path(row['relative_path'])
                require(row['source_split'] == 'train' and not relative.is_absolute() and '..' not in relative.parts, 'unsafe non-train image')
                image_path = REPO/'DataSetStage2/train'/relative
                require(image_path.is_file() and sha(image_path) == row['sha256'], 'train image SHA differs')
                with Image.open(image_path) as image: tensor = transform(image.convert('RGB')).unsqueeze(0)
                with torch.inference_mode(): logits = model(tensor)
                require(tuple(logits.shape) == (1,750) and bool(torch.isfinite(logits).all()), 'CPU train forward failed')
                forward.append(dict(sample_id=row['sample_id'], image_sha256=row['sha256'], shape=[1,750], finite=True))
            budget()
        result = dict(complete=True, actual=time.time(), duration_seconds=time.time()-started, input_sha256=actual_hashes, script_sha256=script_sha,
            python=sys.version, torch=torch.__version__, peak_rss_bytes=resource_info()['peak_rss_bytes'], cpu_threads=2,
            full_clip_model_reconstructed_locally=True, train_forward=forward, test_image_forward=False, no_training=True, no_gpu=True,
            trainable_tensors=152, trainable_parameters=87849216, epoch=20, ac_epochs=20, full_train_count=148695,
            optimizer_each_moment_equal=True, head_optimizer_each_moment_equal=True, scheduler_scaler_restored=True,
            cpu_rng_restored=True, cuda_rng_serialization_only=True, bitwise_cuda_resume_claim=False,
            fetched_files_reverified=len(fetched['files']), full_source_manifest_sha256=inventory['full']['source_manifest_sha256'],
            delivery_source_manifest_sha256=inventory['delivery']['source_manifest_sha256'])
        save(result_name, result)
        save('fv2_source_inventory.json', inventory)
        print(json.dumps(dict(complete=True, cpu_receipt=str(result_path), duration_seconds=result['duration_seconds'], no_gpu=True),ensure_ascii=False),flush=True)
        return result
    except BaseException as error:
        save(f'cpu_failure_{fingerprint[:16]}.json', dict(actual=time.time(), type=type(error).__name__, message=str(error), stage=str(stage_path), complete=False))
        raise


def compare_optimizer(torch, optimizer, params, saved, expected_count):
    require(len(optimizer.state) == expected_count and len(optimizer.param_groups) == 1, 'optimizer state count differs')
    ids = saved['param_groups'][0]['params']
    require(len(ids) == len(params), 'optimizer parameter order length differs')
    for index, parameter in enumerate(params):
        reference = saved['state'][ids[index]]
        restored = optimizer.state[parameter]
        require(set(reference) == set(restored), 'optimizer moment keys differ')
        for key,value in restored.items():
            if isinstance(value,torch.Tensor):
                require(torch.equal(value,reference[key]) and bool(torch.isfinite(value).all()), 'optimizer moment not equal/finite')
            else: require(value == reference[key], 'optimizer scalar state differs')
        require(restored['exp_avg'].shape == restored['exp_avg_sq'].shape == parameter.shape, 'optimizer moment shapes differ')


def resource_info():
    """Read Windows memory/self-process identity without an optional dependency."""
    import ctypes
    from ctypes import wintypes
    require(os.name == 'nt', 'this local acceptance resource probe is Windows only')
    class Memory(ctypes.Structure):
        _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [(name,ctypes.c_ulonglong) for name in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended']]
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('faults',wintypes.DWORD)] + [(name,ctypes.c_size_t) for name in ['peak_working','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile']]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel.GetCurrentProcess()
    memory = Memory(); memory.length = ctypes.sizeof(memory)
    require(kernel.GlobalMemoryStatusEx(ctypes.byref(memory)), 'memory probe failed')
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    require(api.GetProcessMemoryInfo(handle,ctypes.byref(counters),ctypes.sizeof(counters)), 'self RSS probe failed')
    times = [wintypes.FILETIME() for _ in range(4)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    require(kernel.GetProcessTimes(handle,*[ctypes.byref(value) for value in times]), 'self creation-time probe failed')
    creation = ((times[0].dwHighDateTime << 32) + times[0].dwLowDateTime) / 10000000 - 11644473600
    return dict(available_bytes=int(memory.avail_phys),rss_bytes=int(counters.working),peak_rss_bytes=int(counters.peak_working),process_create_time=creation)


def numpy_rng_tuple(value, np):
    """Accept the two actual safe schemas used by H1 and visual checkpoints."""
    if isinstance(value,dict):
        require(set(value) == {'bit_generator','keys','position','has_gauss','cached_gaussian'}, 'unexpected safe NumPy RNG dict')
        keys = value['keys'].detach().cpu().numpy() if hasattr(value['keys'],'detach') else value['keys']
        return (value['bit_generator'],np.asarray(keys,dtype=np.uint32),int(value['position']),int(value['has_gauss']),float(value['cached_gaussian']))
    require(isinstance(value,(tuple,list)) and len(value) == 5, 'unexpected NumPy RNG tuple')
    return (value[0],np.asarray(value[1],dtype=np.uint32),int(value[2]),int(value[3]),float(value[4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['package','sources','cpu'])
    parser.add_argument('--package', type=Path, default=REPO/'submissions/fv2_full_exploratory_20261008')
    parser.add_argument('--expected-checkpoint-sha256', default=EXPECTED['final'])
    args = parser.parse_args()
    if args.action == 'package':
        result = validate_package(args.package, OFFICIAL/'test_samples.csv', OFFICIAL/'classes.csv', args.expected_checkpoint_sha256)
        save('package_acceptance_'+result['manifest_sha256'][:16]+'.json', result)
        print(json.dumps(result,ensure_ascii=False))
    elif args.action == 'sources':
        result = verify_sources(); save('fv2_source_inventory.json', result)
        print(json.dumps(dict(complete=True,full_files=len(result['full']['files']),delivery_files=len(result['delivery']['files']))))
    else: restore_cpu()


if __name__ == '__main__': main()
