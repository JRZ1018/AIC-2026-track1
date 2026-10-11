"""P1 source/asset/lease checks, keeping the original atomic queue implementation."""
import json
import os
import shutil
import time
from pathlib import Path
from p1_contract import ROOT, assets, sha, verify_files, verify_remote_assets, validate_config, selection

def h1_receipt():
    return {"full_train_only":True,"validation_count":0}

def entry(root,phase,resume,config=None):
    import yaml
    from aictrack1.engine.training_queue import read_training_lock
    if Path(root).resolve()!=ROOT or phase!="train" or config is None:
        raise RuntimeError("P1 exact source root/train phase required")
    if sha(ROOT/"source_manifest.json") != os.environ.get("P1_SOURCE_SHA"):
        raise RuntimeError("P1 frozen source binding required")
    manifest=verify_files();payload=verify_remote_assets()
    if manifest["status"]!="ready_requires_positive_selection": raise RuntimeError("full unbound source forbidden")
    config=Path(config).resolve();relative=config.relative_to(ROOT).as_posix()
    arm=config.stem.removeprefix("p1_full_")
    raw=yaml.safe_load(config.read_text())["experiment"]
    validate_config(raw,arm,payload)
    choice=selection(os.environ.get("P1_SELECTION_PATH",""),os.environ.get("P1_SELECTION_SHA",""))
    if choice['selected_arm']!=arm:raise RuntimeError('full arm differs from positive selection')
    if int(os.environ.get('P1_INITIAL_FREE','0'))!=choice['original_initial_free_bytes']:raise RuntimeError('cumulative disk baseline changed')
    import torch
    if (os.environ.get('OMP_NUM_THREADS')!='4' or os.environ.get('MKL_NUM_THREADS')!='4'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8' or torch.get_num_threads()!=4):
        raise RuntimeError('full startup runtime must use explicit OMP4/MKL4/CUBLAS workspace')
    for attempt in range(50):
        record=read_training_lock(payload["canonical_lock"])
        if (record and record.token==os.environ.get("P1_LEASE_TOKEN")
                and record.trainer_pid==os.getpid() and record.queue_pid==os.getppid()
                and record.config==relative): break
        if attempt==49: raise RuntimeError("exact canonical trainer lease mismatch")
        time.sleep(.1)
    output=(config.parent/raw["output_dir"]).resolve()
    if resume is not None and Path(resume).resolve()!=output/"last.pt":
        raise RuntimeError("resume must use this arm's unchanged config and exact last.pt")
    _resource_check(0)

def tensor_storage_bytes(value):
    import torch
    seen=set();total=0
    def visit(item):
        nonlocal total
        if isinstance(item,torch.Tensor):
            storage=item.untyped_storage();key=(storage.data_ptr(),storage.nbytes(),str(item.device))
            if key not in seen: seen.add(key);total+=storage.nbytes()
        elif isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,(tuple,list)):
            for child in item:visit(child)
    visit(value)
    return total

def _resource_check(reserve):
    deadline=float(os.environ.get("P1_DEADLINE","0"))
    initial=int(os.environ.get("P1_INITIAL_FREE","0"))
    diskmax=int(os.environ.get("P1_DISK_MAX",str(20*1024**3)))
    minimum=int(os.environ.get("P1_MIN_FREE",str(3*1024**3)))
    free=shutil.disk_usage(ROOT).free
    if (time.time()>=deadline or initial<=0 or free-reserve<minimum or initial-free+reserve>diskmax):
        raise RuntimeError("P1 deadline/disk reserve exceeded")

def reserve_save(payload):
    _resource_check(tensor_storage_bytes(payload)+64*1024**2)
