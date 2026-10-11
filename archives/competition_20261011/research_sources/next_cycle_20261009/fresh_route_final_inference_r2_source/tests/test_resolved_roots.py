"""Run the actual old/new entry guards against a real filesystem root alias."""
from __future__ import annotations
import importlib.util,json,os,subprocess,sys,time,unittest,uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import torch
import audit_full_checkpoint as current_audit
import run_final as current_run

class ReachedNextCheck(RuntimeError):pass

def old_module(name):
    path=ROOT/'references/final_62367'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('old_62367_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@contextmanager
def real_root_alias():
    folder=ROOT.parent/'operator_tools'/('resolved_root_fixture_'+uuid.uuid4().hex)
    actual=folder/'mounted_data';actual.mkdir(parents=True,exist_ok=False)
    alias=folder/'autodl-tmp';foreign=folder/'foreign';foreign.mkdir()
    if os.name=='nt':
        quote=lambda p:"'"+str(p).replace("'","''")+"'"
        command='New-Item -ItemType Junction -Path '+quote(alias)+' -Target '+quote(actual)+' -ErrorAction Stop | Out-Null'
        done=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,timeout=20)
        if done.returncode:raise RuntimeError('real junction creation failed: '+done.stderr)
        if not alias.is_junction():raise RuntimeError('fixture requires an actual directory junction')
        kind='Windows_directory_junction'
    else:
        alias.symlink_to(actual,target_is_directory=True)
        if not alias.is_symlink():raise RuntimeError('fixture requires an actual directory symlink')
        kind='POSIX_directory_symlink'
    if alias==actual or alias.resolve()!=actual.resolve():raise RuntimeError('real filesystem alias was not constructed')
    (folder/'fixture_evidence.json').write_text(json.dumps(dict(kind=kind,literal_alias=str(alias),resolved_alias=str(alias.resolve()),actual_root=str(actual),foreign_root=str(foreign),actual_server_failure_claim=False))+'\n',encoding='utf-8')
    # Keep bounded local evidence. Never recursively delete a junction or target.
    yield actual,alias,foreign

class ResolvedRootTests(unittest.TestCase):
    def test_actual_audit_guard_old_alias_failure_new_exact_root_and_foreign_rejection(self):
        torch.set_num_threads(4)
        old=old_module('audit_full_checkpoint')
        with real_root_alias() as (actual,alias,foreign):
            args=SimpleNamespace(full_root=alias,source_sha256='0'*64)
            stop=Mock(side_effect=ReachedNextCheck('before checkpoint/source load'))
            with patch.object(old,'FULL_ROOT',alias),patch.object(old,'sys',SimpleNamespace(platform='linux')),patch.object(old,'verify_source'),patch.object(old,'verify_full_source',stop):
                with self.assertRaisesRegex(ValueError,'independent remote FULL root'):old.audit(args)
                stop.assert_not_called()
            with patch.object(current_audit,'FULL_ROOT',alias),patch.object(current_audit,'sys',SimpleNamespace(platform='linux')),patch.object(current_audit,'verify_source'),patch.object(current_audit,'verify_full_source',stop):
                with self.assertRaises(ReachedNextCheck):current_audit.audit(args)
                stop.assert_called_once_with(actual.resolve());stop.reset_mock()
                with self.assertRaisesRegex(ValueError,'independent remote FULL root'):current_audit.audit(SimpleNamespace(full_root=foreign,source_sha256='0'*64))
                stop.assert_not_called()

    def test_actual_controller_guard_old_alias_failure_new_exact_root_and_foreign_rejection(self):
        old=old_module('run_final')
        with real_root_alias() as (actual,alias,foreign):
            argv=['run_final.py','--source-sha256','0'*64,'--cpu-audit',str(actual/'artifacts/full_final_cpu_audit.json'),'--cpu-audit-sha256','0'*64,'--source-git-commit','0'*40,'--deadline-unix',str(time.time()+60),'--initial-free-bytes','13013950464','--disk-max-bytes','8589934592']
            stop=Mock(side_effect=ReachedNextCheck('before source/model/process access'))
            with patch.object(sys,'argv',argv),patch.object(old,'ROOT',actual),patch.object(old,'REMOTE_ROOT',alias),patch.object(old,'sys',SimpleNamespace(platform='linux')),patch.object(old,'verify_source',stop):
                with self.assertRaisesRegex(ValueError,'independent Linux inference ROOT'):old.main()
                stop.assert_not_called()
            with patch.object(sys,'argv',argv),patch.object(current_run,'ROOT',actual),patch.object(current_run,'REMOTE_ROOT',alias),patch.object(current_run,'sys',SimpleNamespace(platform='linux')),patch.object(current_run,'verify_source',stop):
                with self.assertRaises(ReachedNextCheck):current_run.main()
                stop.assert_called_once_with(expected='0'*64);stop.reset_mock()
            with patch.object(sys,'argv',argv),patch.object(current_run,'ROOT',foreign),patch.object(current_run,'REMOTE_ROOT',alias),patch.object(current_run,'sys',SimpleNamespace(platform='linux')),patch.object(current_run,'verify_source',stop):
                with self.assertRaisesRegex(ValueError,'independent Linux inference ROOT'):current_run.main()
                stop.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
