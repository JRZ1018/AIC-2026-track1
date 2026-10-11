"""Actual S loss/optimizer/resume/lease and numerical observability fixtures; CPU only."""
import ast,copy,json,os,subprocess,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'scripts'),str(ROOT/'src')]
from cpu_smoke import Tests,fixture_directory
import torch,yaml
from torch import nn
from train_n1_peft import _base_loss_per_sample,_weighted_base_loss
from supervision_runtime import assert_frozen_head,clip_diagnostics,validate_runtime,RUNTIME_ENV,bind_budget,INITIAL_FREE_BYTES
from p1_contract import assets,validate_assets,validate_config,verify_files,sha
from aictrack1.engine.ema import optimizer_step_completed

class SupervisionTests(Tests):
    def test_alpha_is_only_ce_gradient_increment(self):
        logits=torch.tensor([[-4.,2.,1.],[4.,-2.,-1.]],requires_grad=True);labels=torch.tensor([0,0])
        weights=torch.tensor([.02,1.])
        def gradient(alpha):
            per=_base_loss_per_sample(logits,labels,kind='sce',q=.7,label_smoothing=0.,sce_alpha=alpha)
            return torch.autograd.grad(_weighted_base_loss(per,weights),logits,retain_graph=True)[0]
        ce=torch.autograd.grad(_weighted_base_loss(torch.nn.functional.cross_entropy(logits,labels,reduction='none'),weights),logits)[0]
        torch.testing.assert_close(gradient(.5)-gradient(.1),.4*ce,rtol=2e-5,atol=1e-7)

    def test_clip_coefficient_exact_epsilon_and_nonfinite_exclusion(self):
        row=clip_diagnostics(torch.tensor([0.,.5,1.,2.,float('nan'),float('inf')]),1.,4,2)
        self.assertEqual(row['clip_coefficient_finite_norm_count'],4)
        self.assertEqual(row['clip_coefficient_nonfinite_excluded'],2)
        self.assertAlmostEqual(row['clip_coefficient_quantiles']['0.0'],1/(2+1e-6),places=6)
        self.assertEqual(row['clip_coefficient_quantiles']['1.0'],1.)
        self.assertEqual(row['amp_successful_optimizer_updates'],4);self.assertEqual(row['amp_skipped_optimizer_updates'],2)
        empty=clip_diagnostics(torch.tensor([float('inf')]),1.,0,1)
        self.assertIsNone(empty['clip_coefficient_quantiles'])

    def test_amp_skip_observation_does_not_change_epoch_scheduler(self):
        self.assertFalse(optimizer_step_completed(scaler_enabled=True,before=65536.,after=32768.))
        self.assertTrue(optimizer_step_completed(scaler_enabled=True,before=32768.,after=32768.))
        self.assertTrue(optimizer_step_completed(scaler_enabled=True,before=32768.,after=65536.))
        self.assertTrue(optimizer_step_completed(scaler_enabled=False,before=1.,after=1.))
        source=(ROOT/'scripts/train_n1_peft.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('scheduler.step()'),1)

    def test_strict_152_meta_shape_and_frozen_head(self):
        model=nn.Module();model.visual=nn.ParameterList([nn.Parameter(torch.empty(1,device='meta')) for _ in range(151)]+[nn.Parameter(torch.empty(87849216-151,device='meta'))])
        model.classifier=nn.Linear(2,3)
        for p in model.classifier.parameters():p.requires_grad_(False)
        reference={k:v.detach().clone() for k,v in model.classifier.state_dict().items()}
        assert_frozen_head(model,model.classifier,reference)
        with torch.no_grad():model.classifier.bias.add_(1)
        with self.assertRaises(RuntimeError):assert_frozen_head(model,model.classifier,reference)
        model.classifier.load_state_dict(reference)
        model.classifier.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):assert_frozen_head(model,model.classifier,reference)

    def test_runtime_threads_and_environment_fail_closed(self):
        validate_runtime(dict(environment=dict(RUNTIME_ENV),actual_torch_threads=4))
        for threads in (1,64):
            with self.assertRaises(RuntimeError):validate_runtime(dict(environment=dict(RUNTIME_ENV),actual_torch_threads=threads))
        bad=dict(RUNTIME_ENV);bad['CUBLAS_WORKSPACE_CONFIG']=None
        with self.assertRaises(RuntimeError):validate_runtime(dict(environment=bad,actual_torch_threads=4))

    def test_budget_resumes_without_resetting_global_baseline(self):
        with fixture_directory('S-budget-') as name:
            start=time.time()-1;kw=dict(start=start,deadline=start+21600,initial=INITIAL_FREE_BYTES,diskmax=6*1024**3,minimum=3*1024**3)
            value=bind_budget(name,**kw);self.assertEqual(value,bind_budget(name,**kw))
            for mutation in (dict(start=start+.2,deadline=start+21600+.2),dict(initial=INITIAL_FREE_BYTES+1),dict(deadline=start+21601),dict(minimum=2*1024**3)):
                bad=dict(kw);bad.update(mutation)
                with self.assertRaises(ValueError):bind_budget(name,**bad)

    def test_head_or_schedule_or_augmented_method_cannot_drift(self):
        payload=assets();raw=yaml.safe_load((ROOT/'configs/experiment/p1_S.yaml').read_text())['experiment']
        for key,value in [('base_loss',{'kind':'sce','alpha':.1,'beta':1.,'log_zero':-4.}),('persistent_filter',{'enabled':True}),('visual_prototype_weighting',{'enabled':True})]:
            bad=copy.deepcopy(raw);bad[key]=value
            with self.assertRaises(ValueError):validate_config(bad,'S',payload)
        for key,value in [('head_learning_rate',5e-6),('epochs',8),('seed',20261009)]:
            bad=copy.deepcopy(raw);bad['training'][key]=value
            with self.assertRaises(ValueError):validate_config(bad,'S',payload)
        for key,value in [('split', '1'*64),('h1','2'*64)]:
            bad=copy.deepcopy(payload);bad['assets'][key]['sha256']=value
            with self.assertRaises(ValueError):validate_assets(bad)

    def test_real_cpu_cli_help_and_parse(self):
        for script in ('run_supervision.py','train_n1_peft.py'):
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_supervision.py'),'--arm','J20','--deadline-unix','1','--phase-start-unix','1'],capture_output=True,text=True,timeout=60)
        self.assertNotEqual(result.returncode,0);self.assertIn('invalid choice',result.stderr)

    def test_frozen_manifest_and_smoke_follow_real_alpha(self):
        self.assertEqual(verify_files()['status'],'bound')
        source=(ROOT/'scripts/p1_smoke.py').read_text(encoding='utf-8');ast.parse(source)
        self.assertIn("sce_alpha=raw['base_loss']['alpha']",source)
        self.assertIn('head_restored_equal=True',source)
        self.assertIn('trainable_state_dict(model)',source)

if __name__=='__main__':
    torch.set_num_threads(2)
    start=time.time()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SupervisionTests))
    receipt=dict(schema='aictrack1.supervision.cpu_smoke.v1',status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),duration_seconds=time.time()-start,
        python=sys.executable,torch=torch.__version__,cuda_used=False,server_access=False,test_access=False,
        source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        scope='real loss/AC/RNG/model/AdamW/Cosine20/scaler disk resume; canonical lease mocked CPU launcher;152 meta shape/frozen head; scientific alpha/clip/AMP fixtures;actual CLI')
    (ROOT/'cpu_smoke_result.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
