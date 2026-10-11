"""CPU-only synthetic geometric/no-op/resume/config tests. No model or dataset images."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,io,json,math,sys,time,unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from aictrack1.data.contracts import TransformConfig,_parse_center_guard_fraction,load_dataset_config
from aictrack1.data.transforms import ImageTransform,_guarded_crop_coordinate,center_guard_identity,validate_center_guard_resume
from aictrack1.methods.sce import symmetric_cross_entropy
from p1_contract import ARMS,assets,expected_config,verify_files
from fresh_route_runtime import budget
# The original producer transform is frozen in the parent manifest. The source
# copy below is byte-identical, and uses the same current dataclass optional defaults.
spec=importlib.util.spec_from_file_location('aictrack1.data._CG_original_transform',ROOT/'inputs/original_transforms.py')
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
torch.set_num_threads(4)
def image(w,h):
    y,x=np.indices((h,w));return Image.fromarray(np.stack([(x*13+y)%256,(x+y*7)%256,(x*3+y*5)%256],axis=-1).astype('uint8'))
def rng(seed):return np.random.default_rng(np.random.SeedSequence([seed,3,17]))
class Recorded:
    def __init__(self,seed):self.inner=rng(seed);self.calls=[]
    def uniform(self,*a):self.calls.append(('uniform',a));return self.inner.uniform(*a)
    def integers(self,*a):self.calls.append(('integers',a));return self.inner.integers(*a)
    def random(self):self.calls.append(('random',()));return self.inner.random()
def config(margin=None):return TransformConfig(image_size=336,resize_size=336,center_guard_fraction=margin)
def state(r):return json.dumps(r.bit_generator.state,sort_keys=True)
DIMS=[(321,321),(384,256),(256,384),(960,120),(120,960),(1,1),(1,5),(5,1),(4,2),(2,4)]
CASES={}
class CPU(unittest.TestCase):
    def test_01_real_source_and_original_transform(self):
        m=verify_files(require_bound=False);self.assertEqual(m['arms'],['CG25']);self.assertEqual(ARMS,{'CG25':.25})
        self.assertEqual(hashlib.sha256((ROOT/'inputs/original_transforms.py').read_bytes()).hexdigest(),'cebea0fc613280a87c45c65943a9434a3cbc09bc8d850a705b5ab62ca8ee54e2')
    def test_02_disabled_pixels_and_RNG_30cases(self):
        for w,h in DIMS:
            for seed in [0,1,20261008]:
                raw=image(w,h);ra,rb=rng(seed),rng(seed)
                a=original.ImageTransform('train',config())(raw,rng=ra);b=ImageTransform('train',config())(raw,rng=rb)
                self.assertTrue(torch.equal(a,b));self.assertEqual(state(ra),state(rb))
        CASES['disabled_synthetic_cases']=30
    def test_03_explicit_zero_pixels_and_RNG_30cases(self):
        for w,h in DIMS:
            for seed in [0,1,20261008]:
                raw=image(w,h);ra,rb=rng(seed),rng(seed)
                a=original.ImageTransform('train',config())(raw,rng=ra);b=ImageTransform('train',config(0.))(raw,rng=rb)
                self.assertTrue(torch.equal(a,b));self.assertEqual(state(ra),state(rb))
        CASES['explicit_zero_synthetic_cases']=30
    def test_04_enabled_identical_RNG_calls_bounds_state(self):
        changed=0
        for w,h in DIMS:
            for seed in [0,1,20261008]:
                raw=image(w,h);ra,rb=Recorded(seed),Recorded(seed)
                a=original.ImageTransform('train',config())(raw,rng=ra);b=ImageTransform('train',config(.25))(raw,rng=rb)
                self.assertEqual(ra.calls,rb.calls);self.assertEqual(state(ra.inner),state(rb.inner));self.assertTrue(torch.isfinite(b).all());self.assertEqual(tuple(b.shape),(3,336,336));changed+=not torch.equal(a,b)
        self.assertGreater(changed,0);CASES['enabled_synthetic_changed_cases']=int(changed);CASES['enabled_same_RNG_cases']=30
    def test_05_geometry_monotonic_interval_and_endpoints(self):
        checked=0
        for dimension in range(1,65):
            for side in range(1,dimension+1):
                span=dimension-side;lo=max(0,math.ceil(dimension/2-.75*side));hi=min(span,math.floor(dimension/2-.25*side))
                values=[_guarded_crop_coordinate(i,dimension,side,.25) for i in range(span+1)]
                self.assertEqual(values,sorted(values));self.assertTrue(all(0<=i<=span for i in values))
                if lo<=hi:
                    self.assertEqual(values[0],lo);self.assertEqual(values[-1],hi)
                    self.assertTrue(all(.25*side<=dimension/2-i<=.75*side for i in values))
                else:self.assertTrue(all(i==min(span,max(0,round(span/2))) for i in values))
                checked+=1
        CASES['integer_axis_geometries']=checked
    def test_06_large_square_exact_noop(self):
        for dim in [321,336,512]:
            for seed in [0,1,20261008]:
                raw=image(dim,dim);self.assertTrue(torch.equal(original.ImageTransform('train',config())(raw,rng=rng(seed)),ImageTransform('train',config(.25))(raw,rng=rng(seed))))
    def test_07_validation_unchanged_pixels(self):
        for w,h in DIMS:
            self.assertTrue(torch.equal(original.ImageTransform('validation',config())(image(w,h)),ImageTransform('validation',config(.25))(image(w,h))))
    def test_08_train_requires_explicit_RNG(self):
        with self.assertRaises(ValueError):ImageTransform('train',config(.25))(image(32,24))
    def test_09_parser_valid_and_actual_dataset(self):
        for value in [None,0,.25]:self.assertEqual(_parse_center_guard_fraction(value),value)
        self.assertEqual(load_dataset_config(ROOT/'configs/data/p1_336.yaml').transforms,config(.25))
    def test_10_parser_rejects_invalid(self):
        for value in [True,False,'.25',{},float('nan'),float('inf'),-.1,.5,1.]:
            with self.assertRaises(ValueError):_parse_center_guard_fraction(value)
    def test_11_augmentation_identity_weights_only_roundtrip(self):
        identity=center_guard_identity(config(.25),dataset_sha256='d'*64,source_sha256='s'*64);payload={'center_guard_augmentation':identity}
        buf=io.BytesIO();torch.save(payload,buf);buf.seek(0);validate_center_guard_resume(torch.load(buf,weights_only=True),identity)
    def test_12_augmentation_mismatches_rejected(self):
        identity=center_guard_identity(config(.25),dataset_sha256='d'*64,source_sha256='s'*64)
        for key,value in [('center_guard_fraction',0.),('dataset_config_sha256','e'*64),('transform_source_sha256','t'*64),('train_crop_scale_min',.95),('stochastic_seed_protocol','new')]:
            bad=copy.deepcopy(identity);bad[key]=value
            with self.assertRaises(ValueError):validate_center_guard_resume({'center_guard_augmentation':bad},identity)
        with self.assertRaises(ValueError):validate_center_guard_resume({},identity)
    def test_13_disabled_real_SCE_AC_weighted_loss_gradient_exact(self):
        tree=ast.parse((ROOT/'scripts/train_n1_peft.py').read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_weighted_base_loss')
        ns={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),'real_weighted_loss','exec'),ns)
        baseline=original.ImageTransform('train',config())(image(384,256),rng=rng(20261008));zero=ImageTransform('train',config(0.))(image(384,256),rng=rng(20261008))
        torch.manual_seed(20261008);initial=torch.randn(750,12);labels=torch.tensor([1,749]);weights=torch.tensor([1.,.02])
        results=[]
        for view in [baseline,zero]:
            features=torch.cat([view[:,::84,::84].reshape(-1)[:12].unsqueeze(0),view[:,::84,::84].reshape(-1)[-12:].unsqueeze(0)])
            weight=initial.clone().requires_grad_(True);logits=features@weight.T;per=symmetric_cross_entropy(logits,labels,alpha=.1,beta=1.,log_zero=-4.,reduction='none');loss=ns['_weighted_base_loss'](per,weights);loss.backward();results.append((loss.detach(),weight.grad.detach(),logits.detach()))
        for a,b in zip(*results):self.assertTrue(torch.equal(a,b))
        CASES['loss_gradient_scope']='actual SCE and actual AC weight normalization with synthetic 750-way linear head; not CLIP recovery'
    def test_14_training_config_single_variable_control(self):
        p=assets();actual=yaml.safe_load((ROOT/'configs/experiment/p1_CG25.yaml').read_text())['experiment'];self.assertEqual(actual,expected_config('CG25',p))
        baseline=yaml.safe_load((ROOT/'inputs/baseline_J5.yaml').read_text())['experiment'];actual=copy.deepcopy(actual)
        for k in ['run_id','output_dir']:actual[k]=baseline[k]
        self.assertEqual(actual,baseline);self.assertEqual(actual['base_loss']['alpha'],.1)
        for forbidden in ['persistent_filter','trainable_ema','josrc_soft_repair','attention_feature_augmentation']:self.assertNotIn(forbidden,actual)
        self.assertEqual(actual['confidence_observation_view'],'strong')
    def test_15_same_scientific_training_loop_and_source_closure(self):
        actual=(ROOT/'scripts/train_n1_peft.py').read_text();original_trainer=(ROOT/'inputs/original_train_n1_peft.py').read_text()
        actual=actual.replace('    from aictrack1.data.transforms import center_guard_identity, validate_center_guard_resume\n','').replace('    center_guard_augmentation = center_guard_identity(dataset_config.transforms, dataset_sha256=_sha256(dataset_path), source_sha256=_sha256(ROOT / "src/aictrack1/data/transforms.py"))\n','').replace('        validate_center_guard_resume(state, center_guard_augmentation)\n','').replace('            "center_guard_augmentation": center_guard_augmentation,\n','').replace('        "center_guard_augmentation": center_guard_augmentation,\n','')
        self.assertEqual(actual,original_trainer)
        m=verify_files(require_bound=False);parent=json.loads((ROOT/'inputs/parent_joint_source_manifest.json').read_text())
        for rel in ['src/aictrack1/engine/training_queue.py','scripts/owned_launch.py','scripts/launch_gate.py','scripts/mainline_guard.py','scripts/p1_smoke.py','src/aictrack1/data/indexed_images.py','src/aictrack1/methods/confidence_observation.py','src/aictrack1/methods/confidence.py','src/aictrack1/methods/sce.py']:
            self.assertEqual(m['files'][rel],parent['files'][rel])
    def test_16_prepared_budget_and_source_cannot_execute(self):
        with self.assertRaises(ValueError):budget(ROOT)
        with self.assertRaises(ValueError):verify_files(require_bound=True)
        m=verify_files(require_bound=False);self.assertIsNone(m['future_budget_sha256']);self.assertIsNone(m['future_last_sha256']);self.assertIsNone(m['future_admission_sha256'])
if __name__=='__main__':
    begin=time.time();suite=unittest.defaultTestLoader.loadTestsFromTestCase(CPU);result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(schema='aictrack1.CG25.author_CPU.result.v1',status='passed' if result.wasSuccessful() else 'failed',actual_tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),elapsed_seconds=time.time()-begin,synthetic_image_only=True,official_image_reads=0,model_reads=0,GPU_started=False,server_staged=False,test_access=False,**CASES)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
