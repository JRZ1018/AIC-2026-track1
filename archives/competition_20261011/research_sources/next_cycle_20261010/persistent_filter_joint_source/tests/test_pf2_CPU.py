from pathlib import Path
import ast,copy,hashlib,importlib.util,json,random,sys,tempfile,unittest
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from aictrack1.methods.persistent_filter import PersistentTrainingGate as Gate,ActualSupervisionLedger as Ledger,gated_weighted_loss
from p1_contract import ARMS,assets,expected_config,validate_config,verify_files
from fresh_route_runtime import budget

def fixture(weight=.2,threshold=.3):
    ids=torch.arange(80);labels=torch.cat([torch.zeros(40,dtype=torch.long),torch.ones(40,dtype=torch.long)])
    weights=torch.ones(80);weights[:8]=weight;weights[40:48]=weight
    a=torch.ones(80);a[:8]=0;a[40:48]=0;s=torch.ones(80)
    return Gate(ids,labels,max_ac_weight=threshold),weights,a,s

def update(g,w,a,s,n=3):
    for _ in range(n):g.update(g.sample_indices,w,a,s)

class PF2(unittest.TestCase):
 def test_01_actual_assets_config_budget(self):
    p=assets();r=expected_config('PF30',p);validate_config(r,'PF30',p)
    self.assertEqual(ARMS,{'PF30':.3});self.assertEqual(r['base_loss'],{'kind':'sce','alpha':.1,'beta':1.,'log_zero':-4.})
    self.assertEqual(r['peft']['mode'],'full_visual');self.assertEqual(r['training']['head_learning_rate'],5e-6)
    self.assertEqual(budget(ROOT)['deadline_unix']-budget(ROOT)['phase_start_unix'],10800.)
    bad=copy.deepcopy(r);bad['base_loss']['alpha']=.3
    with self.assertRaises(ValueError):validate_config(bad,'PF30',p)
    with self.assertRaises(ValueError):expected_config('PF50',p)
 def test_02_original_v1_parity(self):
    parent=ROOT.parent/'supervision_joint_r3_source/src/aictrack1/methods/persistent_filter.py'
    spec=importlib.util.spec_from_file_location('old_pf1',parent);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    g,w,a,s=fixture(weight=.08,threshold=.1);o=old.PersistentTrainingGate(g.sample_indices,g.labels)
    for _ in range(5):
      x=g.update(g.sample_indices,w,a,s);y=o.update(o.sample_indices,w,a,s);self.assertEqual(x,y);self.assertTrue(torch.equal(g.keep,o.keep));self.assertTrue(torch.equal(g.streak,o.streak))
    loss=torch.arange(80,dtype=torch.float32,requires_grad=True);n,_=gated_weighted_loss(loss,w,g.keep);grad=torch.autograd.grad(n,loss)[0]
    l=loss.detach().clone().requires_grad_(True);b,_=old.gated_weighted_loss(l,w,o.keep)
    self.assertTrue(torch.equal(n,b));self.assertTrue(torch.equal(grad,torch.autograd.grad(b,l)[0]))
 def test_03_three_observations_next_and_reentry(self):
    g,w,a,s=fixture();update(g,w,a,s,2);self.assertTrue(g.keep.all());update(g,w,a,s,1);self.assertEqual(int((~g.keep).sum()),8)
    update(g,w,torch.ones_like(a),s,1);self.assertTrue(g.keep.all())
 def test_04_threshold_single_axis(self):
    g,w,a,s=fixture(weight=.4,threshold=.3);update(g,w,a,s);self.assertTrue(g.keep.all())
    h,_,_,_=fixture(weight=.4,threshold=.5);update(h,w,a,s);self.assertEqual(int((~h.keep).sum()),8)
 def test_05_class_count_floor_mass(self):
    ids=torch.arange(36);labels=torch.cat([torch.zeros(16,dtype=torch.long),torch.ones(20,dtype=torch.long)]);g=Gate(ids,labels,max_ac_weight=.3)
    w=torch.full((36,),.2);update(g,w,torch.zeros(36),torch.ones(36));self.assertTrue(g.keep[:16].all());self.assertEqual(int((~g.keep[16:]).sum()),2)
    for c in g.update(ids,w,torch.zeros(36),torch.ones(36))['classes']:self.assertGreaterEqual(c['retained_ac_mass']+1e-10,.9*c['original_ac_mass'])
 def test_06_partition_and_signal_rejection(self):
    g,w,a,s=fixture()
    with self.assertRaises(ValueError):g.keep_for(torch.tensor([80]))
    with self.assertRaises(ValueError):g.update(g.sample_indices.flip(0),w,a,s)
    w[0]=float('nan')
    with self.assertRaises(ValueError):g.update(g.sample_indices,w,a,s)
 def test_07_actual_snapshot_not_next_mask(self):
    g,w,a,s=fixture();led=Ledger(g,w);led.observe(g.sample_indices,g.labels);update(g,w,a,s)
    r=led.finish();self.assertEqual(r['excluded'],0);self.assertEqual(r['removed_AC_mass'],0);self.assertEqual(r['mask_source_completed_epoch'],0)
    actual=Ledger(g,w);actual.observe(g.sample_indices,g.labels);r=actual.finish();self.assertEqual(r['excluded'],8);self.assertAlmostEqual(r['removed_AC_mass'],8*.2,places=6)
    self.assertEqual(sum(c['supervised'] for c in r['classes']),r['supervised']);self.assertEqual(sum(c['original_AC_mass'] for c in r['classes']),r['original_AC_mass'])
 def test_08_actual_duplicate_label_and_incomplete(self):
    g,w,a,s=fixture();led=Ledger(g,w);led.observe(g.sample_indices[:2],g.labels[:2])
    with self.assertRaises(ValueError):led.observe(g.sample_indices[:2],g.labels[:2])
    with self.assertRaises(ValueError):led.finish()
    with self.assertRaises(ValueError):Ledger(g,w).observe(g.sample_indices[:2],torch.ones(2,dtype=torch.long))
 def test_09_all_keep_loss_gradient_equivalence(self):
    from train_n1_peft import _weighted_base_loss
    l=torch.tensor([1.,3.,9.],requires_grad=True);w=torch.tensor([.02,.3,1.]);new,_=gated_weighted_loss(l,w,torch.ones(3,dtype=torch.bool));original=_weighted_base_loss(l,w)
    self.assertTrue(torch.equal(new,original));self.assertTrue(torch.equal(torch.autograd.grad(new,l,retain_graph=True)[0],torch.autograd.grad(original,l)[0]))
 def test_10_no_supervision_no_adamw_step(self):
    p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.1,weight_decay=.01);before=p.detach().clone()
    loss,active=gated_weighted_loss(p.square(),torch.ones(2),torch.zeros(2,dtype=torch.bool))
    if active:loss.backward();opt.step()
    self.assertFalse(active);self.assertTrue(torch.equal(before,p));self.assertFalse(opt.state)
 def test_11_disk_gate_optimizer_nextstep_exact(self):
    torch.manual_seed(91);g,w,a,s=fixture();update(g,w,a,s);model=torch.nn.Linear(3,1);opt=torch.optim.AdamW(model.parameters(),lr=.01);x=torch.randn(80,3)
    def step(m,o,gate):
      o.zero_grad();l=m(x).flatten().square();v,active=gated_weighted_loss(l,w,gate.keep)
      if active:v.backward();o.step()
      return float(v.detach())
    step(model,opt,g)
    with tempfile.TemporaryDirectory(dir=ROOT/'tests') as tmp:
      folder=Path(tmp).resolve();assert folder.is_relative_to((ROOT/'tests').resolve());f=folder/'resume.pt'
      torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),gate=g.state_dict(),rng=torch.get_rng_state()),f)
      expected_loss=step(model,opt,g);expected=copy.deepcopy(model.state_dict());state=torch.load(f,map_location='cpu',weights_only=True)
      other=torch.nn.Linear(3,1);oo=torch.optim.AdamW(other.parameters(),lr=.01);other.load_state_dict(state['model']);oo.load_state_dict(state['optimizer']);h,_,_,_=fixture();h.load_state_dict(state['gate'],ac_weights=w);torch.set_rng_state(state['rng']);actual_loss=step(other,oo,h)
      self.assertEqual(expected_loss,actual_loss)
      for k in expected:self.assertTrue(torch.equal(expected[k],other.state_dict()[k]))
 def test_12_state_threshold_v1_and_ac_corruption(self):
    g,w,a,s=fixture();update(g,w,a,s);state=g.state_dict()
    for key,value in [('protocol','persistent-supervision-gate-v1'),('max_ac_weight',.5)]:
      bad=copy.deepcopy(state);bad[key]=value
      with self.assertRaises(ValueError):g.load_state_dict(bad,ac_weights=w)
    with self.assertRaises(ValueError):g.load_state_dict(state,ac_weights=torch.ones_like(w))
    bad=copy.deepcopy(state);bad['keep'][:10]=False;bad['streak'][:10]=3
    with self.assertRaises(ValueError):g.load_state_dict(bad)
 def test_13_original_lifecycle_and_scientific_closure(self):
    parent=ROOT.parent/'supervision_joint_r3_source';m=json.loads((parent/'source_manifest.json').read_text(encoding='utf-8'))
    for rel in m['files']:
      if rel.startswith('src/') and rel!='src/aictrack1/methods/persistent_filter.py' or rel in ('scripts/owned_launch.py','scripts/launch_gate.py','scripts/train_peft.py','scripts/mainline_h1_contract.py','inputs/baseline_J5.yaml','inputs/p1_assets.json','inputs/h1_actual_CPU_complete.json'):
        self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),m['files'][rel],rel)
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
 def test_14_manifest(self):verify_files()

if __name__=='__main__':unittest.main(verbosity=2)
