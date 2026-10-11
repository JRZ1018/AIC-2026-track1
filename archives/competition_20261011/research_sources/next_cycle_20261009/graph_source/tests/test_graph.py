"""Scientific small CPU fixtures, not a 29k graph or a claim of score gain."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import argparse,copy,inspect,io,json,math,shutil,subprocess,sys,time,unittest,uuid
from contextlib import contextmanager,redirect_stdout
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.sparse import csr_matrix
from unittest.mock import patch
from p3_graph import blocked_topk,mutual_graph,smooth,p7,MARGIN,BudgetExceeded
from p3_contract import FEATURE_PROTOCOL,VIEW_PROTOCOL,validate_feature_payload,order_sha,verify_source,sha
torch.set_num_threads(2)

@contextmanager
def fixture():
    path=ROOT/'artifacts'/('fixture-'+uuid.uuid4().hex);path.mkdir(parents=True)
    try:yield path
    finally:
        if not path.resolve().is_relative_to((ROOT/'artifacts').resolve()):raise RuntimeError('fixture escape')
        shutil.rmtree(path)

class Tests(unittest.TestCase):
    def test_directed_reciprocal_and_zero_degree(self):
        angles=np.deg2rad([0,10,25,180]);features=np.c_[np.cos(angles),np.sin(angles)].astype(np.float32)
        features=np.r_[features,np.zeros((1,2),np.float32)]
        ns,ws,stats=blocked_topk(features,1,row_block=2,max_block_bytes=40)
        self.assertEqual(ns[2,0],1);self.assertEqual(ns[1,0],0)
        transition,info,degree=mutual_graph(ns,ws,1)
        self.assertEqual(transition[0,1],1);self.assertEqual(transition[1,0],1);self.assertEqual(transition[2,1],0)
        self.assertEqual(degree.tolist(),[1,1,0,0,0]);self.assertEqual(info['undirected_edges'],1)
        self.assertEqual(stats['zero_feature_rows'],1);self.assertLess(stats['peak_similarity_block_bytes'],len(features)**2*4)
        scores=np.arange(15,dtype=np.float32).reshape(5,3)*.1
        out,report=smooth(scores,transition,degree,.3)
        self.assertTrue(np.array_equal(out[degree==0],scores[degree==0]));self.assertEqual(report['isolated_raw_prediction_changes'],0)

    def test_ties_blocks_and_scalar_cosine_reference(self):
        features=np.array([[1,0],[1,0],[1,0],[-1,0]],np.float32)
        first=blocked_topk(features,1,row_block=1);second=blocked_topk(features,1,row_block=2)
        self.assertTrue(np.array_equal(first[0],second[0]));self.assertEqual(first[0][:3,0].tolist(),[1,0,0])
        self.assertGreater(first[2]['tie_cutoff_rows'],0)
        angles=np.deg2rad([0,15,45,120]);f=np.c_[np.cos(angles),np.sin(angles)].astype(np.float32)
        ns,ws,_=blocked_topk(f,2,row_block=2)
        scalar=f.astype(np.float64)@f.astype(np.float64).T;np.fill_diagonal(scalar,-np.inf)
        for i in range(len(f)):
            reference=np.lexsort((np.arange(len(f)),-scalar[i]))[:2]
            self.assertTrue(np.array_equal(ns[i],reference));self.assertTrue(np.allclose(ws[i],scalar[i,reference],atol=1e-6))
        transition,info,degree=mutual_graph(ns,ws,2);adj=np.zeros((len(f),len(f)))
        for i in range(len(f)):
            for j in ns[i]:
                if i in ns[j] and scalar[i,j]>0:adj[i,j]=scalar[i,j]
        expected=np.divide(adj,adj.sum(1,keepdims=True),out=np.zeros_like(adj),where=adj.sum(1,keepdims=True)>0)
        self.assertTrue(np.allclose(transition.toarray(),expected,atol=1e-6));self.assertTrue(info['symmetric_adjacency'])

    def test_three_synchronous_original_residual_iterations_native_temperature(self):
        values=np.array([[.6,0],[0,.3]],np.float32);w=csr_matrix(np.array([[0,1],[1,0]],np.float32));degree=np.array([1,1])
        out,report=smooth(values,w,degree,.3)
        base=torch.softmax(torch.from_numpy(values),1).numpy();expected=base.copy();wrong=base.copy()
        for _ in range(3):expected=.7*base+.3*(w@expected);wrong=.7*wrong+.3*(w@wrong)
        actual=torch.softmax(torch.from_numpy(out),1).numpy()
        self.assertTrue(np.allclose(actual,expected,atol=2e-7));self.assertGreater(float(np.abs(actual-wrong).max()),1e-3)
        cold=torch.softmax(torch.from_numpy(values)/.5,1).numpy()
        for _ in range(3):cold=.7*torch.softmax(torch.from_numpy(values)/.5,1).numpy()+.3*(w@cold)
        self.assertGreater(float(np.abs(actual-cold).max()),.01);self.assertEqual(report['original_probability_temperature'],1.)
        identity,_=smooth(values,w,degree,0.);self.assertTrue(np.array_equal(identity,values))
        with self.assertRaises(ValueError):smooth(values,w,degree,.3,iterations=2)

    def test_k10_is_refiltered_from_directed_top20_not_mutual_top20(self):
        rng=np.random.default_rng(4);features=rng.normal(size=(41,12)).astype(np.float32)
        features/=np.linalg.norm(features,axis=1,keepdims=True)
        ns20,ws20,_=blocked_topk(features,20,row_block=8);ns10,ws10,_=blocked_topk(features,10,row_block=8)
        self.assertTrue(np.array_equal(ns20[:,:10],ns10))
        derived,_,degree1=mutual_graph(ns20,ws20,10);separate,_,degree2=mutual_graph(ns10,ws10,10)
        self.assertTrue(np.array_equal(degree1,degree2));self.assertTrue(np.array_equal(derived.toarray(),separate.toarray()))

    def test_high_margin_exact_scores_zeros_and_labels_absent(self):
        values=np.array([[5,0,-200],[.1,.3,-200]],np.float32);w=csr_matrix(np.array([[0,1],[1,0]],np.float32));degree=np.array([1,1])
        out,report=smooth(values,w,degree,.3)
        self.assertTrue(np.array_equal(out[0],values[0]));self.assertEqual(report['protected_raw_prediction_changes'],0)
        self.assertEqual(report['margin_change_abs_max_protected'],0);self.assertGreater(report['initial_zero_probability_entries'],0)
        self.assertNotIn('targets',inspect.signature(smooth).parameters);self.assertNotIn('labels',inspect.signature(blocked_topk).parameters)
        first=out.copy();fake_labels=np.array([1,0]);fake_labels[:]=0
        again,_=smooth(values,w,degree,.3);self.assertTrue(np.array_equal(first,again))
        base=np.array([[.3,.1],[.1,.7],[.4,.1]],np.float32)
        self.assertTrue(np.array_equal(p7(base)[0],p7(base+np.array([2,3,-1],np.float32)[:,None])[0]))

    def test_metadata_same_checkpoint_order_and_no_extra_forward(self):
        feature=torch.zeros(4,512);feature[:,0]=1;ids=['a','b','c','d'];target=torch.tensor([0,1,0,1])
        common=dict(checkpoint_sha256='1'*64,experiment_config_sha256='2'*64,split_sha256='3'*64,
            manifest_sha256='4'*64,class_mapping_sha256='5'*64,view_protocol=VIEW_PROTOCOL)
        payload=dict(schema='aictrack1.stage2_train_derived_4v_features.v1',features=feature,sample_ids=ids,targets=target,
            feature_protocol=FEATURE_PROTOCOL,observed_layout='four_calls_B',view_order=['native','native_exact_flip','scaled384_crop336','scaled_exact_flip'],
            checkpoint_role='last',checkpoint_epoch=20,additional_model_forwards=0,test_access=False,**common)
        logits=dict(logits=torch.ones(4,2),sample_ids=ids,targets=target,**common);identity=dict(sample_ids_order_sha256=order_sha(ids),**common)
        original,_=validate_feature_payload(payload,logits,identity)
        changed=copy.deepcopy(payload);other=copy.deepcopy(logits);changed['targets']=1-target;other['targets']=1-target
        after,_=validate_feature_payload(changed,other,identity);self.assertTrue(np.array_equal(original,after))
        for mutation in [dict(additional_model_forwards=1),dict(test_access=True),dict(checkpoint_sha256='9'*64),
                dict(sample_ids=list(reversed(ids))),dict(features=feature.half())]:
            bad=copy.deepcopy(payload);bad.update(mutation)
            with self.assertRaises(ValueError):validate_feature_payload(bad,logits,identity)

    def test_deadline_singletons_and_aggregate_budget(self):
        with self.assertRaises(BudgetExceeded):blocked_topk(np.eye(4,dtype=np.float32),2,deadline=time.monotonic()-1)
        ns,ws,report=blocked_topk(np.array([[1,0]],np.float32),20)
        self.assertEqual(ns.shape,(1,0));self.assertEqual(report['peak_similarity_block_bytes'],0)
        from p3_budget import BudgetLease
        with fixture() as path:
            ledger=path/'budget.json'
            with BudgetLease(ledger):
                with self.assertRaises(FileExistsError):
                    with BudgetLease(ledger):pass
            before=json.loads(ledger.read_text())['spent_wall_seconds']
            with BudgetLease(ledger) as second:self.assertLessEqual(second.remaining,7200-before)
            self.assertGreaterEqual(json.loads(ledger.read_text())['spent_wall_seconds'],before)
            state=json.loads(ledger.read_text());state['spent_wall_seconds']=7200;ledger.write_text(json.dumps(state))
            with self.assertRaises(RuntimeError):
                with BudgetLease(ledger):pass
            self.assertFalse(ledger.with_suffix('.json.lock').exists())

    def test_real_driver_four_points_with_small_injected_cache_and_frozen_source(self):
        import run_graph
        rng=np.random.default_rng(7);n=80;target=np.arange(n)%4
        f=np.eye(512,dtype=np.float32)[target]+rng.normal(0,.0001,(n,512)).astype(np.float32);f/=np.linalg.norm(f,axis=1,keepdims=True)
        scores=np.zeros((n,4),np.float32);scores[np.arange(n),target]=.7
        bad=np.arange(0,n,5);scores[bad,target[bad]]=.2;scores[bad,(target[bad]+1)%4]=.3
        ids=[f'fixture-{i}' for i in range(n)];protocols=[dict(name=name,indices=np.arange(n),columns=np.arange(4),targets=target,
            groups=ids,ids=ids,train_counts=np.full(4,100)) for name in ('natural_unique_images','balanced_supported_unique')]
        data=dict(features=f,scores=scores,protocols=protocols,identity=dict(checkpoint_sha256='1'*64,
            experiment_config_sha256='2'*64,split_sha256='0c902ea38c04581eb33ef4b26fac2d37a09abb8a4fad4ae0893b89e989ac584c',
            sample_ids_order_sha256=order_sha(ids)),precision=dict(note='INJECTED SYNTHETIC FIXTURE ONLY'))
        with fixture() as path:
            args=argparse.Namespace(output=path/'output',assets=path/'unit-unused',assets_sha256='3'*64,stage='explore',
                choice=None,choice_sha256=None,max_wall_seconds=30,budget_ledger=path/'budget.json')
            with patch.object(run_graph,'load_assets',return_value=data),redirect_stdout(io.StringIO()):self.assertEqual(run_graph.run(args),0)
            report=json.loads((args.output/'report.json').read_text());manifest=json.loads((args.output/'result_manifest.json').read_text())
            self.assertTrue(report['complete']);self.assertEqual(len(report['evaluated_grid']),4);self.assertTrue(manifest['complete'])
            for protocol in report['protocols']:
                self.assertEqual(len(protocol['candidates']),4)
                self.assertLess(protocol['neighbor_info']['peak_similarity_block_bytes'],n*n*4)
            self.assertEqual(report['actual_runtime']['torch_num_threads'],2)
            self.assertTrue((args.output/'graph_choice.json').is_file())
            choice=args.output/'graph_choice.json';old_spent=json.loads(args.budget_ledger.read_text())['spent_wall_seconds']
            confirm=copy.copy(args);confirm.output=path/'confirmation';confirm.stage='confirm';confirm.choice=choice;confirm.choice_sha256=sha(choice)
            fresh=copy.deepcopy(data);fresh['identity']['split_sha256']='57189c0bd5bf195fc9d582663eb1b31282448e82ab99ca068c76c53097a658b7'
            with patch.object(run_graph,'load_assets',return_value=fresh),redirect_stdout(io.StringIO()):self.assertEqual(run_graph.run(confirm),0)
            confirmation=json.loads((confirm.output/'report.json').read_text())
            self.assertEqual(len(confirmation['evaluated_grid']),1);self.assertGreater(json.loads(args.budget_ledger.read_text())['spent_wall_seconds'],old_spent)

    def test_frozen_raw_manifest_and_cli_interfaces(self):
        verify_source()
        for name in ('run_graph.py','bind_graph_assets.py'):
            result=subprocess.run([sys.executable,str(ROOT/name),'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt=dict(schema='aictrack1.p3.cpu_test_receipt.v1',status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        graph_protocol_sha256=sha(ROOT/'graph_protocol.json'),torch_num_threads=torch.get_num_threads(),
        synthetic_max_rows=80,real_29737_graph_executed=False,server_access=False,cuda_used=False,test_access=False)
    (ROOT/'CPU_TEST_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
