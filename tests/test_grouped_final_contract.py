import copy, unittest
from aictrack1.evaluation.grouped_final_contract import validate_grouped_final, bound_full_train_rows
class Vector:
    def __init__(self, x): self.x = x
    def tolist(self): return list(self.x)
def fixture(n=2):
    p=dict(full_count=3,selected_count=n,test_access=False)
    h=[dict(epoch=i,validation={},training_pool=dict(protocol=p,physical_exposures=3,
       unique_ac_observations=n,attempted_optimizer_steps=1859,optimizer_steps=1858,
       amp_overflow_steps=1)) for i in range(1,21)]
    c=dict(competition_stage='stage2',config_sha256='cfg',train_manifest_sha256='m',
       split_sha256='s',class_mapping_sha256='map',num_classes=750,epoch=20,history=h,
       training_pool_protocol=p,confidence_weighting=dict(sample_indices=Vector(range(n)),
       class_ids=Vector(range(n)),epochs_updated=20,seen=Vector([1]*n)))
    m=dict(training_scope='fixed_split_validation',config_sha256='cfg',manifest_sha256='m',
       split_sha256='s',train_count=n,validation_count=1,history=h)
    k=dict(manifest_sha256='m',split_sha256='s',class_mapping_sha256='map',num_classes=750,
       train_count=3,validation_count=1,expected_total_count=4,protocol=p,
       selected_global_indices=list(range(n)),selected_labels=list(range(n)))
    return c,m,dict(training=dict(epochs=20)),k
class Tests(unittest.TestCase):
    def test_canonical_split_derives_partition_without_mutation(self):
        rows=[dict(sample_id=str(i),source_split='train',internal_class_index=i) for i in range(4)]
        before=copy.deepcopy(rows);parts={str(i):('validation' if i==3 else 'train') for i in range(4)}
        out=bound_full_train_rows(rows,parts,expected_total_count=4,expected_train_count=3,expected_validation_count=1)
        self.assertEqual(rows,before);self.assertEqual(len(out),3)
        self.assertTrue(all(r['partition']=='train' for r in out))
        self.assertEqual([r['internal_class_index'] for r in out],[0,1,2])
    def test_reject_unknown_missing_split_and_nontrain_source(self):
        for wrong in ('missing','source','partition'):
            rows=[dict(sample_id=str(i),source_split='train') for i in range(4)]
            parts={str(i):('validation' if i==3 else 'train') for i in range(4)}
            if wrong=='missing':parts.pop('0')
            elif wrong=='source':rows[0]['source_split']='test'
            else:rows[0]['partition']='validation'
            with self.assertRaises(ValueError):
                bound_full_train_rows(rows,parts,expected_total_count=4,expected_train_count=3,expected_validation_count=1)
    def test_reject_duplicate_manifest_id(self):
        rows=[dict(sample_id='0',source_split='train') for i in range(4)]
        with self.assertRaises(ValueError):
            bound_full_train_rows(rows,{'0':'train'},expected_total_count=4,expected_train_count=3,expected_validation_count=1)
    def test_subset_and_full_pass_without_changing_counts(self):
        for n in (2,3):
            c,m,e,k=fixture(n);before=copy.deepcopy(m)
            validate_grouped_final(c,m,e,**k);self.assertEqual(m,before)
    def test_reject_full_scope(self):
        c,m,e,k=fixture();c['training_scope']='full_train'
        with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
    def test_reject_fake_full_train_count(self):
        c,m,e,k=fixture();m['train_count']=3
        with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
    def test_reject_same_count_wrong_global_ids(self):
        c,m,e,k=fixture();c['confidence_weighting']['sample_indices']=Vector([0,3])
        with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
    def test_reject_same_count_wrong_labels(self):
        c,m,e,k=fixture();c['confidence_weighting']['class_ids']=Vector([1,0])
        with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
    def test_reject_missing_history_or_coverage(self):
        for change in ('history','physical_exposures','unique_ac_observations','attempted_optimizer_steps'):
            c,m,e,k=fixture()
            if change=='history':c['history']=c['history'][:-1]
            else:c['history'][0]['training_pool'][change]-=1
            with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
    def test_reject_source_and_protocol_drift(self):
        for key in ('split_sha256','class_mapping_sha256','train_manifest_sha256','config_sha256'):
            c,m,e,k=fixture();c[key]='wrong'
            with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
    def test_reject_unseen_or_wrong_epoch(self):
        for key,val in (('seen',Vector([1,0])),('epochs_updated',19)):
            c,m,e,k=fixture();c['confidence_weighting'][key]=val
            with self.assertRaises(ValueError): validate_grouped_final(c,m,e,**k)
if __name__=='__main__': unittest.main()
