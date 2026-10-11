"""Meaningful corruption checks; synthetic fixtures, no official image inference."""
import csv
import hashlib
import io
import json
import uuid
import unittest
import zipfile
from pathlib import Path

import accept_fv2 as acceptance


class PackageContract(unittest.TestCase):
    def setUp(self):
        self.root = acceptance.OUT / 'test_fixtures'
        self.root.mkdir(exist_ok=True)
        self.fixture = (self.root/('synthetic-'+uuid.uuid4().hex)).resolve()
        self.fixture.mkdir()
        self.package = self.fixture/'package'; self.package.mkdir()
        self.test = self.fixture/'test.csv'
        self.mapping = self.fixture/'classes.csv'
        names = [f'synthetic-{i:05d}.jpg' for i in range(37444)]
        with self.test.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n')
            writer.writerow(['relative_path','source_split','internal_class_index','official_class_id'])
            writer.writerows((name,'test','','') for name in names)
        with self.mapping.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n')
            writer.writerow(['official_class_id','internal_class_index','class_name'])
            writer.writerows((f'{i:04d}',i,'') for i in range(750))
        payload=''.join(f'{name},{i%750:04d}\n' for i,name in enumerate(names)).encode()
        self.csv=self.package/'pred_results.csv'; self.csv.write_bytes(payload)
        self.zip=self.package/'pred_results.zip'
        with zipfile.ZipFile(self.zip,'w',compression=zipfile.ZIP_DEFLATED) as archive: archive.writestr('pred_results.csv',payload)
        _,mapping_sha=acceptance.mapping_identity(self.mapping)
        self.manifest=dict(spec=dict(csv_name='pred_results.csv',include_header=False,filename_mode='basename',official_class_id_pattern=r'\d{4}'),
            record_count=37444,csv_sha256=acceptance.sha(self.csv),zip_sha256=acceptance.sha(self.zip),
            class_mapping_sha256=mapping_sha,expected_filenames_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest(),
            run_metadata=dict(run_id='synthetic-not-event-data',checkpoint_sha256='1'*64,config_hash='2'*64,
                              data_hash=acceptance.sha(self.test),test_manifest_sha256=acceptance.sha(self.test),
                              git_commit='3'*40,inference_command='synthetic fixture, never run',class_mapping_sha256=mapping_sha))
        self.flush_manifest()

    def tearDown(self):
        acceptance.require(self.fixture.is_relative_to(self.root.resolve()),'temporary cleanup escaped fixture root')
        for path in [self.csv,self.zip,self.package/'submission_manifest.json',self.test,self.mapping]:
            path.unlink(missing_ok=True)
        self.package.rmdir()
        self.fixture.rmdir()

    def flush_manifest(self):
        (self.package/'submission_manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')

    def validate(self):
        return acceptance.validate_package(self.package,self.test,self.mapping,'1'*64)

    def test_valid_synthetic_contract(self):
        self.assertTrue(self.validate()['complete'])

    def test_wrong_order_rejected(self):
        lines=self.csv.read_bytes().splitlines(keepends=True);lines[0],lines[1]=lines[1],lines[0]
        self.csv.write_bytes(b''.join(lines))
        with self.assertRaisesRegex(ValueError,'filename order'): self.validate()

    def test_unknown_id_rejected(self):
        self.csv.write_bytes(self.csv.read_bytes().replace(b',0000\n',b',9999\n',1))
        with self.assertRaisesRegex(ValueError,'official label'): self.validate()

    def test_missing_row_rejected(self):
        lines=self.csv.read_bytes().splitlines(keepends=True);self.csv.write_bytes(b''.join(lines[:-1]))
        with self.assertRaisesRegex(ValueError,'row count'): self.validate()

    def test_bom_rejected(self):
        self.csv.write_bytes(b'\xef\xbb\xbf'+self.csv.read_bytes())
        with self.assertRaisesRegex(ValueError,'BOM'): self.validate()

    def test_nested_zip_rejected(self):
        with zipfile.ZipFile(self.zip,'w') as archive: archive.writestr('nested/pred_results.csv',self.csv.read_bytes())
        with self.assertRaisesRegex(ValueError,'one root CSV'): self.validate()

    def test_wrong_checkpoint_rejected(self):
        self.manifest['run_metadata']['checkpoint_sha256']='4'*64;self.flush_manifest()
        with self.assertRaisesRegex(ValueError,'wrong model'): self.validate()

    def test_missing_provenance_rejected(self):
        del self.manifest['run_metadata']['inference_command'];self.flush_manifest()
        with self.assertRaisesRegex(ValueError,'provenance is missing'): self.validate()


class SafeRNG(unittest.TestCase):
    def test_actual_safe_numpy_schemas_roundtrip(self):
        import numpy as np
        import torch
        original=np.random.get_state()
        seeded=np.random.RandomState(20261009).get_state()
        values=[(seeded[0],seeded[1].tolist(),*seeded[2:]),
                dict(bit_generator=seeded[0],keys=torch.from_numpy(seeded[1].astype(np.int64)),position=seeded[2],has_gauss=seeded[3],cached_gaussian=seeded[4])]
        try:
            for value in values:
                parsed=acceptance.numpy_rng_tuple(value,np);np.random.set_state(parsed);actual=np.random.get_state()
                self.assertEqual(actual[0],seeded[0]);self.assertTrue(np.array_equal(actual[1],seeded[1]));self.assertEqual(actual[2:],seeded[2:])
        finally:np.random.set_state(original)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    acceptance.save('acceptance_contract_tests_'+acceptance.sha(__file__)[:16]+'.json',dict(complete=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),synthetic_only=True,test_source_sha256=acceptance.sha(__file__),acceptance_source_sha256=acceptance.sha(acceptance.__file__)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
