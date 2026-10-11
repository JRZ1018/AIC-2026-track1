"""One persistent two-hour wall ledger for P3 exploration plus confirmation."""
import json,math,os,time,uuid
from pathlib import Path
from p3_contract import ROOT,sha

class BudgetLease:
    def __init__(self,path):
        self.path=Path(path).resolve();self.lock=self.path.with_suffix(self.path.suffix+'.lock');self.token=uuid.uuid4().hex
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(self.lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump(dict(token=self.token,pid=os.getpid(),started_unix=time.time()),f)
        try:
            self.payload=json.loads(self.path.read_text()) if self.path.exists() else dict(schema='aictrack1.p3.aggregate_budget.v1',
                source_manifest_sha256=sha(ROOT/'source_manifest.json'),total_wall_seconds=7200.,spent_wall_seconds=0.)
            if (self.payload.get('schema')!='aictrack1.p3.aggregate_budget.v1'
                    or self.payload.get('source_manifest_sha256')!=sha(ROOT/'source_manifest.json')
                    or self.payload.get('total_wall_seconds')!=7200.
                    or not isinstance(self.payload.get('spent_wall_seconds'),(int,float))
                    or not math.isfinite(self.payload['spent_wall_seconds']) or self.payload['spent_wall_seconds']<0):
                raise ValueError('original shared P3 budget identity differs')
            self.remaining=7200.-self.payload['spent_wall_seconds']
            if self.remaining<=0:raise RuntimeError('P3 aggregate budget already exhausted')
            self.started=time.monotonic();return self
        except BaseException:
            self.lock.unlink();raise
    def __exit__(self,typ,value,tb):
        elapsed=time.monotonic()-self.started;self.payload['spent_wall_seconds']+=elapsed
        self.payload['last_run_status']='complete' if typ is None else 'failed_or_interrupted'
        temp=self.path.with_name(self.path.name+'.'+self.token+'.tmp')
        temp.write_text(json.dumps(self.payload,indent=2)+'\n',encoding='utf-8');temp.replace(self.path)
        current=json.loads(self.lock.read_text())
        if current.get('token')!=self.token:raise RuntimeError('P3 lease changed; do not remove')
        self.lock.unlink()
