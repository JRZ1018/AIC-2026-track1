"""Local engineering freeze; deploy only listed raw files, never re-freeze live code."""
import ast,datetime,json,platform
from pathlib import Path
from p3_contract import ROOT,sha

def main():
    import numpy,torch,scipy
    files={};metadata={}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if (p.is_file() and not {'artifacts','__pycache__'}.intersection(rel.parts)
                and p.name not in {'source_manifest.json','CPU_TEST_RECEIPT.json'} and p.suffix in ('.py','.json','.md')):
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
            files[rel.as_posix()]=sha(p);metadata[rel.as_posix()]=dict(bytes=p.stat().st_size)
    manifest=dict(schema='aictrack1.p3.graph_source.v1',status='cpu_engineering_only_not_candidate_result',files=files,file_metadata=metadata,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),graph_protocol_sha256=sha(ROOT/'graph_protocol.json'),
        dependencies=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,torch=torch.__version__),
        real_graph_executed=False,test_access=False,model_updates=0,third_party_source_copied=False)
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(dict(source_manifest_sha256=sha(ROOT/'source_manifest.json'),files=len(files),bytes=sum(v['bytes'] for v in metadata.values()))))
if __name__=='__main__':main()
