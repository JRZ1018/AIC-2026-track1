"""Verify the complete official-layout L35 work and its submitted CSV/ZIP."""
import argparse,csv,io,json,os,re,zipfile
from pathlib import Path,PurePosixPath
import release_common as common

ARCHIVE_REL='AIC-2026-43080930-可执行模型文件或容器化复现环境/container-image.tar'
CHECKSUM_REL=ARCHIVE_REL+'.sha256'
EXCLUDED={ARCHIVE_REL,CHECKSUM_REL}

def contained_file(relative):
    if not isinstance(relative,str) or '\\' in relative or ':' in relative:
        raise ValueError('Manifest paths must be root-relative POSIX paths')
    clean=PurePosixPath(relative)
    if clean.is_absolute() or clean.as_posix()!=relative or not clean.parts or any(x in {'.','..'} for x in clean.parts):
        raise ValueError('Invalid manifest path: '+relative)
    path=(common.PACKAGE/relative).resolve()
    if not path.is_relative_to(common.PACKAGE) or not path.is_file():
        raise ValueError('File escapes or is absent from the work root: '+relative)
    return path

def inventory_files(manifest):
    result=set()
    # The operator may add the official dataset and new run outputs later.
    # Exclude only these two root-level dynamic directories.
    for base,dirs,files in os.walk(common.PACKAGE,topdown=True,followlinks=False):
        base=Path(base)
        kept=[]
        for name in dirs:
            directory=base/name
            relative=directory.relative_to(common.PACKAGE)
            if name=='__pycache__' or (len(relative.parts)==1 and name in {'runtime_outputs','DataSetStage2'}):continue
            if directory.is_symlink() or getattr(directory,'is_junction',lambda:False)():
                raise ValueError('Release links are not accepted: '+relative.as_posix())
            kept.append(name)
        dirs[:]=kept
        for name in files:
            file=base/name
            relative=file.relative_to(common.PACKAGE)
            if file.is_symlink():raise ValueError('Release links are not accepted: '+relative.as_posix())
            if file==manifest or file.suffix=='.pyc' or relative.as_posix() in EXCLUDED:continue
            result.add(relative.as_posix())
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=common.CODE/'RELEASE_SHA256.json')
    args=p.parse_args()
    manifest=args.manifest.resolve()
    if not manifest.is_relative_to(common.PACKAGE) or not manifest.is_file():
        raise ValueError('Release manifest must be a file inside the work root')
    allowed={common.MODELS.name,common.CODE.name,common.LIBRARY.name,'DataSetStage2','runtime_outputs'}
    extra=[p.name for p in common.PACKAGE.iterdir() if not p.is_dir() or p.name not in allowed]
    if extra:raise ValueError('Unexpected root entries: '+str(sorted(extra)))
    inventory=json.loads(manifest.read_text(encoding='utf-8'))
    entries=inventory['files']
    expected={x['path'] for x in entries}
    if len(expected)!=len(entries):raise ValueError('Duplicate release manifest paths')
    actual=inventory_files(manifest)
    if actual!=expected:raise ValueError('Release missing/extra files: '+str(sorted(actual^expected)[:10]))
    for item in entries:
        path=contained_file(item['path'])
        if path.stat().st_size!=item['bytes'] or common.sha(path)!=item['sha256']:
            raise ValueError('File mismatch: '+item['path'])
    output=common.RESULTS
    raw=(output/'pred_results.csv').read_bytes()
    if common.sha(output/'pred_results.csv')!=common.CSV_SHA:raise ValueError('CSV SHA mismatch')
    if common.sha(common.MODELS/'final.pt')!=common.FINAL_SHA:raise ValueError('L35 final checkpoint SHA mismatch')
    if common.sha(common.MODELS/'openai_clip_ViT-B-32.pt')!=common.OFFICIAL_SHA:raise ValueError('Official CLIP SHA mismatch')
    rows=list(csv.reader(io.StringIO(raw.decode('utf-8'))))
    names={r['relative_path'] for r in csv.DictReader((common.DATA/'test_manifest.csv').open(encoding='utf-8-sig',newline=''))}
    classes={r['official_class_id'] for r in csv.DictReader((common.DATA/'class_mapping.csv').open(encoding='utf-8-sig',newline=''))}
    if len(rows)!=37444 or len({r[0] for r in rows})!=37444 or {r[0] for r in rows}!=names or len(classes)!=750:
        raise ValueError('Official image/class contract mismatch')
    if any(len(r)!=2 or r[1] not in classes or re.fullmatch(r'\d{4}',r[1]) is None for r in rows):raise ValueError('CSV format mismatch')
    with zipfile.ZipFile(output/'pred_results.zip') as z:
        if z.namelist()!=['pred_results.csv'] or z.testzip() is not None or z.read('pred_results.csv')!=raw:
            raise ValueError('Submission ZIP member/CRC/bytes mismatch')
    archive=common.MODELS/'container-image.tar'
    checksum=common.MODELS/'container-image.tar.sha256'
    if archive.exists() or checksum.exists():
        if not archive.is_file() or not checksum.is_file():raise ValueError('Container archive and SHA sidecar must appear together')
        fields=checksum.read_text(encoding='ascii').strip().split()
        if len(fields)!=2 or re.fullmatch(r'[0-9a-f]{64}',fields[0]) is None or fields[1]!='container-image.tar':
            raise ValueError('Invalid container archive SHA sidecar')
        if common.sha(archive)!=fields[0]:raise ValueError('Container archive SHA mismatch')
    print(json.dumps({'container_archive_sha_pass':True if archive.exists() else None,'complete':True,'sha_files':len(expected),
        'rows':37444,'classes':750,'submission_zip_crc_pass':True,'csv_sha256':common.CSV_SHA,
        'model_sha256':common.FINAL_SHA,'layout':'three required categories; shared root-relative resources'},ensure_ascii=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
