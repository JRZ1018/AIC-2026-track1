from pathlib import Path
import hashlib, json, zipfile
from lxml import etree
from pypdf import PdfReader

root = Path(r'D:\aicomp\AICtrack1')
out = root / 'reports' / 'review_20261008'
out.mkdir(parents=True, exist_ok=True)
inventory=[]
for p in sorted((root/'docs').rglob('*')):
    if not p.is_file(): continue
    b=p.read_bytes()
    inventory.append({'path':p.relative_to(root).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'lines':len(b.decode('utf-8-sig').splitlines()) if p.suffix in ['.md','.txt','.json','.yaml'] else None})
(out/'source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
reader=PdfReader(root/'docs'/'赛题规则.pdf')
text='\n\n'.join(f'PAGE {i+1}\n'+(p.extract_text() or '') for i,p in enumerate(reader.pages))
(out/'rules_pdf_text.txt').write_text(text,encoding='utf-8')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(root/'docs'/'技术报告.docx') as z:
    parts=[n for n in z.namelist() if n.startswith('word/') and n.endswith('.xml') and any(k in n for k in ['document.xml','header','footer','footnotes','endnotes','comments'])]
    texts=[]
    for name in parts:
        xml=etree.fromstring(z.read(name))
        texts.append('PART '+name+'\n'+'\n'.join(''.join(p.xpath('.//w:t/text()',namespaces=ns)) for p in xml.xpath('//w:p',namespaces=ns)))
    images=[n for n in z.namelist() if n.startswith('word/media/')]
    for name in images:
        (out/Path(name).name).write_bytes(z.read(name))
(out/'technical_report_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
print(json.dumps({'docs_files':len(inventory),'docs_bytes':sum(x['bytes'] for x in inventory),'pdf_pages':len(reader.pages),'pdf_text_chars':len(text),'docx_media':images},ensure_ascii=False))
