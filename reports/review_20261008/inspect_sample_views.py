from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import random, json, hashlib

root=Path(r'D:\aicomp\AICtrack1')
out=root/'reports'/'review_20261008'
files=sorted(p for p in (root/'DataSetStage2'/'test').iterdir() if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png'])
chosen=random.Random(20261008).sample(files,12)
canvas=Image.new('RGB',(1600,1800),'#eeeeee')
draw=ImageDraw.Draw(canvas)
records=[]
for i,p in enumerate(chosen):
    with Image.open(p) as raw:
        im=raw.convert('RGB')
    w,h=im.size
    r=336/min(w,h)
    resized=im.resize((round(w*r),round(h*r)),Image.Resampling.BICUBIC)
    left=(resized.width-336)//2;top=(resized.height-336)//2
    crop=resized.crop((left,top,left+336,top+336))
    x=(i%4)*400;y=(i//4)*600
    full=ImageOps.contain(im,(390,260),Image.Resampling.LANCZOS)
    canvas.paste(full,(x+(400-full.width)//2,y+35+(260-full.height)//2))
    crop=crop.resize((280,280),Image.Resampling.LANCZOS)
    canvas.paste(crop,(x+60,y+310))
    draw.text((x+10,y+8),f'{i+1:02} original {w}x{h} / center crop',fill='black')
    records.append({'index':i+1,'file':p.name,'width':w,'height':h,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
canvas.save(out/'unlabeled_test_12_views.jpg',quality=93)
(out/'unlabeled_test_12_manifest.json').write_text(json.dumps({'seed':20261008,'purpose':'read-only composition audit; no labels, model, gradients or parameter selection','samples':records},ensure_ascii=False,indent=2),encoding='utf-8')
print('Decoded 12 fixed samples; saved original and center-crop comparison.')
