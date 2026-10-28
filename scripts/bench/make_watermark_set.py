"""Builds a synthetic watermark benchmark from clean photos.

The clean photo is the ground truth, so removal quality can be measured exactly. Writes, per case, raw RGBA files
(`<case>.clean.rgba`, `<case>.marked.rgba`, `<case>.mask.bin`) plus `manifest.json`.

    python -I scripts/bench/make_watermark_set.py <photos_dir> <out_dir> [count] [edge]

Three watermark kinds are composited: tiled diagonal translucent text, an opaque corner stamp, and a large centred
translucent mark. The mask is the pixels the watermark actually changed, grown by 2 px (what a good detector would hand over).
"""
import json
import math
import os
import random
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FONT = r'C:\Windows\Fonts\arialbd.ttf'
KINDS = ('tiled', 'corner', 'center')


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def layer_for(kind, size, rng):
    """RGBA layer holding only the watermark."""
    w, h = size
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if kind == 'tiled':
        f = font(max(18, w // 28))
        text = 'SAMPLE'
        tile = Image.new('RGBA', (w * 2, h * 2), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)
        step_x, step_y = w // 4, h // 5
        for y in range(0, h * 2, step_y):
            for x in range(0, w * 2, step_x):
                td.text((x, y), text, font=f, fill=(255, 255, 255, 110))
        layer = tile.rotate(30, resample=Image.BICUBIC).crop((w // 2, h // 2, w // 2 + w, h // 2 + h))
    elif kind == 'corner':
        f = font(max(20, w // 22))
        text = 'BRANDNAME.COM'
        bw = d.textlength(text, font=f)
        d.text((w - bw - w * 0.03, h - f.size * 1.6), text, font=f, fill=(255, 255, 255, 255))
    else:
        f = font(max(40, w // 7))
        text = 'PREVIEW'
        bw = d.textlength(text, font=f)
        d.text(((w - bw) / 2, h * 0.45), text, font=f, fill=(255, 255, 255, 120))
    return layer


def main():
    src, out = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    edge = int(sys.argv[4]) if len(sys.argv) > 4 else 768
    rng = random.Random(7)
    files = sorted(
        os.path.join(root, f)
        for root, _, names in os.walk(src)
        for f in names
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    rng.shuffle(files)
    os.makedirs(out, exist_ok=True)
    cases = []
    for i, path in enumerate(files[:count]):
        kind = KINDS[i % len(KINDS)]
        im = Image.open(path).convert('RGB')
        im.thumbnail((edge, edge), Image.LANCZOS)
        layer = layer_for(kind, im.size, rng)
        marked = Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB')
        diff = ImageChops.difference(im, marked).convert('L').point(lambda v: 255 if v > 2 else 0)
        mask = diff.filter(ImageFilter.MaxFilter(5))
        name = '%02d_%s' % (i, kind)
        w, h = im.size
        with open(os.path.join(out, name + '.clean.rgba'), 'wb') as fh:
            fh.write(im.convert('RGBA').tobytes())
        with open(os.path.join(out, name + '.marked.rgba'), 'wb') as fh:
            fh.write(marked.convert('RGBA').tobytes())
        with open(os.path.join(out, name + '.mask.bin'), 'wb') as fh:
            fh.write(mask.point(lambda v: 1 if v else 0).tobytes())
        marked.save(os.path.join(out, name + '.marked.png'))
        area = sum(mask.point(lambda v: 1 if v else 0).tobytes()) / (w * h) * 100
        cases.append({'name': name, 'kind': kind, 'w': w, 'h': h, 'source': os.path.relpath(path, src), 'maskPercent': round(area, 2)})
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        json.dump(cases, fh, indent=1)
    print(len(cases), 'cases ->', out)


if __name__ == '__main__':
    main()
