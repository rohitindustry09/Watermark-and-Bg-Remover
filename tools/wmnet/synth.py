"""Synthetic training pictures for the watermark detector.

The "Dharma Statues" mark is drawn (Century Gothic: "Dharma" regular and red, "Statues" bold and orange) at a random size, position, transparency and blur
onto crops of statue photos taken from the part of the picture that carries no mark. The answer (which pixels are lettering) is known exactly, so there
is unlimited training data. Half of the crops get no mark at all, so the model also learns what is NOT lettering: carving, red enamel, orange inlay.
"""
from __future__ import annotations

import io
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_REG = os.environ.get("WMNET_FONT_REG", "C:/Windows/Fonts/GOTHIC.TTF")
FONT_BOLD = os.environ.get("WMNET_FONT_BOLD", "C:/Windows/Fonts/GOTHICB.TTF")
WORK = 512  # the working size of a whole photo (the long side); the model is fully convolutional, so any size runs
CROP = 192  # training crop


def load_pool(folder: str, count: int, seed: int = 1) -> list[np.ndarray]:
    """`count` photos, each reduced to WORK x WORK (short side scaled to WORK, centre cropped), as uint8 arrays."""
    rng = random.Random(seed)
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in (".webp", ".jpg", ".jpeg", ".png")]
    rng.shuffle(files)
    pool: list[np.ndarray] = []
    for f in files:
        if len(pool) >= count:
            break
        try:
            with Image.open(f) as im:
                if min(im.size) < 400:
                    continue  # tiny images (the certificate badge) are not statue photos
                im = im.convert("RGB")
                s = WORK / min(im.size)
                im = im.resize((max(WORK, round(im.width * s)), max(WORK, round(im.height * s))), Image.LANCZOS)
                left = (im.width - WORK) // 2
                top = (im.height - WORK) // 2
                pool.append(np.asarray(im.crop((left, top, left + WORK, top + WORK))))
        except Exception:
            continue
    return pool


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render_mark(rng: random.Random, width: int) -> tuple[np.ndarray, np.ndarray]:
    """The mark as (coverage 0..1 HxW, colour HxWx3 uint8), `width` pixels wide, drawn at 3x and reduced so the edges are soft like a real photo's."""
    ss = 3
    size = 100
    gap_font = _font(FONT_REG, size)
    reg = _font(FONT_REG, size)
    bold = _font(FONT_BOLD, size)
    w1 = reg.getlength("Dharma")
    wg = reg.getlength(" ")
    w2 = bold.getlength("Statues")
    scale = (width * ss) / (w1 + wg + w2)
    size = max(8, round(size * scale))
    reg = _font(FONT_REG, size)
    bold = _font(FONT_BOLD, size)
    w1 = reg.getlength("Dharma")
    wg = reg.getlength(" ")
    w2 = bold.getlength("Statues")
    H = round(size * 1.5)
    W = round(w1 + wg + w2) + 4
    cov = Image.new("L", (W, H), 0)
    col = Image.new("RGB", (W, H), (0, 0, 0))
    red = tuple(int(np.clip(c + rng.uniform(-30, 30), 0, 255)) for c in (190, 45, 40))
    orange = tuple(int(np.clip(c + rng.uniform(-30, 30), 0, 255)) for c in (228, 118, 28))
    if rng.random() < 0.15:
        red = orange  # some copies are one colour
    d_cov = ImageDraw.Draw(cov)
    d_col = ImageDraw.Draw(col)
    y = round(size * 0.12)
    d_cov.text((2, y), "Dharma", font=reg, fill=255)
    d_cov.text((2 + w1 + wg, y), "Statues", font=bold, fill=255)
    # colour: the left part red, the right part orange
    d_col.rectangle((0, 0, round(2 + w1 + wg * 0.5), H), fill=red)
    d_col.rectangle((round(2 + w1 + wg * 0.5), 0, W, H), fill=orange)
    w_out = max(4, W // ss)
    h_out = max(4, H // ss)
    cov = cov.resize((w_out, h_out), Image.LANCZOS)
    col = col.resize((w_out, h_out), Image.LANCZOS)
    return np.asarray(cov, dtype=np.float32) / 255.0, np.asarray(col)


def make_sample(rng: random.Random, pool: list[np.ndarray], p_mark: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """One training pair: (CROP x CROP x 3 uint8 picture, CROP x CROP float mask of lettering pixels)."""
    src = pool[rng.randrange(len(pool))]
    # a clean band: the upper middle of the photo, below the corner logo, above where the marks sit
    scale = rng.uniform(0.75, 1.35)
    s = max(CROP + 8, round(WORK * scale))
    img = Image.fromarray(src).resize((s, s), Image.BILINEAR)
    y_lo = round(0.12 * s)
    y_hi = max(y_lo, round(0.52 * s) - CROP)
    y0 = rng.randint(y_lo, y_hi)
    x0 = rng.randint(0, s - CROP)
    crop = img.crop((x0, y0, x0 + CROP, y0 + CROP))
    if rng.random() < 0.5:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    pic = np.asarray(crop).astype(np.float32)
    mask = np.zeros((CROP, CROP), dtype=np.float32)

    if rng.random() < p_mark:
        for _ in range(1 if rng.random() < 0.9 else 2):
            width = int(math.exp(rng.uniform(math.log(36), math.log(330))))  # 36..330 px wide at the 512 working size
            cov, colour = render_mark(rng, width)
            ang = rng.uniform(-2.5, 2.5)
            if abs(ang) > 0.3:
                rot_c = Image.fromarray((cov * 255).astype(np.uint8)).rotate(ang, expand=True, resample=Image.BILINEAR)
                rot_k = Image.fromarray(colour).rotate(ang, expand=True, resample=Image.BILINEAR)
                cov = np.asarray(rot_c, dtype=np.float32) / 255.0
                colour = np.asarray(rot_k)
            h, w = cov.shape
            # the mark may run off the crop a little, as real ones are cut by a crop
            px = rng.randint(-w // 5, CROP - w + w // 5) if w < CROP * 1.4 else rng.randint(CROP - w, 0)
            py = rng.randint(-h // 3, CROP - h + h // 3)
            ax0, ay0 = max(0, px), max(0, py)
            ax1, ay1 = min(CROP, px + w), min(CROP, py + h)
            if ax1 - ax0 < 4 or ay1 - ay0 < 3:
                continue
            c = cov[ay0 - py : ay1 - py, ax0 - px : ax1 - px]
            k = colour[ay0 - py : ay1 - py, ax0 - px : ax1 - px].astype(np.float32)
            alpha = rng.uniform(0.2, 0.8)
            a = (c * alpha)[..., None]
            pic[ay0:ay1, ax0:ax1] = pic[ay0:ay1, ax0:ax1] * (1 - a) + k * a
            mask[ay0:ay1, ax0:ax1] = np.maximum(mask[ay0:ay1, ax0:ax1], (c > 0.35).astype(np.float32))

    im = Image.fromarray(np.clip(pic, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    if rng.random() < 0.6:  # the photos were saved as lossy images
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(55, 95))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return np.asarray(im), mask


def preview(pool: list[np.ndarray], path: str, n: int = 24, seed: int = 5) -> None:
    rng = random.Random(seed)
    cols = 6
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * (CROP + 4), rows * (CROP + 4)), (30, 30, 30))
    for i in range(n):
        pic, mask = make_sample(rng, pool, p_mark=0.75)
        sheet.paste(Image.fromarray(pic), ((i % cols) * (CROP + 4), (i // cols) * (CROP + 4)))
    sheet.save(path, quality=90)


if __name__ == "__main__":
    import sys

    folder = sys.argv[1]
    out = sys.argv[2]
    preview(load_pool(folder, 60), out)
    print("preview written", out)
