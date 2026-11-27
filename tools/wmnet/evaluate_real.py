"""How does the detector do on REAL pictures? The synthetic validation only says it learned what we showed it; this says whether it works on your photos.

    python evaluate_real.py <weights.pt> <photos folder> <labels.json> [--sheet out.jpg]

labels.json is a list of {"file": relative path, "label": "REAL" | "fake", "x", "y", "width", "height"}: the box of a mark I checked by eye (REAL), or the box
of a false alarm on a red part of the statue (fake). The model should mark pixels inside REAL boxes and leave the fake boxes alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import synth
from model import TinyUNet, prepare


def predict(model: torch.nn.Module, im: Image.Image) -> tuple[np.ndarray, float]:
    """Probability map at the photo's own size, and the scale used (the long side is brought to WORK pixels)."""
    s = synth.WORK / max(im.size)
    small = im.convert("RGB").resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.LANCZOS)
    with torch.no_grad():
        p = torch.sigmoid(model(prepare(np.asarray(small))))[0, 0].numpy()
    return p, s


def main() -> None:
    weights, photos, labels = sys.argv[1], Path(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
    sheet = sys.argv[sys.argv.index("--sheet") + 1] if "--sheet" in sys.argv else None
    model = TinyUNet()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    real_hits, real_n, fake_hits, fake_n = 0, 0, 0, 0
    tiles = []
    for l in labels:
        f = photos / l["file"]
        if not f.exists():
            continue
        im = Image.open(f).convert("RGB")
        p, s = predict(model, im)
        x0, y0 = int(l["x"] * s), int(l["y"] * s)
        x1, y1 = int((l["x"] + l["width"]) * s) + 1, int((l["y"] + l["height"]) * s) + 1
        inside = p[max(0, y0) : y1, max(0, x0) : x1]
        share = float((inside > 0.5).mean()) if inside.size else 0.0
        if l["label"] == "REAL":
            real_n += 1
            real_hits += int(share >= 0.12)  # letters fill about a third of their box; 12% of the box marked is a clear detection
        else:
            fake_n += 1
            fake_hits += int(share >= 0.12)
        if sheet and len(tiles) < 36:
            ov = np.asarray(im.resize((round(im.width * s), round(im.height * s)))).copy()
            ov[p > 0.5] = (0, 255, 255)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            crop = Image.fromarray(ov).crop((max(0, cx - 150), max(0, cy - 55), cx + 150, cy + 55)).resize((300, 110))
            tiles.append((l["label"], crop))
    print(f"REAL marks found by the model: {real_hits}/{real_n}")
    print(f"false alarms the model also fires on: {fake_hits}/{fake_n}  (lower is better)")
    if sheet and tiles:
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        out = Image.new("RGB", (cols * 304, rows * 114), (30, 30, 30))
        for i, (_, t) in enumerate(tiles):
            out.paste(t, ((i % cols) * 304, (i // cols) * 114))
        out.save(sheet, quality=88)
        print("sheet", sheet)


if __name__ == "__main__":
    main()
