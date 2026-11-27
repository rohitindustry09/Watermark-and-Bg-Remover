# wmnet: a small detector for the "Dharma Statues" lettering

A tiny U-Net trained on synthetic pictures (the mark drawn onto clean statue backgrounds), so the answer is known exactly. It is meant to tell lettering
from red enamel and carving, which the pattern matcher confuses.

    python train.py <photos folder> --steps 6000 --out wmnet.pt        # train (GPU is used when present)
    python evaluate_real.py wmnet.pt <photos folder> <labels.json>      # test on real, hand-checked boxes

Needs: python 3, torch, numpy, pillow, and the Century Gothic font (set WMNET_FONT_REG / WMNET_FONT_BOLD to other .ttf files if it is not installed).
