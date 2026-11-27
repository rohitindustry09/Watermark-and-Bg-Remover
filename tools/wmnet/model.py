"""A small U-Net that marks the pixels of the "Dharma Statues" lettering. About 300k parameters: fast enough to run on every photo."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, ch: tuple[int, ...] = (8, 16, 32, 48)) -> None:
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 3
        for c in ch:
            self.enc.append(block(cin, c))
            cin = c
        self.mid = block(ch[-1], ch[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = ch[-1] * 2
        for c in reversed(ch):
            self.up.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.dec.append(block(c * 2, c))
            prev = c
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:  # odd sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)  # logits: > 0 means lettering


def prepare(img_uint8_hwc) -> torch.Tensor:
    """HxWx3 uint8 -> 1x3xHxW float, roughly centred."""
    t = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).float() / 255.0
    return ((t - 0.5) / 0.25).unsqueeze(0)


if __name__ == "__main__":
    m = TinyUNet()
    print("parameters:", sum(p.numel() for p in m.parameters()))
    print(m(torch.zeros(2, 3, 192, 192)).shape, m(torch.zeros(1, 3, 512, 512)).shape)
