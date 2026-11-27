"""Trains the lettering detector on synthetic pictures.

    python train.py <photos folder> [--steps 4000] [--out wmnet.pt] [--workers 5]

The photos are only a source of statue backgrounds; their own watermarks sit low in the picture and the crops are taken from the clean upper band.
Validation uses photos the model never trains on.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

import synth
from model import TinyUNet, prepare


class Stream(IterableDataset):
    def __init__(self, pool_path: str, lo: int, hi: int, seed: int, p_mark: float = 0.5) -> None:
        self.pool_path, self.lo, self.hi, self.seed, self.p_mark = pool_path, lo, hi, seed, p_mark

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        pool = np.load(self.pool_path, mmap_mode="r")[self.lo : self.hi]
        view = [pool[i] for i in range(len(pool))]
        while True:
            pic, mask = synth.make_sample(rng, view, self.p_mark)
            x = (torch.from_numpy(np.ascontiguousarray(pic)).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.25
            yield x, torch.from_numpy(mask)[None]


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(6.0))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, batches: int) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0.0
    empty_pics = empty_hits = 0
    for i, (x, y) in enumerate(loader):
        if i >= batches:
            break
        pred = (model(x) > 0).float()
        tp += (pred * y).sum().item()
        fp += (pred * (1 - y)).sum().item()
        fn += ((1 - pred) * y).sum().item()
        for b in range(x.shape[0]):
            if y[b].sum() == 0:
                empty_pics += 1
                empty_hits += int(pred[b].sum().item() > 20)  # a picture with no mark where the model draws more than a speck
    model.train()
    return {"iou": tp / max(1.0, tp + fp + fn), "recall": tp / max(1.0, tp + fn), "precision": tp / max(1.0, tp + fp), "false_pictures": empty_hits / max(1, empty_pics)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", default="wmnet.pt")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--pool", type=int, default=1200)
    ap.add_argument("--cache", default="pool.npy")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cache = Path(args.cache)
    if not cache.exists():
        print(f"decoding {args.pool} photos ...", flush=True)
        pool = synth.load_pool(args.photos, args.pool)
        np.save(cache, np.stack(pool))
        print(f"cached {len(pool)} photos in {cache}", flush=True)
    n = np.load(cache, mmap_mode="r").shape[0]
    split = int(n * 0.88)  # the last 12% of the photos are for validation only
    train_dl = DataLoader(Stream(str(cache), 0, split, 1), batch_size=args.batch, num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    val_dl = DataLoader(Stream(str(cache), split, n, 7, p_mark=0.5), batch_size=args.batch, num_workers=2)

    model = TinyUNet()
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.steps, pct_start=0.1)
    started = time.time()
    run = 0.0
    it = iter(train_dl)
    for step in range(1, args.steps + 1):
        x, y = next(it)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0:
            print(f"step {step}/{args.steps}  loss {run:.3f}  {(time.time() - started) / step:.2f}s/step", flush=True)
        if step % 500 == 0 or step == args.steps:
            m = validate(model, val_dl, 20)
            print(f"  validation: IoU {m['iou']:.3f}  recall {m['recall']:.3f}  precision {m['precision']:.3f}  pictures without a mark that get one drawn: {m['false_pictures']:.3f}", flush=True)
            torch.save(model.state_dict(), args.out)
    print("done", args.out, f"{(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
