#!/usr/bin/env python
"""
Fine-tune the contextual sense encoder end-to-end for homograph sense classification.

Complements scripts/build_sense_lexicon.py (frozen-encoder classifiers): here the encoder itself
is trained, on the same Wikipedia Homograph Data, to classify each homograph occurrence from its
clause-bounded context window. At runtime the two models' probabilities are averaged
(MemoryConfig.sense_model_weight).

Measured on the 1,614 held-out WHD sentences (model trained on the train split only):
    frozen lexicon 96.8%, fine-tuned 98.1%, 50/50 average 97.5% (best on mixed-sense probes).

Usage (GPU is used automatically when available; ~25 min on 12 CPU cores, minutes on a GPU):
    python scripts/train_sense_model.py --whd /path/to/WikipediaHomographData-master
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import MemoryConfig  # noqa: E402
from flowedit.memory.sense_model import SenseModel  # noqa: E402
from build_sense_lexicon import load_whd, read_split  # noqa: E402


def main():
    cfg = MemoryConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whd", required=True, help="WikipediaHomographData directory, zip file or URL")
    ap.add_argument("--out", default=cfg.sense_model_path)
    ap.add_argument("--base", default=cfg.sense_encoder_model)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--window", type=int, default=8, help="context words on each side (clause-bounded)")
    ap.add_argument("--final-on-all", action=argparse.BooleanOptionalAction, default=True,
                    help="train on train+eval for the shipped model (held-out accuracy is then not reported)")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())

    root = load_whd(args.whd)
    train, test = read_split(root, "train"), read_split(root, "eval")
    if args.final_on_all:
        train, test = train + test, []
    labels = sorted({r[1] for r in train})
    word_labels = {}
    for w, l, *_ in train:
        word_labels.setdefault(w, set()).add(l)

    model = SenseModel.from_base(args.base, labels, {w: sorted(s) for w, s in word_labels.items()}, args.window,
                                 cfg.sense_clause_delimiters).to(device)
    lab_idx = {l: i for i, l in enumerate(labels)}
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 5e-5},
        {"params": model.head.parameters(), "lr": 2e-3},
    ], weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-5, 2e-3], total_steps=steps, pct_start=0.1)

    def accuracy(rows):
        model.eval()
        ok = 0
        with torch.no_grad():
            for i in range(0, len(rows), 64):
                ch = rows[i:i + 64]
                probs = model.predict([(w, t, a, b) for w, _, t, a, b in ch])
                ok += sum(max(p, key=p.get) == r[1] for p, r in zip(probs, ch))
        model.train()
        return ok / max(1, len(rows))

    print(f"{len(train)} training sentences, {len(labels)} senses, device={device}", flush=True)
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            ch = train[i:i + args.batch]
            logits = model([(w, t, a, b) for w, _, t, a, b in ch])
            loss = F.cross_entropy(logits, torch.tensor([lab_idx[r[1]] for r in ch], device=logits.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        msg = f"epoch {ep + 1}/{args.epochs} ({time.time() - t0:.0f}s)"
        if test:
            msg += f" held-out accuracy {accuracy(test):.4f}"
        print(msg, flush=True)

    model.save(args.out)
    print(f"saved fine-tuned sense model to {args.out}")


if __name__ == "__main__":
    main()
