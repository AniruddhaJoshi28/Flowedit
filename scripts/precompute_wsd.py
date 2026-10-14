#!/usr/bin/env python
"""
Score every WordNet sense of every Wikipedia Homograph Data occurrence with GlossBERT.

Output feeds scripts/build_wsd_tables.py. Run on a GPU (minutes; ~75 min on 12 CPU cores):
    CUDA_VISIBLE_DEVICES=1 python scripts/precompute_wsd.py --whd /path/to/WikipediaHomographData-master \
        --glossbert models/glossbert --out glossbert_whd.pt

Requires NLTK WordNet data (set NLTK_DATA if it lives in a private folder).
"""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import MemoryConfig  # noqa: E402
from flowedit.memory.sense_encoder import ContextualSenseEncoder  # noqa: E402


def main():
    cfg = MemoryConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whd", required=True)
    ap.add_argument("--glossbert", default=cfg.sense_wsd_model_path)
    ap.add_argument("--out", default="glossbert_whd.pt")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from nltk.corpus import wordnet as wn
    from transformers import BertForSequenceClassification, BertTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(os.cpu_count())
    rows = []
    for split in ("train", "eval"):
        d = os.path.join(args.whd, "data", split)
        for fn in sorted(os.listdir(d)):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    rows.append((split, r["homograph"].lower(), r["wordid"], r["sentence"], int(r["start"]), int(r["end"])))

    tok = BertTokenizer(os.path.join(args.glossbert, "vocab.txt"), do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(args.glossbert, num_labels=2).eval()
    model = (model.half() if dev == "cuda" else model).to(dev)
    win = ContextualSenseEncoder("", (0,), cfg.sense_wsd_window_words, cfg.sense_clause_delimiters, cache_size=0)

    senses, pairs, owners = {}, [], []
    for i, (_, w, _, s, a, b) in enumerate(rows):
        t, cs, ce = win._window(s, a, b)
        senses.setdefault(w, [(x.name(), x.definition()) for x in wn.synsets(w)])
        for name, gloss in senses[w]:
            pairs.append((t[:cs] + '" ' + t[cs:ce] + ' "' + t[ce:], f"{t[cs:ce]} : {gloss}"))
            owners.append((i, name))
    print(f"{len(rows)} occurrences, {len(pairs)} context-definition pairs, device={dev}", flush=True)

    scores = [dict() for _ in rows]
    t0 = time.time()
    for k in range(0, len(pairs), args.batch):
        ch = pairs[k:k + args.batch]
        enc = tok([c for c, _ in ch], [g for _, g in ch], padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1].tolist()
        for (i, name), v in zip(owners[k:k + args.batch], p):
            scores[i][name] = v
        if (k // args.batch) % 100 == 0:
            print(f"{k}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)
    torch.save({"rows": rows, "scores": scores}, args.out)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
