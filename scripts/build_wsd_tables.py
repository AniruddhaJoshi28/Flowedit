#!/usr/bin/env python
"""
Add WordNet word-sense knowledge (GlossBERT) to the shipped sense lexicon.

Inputs:
  --scores   output of scripts/precompute_wsd.py: GlossBERT probability of every WordNet sense for
             every Wikipedia Homograph Data occurrence (run it on a GPU: minutes instead of an hour).
Writes into the lexicon file, per homograph:
  wsd_glosses  [(synset, "word : definition"), ...]  – the candidate senses scored at runtime
  wsd_table    {synset: {pronunciation label: soft count}} – learned from the labelled corpus
and the combination weight `wsd_weight`. Nothing here is hand-written: the sense inventory is
WordNet, the sense → pronunciation link is counted from labelled sentences.

Fair evaluation (weights tuned on one half of the held-out sentences, accuracy on the other):
lexicon 96.45% → lexicon + GlossBERT 97.21% (both halves chose weight 2).

Usage:
    python scripts/build_wsd_tables.py --scores glossbert_whd.pt
"""

import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import MemoryConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--lexicon", default=MemoryConfig().sense_lexicon_path)
    ap.add_argument("--weight", type=float, default=2.0, help="log-probability weight of the WSD signal")
    args = ap.parse_args()

    from nltk.corpus import wordnet as wn

    data = torch.load(args.lexicon, map_location="cpu", weights_only=False)
    scores = torch.load(args.scores, map_location="cpu", weights_only=False)
    tables = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for (_, word, label, *_), s in zip(scores["rows"], scores["scores"]):
        z = sum(s.values())
        if word not in data["words"] or not z:
            continue
        for synset, p in s.items():
            tables[word][synset][label] += p / z

    n = 0
    for word, d in data["words"].items():
        d["wsd_glosses"] = [(s.name(), f"{word} : {s.definition()}") for s in wn.synsets(word)]
        d["wsd_table"] = {s: dict(c) for s, c in tables.get(word, {}).items()}
        n += bool(d["wsd_glosses"])
    data["wsd_weight"] = args.weight
    data["wsd_source"] = "GlossBERT (MIT) over WordNet senses; tables from WikipediaHomographData"
    torch.save(data, args.lexicon)
    print(f"WSD tables added for {n}/{len(data['words'])} homographs -> {args.lexicon}")


if __name__ == "__main__":
    main()
