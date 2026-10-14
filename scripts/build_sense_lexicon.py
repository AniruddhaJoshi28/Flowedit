#!/usr/bin/env python
"""
Build the pretrained homograph sense lexicon used for automatic pronunciation disambiguation.

Training data: Wikipedia Homograph Data (Google, Apache-2.0) — ~16k sentences, each with one
homograph occurrence labelled by sense and IPA pronunciation (162 homographs).
https://github.com/google/WikipediaHomographData

For every homograph a multinomial logistic-regression classifier is trained on contextual
occurrence embeddings (the same shared encoder the Hopfield memory uses, at several
clause-bounded window sizes). For every sense a respelling the TTS will read with that sense's
pronunciation is derived: an unambiguous CMU-dictionary word with the identical phone sequence,
preferring words common enough to be whole tokens of the encoder vocabulary (e.g. "red", "reed").
Senses without such a homophone get no respelling and are left to the TTS model's own reading.

Usage:
    python scripts/build_sense_lexicon.py --whd /path/to/WikipediaHomographData-master
    python scripts/build_sense_lexicon.py --whd https://github.com/google/WikipediaHomographData/archive/refs/heads/master.zip
"""

import argparse
import collections
import csv
import io
import os
import re
import sys
import tempfile
import time
import zipfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import MemoryConfig  # noqa: E402
from flowedit.memory.sense_lexicon import featurize, fit_classifier  # noqa: E402

# IPA → ARPAbet symbol conversion (notation only; ordered so diphthongs/long vowels match first)
IPA_TO_ARPA = [
    ("aɪ", "AY"), ("aʊ", "AW"), ("ɔɪ", "OY"), ("eɪ", "EY"), ("oʊ", "OW"),
    ("iː", "IY"), ("uː", "UW"), ("ɑː", "AA"), ("ɔː", "AO"), ("ɜː", "ER"),
    ("ʤ", "JH"), ("ʧ", "CH"),
    ("i", "IY"), ("ɪ", "IH"), ("e", "EH"), ("ɛ", "EH"), ("æ", "AE"), ("ɑ", "AA"), ("a", "AA"),
    ("ɔ", "AO"), ("o", "OW"), ("ʊ", "UH"), ("u", "UW"), ("ʌ", "AH"), ("ə", "AH"), ("ɚ", "ER"), ("ɝ", "ER"),
    ("p", "P"), ("b", "B"), ("t", "T"), ("d", "D"), ("k", "K"), ("ɡ", "G"), ("g", "G"),
    ("f", "F"), ("v", "V"), ("θ", "TH"), ("ð", "DH"), ("s", "S"), ("z", "Z"), ("ʃ", "SH"), ("ʒ", "ZH"),
    ("h", "HH"), ("m", "M"), ("n", "N"), ("ŋ", "NG"), ("l", "L"), ("ɹ", "R"), ("r", "R"), ("j", "Y"), ("w", "W"),
]


def ipa_to_arpa(ipa: str):
    s = re.sub(r"[ˈˌ'ː0-9.]", lambda m: "ː" if m.group(0) == "ː" else "", ipa)
    out, i = [], 0
    while i < len(s):
        for sym, arpa in IPA_TO_ARPA:
            if s.startswith(sym, i):
                out.append(arpa)
                i += len(sym)
                break
        else:
            i += 1  # unknown mark
    # Unstressed "ə + ɹ" is the CMU "ER"
    merged = []
    for p in out:
        if p == "R" and merged and merged[-1] == "AH":
            merged[-1] = "ER"
        else:
            merged.append(p)
    return tuple(merged)


def strip_stress(phones):
    return tuple(re.sub(r"\d", "", p) for p in phones)


def load_whd(src: str) -> str:
    if os.path.isdir(src):
        return src
    tmp = tempfile.mkdtemp(prefix="whd_")
    if re.match(r"https?://", src):
        import urllib.request
        data = urllib.request.urlopen(src, timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    else:
        zipfile.ZipFile(src).extractall(tmp)
    roots = [os.path.join(tmp, d) for d in os.listdir(tmp)]
    return next(r for r in roots if os.path.isdir(os.path.join(r, "data")))


def read_split(root, split):
    rows = []
    d = os.path.join(root, "data", split)
    for fn in sorted(os.listdir(d)):
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows.append((r["homograph"].lower(), r["wordid"], r["sentence"], int(r["start"]), int(r["end"])))
    return rows


ARPA_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
LETTER_VOWELS = set("aeiouy")


def _split_word(word, phones):
    """Monosyllable → ((onset phones, onset letters), (rime phones, rime letters)), else None."""
    vowel_idx = [i for i, p in enumerate(phones) if p in ARPA_VOWELS]
    if len(vowel_idx) != 1:
        return None
    k = next((i for i, ch in enumerate(word) if ch in LETTER_VOWELS and not (ch == "y" and i == 0)), None)
    if k is None:
        return None
    v = vowel_idx[0]
    return (tuple(phones[:v]), word[:k]), (tuple(phones[v:]), word[k:])


def _spelling_model(cmu):
    """Onset/rime phone → spelling frequencies learned from unambiguous CMU monosyllables."""
    onsets, rimes = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for w, prons in cmu.items():
        if not w.isalpha() or len({strip_stress(p) for p in prons}) != 1:
            continue
        sp = _split_word(w, strip_stress(prons[0]))
        if sp:
            onsets[sp[0][0]][sp[0][1]] += 1
            rimes[sp[1][0]][sp[1][1]] += 1
    return onsets, rimes


def respellings(senses, vocab, top_k=6):
    """For each (word, label): a spelling that a G2P reader pronounces as that sense, or None.

    Candidates: unambiguous CMU homophones, then onset+rime compositions spelled with the most
    frequent English spellings of those sounds. Every candidate is verified with g2p_en: it must
    read back exactly the sense's phones and must not be the homograph itself.
    """
    from nltk.corpus import cmudict
    from g2p_en import G2p
    g2p = G2p()
    cmu = cmudict.dict()
    by_phones = collections.defaultdict(list)
    for w, prons in cmu.items():
        if w.isalpha() and len({strip_stress(p) for p in prons}) == 1:
            by_phones[strip_stress(prons[0])].append(w)
    onsets, rimes = _spelling_model(cmu)

    def reads_as(spelling, target):
        got = tuple(re.sub(r"\d", "", p) for p in g2p(spelling) if re.fullmatch(r"[A-Z]+\d?", p))
        return got == target

    out = {}
    for (word, label), ipa in senses.items():
        target = ipa_to_arpa(ipa)
        cands = [(0 if c in vocab else 1, 0, len(c), c) for c in by_phones.get(target, [])]
        v = [i for i, p in enumerate(target) if p in ARPA_VOWELS]
        if len(v) == 1:
            on, ri = target[:v[0]], target[v[0]:]
            on_sp = onsets[on].most_common(top_k) if on else [("", 1)]
            for o, o_freq in on_sp:
                for r, r_freq in rimes[ri].most_common(top_k):
                    cands.append((2, -(o_freq * r_freq), len(o + r), o + r))
        chosen = None
        for *_, c in sorted(set(cands)):
            if c == word or (c in cmu and strip_stress(cmu[c][0]) != target):
                continue
            if reads_as(c, target):
                chosen = c
                break
        out[(word, label)] = (chosen, " ".join(target))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whd", required=True, help="WikipediaHomographData directory, zip file or URL")
    ap.add_argument("--out", default=MemoryConfig().sense_lexicon_path)
    ap.add_argument("--windows", default="3,6", help="clause-bounded context window sizes (words)")
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--final-on-all", action="store_true", default=True,
                    help="after reporting held-out accuracy, retrain on train+eval for the shipped model")
    args = ap.parse_args()

    cfg = MemoryConfig()
    windows = tuple(int(w) for w in args.windows.split(","))
    root = load_whd(args.whd)
    train, test = read_split(root, "train"), read_split(root, "eval")
    with open(os.path.join(root, "data", "wordids.tsv"), encoding="utf-8") as f:
        senses_meta = {(r["homograph"].lower(), r["wordid"]): r for r in csv.DictReader(f, delimiter="\t")}
    print(f"WHD: {len(train)} train / {len(test)} eval sentences, {len({r[0] for r in train})} homographs")

    t0 = time.time()
    Xtr = featurize(cfg, windows, [(r[2], r[3], r[4]) for r in train])
    Xte = featurize(cfg, windows, [(r[2], r[3], r[4]) for r in test])
    print(f"features: {Xtr.shape[1]}-d in {time.time() - t0:.0f}s")

    groups = collections.defaultdict(lambda: ([], []))
    for i, r in enumerate(train):
        groups[r[0]][0].append(i)
    for i, r in enumerate(test):
        groups[r[0]][1].append(i)

    ok = n = 0
    words = {}
    for word, (tr_idx, te_idx) in sorted(groups.items()):
        labels = sorted({train[i][1] for i in tr_idx})
        y_tr = torch.tensor([labels.index(train[i][1]) for i in tr_idx])
        W, b = fit_classifier(Xtr[tr_idx], y_tr, len(labels), reg=args.reg)
        te_idx = [i for i in te_idx if test[i][1] in labels]
        if te_idx:
            y_te = torch.tensor([labels.index(test[i][1]) for i in te_idx])
            pred = (Xte[te_idx] @ W + b).argmax(1)
            ok += int((pred == y_te).sum())
            n += len(te_idx)
        if args.final_on_all and te_idx:
            all_x = torch.cat([Xtr[tr_idx], Xte[te_idx]])
            all_y = torch.cat([y_tr, torch.tensor([labels.index(test[i][1]) for i in te_idx])])
            W, b = fit_classifier(all_x, all_y, len(labels), reg=args.reg)
        counts = collections.Counter(train[i][1] for i in tr_idx) + collections.Counter(test[i][1] for i in te_idx)
        words[word] = {
            "labels": labels, "W": W, "b": b, "counts": {l: int(counts[l]) for l in labels},
            # mean occurrence feature: subtracted before comparing contexts (removes word identity)
            "mean": Xtr[tr_idx].mean(0),
        }
    print(f"held-out sense accuracy: {ok}/{n} = {ok / max(1, n):.4f}")

    from flowedit.memory.sense_encoder import get_sense_encoder
    vocab = set(get_sense_encoder(cfg)._tokenizer.get_vocab())
    ipa = {(w, l): senses_meta[(w, l)]["pronunciation"] for w, d in words.items() for l in d["labels"] if (w, l) in senses_meta}
    resp = respellings(ipa, vocab)
    n_resp = 0
    for w, d in words.items():
        d["pronunciation"] = {l: ipa.get((w, l)) for l in d["labels"]}
        d["phones"] = {l: resp[(w, l)][1] if (w, l) in resp else None for l in d["labels"]}
        d["respelling"] = {l: resp[(w, l)][0] if (w, l) in resp else None for l in d["labels"]}
        d["description"] = {l: senses_meta.get((w, l), {}).get("label", l) for l in d["labels"]}
        n_resp += sum(1 for v in d["respelling"].values() if v)
    print(f"respellings found for {n_resp}/{sum(len(d['labels']) for d in words.values())} senses")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        "source": "WikipediaHomographData (Apache-2.0)",
        "encoder": cfg.sense_encoder_model,
        "layers": list(cfg.sense_encoder_layers),
        "windows": list(windows),
        "clause_delimiters": cfg.sense_clause_delimiters,
        "heldout_accuracy": ok / max(1, n),
        "words": words,
    }, args.out)
    print(f"saved {len(words)} homographs to {args.out}")
    for w in ("lead", "read", "bow", "tear", "wind", "live", "bass", "close", "wound", "row", "record"):
        if w in words:
            print(f"  {w:7s} " + "  ".join(f"{l}={words[w]['respelling'][l]}" for l in words[w]["labels"]))


if __name__ == "__main__":
    main()
