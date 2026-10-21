#!/usr/bin/env python
"""
Sense-selection benchmark for homographs on the real retrieval path (real F5-TTS text encoder).

Measures which learned pronunciation the memory applies to held-out sentences, for:
  pooled    - original behaviour (one EMA-pooled key per sense), FLOWEDIT_SENSE_CONTEXT=0
  context   - context sense model (flowedit/memory/sense_context.py), the default

Held-out sentences pass through retrieval one by one, so their tokens enter the background
statistics exactly as synthesized text does in deployment (no labels are used for that).

Scenarios:
  single_<sense>  only that pronunciation is corrected (1 sentence); every held-out sentence of both
                  senses is tested. Same-sense sentences should get the correction, the other
                  sense's sentences should not.
  both_k{1,2,3}   k correction sentences per pronunciation; held-out sentences must get the right one.

This isolates the selection decision: memory entries carry the spelling label of the correction and
the key computed from its sentence; δ* is not involved (zero), so no pronunciation quality is claimed
here. Audio is evaluated separately with real corrections.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowedit.config import FlowEditConfig, MemoryConfig  # noqa: E402
from flowedit.memory.hopfield_memory import HopfieldMemory  # noqa: E402
from flowedit.memory.s3_storage import s3_spelling_store  # noqa: E402

# word -> sense spelling -> {"corrections": [...3], "tests": [...]}
DATA = {
    "lead": {
        "led": {
            "corrections": ["This toy is made up of lead.", "The old pipes were full of lead.",
                            "Lead paint is dangerous for children."],
            "tests": ["The pipes in the basement are made of lead.", "Lead is a soft and heavy metal.",
                      "They found lead in the drinking water.", "The bullet was made of lead and copper.",
                      "Old paint often contains lead.", "Exposure to lead harms the nervous system.",
                      "The fishing weights were cast from solid lead.", "A thin sheet of lead blocks the radiation.",
                      "The miners extracted lead and zinc from the ore.", "Children should never play with lead toys."],
        },
        "leed": {
            "corrections": ["She is the lead of data science.", "He will lead the team next year.",
                            "Our team lead approved the plan."],
            "tests": ["She will lead the meeting tomorrow.", "He was asked to lead the project.",
                      "Who will lead the team after she retires?", "The lead singer walked onto the stage.",
                      "Our team lead scheduled a review.", "The guide will lead us through the forest.",
                      "She took the lead in the final lap.", "Priya is the lead engineer on this product.",
                      "These roads lead to the old city.", "Let me lead the discussion today."],
        },
    },
    "read": {
        "red": {
            "corrections": ["Yesterday she read the whole book.", "I have already read that report.",
                            "He read the letter twice last night."],
            "tests": ["She read the report last night.", "I read that book when I was a child.",
                      "He had already read the email before the meeting.", "Yesterday we read three chapters together.",
                      "Have you read the new policy?", "They read the results and smiled.",
                      "The letter was read aloud at the ceremony.", "I read about the accident in the newspaper last week.",
                      "She read every page twice before signing.", "Last year he read over fifty novels."],
        },
        "reed": {
            "corrections": ["Please read the instructions carefully.", "I like to read before bed.",
                            "Can you read this sentence aloud?"],
            "tests": ["I read the news every morning.", "Please read the label before use.",
                      "Children love to read comic books.", "Can you read the second paragraph?",
                      "I will read your proposal tomorrow.", "She likes to read in the garden.",
                      "Read the question carefully before answering.", "We read together every evening.",
                      "It is hard to read without my glasses.", "They want to read the contract first."],
        },
    },
    "wound": {
        "woond": {
            "corrections": ["The nurse cleaned the wound carefully.", "The soldier had a deep wound in his leg.",
                            "Words can wound people deeply."],
            "tests": ["The wound healed after two weeks.", "She pressed a bandage on the wound.",
                      "A small wound can still get infected.", "The doctor stitched the open wound.",
                      "He has a wound on his arm.", "Keep the wound clean and dry.",
                      "The bullet left a wound in his shoulder.", "His harsh comments will wound her."],
        },
        "wownd": {
            "corrections": ["She wound the clock before bed.", "He wound the rope around the post.",
                            "The path wound through the hills."],
            "tests": ["They wound the cable onto the spool.", "I wound the string around my finger.",
                      "The river wound slowly to the sea.", "She wound up the toy car.",
                      "We wound the yarn into a ball.", "The road wound up the mountain.",
                      "He wound the bandage around her wrist.", "Grandpa wound his watch every morning."],
        },
    },
    "live": {
        "liv": {
            "corrections": ["We live in a small town.", "Where do you live now?", "I want to live near the sea."],
            "tests": ["They live next to the station.", "My parents live abroad.", "Plants need water to live.",
                      "Do they still live here?", "You only live once.", "We will live with my aunt.",
                      "Many animals live in the forest.", "I could live without coffee."],
        },
        "lyve": {
            "corrections": ["The band played live music.", "Be careful with live wires.",
                            "The match is broadcast live tonight."],
            "tests": ["We watched a live concert.", "The show was streamed live online.", "Never touch a live wire.",
                      "He caught a live fish.", "The news is coming to you live.", "She loves live theatre.",
                      "A live audience laughed.", "They sell live bait."],
        },
    },
    "tear": {
        "teer": {
            "corrections": ["A tear rolled down her cheek.", "He wiped a tear from his eye.",
                            "There was a tear in his eye."],
            "tests": ["A single tear fell onto the letter.", "She shed a tear at the wedding.",
                      "His eyes filled with a tear of joy.", "A tear ran down the child's face."],
        },
        "tair": {
            "corrections": ["Do not tear the paper.", "Be careful not to tear your shirt.",
                            "There is a tear in my jeans."],
            "tests": ["Please do not tear the poster.", "You can tear the coupon along the line.",
                      "They will tear down the old building.", "I tried to tear the package open."],
        },
    },
}

METHODS = {"pooled": False, "context": True}


def build_memory(encode, space, dim, method, corrections):
    """corrections: [(word, spelling, sentence)] written through HopfieldMemory.write in order."""
    cfg = MemoryConfig(sense_context=METHODS[method])
    mem = HopfieldMemory(cfg, embedding_dim=dim, embedding_space=space, text_encoder=encode)
    for word, spelling, sentence in corrections:
        key = mem._carrier_key(sentence, word)
        span = mem._find_word_char_spans(sentence, word)[0]
        mem.write(key=key, value=torch.zeros(dim), word=word, carrier_text=sentence, phonetic_text=spelling,
                  char_span=span)
    return mem


def applied_spelling(mem, encode, tok, text, word):
    q = encode(text, "en")
    r = mem.retrieve(q, text=text, tokenizer=tok, language="en")
    for s in r.matched_spans:
        if s.word.lower() == word:
            return (s.phonetic_text or "").lower()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    s3_spelling_store._dictionary = {}
    s3_spelling_store._ensure_loaded = lambda: None
    s3_spelling_store._save_local_cache = lambda: False

    cfg = FlowEditConfig()
    cfg.backbone.backbone_type = "f5tts"
    from flowedit.backbone import create_backbone
    bb = create_backbone(cfg.backbone)
    bb.load_model()
    encode, space, dim = bb.encode_text, bb.embedding_space, bb.embedding_dim
    tok = getattr(bb, "tokenizer_instance", None)

    rows = []
    for word, senses in DATA.items():
        names = list(senses)
        scenarios = []
        for s in names:
            scenarios.append((f"single_{s}", [(word, s, senses[s]["corrections"][0])]))
        for k in (1, 2, 3):
            corr = [(word, s, senses[s]["corrections"][i]) for i in range(k) for s in names]
            scenarios.append((f"both_k{k}", corr))
        for scen, corr in scenarios:
            for method in METHODS:
                mem = build_memory(encode, space, dim, method, corr)
                for expected in names:
                    for text in senses[expected]["tests"]:
                        got = applied_spelling(mem, encode, tok, text, word)
                        rows.append({"word": word, "scenario": scen, "method": method, "expected": expected,
                                     "applied": got or "NONE", "text": text,
                                     "decision": json.dumps(mem.last_sense_decisions, default=str)})
        print(f"{word}: done", flush=True)

    with open(os.path.join(args.out, "selection_rows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summaries
    summary = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (r["scenario"].split("_")[0] if r["scenario"].startswith("single") else r["scenario"], r["method"])
        if r["scenario"].startswith("single"):
            learned = r["scenario"][len("single_"):]
            if r["expected"] == learned:
                summary[key]["same_sense_applied" if r["applied"] == learned else "same_sense_missed"] += 1
            else:
                summary[key]["other_sense_forced_wrong" if r["applied"] == learned else "other_sense_left_alone"] += 1
        else:
            summary[key]["correct" if r["applied"] == r["expected"] else
                         ("none" if r["applied"] == "NONE" else "wrong")] += 1
    out = {f"{s}|{m}": dict(v) for (s, m), v in sorted(summary.items())}
    per_word = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["scenario"] == "both_k3":
            per_word[f"{r['word']}|{r['method']}"]["correct" if r["applied"] == r["expected"] else "not"] += 1
    out["per_word_both_k3"] = {k: dict(v) for k, v in sorted(per_word.items())}
    json.dump(out, open(os.path.join(args.out, "selection_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
