# Contextual Homograph Disambiguation — Models in the FlowEdit Pipeline

## The problem

Some words are spelled the same but pronounced differently depending on meaning or tense:

| Word | Meaning A | Meaning B |
|---|---|---|
| lead | metal → "led" | to guide / leader → "leed" |
| read | yesterday → "red" | now / future → "reed" |
| bow | ribbon, weapon → "boh" | bending, gesture → "bau" |
| tear | crying → "teer" | ripping → "tare" |

The TTS has to work out from the sentence which meaning is intended. It does this **without
hand-written word lists and without a large language model**.

## How the pipeline handles it

- **Each occurrence is judged separately.** Every occurrence of a word is decided using only its
  own part of the sentence. *"The old paint contains **lead**, and Priya is the **lead**
  researcher"* therefore gets two independent decisions: "led", then "leed".
- **The meaning is turned into a respelling** the TTS pronounces correctly ("led", "reed", "bo").
- **The common meaning is left as written.** The TTS already reads it correctly, so only the rarer
  meaning is respelled.

Three components make the decision, in this order.

### 1. Context encoder + per-word sense classifier (core)

- **What it is.** A compact sentence encoder, MiniLM, converts the words around the target into a
  numeric "context fingerprint". A small classifier for each word maps that fingerprint to a
  pronunciation.
- **Training data.** Google's Wikipedia Homograph Data: 16,000 real sentences labelled with the
  meaning and pronunciation of 162 homographs.
- **How it helps with sense.** It learns the typical surroundings of each meaning. Paint, pipes
  and metal point to "led". Team, role and project point to "leed".
- **Result.** **96.8% correct on unseen sentences.** Always guessing the most common pronunciation
  scores 84%.

### 2. GlossBERT — dictionary-meaning model (MIT licence)

- **What it is.** A model that compares the sentence against each **dictionary definition** of the
  word (WordNet) and scores how well each one fits. A table learned from the labelled sentences
  then turns the chosen definition into a pronunciation.
- **How it helps with sense.** It brings general English knowledge the training sentences lack.
  For *"the actor took a **bow** after the performance"* it picks *"bow: an appearance by actors
  at the end of a play to acknowledge the applause"*, the "bau" meaning.
- **When it runs.** Only when the core classifier is less than 95% sure, which is about 9% of
  occurrences. That keeps it fast. It runs on the GPU.
- **Result.** Raises accuracy on unseen sentences to **97.2%**.

### 3. Learning from user corrections

- **What it is.** When a user corrects a pronunciation, for example *"I read books every day"* →
  "reed", the corrected sentence is stored as an example.
- **How it helps with sense.** New sentences whose context is very close to a corrected example
  follow the correction. Everything else keeps the model's decision.
- **Result.** It fixed the reported failing sentences **without changing any other sentence's
  output**, verified on held-out data.

## Overall

| Stage | Accuracy on unseen sentences |
|---|---|
| Before: hard-coded keyword table, same pronunciation for every occurrence | not measured |
| Context classifier | 96.8% |
| + GlossBERT | **97.2%** |
| + user corrections | fixes specific phrasings with no side effects |

## Known limits

- **Everyday present-tense *read*** ("we read the news every morning") is rare in the training
  data. It needs a one-time correction.
- **Two competing cues in one sentence** ("caught a **bass** while his brother played **bass**
  guitar") can still be resolved wrongly.
- **Stress-only pairs** (REcord vs reCORD) cannot be expressed by respelling and are left to the
  TTS.
