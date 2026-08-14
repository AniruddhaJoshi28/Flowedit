# FlowEdit Speech Generation Core: Research Design & Architectural Specification

**Author:** Antigravity AI Engineering  
**Version:** 1.0.0  
**Date:** 2026-08-04  
**Target Architecture:** F5-TTS (Flow-Matching DiT), CosyVoice 2 (Speech LLM + CFM), XTTS-v2 (Autoregressive GPT + VQ-VAE)  

---

## 1. Executive Architecture Overview

FlowEdit provides a unified internal speech correction framework operating under a model-agnostic abstraction (`TTSBackbone`). However, due to structural differences across backbones—F5-TTS uses flow matching with continuous text embeddings; CosyVoice 2 uses discrete semantic token generation; XTTS-v2 uses autoregressive GPT token sampling—each backbone defines an explicit capability contract (`BackboneCapabilities`) and a model-specific correction strategy (`CorrectionStrategy`).

```
                              +-----------------------+
                              |   Incoming API Request |
                              | /api/synthesize, /correct|
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Shared Audio Processor|
                              | & Prompt Validator    |
                              | (PreparedAudio / 422) |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Speech Text Processor |
                              | & Duration Planner    |
                              | (Entity-Safe Chunking)|
                              +-----------+-----------+
                                          |
                   +----------------------+----------------------+
                   |                      |                      |
                   v                      v                      v
        +--------------------+  +-------------------+  +-------------------+
        |   F5-TTS Wrapper   |  | CosyVoice 2 Wrapr |  |   XTTS-v2 Wrapper |
        | (Continuous Flow)  |  | (LLM + CFM Stack) |  | (Autoregressive)  |
        +---------+----------+  +---------+---------+  +---------+---------+
                  |                       |                      |
                  +-----------------------+----------------------+
                                          |
                                          v
                              +-----------------------+
                              |   2-Tier Evaluator    |
                              | & Candidate Selector  |
                              +-----------+-----------+
                                          |
                                          v
                              +-----------------------+
                              | Versioned Hopfield    |
                              | Pronunciation Memory  |
                              +-----------------------+
```

---

## 2. Mathematical Formulations & Algorithms

### 2.1 Dynamic Duration & Pacing Planner
Speech rate $\mathcal{R}_{\text{ref}}$ (syllables per second) is derived from the reference audio voiced duration $T_{\text{ref\_voiced}}$:

$$\mathcal{R}_{\text{ref}} = \operatorname{clamp}\left(\frac{N_{\text{ref\_syllables}}}{\max(0.5, T_{\text{ref\_voiced}})}, \mathcal{R}_{\min}, \mathcal{R}_{\max}\right)$$

Target duration $T_{\text{planned}}$ and dynamic speed factor $s_{\text{dynamic}}$ are computed as:

$$T_{\text{planned}} = \frac{N_{\text{target\_syllables}}}{\mathcal{R}_{\ref}} + \tau_{\text{punctuation}} + \tau_{\text{sentence\_padding}}$$

$$s_{\text{dynamic}} = \operatorname{clamp}\left(\frac{T_{\text{default}}}{T_{\text{planned}}}, 0.82, 1.02\right)$$

### 2.2 Relative Trust-Region Projection
To prevent catastrophic forgetting and preserve voice quality, perturbation $\delta$ is clamped after every optimization step relative to base embedding norm $\|c\|$:

$$\delta^{(k+1)} = \delta^{(k)} \cdot \min\left(1.0, \frac{\eta \cdot \|c\|}{\|\delta^{(k)}\|}\right)$$

where strategy-specific ratio $\eta$ is defined as:
- F5-TTS Conditioning: $\eta = 0.03$
- XTTS GPT Conditioning: $\eta = 0.02$
- CosyVoice Adapter Hidden: $\eta = 0.01$

### 2.3 CosyVoice 2 Priority Fallback Hierarchy
When synthesizing with CosyVoice 2, corrections are resolved through a 7-tier priority hierarchy:
1. **Normalized Canonical Text**
2. **Pronunciation Lexicon Alias**
3. **Native-Script Entity Substitution** (e.g. Devanagari rendering for Marathi/Hindi names)
4. **Cross-Lingual Inference** (when prompt language != target language)
5. **Staged Candidate Retry**
6. **Verified Exemplar Conditioning**
7. **Pronunciation Adapter** (active *only* after 8-step hidden-state injection is verified)

### 2.4 Composite Gibberish Classifier
A multi-signal composite score $S_{\text{gibberish}} \in [0.0, 1.0]$ prevents returning degraded audio:

$$S_{\text{gibberish}} = 0.30(1 - S_{\text{ASR\_overlap}}) + 0.25(1 - S_{\text{phoneme\_align}}) + 0.20 S_{\text{repetition}} + 0.15 \max(0, 1 - r_{\text{duration}}) + 0.10 \max(0, F_{\text{spectral}} - 0.40)$$

---

## 3. Pronunciation Memory Isolation Schema

Stored corrections (`PronunciationCorrection`) enforce strict isolation by `backbone`, `model_version`, and `embedding_schema_version`:

$$\mathcal{K}_{\text{valid}} = \{ K_i \mid \operatorname{metadata}_i.\text{backbone} = \text{B} \land \operatorname{metadata}_i.\text{schema\_version} = \text{V} \}$$

Memory lookup operates over $\mathcal{K}_{\text{valid}}$ only, completely eliminating cross-backbone embedding corruption.
