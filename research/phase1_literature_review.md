# Phase 1 focused literature review

Status: initial verified review; citation-network expansion remains open. Sources were checked against primary paper pages/PDFs or official proceedings where available.

## Research question

Which parts of contextual lifelong trajectory editing are already established, and what remains both scientifically distinct and feasible with a frozen F5-TTS checkpoint?

## Core verified works

### FlowEdit: Associative Memory for Lifelong Pronunciation Adaptation in Flow-Matching TTS

Harshit Singh, Ayush Pratap Singh, Nityanand Mathur (2026), arXiv:2606.20518. Source: https://arxiv.org/abs/2606.20518

- Frozen F5-TTS; optimize a target-token text-conditioning perturbation against reference mel loss plus L2 penalty.
- Paper states differentiation through the flow with adjoint sensitivity, 50 Adam steps, and a target-position mask.
- Stores pooled contextual keys and perturbation values in a gated Modern Hopfield memory.
- Already covers latent conditioning edits, contextual keys, episodic reuse, and similarity gating. These cannot be claimed as new.
- The paper's stated “zero forgetting” follows only when the retrieval gate is effectively zero; a sigmoid at finite logit is not exactly zero. Its homograph treatment is a contextual pooling note, not a demonstrated open-set homograph benchmark.
- Repository reproduction is presently unfaithful: custom solver equivalence and trained correction provenance are not verified.

### F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, Xie Chen (ACL 2025; arXiv 2024). Source: https://aclanthology.org/2025.acl-long.313/

- Fully non-autoregressive conditional flow-matching TTS with a DiT and ConvNeXt-refined text representation.
- Avoids an explicit duration model and phoneme alignment; pads text/filler conditioning to acoustic length.
- Introduces sway sampling; official inference uses a frozen vector field, CFG, and numerical ODE integration.
- This architecture makes “target word frames” latent and uncertain. Selective acoustic intervention therefore needs an independently justified alignment or learned sensitivity support.

### D-Flow: Differentiating through Flows for Controlled Generation

Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, Yaron Lipman (ICML 2024). Source: https://proceedings.mlr.press/v235/ben-hamu24a.html

- Controls a frozen flow by differentiating an outcome loss through the ODE and optimizing the source noise.
- Establishes source-point optimization as a general controlled-generation method and motivates the flow-induced prior.
- Therefore “differentiate through a frozen generative flow” is established. A new contribution must involve speech-specific selective controllability, constraints, transfer, or reliable gating—not differentiation alone.

### SonoEdit: Null-Space Constrained Knowledge Editing for Pronunciation Correction in LLM-Based TTS

Ayush Pratap Singh, Harshit Singh, Nityanand Mathur, Akshat Mandloi, Sudarshan Kamath (2026), arXiv:2601.17086. Source: https://arxiv.org/abs/2601.17086

- Uses acoustic causal tracing to select pronunciation-relevant transformer layers.
- Computes a null-space projection from general-speech hidden-state covariance and applies a closed-form weight edit.
- States zero first-order change on the preserved key subspace; this is not a global nonlinear preservation guarantee.
- Null-space projection, preserved-corpus covariance, and first-order orthogonality are established for pronunciation editing. Applying the same algebra to flow controls without a distinct mechanism would be an application, not a new mathematical contribution.

### Flow Matching for Generative Modeling

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le (ICLR 2023). Source: https://openreview.net/forum?id=PqvMRDCJT9t

- Simulation-free vector-field regression for continuous normalizing flows along chosen conditional probability paths.
- Provides the flow-matching foundation; it does not solve post-deployment pronunciation correction.

### SpeechPainter: Text-conditioned Speech Inpainting

Eliya Nachmani, Darius Petrescu, Yossi Adi (Interspeech 2022). Source: https://arxiv.org/abs/2202.07273

- Text-conditioned speech inpainting replaces a masked acoustic interval and evaluates perceptual quality.
- Establishes localized speech regeneration as a baseline family. Waveform/mel splicing is not novel and must be evaluated for seams, coarticulation, and non-target change.

### Selective Classification for Deep Neural Networks

Yonatan Geifman, Ran El-Yaniv (NeurIPS 2017). Source: https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

- Formalizes coverage and selective risk and learns a reject function for a target risk.
- Supports risk-coverage evaluation; an abstention is neither a correct pronunciation nor an error-free edit.
- A raw softmax or one-prototype margin is not a calibrated open-set guarantee.

### Control-theoretic neural ODE analysis

Controllability Gramians for linearized time-varying systems are classical. A recent neural-ODE-specific empirical treatment is Md Saiful Islam and Rahul Bhadani, “Control-Theoretic View of Neural ODEs: Empirical Controllability and Observability” (2026), https://arxiv.org/abs/2606.08431. It studies local trajectory linearization and empirical Gramians, not TTS editing. This supports using a Gramian as a diagnostic, but not claiming the diagnostic itself as novel.

## Synthesis and limitations

The literature already supplies each isolated ingredient: latent/source optimization, null-space projection, flow differentiation, associative retrieval, localized inpainting, and selective prediction. The defensible opportunity is narrower: test whether a **time-indexed, low-dimensional control basis derived from paired target/protected acoustic sensitivities** gives a measurable Pareto improvement over static conditioning and unconstrained controls, then store only controls whose applicability and actuator feasibility are calibrated separately. Novelty remains unverified until broader citation-network review.

No paper reviewed establishes all of the following together: contextual homograph sense inference, matrix-free target-versus-protected flow controllability, time-indexed low-rank controls, feasibility-based abstention, and lifelong reuse of successful controls in frozen flow-matching TTS. This is a candidate gap, not a novelty claim.

