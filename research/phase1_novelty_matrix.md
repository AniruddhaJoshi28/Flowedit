# Phase 1 novelty matrix

| Candidate element | Closest verified prior | Classification | Consequence |
|---|---|---|---|
| Optimize text-conditioning perturbation through frozen F5-TTS | FlowEdit | Established method | Baseline only |
| Optimize source noise through a frozen flow | D-Flow | Established method | Baseline only |
| Null-space projection for pronunciation preservation | SonoEdit | Established method | Baseline/constraint component only |
| Context-keyed Hopfield correction memory | FlowEdit | Established method | Baseline only |
| Cosine k-NN/EMA prototypes | broad metric/continual learning practice | Established method | Simple mandatory baseline |
| Risk-coverage/open-set abstention | selective classification literature | Established method applied to pronunciation routing | Must calibrate on development data |
| Local waveform or mel inpainting | SpeechPainter and speech-editing literature | Established method | Acoustic baseline only |
| Time-dependent additive control of the F5-TTS vector field | continuous-time control plus guided flow literature | Established method applied to a different setting | Insufficient alone |
| Matrix-free target/protected JVP subspace for a frozen TTS trajectory | SonoEdit-like preservation plus local flow sensitivity | Potentially new algorithmic combination | Must beat static projection under matched compute |
| Generalized-eigenvector control basis maximizing target sensitivity per protected change and energy | classical constrained control/Rayleigh quotient | Established mathematics applied to a different setting | Value is empirical mechanism, not theorem novelty |
| Time-windowed basis selected by conditional target/protected controllability ratio | control Gramians plus TTS trajectory | Potentially new mathematical mechanism | Strongest candidate; novelty unverified |
| Store reusable controls with separate semantic applicability and acoustic-feasibility gates | FlowEdit memory plus selective prediction plus control diagnostics | Potentially new algorithmic combination | Strong lifelong-learning candidate |
| Global preservation guarantee for nonlinear TTS | none justified by local linearization | Unsupported | Must not claim |

## Three strongest research opportunities

1. **Target/protected controllability quotient over time.** Estimate matrix-free sensitivities of a validated target-pronunciation representation and protected non-target representation to compact controls at each flow interval. Compare the leading generalized eigenvalue and subspace against achieved edit selectivity. Distinctiveness: time-resolved target/protected actuation, not generic null-space projection. Feasibility: small randomized JVP/VJP rank on one sentence first.

2. **Feasibility-aware selective memory.** Separate `P(sense | context)` from `P(edit succeeds without collateral damage | control, trajectory)`. Apply a correction only if both calibrated gates pass. This directly addresses the single-prototype margin failure: with one sense stored, semantic competition contains no negative evidence, while actuator feasibility and distance-to-support still can reject.

3. **Transported low-rank control policies.** Store coefficients in a shared compact basis plus trajectory/time support, not full mel perturbations. Transfer only after aligning the new target sensitivity subspace to the stored one (e.g., principal angles/projection residual). This could give a mathematical transfer condition, but computation and alignment robustness are uncertain.

## Smallest discriminating experiment

Use one `lead` pair, one speaker reference, one seed, identical duration and official time grid. First verify a real source-noise or text-conditioning actuator changes the pronunciation-sensitive target representation. Then compare matched-energy controls:

1. static conditioning delta;
2. the same direction applied at every trajectory step;
3. a single early, middle, or late time window;
4. a target/protected projected time-window control.

Record target loss change, protected mel change outside an alignment uncertainty band, solver evaluations, wall time, and parameter hashes. The hypothesis survives only if a time-windowed or projected intervention improves the target/protected Pareto frontier across more than one seed. A change in mel MSE alone is not pronunciation success.

Stop after this actuator gate if no method changes a validated pronunciation judgment.

