# Controllability mathematics and incremental research direction

## Local controlled flow

Let the frozen F5 state obey `x_(k+1) = x_k + dt_k v_theta(x_k, t_k, c + M delta)`, where `M` restricts the perturbation to target-character embeddings. Linearizing about `delta=0` gives

`z_(k+1) = (I + dt_k A_k) z_k + dt_k B_k M delta`,

with `A_k = d v_theta / d x` and `B_k = d v_theta / d c`. The final acoustic tangent is `z_N = J_delta delta`. `research.instrumentation.sensitivity` computes products with this operator without materializing the full Jacobian and validates them against central differences.

For target and protected feature projections `C_T` and `C_P`, a finite control basis `U` produces response matrices `G_T = C_T J_delta U` and `G_P = C_P J_delta U`. The locally most selective coefficient solves

`max_a  a^T G_T^T G_T a / a^T (G_P^T G_P + rho I) a`.

This is a generalized Rayleigh quotient. Its mathematics is established; it provides a diagnostic and basis-selection rule, not an originality claim. A large leading value indicates local target/protected separation. Rank deficiency, poor conditioning, or failure at finite step sizes invalidates the local prediction.

## Paper-worthy strategy if the actuator succeeds

The potentially incremental mechanism is **counterfactual tangent distillation**:

1. generate paired oracle acoustic counterfactuals with identical noise and speaker;
2. extract the desired local acoustic displacement only inside an uncertainty-aware target region;
3. solve for a minimum-energy conditioning edit in a matrix-free target/protected tangent subspace;
4. distill the successful edit into a context-conditioned policy only after measuring transfer and opposite-sense rejection.

Unlike ordinary FlowEdit reference matching, the proposed research object is the *projected counterfactual displacement and its controllability certificate*: estimated rank, conditioning, protected leakage, and nonlinear realization error. Unlike null-space editing, it does not update weights and makes no global preservation guarantee. Novelty remains conditional on a broader prior-art check and empirical superiority over the weighted-loss baseline now implemented.

## Scientific status

The CPU tests establish correct local differential calculations on controlled systems, not F5 pronunciation control. Until the IIT run produces real WAVs and pronunciation-sensitive judgments, the only defensible decision is Outcome B: implementation ready, evidence incomplete.
