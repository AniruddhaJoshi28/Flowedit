# Phase 2 competing hypotheses and gates

## A. Context-conditioned trajectory control

Question: at matched control energy, does time-dependent intervention improve target pronunciation versus a static edit? Evidence: paired real-checkpoint trajectories, identical noise/conditioning, acoustic judgment, protected-region change. Failure condition: no Pareto improvement over static conditioning or prohibitive CPU/GPU cost.

## B. Acoustic controllability and protected subspaces

Question: does the local control-to-target Jacobian contain directions with materially smaller protected-output gain? Evidence: matrix-free JVP/VJP singular values, generalized target/protected quotient, numerical residuals, and realized nonlinear edits. Failure condition: target and protected sensitivities are inseparable, ill-conditioned, or linear predictions do not transfer to finite edits.

## C. Context-conditioned lifelong control memory

Question: can successful compact controls transfer only when both semantic context and acoustic sensitivity are compatible? Evidence: one-shot transfer, opposite-sense protection before and after the conflicting sense is taught, calibrated risk-coverage, sequential retention. Failure condition: simpler k-NN or respelling dominates, or single-prototype intrusion remains unbounded.

## Current ranking

Start with B as a diagnostic and A as the actuator comparison. Do not implement C until a real acoustic actuator passes. The strongest combined candidate is a time-windowed low-rank control selected by a target/protected controllability quotient, with semantic and feasibility gates kept distinct.

