# Review of the Assignment 2 plan

Reviewed against the four-page uploaded assignment, original-paper details already read, and the pasted handoff. This is a plan review, not evidence that assignment experiments have run.

## Findings and resolutions

1. **High — PGD convergence was underspecified.** At x near 1000, small gradients can make unit-step descent appear converged while the objective remains materially suboptimal. Added step growth, projected Armijo backtracking, gradient-mapping diagnostics, solver-aware acceptance, separate termination statuses and independent reference-gap checks. Actual convergence of all 12 starts remains an experiment, not a promise.

2. **High — “exact reference” needed a stronger numerical qualification.** Enumerating every configuration and freshly checking only the apparent winner does not resolve nearly tied optima. Added small-system conditioning/residual checks, refinement of all competitive configurations, a runner-up gap and a deterministic reference-tie policy. Float64 exhaustive search remains exhaustive up to stated numerical uncertainty, not a formal exact-arithmetic proof.

3. **High — Random and bonus costs were insufficiently controlled.** Full inverse updates for every prefix and rejected swap would cause unnecessary dense work. Added explicit simultaneous-update formulas for fixed-base Random/exhaustive evaluations and rank-two scoring for swaps. Large-state changes occur only on accepted bonus moves. Standalone setup cost must still be counted.

4. **Medium — a Bellman reference vector was not explicitly connected to an attainable policy.** Added endpoint-policy extraction, an independent equilibrium solve, and a Bellman-residual check on that policy. Also separated fixed-l epsilon complexity from general exact bit-complexity claims.

5. **Medium — tolerance-based pairwise tie selection can depend on iteration order.** Added a batch-best comparison, uncertainty refinement and a canonical ID/endpoint rule. Added cache-update equations and the requirement to use old state/copy aliased views.

6. **Medium — measurement and comparison scope needed tightening.** Added fresh-interpreter memory workers, cold versus amortized setup, resumed-run accounting, Random aggregation rules, and an explicit U/P-max-only scope for T* comparisons. The required main table covers four methods, three scenarios and three budgets.

7. **Low — “self-contained entry points” contradicted the planned shared imports.** Clarified that scripts may import routines from other included required task scripts, with no external workspace helper dependency.

All seven findings were addressed in IMPLEMENTATION_PLAN.md. No assignment implementation was added.

## Mathematical checks executed during this review

A deterministic seven-node weighted undirected numerical probe checked:

- Rank-one updated equilibrium versus an independent dense solve.
- Updated inverse column sums and diag(PG) versus independently recomputed caches.
- A simultaneous three-node endpoint change using the small Woodbury system versus a fresh equilibrium solve.
- The X-coordinate adjoint gradient versus central finite differences, maximum absolute discrepancy approximately 4.60e-11.

The weighted probe checks algebra for row-stochastic P; it is not a replacement for required tests on simple unweighted graphs or Facebook. The two-node non-convexity/non-concavity counterexample had already been checked with exact fractions in the planning turn.

## Remaining execution risks

- PGD may not meet a global reference tolerance from every start within the initial iteration budget; investigate scaling and stopping behavior without replacing the assigned algorithm or fabricating agreement.
- The corrected-paper attribution still needs source inspection before use in the report. The counterexample independently supports the convexity discrepancy.
- Full-graph inverse setup, solver conditioning, bonus scan counts and witness discovery need measured pilots. No production runtime estimate is established.
- Required non-submodularity evidence must be found on Facebook. A separate supermodularity witness is planned to substantiate the broader claim; failure to find it is not proof that no such witness exists.

The revised plan covers the rubric and resolved user conventions. The next implementation milestone is deterministic data preparation plus the numerical validation suite, followed by a one-round performance pilot. No additional user decision is currently needed.
