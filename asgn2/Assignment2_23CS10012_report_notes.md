# Assignment 2: material for writing your own report

These are fixed reference notes from the saved `results-local/` experiment.
They contain the equations, explanations, numerical comparisons, witnesses and
plot guidance for writing your report. Rerun the task scripts and check their
outputs against these notes yourself; the Python code does not update this file.
Use your new run's values and timings when they differ from this snapshot.

The quick findings table below summarizes the saved run. Detailed tables follow
the writing guide. `True` means the saved comparisons support the claim for this
seeded experiment. You choose the report's wording and layout.

| profile | direction | initial | greedy_k30 | improvement | beats_both_baselines_at_every_budget | increasing_marginal_steps |
| --- | --- | --- | --- | --- | --- | --- |
| U | max | 2034.815972 | 2087.060183 | 52.244210 | True | 3 |
| P | max | 2006.049365 | 2091.745296 | 85.695931 | True | 2 |
| U | min | 2034.815972 | 1980.341463 | 54.474510 | True | 1 |


## 1. Problem, graph and experimental setup

Explain that resistance controls how much a node retains its innate opinion.
Changing resistance changes the equilibrium of the whole network; the objective
is the **sum** of equilibrium opinions, not their mean.

Use the saved manifest for the dataset checksum, software versions, machine,
seed, node/edge counts and preprocessing. The expected ego-Facebook graph has
4,039 nodes, 88,234 undirected edges and one connected component. Original SNAP
node labels are retained. Report the observed numbers of removed loops and
collapsed duplicate edges rather than assuming they are zero.

State these conventions:

- Resistance bounds: `LOW = 0.001`, `HIGH = 1`.
- Initial resistance: Uniform[0.001, 1], shared between the two opinion profiles.
- U: innate opinions Uniform[0, 1]. P: Beta(0.5, 0.5), concentrating more mass
  near the two extremes. These are fixed draws, not repeated profile samples.
- Independent `SeedSequence([2026, stream])` streams: alpha=0, U=1, P=2.
  Random uses `1000 + 100*pool_code + trial`, with all-node pool code 0,
  shortlist code 1 and restricted C code 2; the local random X start uses 3000.
- Budgeted comparisons: U-max, P-max and U-min at k = 1, 2, 3, 4, 5, 10, 20, 30.
  Unbudgeted local search covers both min/max for both U/P.
- Random: 20 seeded trials; shade plots by one **sample standard deviation**,
  not a confidence interval. This variability is conditional on this graph and
  these fixed initial resistances/opinions.
- Time includes setup and numerical verification. Each main worker runs in a
  fresh subprocess. RSS is Linux total-process peak `ru_maxrss / 1024` MiB,
  including Python and native arrays. It is not an incremental allocation count.
  Random's time includes setup plus the sum of the 20 trial-prefix times.

## 2. Task 1: equilibrium, endpoints and fresh-solve greedy

Write the model, with A = diag(alpha), row-stochastic neighbor matrix P and
innate opinion vector s:

```text
z(t+1) = A s + (I-A) P z(t)
M = I - (I-A) P
z* = M^(-1) A s                  f(alpha) = 1^T z*
```

Because alpha_i >= 0.001, the update contracts in the infinity norm by at most
0.999. This gives a unique equilibrium. The implementation solves the sparse
linear system using checked BiCGSTAB (`rtol = atol = 2e-13`), with a checked
sparse direct solve on failure. The saved `iteration_error` is the maximum
coordinate difference between a separate fixed-point iteration and the linear
solve; the acceptance threshold is 1e-8. Give the actual U/P errors from the
initial-equilibrium table.

For a two-node edge, let D = alpha_a + alpha_b - alpha_a*alpha_b:

```text
z_a = [alpha_a*s_a + (1-alpha_a)*alpha_b*s_b] / D
z_b = [alpha_b*s_b + (1-alpha_b)*alpha_a*s_a] / D
```

The saved analytic checks compare this formula with numerical solutions.

Explain the endpoint argument: with other resistances fixed, f is a
linear-fractional function of one resistance. Its denominator stays positive,
and its derivative has constant sign (or is zero). Therefore one endpoint is
at least as good as every interior point. Replacing coordinates one by one
shows an endpoint optimum exists. Flat coordinates can also be optimal inside
the interval. Use `endpoint_sweeps.png` as illustrative numerical evidence; the
argument, rather than the finite sweep alone, establishes the property.

Describe fresh-solve greedy: evaluate both endpoints for each remaining node
among the 200 highest-degree candidates, on the **full graph**, retain previous
assignments, and accept the best choice. Degree ties use original node ID;
objective ties within 1e-9 use ID then lower endpoint. Ten rounds use
`2 * (200 + 199 + ... + 191) = 3910` candidate equilibrium solves per profile.
Verification/fallback solves are counted separately.

Use the per-round table for node IDs, endpoints, objective, marginal gain,
cumulative time and memory. Include k=1,2,3,4,5,10 checkpoints if space is limited.

## 3. Task 2: fast greedy, baselines and Random

Derive the rank-one update. With G=M^(-1), an update at node v has
`delta = alpha_new[v] - alpha[v]`, `g = G[:,v]`, `h = P[v,:]G` and
`d = 1 + delta*h[v] > 0`:

```text
M_new = M + delta*e_v*P[v,:]
G_new = G - (delta/d)*g*h
z_new = z + (delta/d)*(s[v] - (Pz)[v])*g
f_new - f = (delta/d)*(s[v] - (Pz)[v])*sum(g)
```

Cache `c = 1^T G` and `t = diag(PG)` to score each candidate from scalars.
For beta=delta/d, update `c -= beta*c[v]*h` and `t -= beta*(P*g)*h` using the
old values. No candidate equilibrium solve is needed. Accepted updates are
checked with an independent solver. There is no division by `1-alpha`, so the
upper bound 1 is supported. Initial dense inversion and each accepted rank-one
update still have substantial cost/memory; avoid claiming the whole algorithm
has constant cost.

Compare fresh versus fast on the **same shortlist and ten rounds** using the
runtime table. State whether every selected node, endpoint and objective agrees
(within 1e-7). The separate all-node fast run has a larger candidate universe;
its runtime is useful context, but is not the like-for-like speedup denominator.

Define the alternatives precisely:

| Method | Selection and resistance assignment |
| --- | --- |
| Greedy (`fast`) | Recompute the best marginal over every eligible node and both endpoints; retain previous assignments. |
| Baseline I | Rank original innate opinions s_i descending; set selected resistance to 1. |
| Baseline II | Rank `[degree_i/(2m)] * s_i / sum_neighbor(s)` descending; set selected resistance to 1. Positive/zero is infinity and 0/0 is zero. |
| Random | Draw 20 uniform node permutations and use nested prefixes. For each new node, choose its better endpoint given the earlier assignments. Earlier endpoints are retained. |

For minimization, baseline rankings use 1-s and their final objectives use s.
Random does not jointly optimize all endpoints of its selected set. Small
Woodbury systems evaluate its changes against a fixed inverse.

## 4. Task 2 part 2: corrected non-convex local search

Explain the Section 6.1 basic single-coordinate strategy in the corrected
paper, following the TA clarification of 11 November 2026. The implemented
method is non-convex local search. Do not describe it as projected gradient
descent, or as the approximate/optimistic algorithm of Section 6.3.

For minimization, solve equilibrium and scan nodes by original ID. Compute
`d_i = s_i - (Pz)_i`. A positive value favors lowering alpha_i to 0.001;
a negative value favors raising it to 1. Accept the first reliably improving
coordinate, then solve equilibrium again. Keep exact or unresolved ties
unchanged, and do not perturb the input opinions. Maximize by minimizing 1-s;
use a fresh solve with original s for the reported objective.

Explain the upper-bound adaptation: the paper states bounds strictly below 1.
At alpha_i=1, `s_i-z_i=0` can falsely suggest no move. The derivative is instead
`M^(-1)e_i * (s_i-(Pz)_i)`, with a nonnegative inverse column of positive sum.
The neighbor-average test remains useful at alpha_i=1.

The canonical all-upper start only flips upper to lower. In exact arithmetic,
accepted minimizing moves decrease z componentwise, so s-Pz cannot decrease;
a coordinate cannot need a reverse flip. Hence at most n accepted moves.
This guarantee does not apply to the supplementary starts.

| Start | Initial resistance | Explanation |
| --- | --- | --- |
| upper | All 1 | Canonical strategy; at most n moves. |
| initial | Sampled assignment alpha | Original initial-resistance start. |
| x_midpoint | All 1/500.5 | Invert the midpoint of the X box [1,1000]. |
| x_random | Reciprocal of Uniform[1,1000], stream 3000 | Invert the random X start. |

Include the 16 saved runs: starting/final objective, moves, endpoint count,
status, objective error bound, reference gap, runtime and memory. Compare all
four starts for each scenario. Equal objectives do not require identical
resistance vectors; supplementary starts may retain interior coordinates at ties.

State the actual numerical acceptance rules. Each move must improve the
objective by more than combined equilibrium-solve uncertainty. If a scan finds
no resolvable move, repeat using direct solves. The final direct equilibrium
is checked against the endpoint Bellman map:

```text
H_i(z) = min or max over a in {0.001,1} of [a*s_i + (1-a)*(Pz)_i]
objective error estimate = n*||H(z)-z||_infinity / 0.001
```

The code adds roundoff and equilibrium-solve allowances. Every run must have
status `certified`, bound <=1e-5 and absolute independent reference gap <=1e-5;
the four-start spread must also be <=1e-5. The independent reference uses
simultaneous endpoint policy improvement. These are floating-point residual
diagnostics, not rigorous interval certificates. A move cap is not convergence.

Optional non-convexity explanation: in reciprocal coordinates, the two-node
example s=(1,0) gives `f(x,y)=(2y-1)/(x+y-1)`. At (2,2), f=1 exceeds
`[f(2,1)+f(2,3)]/2=7/8` but is below
`[f(1,2)+f(3,2)]/2=9/8`; therefore it is neither convex nor concave.

## 5. Task 2 claims: use explicit witnesses and signed gaps

Define F(S) as the best maximization objective after changing only nodes in S.
For A contained in B and v outside B, compare
`small = F(A+v)-F(A)` and `large = F(B+v)-F(B)`.

- `large > small` violates submodularity.
- `large < small` violates supermodularity.

Each saved witness gives original node IDs A, B, v, the four objectives in the
order **A, A+v, B, B+v**, and the corresponding optimized endpoint assignments
(in sorted set order). Copy these concrete sets/values and compare the gap
with the recorded numerical error bound. The algorithm enumerates all endpoint
assignments for each witness set and checks them with fresh equilibrium solves.
Separate witnesses are needed for the two properties.

The triangle example has s=(1,0,0) and original alpha=(0.1,0.1,0.1).
The paper calls it Example 4; the assignment calls it Example 13. The saved
values are for sets empty, {1}, {3}, {1,2}, {1,3}, {1,2,3} in that order.
The printed marginal about 1.493 omits F(empty)=1; the corrected marginal is
about 0.493. The non-supermodularity violation remains. Use the saved example
JSON for the full numerical comparison.

For the greedy-versus-baselines claim, use every requested budget. Define the
signed gap as `f_greedy-f_baseline` for max and `f_baseline-f_greedy` for min.
The claim of strict superiority everywhere holds for a scenario only if **all**
its reported gaps exceed 1e-8. Report the actual verdict; an empirical win on one
seed does not establish an approximation guarantee.

## 6. Task 3: exact restricted ground truth

Define C as the union of the 15 highest-degree nodes and the 15 nodes with the
largest best singleton gain. Include |C| and the overlap of these two lists.
For each k=2,3, enumerate every subset of C and every endpoint assignment:
`binomial(|C|,k) * 2^k` configurations. Competitive winners are independently
verified; objective ties use tolerance 1e-9, then IDs/endpoints.

Use the exact-reference table for the winning IDs, endpoints, objective,
configuration/subset counts, cumulative runtime and peak memory. These are
numerically checked optima **within C**, not proven all-node optima.

The Woodbury evaluation for a changed set S is:

```text
E = columns selecting S; D = diag(alpha_new[S]-alpha[S]); R = P[S,:]
(I + D R G E) w = D (s[S] - R z)
f_new = f_base + (1^T G E) w
```

For a maximization method T at the same k, report:

```text
gain_ratio = [f(T)-f(initial)] / [f(T*_C)-f(initial)]
Jaccard = |T intersect T*_C| / |T union T*_C|
```

Label the candidate universe as `all` or `C`. All-node ratios can exceed 1
because the reference is restricted. Random ratios and Jaccards are means of
the 20 trial statistics; do not compare an average objective to an invented
average target set. If the denominator is zero, the ratio is undefined.

## 7. Task 4: comparisons and interpretation

Use the budgeted table for f, signed improvement, normalized `gain_fraction`,
Random sample standard deviation, cumulative seconds and peak MiB. Put k=5,10,30
in the main report if you need a compact table; all eight budgets are included
in the notes for complete comparisons.

The normalized fraction uses the **unbudgeted canonical local-search reference**:
`sign*(f_method-f_initial) / [sign*(f_unbudgeted-f_initial)]`, with sign=+1 for
max and -1 for min. This differs from the restricted-k gain ratio in Task 3.
Do not call the unbudgeted value an optimum under budget k.

Useful figures and what to say about them:

| Figure | Interpretation/caption guidance |
| --- | --- |
| objective_U_max / objective_P_max / objective_U_min | Left: absolute objectives with initial and unbudgeted reference lines. Right: zoom on budgeted methods. Random shading is one sample standard deviation over 20 trials. |
| runtime_U / runtime_P | Cumulative time versus k. Compare fresh/fast on matching shortlist through k=10; all-node fast is additional context. |
| targets_U_max / targets_P_max / targets_U_min | Selected nodes' innate opinions versus **pre-intervention** mean neighbor equilibrium, marked by assigned endpoint. |
| endpoint_sweeps | Three one-coordinate sweeps illustrate endpoint behavior. |
| bonus_U_max / bonus_P_max / bonus_U_min | Extra improvement from flip/swap search at each budget, including zero values. |

For maximization, raising resistance helps when s_i exceeds the **current**
neighbor average, while lowering it helps when s_i is below that average.
Minimization reverses these choices. The scatter uses the initial neighbor
average, so later-round decisions need not lie on its expected side of the
initial diagonal. Retained earlier assignments can also become less attractive.

Report how often consecutive greedy marginal gains increase by >1e-8. Such
increases warn against assuming diminishing gains along this trajectory; use
the explicit set-function witnesses to establish failure of submodularity.

## 8. Bonus: flip/swap improvement

Starting from greedy at each requested budget, consider endpoint flips and
swapping a selected node with an unselected node in the 200-degree shortlist.
Removing a selected node restores its original alpha; inserted nodes try both
endpoints. Scan selected original IDs ascending, then shortlist order, lower
endpoint first. Apply the best improvement above 1e-8 and repeat until a full
scan finds none. Accepted moves receive independent equilibrium checks.

Use the bonus table to report the improvement at every budget and the fraction
of starts already locally optimal. Zero is a valid result. A completed scan
establishes local optimality only for this specified flip/shortlist-swap
neighborhood and tolerance. It does not establish global budgeted optimality.
Flips can revise earlier endpoint decisions; swaps can undo earlier node choices.

## 9. Validation, limitations and references

Use the validation evidence to describe two-node analytic checks, separate
fixed-point iteration, cache/Woodbury comparisons, dense exhaustive small-graph
local-search optima, replayed moves, complement symmetry, constant-opinion ties,
original-ID ordering and the alpha=1 boundary. On two nodes with s=(1,0), the
unbudgeted extrema are 0.001 and 1.999. A zero-move cap must return an
uncertified result on a non-optimal start.

Separate numerical accuracy from statistical generality: one graph and one
seeded draw per profile do not establish behavior over arbitrary networks or
opinion distributions. Timings depend on the recorded environment. Use its
manifest for the run you use in your report. The facts below retain the
provenance of the earlier saved run; check new outputs and update your report yourself.

References to include (check your course's citation format):

- SNAP ego-Facebook dataset: https://snap.stanford.edu/data/ego-Facebook.html
- Abebe, Kleinberg, Parkes and Tsourakakis, *Opinion Dynamics with Varying
  Susceptibility to Persuasion*, KDD 2018, pp. 1089–1098.
  DOI: 10.1145/3219819.3219983. Endpoint, greedy and baseline definitions.
- Abebe, Chan, Kleinberg, Liang, Parkes, Sozio and Tsourakakis, *Opinion Dynamics
  with Varying Susceptibility to Persuasion via Non-Convex Local Search*,
  arXiv:2011.02583v1 (2020), Sections 5 and 6.1.
  https://arxiv.org/abs/2011.02583
- Your supplied CS60017 assignment and the TA clarification of 11 November 2026.

Before exporting your PDF, check that every numerical claim has its table/raw
record, the restricted and unbudgeted references are labeled correctly, and
all figure axes/captions identify the objective direction and profile.


## Saved numerical facts

These tables are a fixed snapshot of the saved experiment. The hashes below were recorded when the notes were prepared; they are not updated by later source edits or reruns.

```json
{
  "recorded_experiment_hash": "e9e8c52c2dc6c346229b50afe8d69ef0f9e9462bbece83862f3970da9874ea58",
  "analysis_source_hash_when_notes_were_prepared": "a8d75459b62a6554a535e55f48d4902a8fcadcb3cfc3c31281e8aaa2e5df5622",
  "dataset_sha256": "f41c026ed8af3cc3359f1ca5573d0605fb09ae0eefa34544b820fd8c6e2ef296",
  "results_directory": "/home/chrisrex/acads/soccomp/asgn2/results-local",
  "recorded_environment": {
    "dataset_sha256": "f41c026ed8af3cc3359f1ca5573d0605fb09ae0eefa34544b820fd8c6e2ef296",
    "nodes": 4039,
    "edges": 88234,
    "components": 1,
    "raw_lines": 88234,
    "removed_loops": 0,
    "collapsed_duplicates": 0,
    "seed": 2026,
    "streams": {
      "alpha": 0,
      "U": 1,
      "P": 2,
      "random": "1000 + pool_code*100 + trial",
      "local_x_start": 3000
    },
    "python": "3.12.14",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "networkx": "3.6.1",
    "pandas": "3.1.1",
    "matplotlib": "3.11.1",
    "machine": "x86_64",
    "processor": "x86_64",
    "cpus": 96,
    "threads": {
      "OPENBLAS_NUM_THREADS": null,
      "OMP_NUM_THREADS": null,
      "MKL_NUM_THREADS": null
    },
    "experiment_hash": "e9e8c52c2dc6c346229b50afe8d69ef0f9e9462bbece83862f3970da9874ea58"
  }
}
```

**Source revision differs from the saved run.** These are recorded-run results; analysis exports do not certify a new numerical execution. The saved manifest and completion hash are preserved.


### Initial equilibrium

| profile | f | iteration_error | steps |
| --- | --- | --- | --- |
| U | 2034.815972 | 1.00614e-12 | 51 |
| P | 2006.049365 | 1.46544e-12 | 53 |


### Task 1: every greedy round (200-node shortlist)

| profile | k | added | alpha | f | gain | seconds | peak_mib | candidate_solves |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U | 1 | 1684 | 0.001000 | 2038.604013 | 3.788041 | 2.122204 | 114.000000 | 400 |
| U | 2 | 348 | 1.000000 | 2042.216701 | 3.612688 | 4.225958 | 114.000000 | 798 |
| U | 3 | 414 | 1.000000 | 2045.720239 | 3.503538 | 6.274140 | 114.000000 | 1194 |
| U | 4 | 3437 | 0.001000 | 2049.122411 | 3.402171 | 8.350094 | 114.000000 | 1588 |
| U | 5 | 1431 | 0.001000 | 2050.967348 | 1.844937 | 10.388280 | 114.000000 | 1980 |
| U | 6 | 1621 | 0.001000 | 2052.579904 | 1.612556 | 12.429717 | 114.000000 | 2370 |
| U | 7 | 1211 | 0.001000 | 2054.181838 | 1.601934 | 14.436507 | 114.000000 | 2758 |
| U | 8 | 2543 | 0.001000 | 2055.639448 | 1.457610 | 16.433494 | 114.000000 | 3144 |
| U | 9 | 1559 | 1.000000 | 2056.830346 | 1.190898 | 18.413809 | 114.000000 | 3528 |
| U | 10 | 2007 | 0.001000 | 2057.992578 | 1.162232 | 20.382563 | 114.000000 | 3910 |
| P | 1 | 107 | 0.001000 | 2025.000251 | 18.950887 | 2.091887 | 114.000000 | 400 |
| P | 2 | 1684 | 1.000000 | 2032.726543 | 7.726292 | 4.162935 | 114.000000 | 798 |
| P | 3 | 3437 | 0.001000 | 2039.514408 | 6.787864 | 6.211963 | 114.000000 | 1194 |
| P | 4 | 1912 | 0.001000 | 2043.875718 | 4.361310 | 8.251142 | 114.000000 | 1588 |
| P | 5 | 0 | 0.001000 | 2047.459113 | 3.583396 | 10.282527 | 114.000000 | 1980 |
| P | 6 | 414 | 0.001000 | 2049.935490 | 2.476376 | 12.391698 | 114.000000 | 2370 |
| P | 7 | 1431 | 0.001000 | 2051.975625 | 2.040135 | 14.487842 | 114.000000 | 2758 |
| P | 8 | 1800 | 0.001000 | 2053.608515 | 1.632890 | 16.575777 | 114.000000 | 3144 |
| P | 9 | 1610 | 0.001000 | 2055.241373 | 1.632858 | 18.650429 | 114.000000 | 3528 |
| P | 10 | 2333 | 0.001000 | 2056.502117 | 1.260744 | 20.713108 | 114.000000 | 3910 |


### Task 1: matching-shortlist runtime

| profile | fresh_seconds | fast_seconds | speedup |
| --- | --- | --- | --- |
| U | 20.382563 | 3.647248 | 5.588477 |
| P | 20.713108 | 2.559285 | 8.093319 |


### Task 2/4: all budgeted comparisons

| profile | direction | method | k | f | improvement | std | gain_fraction | seconds | peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U | max | fast | 1 | 2038.604013 | 3.788041 |  | 0.002026 | 1.919452 | 529.328125 |
| U | max | fast | 2 | 2042.216701 | 7.400729 |  | 0.003958 | 1.993124 | 529.328125 |
| U | max | fast | 3 | 2045.720239 | 10.904267 |  | 0.005832 | 2.066714 | 529.328125 |
| U | max | fast | 4 | 2049.122411 | 14.306438 |  | 0.007652 | 2.141545 | 529.328125 |
| U | max | fast | 5 | 2051.966521 | 17.150549 |  | 0.009173 | 2.215037 | 529.328125 |
| U | max | fast | 10 | 2061.520692 | 26.704719 |  | 0.014283 | 2.586076 | 529.328125 |
| U | max | fast | 20 | 2075.305882 | 40.489910 |  | 0.021656 | 3.335692 | 529.328125 |
| U | max | fast | 30 | 2087.060183 | 52.244210 |  | 0.027943 | 4.098352 | 529.328125 |
| U | max | I | 1 | 2035.100368 | 0.284396 |  | 0.000152 | 0.011804 | 114.000000 |
| U | max | I | 2 | 2035.842692 | 1.026719 |  | 0.000549 | 0.017601 | 114.000000 |
| U | max | I | 3 | 2036.831428 | 2.015455 |  | 0.001078 | 0.023330 | 114.000000 |
| U | max | I | 4 | 2037.838349 | 3.022377 |  | 0.001617 | 0.029016 | 114.000000 |
| U | max | I | 5 | 2038.396340 | 3.580368 |  | 0.001915 | 0.034680 | 114.000000 |
| U | max | I | 10 | 2040.879236 | 6.063263 |  | 0.003243 | 0.062359 | 114.000000 |
| U | max | I | 20 | 2046.157503 | 11.341531 |  | 0.006066 | 0.116687 | 114.000000 |
| U | max | I | 30 | 2050.902221 | 16.086248 |  | 0.008604 | 0.170375 | 114.000000 |
| U | max | II | 1 | 2035.202322 | 0.386350 |  | 0.000207 | 0.011880 | 114.000000 |
| U | max | II | 2 | 2035.844630 | 1.028658 |  | 0.000550 | 0.017631 | 114.000000 |
| U | max | II | 3 | 2036.062691 | 1.246718 |  | 0.000667 | 0.023345 | 114.000000 |
| U | max | II | 4 | 2036.270853 | 1.454881 |  | 0.000778 | 0.028975 | 114.000000 |
| U | max | II | 5 | 2036.501587 | 1.685615 |  | 0.000902 | 0.034574 | 114.000000 |
| U | max | II | 10 | 2037.375353 | 2.559380 |  | 0.001369 | 0.062025 | 114.000000 |
| U | max | II | 20 | 2038.954054 | 4.138082 |  | 0.002213 | 0.115665 | 114.000000 |
| U | max | II | 30 | 2042.045296 | 7.229324 |  | 0.003867 | 0.169293 | 114.000000 |
| U | max | random | 1 | 2035.010838 | 0.194865 | 0.099897 | 0.000104 | 1.842069 | 530.718750 |
| U | max | random | 2 | 2035.277813 | 0.461840 | 0.311322 | 0.000247 | 1.844355 | 530.718750 |
| U | max | random | 3 | 2035.488089 | 0.672116 | 0.378560 | 0.000359 | 1.846066 | 530.718750 |
| U | max | random | 4 | 2035.757993 | 0.942020 | 0.394987 | 0.000504 | 1.847714 | 530.718750 |
| U | max | random | 5 | 2036.128366 | 1.312394 | 0.502875 | 0.000702 | 1.849357 | 530.718750 |
| U | max | random | 10 | 2037.303267 | 2.487294 | 0.746372 | 0.001330 | 1.857812 | 530.718750 |
| U | max | random | 20 | 2039.580043 | 4.764070 | 0.925540 | 0.002548 | 1.876391 | 530.718750 |
| U | max | random | 30 | 2042.575947 | 7.759974 | 1.484936 | 0.004150 | 1.991069 | 530.718750 |
| P | max | fast | 1 | 2025.000251 | 18.950887 |  | 0.009532 | 1.913855 | 530.066406 |
| P | max | fast | 2 | 2032.726543 | 26.677179 |  | 0.013418 | 1.985502 | 530.066406 |
| P | max | fast | 3 | 2039.514408 | 33.465043 |  | 0.016832 | 2.058083 | 530.066406 |
| P | max | fast | 4 | 2044.395388 | 38.346023 |  | 0.019287 | 2.129256 | 530.066406 |
| P | max | fast | 5 | 2048.756722 | 42.707357 |  | 0.021481 | 2.202602 | 530.066406 |
| P | max | fast | 10 | 2061.944625 | 55.895261 |  | 0.028114 | 2.563210 | 530.066406 |
| P | max | fast | 20 | 2077.992468 | 71.943103 |  | 0.036186 | 3.288393 | 530.066406 |
| P | max | fast | 30 | 2091.745296 | 85.695931 |  | 0.043104 | 4.026795 | 530.066406 |
| P | max | I | 1 | 2006.741837 | 0.692473 |  | 0.000348 | 0.011268 | 114.000000 |
| P | max | I | 2 | 2007.496039 | 1.446675 |  | 0.000728 | 0.016826 | 114.000000 |
| P | max | I | 3 | 2007.995945 | 1.946580 |  | 0.000979 | 0.022398 | 114.000000 |
| P | max | I | 4 | 2008.826532 | 2.777167 |  | 0.001397 | 0.027927 | 114.000000 |
| P | max | I | 5 | 2008.939537 | 2.890173 |  | 0.001454 | 0.033385 | 114.000000 |
| P | max | I | 10 | 2010.803477 | 4.754112 |  | 0.002391 | 0.060289 | 114.000000 |
| P | max | I | 20 | 2015.835964 | 9.786600 |  | 0.004923 | 0.112522 | 114.000000 |
| P | max | I | 30 | 2020.844202 | 14.794837 |  | 0.007442 | 0.164590 | 114.000000 |
| P | max | II | 1 | 2006.493725 | 0.444360 |  | 0.000224 | 0.012000 | 114.000000 |
| P | max | II | 2 | 2007.199599 | 1.150234 |  | 0.000579 | 0.017653 | 114.000000 |
| P | max | II | 3 | 2007.380662 | 1.331298 |  | 0.000670 | 0.023246 | 114.000000 |
| P | max | II | 4 | 2007.619127 | 1.569763 |  | 0.000790 | 0.028809 | 114.000000 |
| P | max | II | 5 | 2007.619648 | 1.570284 |  | 0.000790 | 0.034299 | 114.000000 |
| P | max | II | 10 | 2008.846404 | 2.797040 |  | 0.001407 | 0.061382 | 114.000000 |
| P | max | II | 20 | 2009.384033 | 3.334669 |  | 0.001677 | 0.113972 | 114.000000 |
| P | max | II | 30 | 2010.667220 | 4.617856 |  | 0.002323 | 0.166153 | 114.000000 |
| P | max | random | 1 | 2006.280517 | 0.231152 | 0.231699 | 0.000116 | 1.859158 | 530.839844 |
| P | max | random | 2 | 2006.607721 | 0.558357 | 0.263790 | 0.000281 | 1.861427 | 530.839844 |
| P | max | random | 3 | 2006.833436 | 0.784071 | 0.454245 | 0.000394 | 1.863202 | 530.839844 |
| P | max | random | 4 | 2007.089111 | 1.039746 | 0.497088 | 0.000523 | 1.864861 | 530.839844 |
| P | max | random | 5 | 2007.392872 | 1.343507 | 0.609688 | 0.000676 | 1.866551 | 530.839844 |
| P | max | random | 10 | 2008.895815 | 2.846450 | 0.609908 | 0.001432 | 1.875345 | 530.839844 |
| P | max | random | 20 | 2011.952583 | 5.903218 | 1.382083 | 0.002969 | 1.894802 | 530.839844 |
| P | max | random | 30 | 2014.793613 | 8.744248 | 1.786993 | 0.004398 | 2.007357 | 530.839844 |
| U | min | fast | 1 | 2023.303311 | 11.512661 |  | 0.006052 | 1.903532 | 530.710938 |
| U | min | fast | 2 | 2018.995571 | 15.820401 |  | 0.008317 | 1.977299 | 530.710938 |
| U | min | fast | 3 | 2016.500485 | 18.315487 |  | 0.009628 | 2.050414 | 530.710938 |
| U | min | fast | 4 | 2014.038699 | 20.777273 |  | 0.010922 | 2.123549 | 530.710938 |
| U | min | fast | 5 | 2012.219707 | 22.596266 |  | 0.011879 | 2.196425 | 530.710938 |
| U | min | fast | 10 | 2004.715419 | 30.100554 |  | 0.015823 | 2.563784 | 530.710938 |
| U | min | fast | 20 | 1991.962672 | 42.853300 |  | 0.022527 | 3.305730 | 530.710938 |
| U | min | fast | 30 | 1980.341463 | 54.474510 |  | 0.028636 | 4.060390 | 530.710938 |
| U | min | I | 1 | 2034.307506 | 0.508467 |  | 0.000267 | 0.011834 | 114.000000 |
| U | min | I | 2 | 2033.975837 | 0.840135 |  | 0.000442 | 0.017611 | 114.000000 |
| U | min | I | 3 | 2033.755875 | 1.060097 |  | 0.000557 | 0.023331 | 114.000000 |
| U | min | I | 4 | 2033.666461 | 1.149511 |  | 0.000604 | 0.029021 | 114.000000 |
| U | min | I | 5 | 2033.311955 | 1.504018 |  | 0.000791 | 0.034684 | 114.000000 |
| U | min | I | 10 | 2031.095424 | 3.720549 |  | 0.001956 | 0.062225 | 114.000000 |
| U | min | I | 20 | 2025.845466 | 8.970506 |  | 0.004716 | 0.116030 | 114.000000 |
| U | min | I | 30 | 2021.212449 | 13.603523 |  | 0.007151 | 0.169148 | 114.000000 |
| U | min | II | 1 | 2034.287432 | 0.528541 |  | 0.000278 | 0.012077 | 114.000000 |
| U | min | II | 2 | 2034.041479 | 0.774493 |  | 0.000407 | 0.017817 | 114.000000 |
| U | min | II | 3 | 2033.699569 | 1.116404 |  | 0.000587 | 0.023599 | 114.000000 |
| U | min | II | 4 | 2033.551619 | 1.264354 |  | 0.000665 | 0.029301 | 114.000000 |
| U | min | II | 5 | 2033.332104 | 1.483868 |  | 0.000780 | 0.034961 | 114.000000 |
| U | min | II | 10 | 2032.027747 | 2.788226 |  | 0.001466 | 0.062571 | 114.000000 |
| U | min | II | 20 | 2030.109523 | 4.706449 |  | 0.002474 | 0.116402 | 114.000000 |
| U | min | II | 30 | 2025.756490 | 9.059483 |  | 0.004762 | 0.169201 | 114.000000 |
| U | min | random | 1 | 2034.475244 | 0.340728 | 0.307714 | 0.000179 | 1.864975 | 529.226562 |
| U | min | random | 2 | 2034.309775 | 0.506197 | 0.376330 | 0.000266 | 1.867202 | 529.226562 |
| U | min | random | 3 | 2034.069837 | 0.746136 | 0.353438 | 0.000392 | 1.868905 | 529.226562 |
| U | min | random | 4 | 2033.802720 | 1.013253 | 0.525743 | 0.000533 | 1.870525 | 529.226562 |
| U | min | random | 5 | 2033.509981 | 1.305991 | 0.627678 | 0.000687 | 1.872189 | 529.226562 |
| U | min | random | 10 | 2032.397418 | 2.418554 | 0.792070 | 0.001271 | 1.880823 | 529.226562 |
| U | min | random | 20 | 2030.192423 | 4.623550 | 0.986024 | 0.002431 | 1.899561 | 529.226562 |
| U | min | random | 30 | 2027.681036 | 7.134937 | 1.263370 | 0.003751 | 2.012139 | 529.226562 |


### Task 2: all unbudgeted starts

| profile | direction | start_name | initial | f | moves | endpoint_count | status | objective_error_bound | reference_gap | seconds | peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U | max | upper | 2040.720426 | 3904.478179 | 3901 | 4039 | certified | 3.74666e-08 | -3.09228e-11 | 31.822240 | 114.726562 |
| U | max | initial | 2034.815972 | 3904.478179 | 5290 | 4039 | certified | 3.74666e-08 | -3.09228e-11 | 63.593700 | 119.925781 |
| U | max | x_midpoint | 2035.514795 | 3904.478179 | 4258 | 4039 | certified | 3.74666e-08 | -3.09228e-11 | 96.529515 | 128.000000 |
| U | max | x_random | 2064.136438 | 3904.478179 | 4408 | 4039 | certified | 3.74666e-08 | -3.09228e-11 | 92.367478 | 137.343750 |
| U | min | upper | 2040.720426 | 132.539702 | 3902 | 4039 | certified | 1.5068e-08 | 3.9222e-12 | 31.158037 | 132.734375 |
| U | min | initial | 2034.815972 | 132.539702 | 5276 | 4039 | certified | 1.5068e-08 | 3.9222e-12 | 62.299006 | 132.734375 |
| U | min | x_midpoint | 2035.514795 | 132.539702 | 4266 | 4039 | certified | 1.5068e-08 | 3.9222e-12 | 92.660497 | 132.734375 |
| U | min | x_random | 2064.136438 | 132.539702 | 4392 | 4039 | certified | 1.5068e-08 | 3.9222e-12 | 89.308084 | 140.441406 |
| P | max | upper | 2029.716658 | 3994.183406 | 3768 | 4039 | certified | 3.01698e-08 | -7.50333e-11 | 24.399652 | 147.972656 |
| P | max | initial | 2006.049365 | 3994.183406 | 5258 | 4039 | certified | 3.01698e-08 | -7.50333e-11 | 50.808946 | 147.972656 |
| P | max | x_midpoint | 2031.241025 | 3994.183406 | 4276 | 4039 | certified | 3.01698e-08 | -7.50333e-11 | 83.298291 | 147.972656 |
| P | max | x_random | 2025.962957 | 3994.183406 | 4415 | 4039 | certified | 3.01698e-08 | -7.50333e-11 | 81.026822 | 147.972656 |
| P | min | upper | 2029.716658 | 47.759809 | 3773 | 4039 | certified | 1.45579e-08 | 1.06581e-12 | 26.108464 | 147.972656 |
| P | min | initial | 2006.049365 | 47.759809 | 5173 | 4039 | certified | 1.45579e-08 | 1.06581e-12 | 54.377901 | 147.972656 |
| P | min | x_midpoint | 2031.241025 | 47.759809 | 4246 | 4039 | certified | 1.45579e-08 | 1.06581e-12 | 84.214264 | 147.972656 |
| P | min | x_random | 2025.962957 | 47.759809 | 4384 | 4039 | certified | 1.45579e-08 | 1.06581e-12 | 82.503971 | 147.972656 |


### Task 2: non-submodularity and non-supermodularity witnesses

| profile | kind | A | B | v | f | assignments | marginal_small | marginal_large | numerical_error_bound |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U | nonsubmodular | [2593] | [2593, 1048] | 959 | [2035.3221459736797, 2035.6416070757998, 2035.4126097855328, 2035.7323554416894] | [[1.0], [1.0, 1.0], [0.001, 1.0], [1.0, 0.001, 1.0]] | 0.319461 | 0.319746 | 1.31496e-05 |
| U | nonsupermodular | [2496] | [2496, 2399] | 2194 | [2034.8886893179329, 2035.4053014864078, 2035.0257616872407, 2035.5368510729709] | [[1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0, 1.0]] | 0.516612 | 0.511089 | 1.34953e-05 |
| P | nonsubmodular | [] | [2614] | 2621 | [2006.0493646605369, 2006.1177818048056, 2006.8050373424048, 2006.873757715332] | [[], [0.001], [1.0], [1.0, 0.001]] | 0.068417 | 0.068720 | 2.0659e-05 |
| P | nonsupermodular | [2412] | [2412, 1645] | 1740 | [2006.2768153075262, 2007.1912030162857, 2006.3415482267405, 2007.2515787667812] | [[1.0], [0.001, 1.0], [1.0, 1.0], [1.0, 0.001, 1.0]] | 0.914388 | 0.910031 | 2.03971e-05 |


### Task 3: exact restricted references

| profile | k | pool_size | pool_overlap | ids | alpha | f | subsets | configurations | total_seconds | peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U | 2 | 26 | 4 | [348, 1684] | [1.0, 0.001] | 2042.216701 | 325 | 1300 | 1.889448 | 529.605469 |
| U | 3 | 26 | 4 | [348, 414, 1684] | [1.0, 1.0, 0.001] | 2045.720239 | 2600 | 20800 | 2.542768 | 529.605469 |
| P | 2 | 24 | 6 | [107, 1684] | [0.001, 1.0] | 2032.726543 | 276 | 1104 | 1.871107 | 529.550781 |
| P | 3 | 24 | 6 | [107, 1684, 3437] | [0.001, 1.0, 0.001] | 2039.514408 | 2024 | 16192 | 2.393368 | 529.550781 |


### Task 3/4: gain ratios and Jaccard overlaps

| profile | pool | method | k | gain_ratio | jaccard |
| --- | --- | --- | --- | --- | --- |
| U | all | fast | 2 | 1.000000 | 1.000000 |
| U | all | fast | 3 | 1.000000 | 1.000000 |
| U | all | I | 2 | 0.138732 | 0.000000 |
| U | all | I | 3 | 0.184832 | 0.000000 |
| U | all | II | 2 | 0.138994 | 0.000000 |
| U | all | II | 3 | 0.114333 | 0.000000 |
| U | all | random | 2 | 0.062405 | 0.000000 |
| U | all | random | 3 | 0.061638 | 0.000000 |
| U | C | fast | 2 | 1.000000 | 1.000000 |
| U | C | fast | 3 | 1.000000 | 1.000000 |
| U | C | I | 2 | 0.961558 | 0.333333 |
| U | C | I | 3 | 0.669833 | 0.500000 |
| U | C | II | 2 | 0.961558 | 0.333333 |
| U | C | II | 3 | 0.669833 | 0.500000 |
| U | C | random | 2 | 0.358003 | 0.050000 |
| U | C | random | 3 | 0.393214 | 0.095000 |
| P | all | fast | 2 | 1.000000 | 1.000000 |
| P | all | fast | 3 | 1.000000 | 1.000000 |
| P | all | I | 2 | 0.054229 | 0.000000 |
| P | all | I | 3 | 0.058168 | 0.000000 |
| P | all | II | 2 | 0.043117 | 0.000000 |
| P | all | II | 3 | 0.039782 | 0.000000 |
| P | all | random | 2 | 0.020930 | 0.000000 |
| P | all | random | 3 | 0.023430 | 0.000000 |
| P | C | fast | 2 | 1.000000 | 1.000000 |
| P | C | fast | 3 | 1.000000 | 1.000000 |
| P | C | I | 2 | 0.147184 | 0.000000 |
| P | C | I | 3 | 0.171581 | 0.000000 |
| P | C | II | 2 | 0.158482 | 0.000000 |
| P | C | II | 3 | 0.130407 | 0.000000 |
| P | C | random | 2 | 0.200566 | 0.050000 |
| P | C | random | 3 | 0.217275 | 0.040000 |


### Bonus: every tested budget

| profile | direction | k | initial | f | added_gain | already_local | scans | evaluations | seconds | peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U | max | 1 | 2038.604013 | 2038.604013 | 0.000000 | True | 1 | 399 | 1.863619 | 529.968750 |
| U | max | 2 | 2042.216701 | 2042.216701 | 0.000000 | True | 1 | 794 | 1.865353 | 545.894531 |
| U | max | 3 | 2045.720239 | 2045.720239 | 0.000000 | True | 1 | 1185 | 1.872816 | 545.894531 |
| U | max | 4 | 2049.122411 | 2049.122411 | 0.000000 | True | 1 | 1572 | 1.883749 | 545.894531 |
| U | max | 5 | 2051.966521 | 2051.966521 | 0.000000 | True | 1 | 1965 | 1.897806 | 545.894531 |
| U | max | 10 | 2061.520692 | 2061.520692 | 0.000000 | True | 1 | 3870 | 1.956397 | 545.894531 |
| U | max | 20 | 2075.305882 | 2075.305882 | 0.000000 | True | 1 | 7700 | 2.081980 | 545.894531 |
| U | max | 30 | 2087.060183 | 2087.060183 | 0.000000 | True | 1 | 11370 | 2.191612 | 545.894531 |
| P | max | 1 | 2025.000251 | 2025.000251 | 0.000000 | True | 1 | 399 | 1.866163 | 529.992188 |
| P | max | 2 | 2032.726543 | 2032.726543 | 0.000000 | True | 1 | 794 | 1.861255 | 545.894531 |
| P | max | 3 | 2039.514408 | 2039.514408 | 0.000000 | True | 1 | 1185 | 1.874560 | 545.894531 |
| P | max | 4 | 2044.395388 | 2044.395388 | 0.000000 | True | 1 | 1580 | 1.890656 | 545.898438 |
| P | max | 5 | 2048.756722 | 2048.756722 | 0.000000 | True | 1 | 1965 | 1.898507 | 545.898438 |
| P | max | 10 | 2061.944625 | 2061.944625 | 0.000000 | True | 1 | 3870 | 1.959280 | 545.898438 |
| P | max | 20 | 2077.992468 | 2077.992468 | 0.000000 | True | 1 | 7660 | 2.086865 | 545.898438 |
| P | max | 30 | 2091.745296 | 2091.745296 | 0.000000 | True | 1 | 11490 | 2.195291 | 545.898438 |
| U | min | 1 | 2023.303311 | 2023.303311 | -0.000000 | True | 1 | 399 | 1.865811 | 529.449219 |
| U | min | 2 | 2018.995571 | 2018.995571 | -0.000000 | True | 1 | 794 | 1.866809 | 545.875000 |
| U | min | 3 | 2016.500485 | 2016.500485 | -0.000000 | True | 1 | 1185 | 1.877298 | 545.875000 |
| U | min | 4 | 2014.038699 | 2014.038699 | -0.000000 | True | 1 | 1572 | 1.888651 | 545.878906 |
| U | min | 5 | 2012.219707 | 2012.219707 | -0.000000 | True | 1 | 1965 | 1.910495 | 545.878906 |
| U | min | 10 | 2004.715419 | 2004.715419 | -0.000000 | True | 1 | 3890 | 1.965614 | 545.878906 |
| U | min | 20 | 1991.962672 | 1991.962672 | -0.000000 | True | 1 | 7700 | 2.081396 | 545.878906 |
| U | min | 30 | 1980.341463 | 1980.341463 | -0.000000 | True | 1 | 11370 | 2.199756 | 545.878906 |


### Claim verdicts and marginal increases

Positive signed gaps mean greedy is better. Strict superiority at every requested budget requires every gap > 1e-8. Increasing greedy marginals alone do not prove non-submodularity.

```json
{
  "marginal_increases": {
    "U_max": 3,
    "P_max": 2,
    "U_min": 1
  },
  "greedy_minus_baselines": {
    "U_max": {
      "1": {
        "I": 3.503644903227041,
        "II": 3.4016912024310386
      },
      "2": {
        "I": 6.374009159375646,
        "II": 6.3720705774746875
      },
      "3": {
        "I": 8.888811452744903,
        "II": 9.65754849390055
      },
      "4": {
        "I": 11.284061305004116,
        "II": 12.851557157742718
      },
      "5": {
        "I": 13.570180552913826,
        "II": 15.464933534075499
      },
      "10": {
        "I": 20.641456065672628,
        "II": 24.145338857885235
      },
      "20": {
        "I": 29.14837881580138,
        "II": 36.35182784676135
      },
      "30": {
        "I": 36.15796201985313,
        "II": 45.01488624140029
      }
    },
    "P_max": {
      "1": {
        "I": 18.258413886406743,
        "II": 18.506526104919885
      },
      "2": {
        "I": 25.230504083539017,
        "II": 25.526944417810455
      },
      "3": {
        "I": 31.518462914790916,
        "II": 32.133745651369736
      },
      "4": {
        "I": 35.568856442508604,
        "II": 36.77626065434242
      },
      "5": {
        "I": 39.81718457765692,
        "II": 41.13707348909452
      },
      "10": {
        "I": 51.14114862001452,
        "II": 53.09822102934936
      },
      "20": {
        "I": 62.15650359148685,
        "II": 68.60843468021721
      },
      "30": {
        "I": 70.90109362291992,
        "II": 81.07807538429051
      }
    },
    "U_min": {
      "1": {
        "I": 11.00419472133899,
        "II": 10.984120590305793
      },
      "2": {
        "I": 14.98026596485306,
        "II": 15.045907924660014
      },
      "3": {
        "I": 17.255389777206346,
        "II": 17.19908331304873
      },
      "4": {
        "I": 19.62776241387246,
        "II": 19.512919602411102
      },
      "5": {
        "I": 21.092248260004908,
        "II": 21.112397647168564
      },
      "10": {
        "I": 26.380005206299984,
        "II": 27.31232810633469
      },
      "20": {
        "I": 33.882793688021366,
        "II": 38.14685116178907
      },
      "30": {
        "I": 40.87098659537378,
        "II": 45.415027327548614
      }
    }
  }
}
```


### Validation evidence

Two-node analytic checks and endpoint sweeps: `/home/chrisrex/acads/soccomp/asgn2/results-local/checks/task1_checks.json`. Full small-graph validation: `/home/chrisrex/acads/soccomp/asgn2/results-local/checks/validation.json`. Paper triangle: 

```json
{
  "sets": [
    [],
    [
      1
    ],
    [
      3
    ],
    [
      1,
      2
    ],
    [
      1,
      3
    ],
    [
      1,
      2,
      3
    ]
  ],
  "f": [
    1.0000000000000004,
    2.6363636363636367,
    1.4927824788451978,
    2.8047018607501046,
    2.804701860750105,
    2.9960039960039966
  ],
  "nonsubmodular": {
    "small": 0.1683382243864684,
    "large": 0.19130213525389195
  },
  "nonsupermodular": {
    "small": 0.4927824788451973,
    "large": 0.1683382243864684
  }
}
```


### Figures to use

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/objective_U_max.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/targets_U_max.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/bonus_U_max.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/objective_P_max.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/targets_P_max.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/bonus_P_max.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/objective_U_min.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/targets_U_min.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/bonus_U_min.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/runtime_U.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/runtime_P.png`

- `/home/chrisrex/acads/soccomp/asgn2/results-local/figures/endpoint_sweeps.png`


### Recorded commands

```text
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_greedy_task1.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile U
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method fast --profile U --pool shortlist --k 10
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_greedy_task1.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile P
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method fast --profile P --pool shortlist --k 10
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method fast --profile U --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method I --profile U --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method II --profile U --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method random --profile U --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method fast --profile P --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method I --profile P --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method II --profile P --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method random --profile P --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method fast --profile U --direction min
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method I --profile U --direction min
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method II --profile U --direction min
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method random --profile U --direction min
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method local --profile U --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method local --profile U --direction min
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method witness --profile U
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_groundtruth_task3.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile U --restricted-comparisons
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method local --profile P --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method local --profile P --direction min
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_paper_task2.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --method witness --profile P
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_groundtruth_task3.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile P --restricted-comparisons
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_bonus_task4.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile U --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_bonus_task4.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile P --direction max
/home/bt3/23CS10012/soccomp/asgn2/.venv/bin/python /home/bt3/23CS10012/soccomp/asgn2/Assignment2_23CS10012_bonus_task4.py --dataset /home/bt3/23CS10012/soccomp/asgn2/data/facebook_combined.txt --output /home/bt3/23CS10012/soccomp/asgn2/results-local --profile U --direction min
```
